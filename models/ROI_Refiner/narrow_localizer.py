"""ROI point-refiner for narrow-scene fire 3D localisation.

This module deliberately does not classify fire. An upstream detector supplies
a coarse 2D point (or bbox/mask). The refiner learns only the image location
of the fire-ground contact point inside a square ROI, then the 3D pipeline
uses that refined point for camera ray casting.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from train_week6 import load_records, save_checkpoint, seed_everything, split_records


ROI_ARCHITECTURE = "narrow_roi_localizer_v1"
ROI_IMAGE_SIZE = (224, 224)
ROI_MEAN = (0.485, 0.456, 0.406)
ROI_STD = (0.229, 0.224, 0.225)


@dataclass
class ROIBox:
    left: float
    top: float
    side: float
    width: int
    height: int


@dataclass
class RefinedPoint:
    """Point returned by the ROI refiner in original-image coordinates."""

    point: tuple[float, float]
    confidence: float
    roi: ROIBox
    heatmap: Optional[np.ndarray] = None


def _crop_square(image: Image.Image, coarse_xy: Sequence[float], roi_fraction: float = 0.70):
    """Crop a square around a coarse point, padding outside image if needed."""
    image = image.convert("RGB")
    width, height = image.size
    side = float(max(32.0, min(width, height) * roi_fraction))
    coarse = np.asarray(coarse_xy, dtype=np.float64).reshape(2)
    left = int(round(float(coarse[0] - side / 2.0)))
    top = int(round(float(coarse[1] - side / 2.0)))
    right = left + int(round(side))
    bottom = top + int(round(side))
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
    crop = image.crop((left + pad_left, top + pad_top, right + pad_left, bottom + pad_top))
    return crop, ROIBox(float(left), float(top), float(side), width, height)


def _prepare_image(image: Image.Image):
    image = image.resize(ROI_IMAGE_SIZE, Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, ROI_MEAN, ROI_STD)


class NarrowROIDataset(Dataset):
    """Positive fire records with synthetic detector-point perturbations.

    Existing labels give the true base point. During training the ROI centre
    is perturbed to simulate an upstream detector. This trains the refiner to
    correct point errors without re-training fire classification.
    """

    def __init__(self, records, train=False, repeats=3, roi_fraction=0.70,
                 noise_std=0.08, seed=42, coarse_manifest=None):
        self.records = list(records)
        self.train = bool(train)
        self.repeats = max(1, int(repeats if train else 1))
        self.roi_fraction = float(roi_fraction)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.coarse_manifest = coarse_manifest or {}

    def __len__(self):
        return len(self.records) * self.repeats

    def __getitem__(self, index):
        record = self.records[index % len(self.records)]
        rng = np.random.default_rng(self.seed + index * 1009)
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt = np.array([record.x_norm * width, record.y_norm * height], dtype=np.float64)
        manifest_item = self.coarse_manifest.get(record.image_path)
        if manifest_item is None:
            manifest_item = self.coarse_manifest.get(Path(record.image_path).name)
        manifest_point = None
        if isinstance(manifest_item, dict):
            manifest_point = manifest_item.get("point")
            manifest_size = manifest_item.get("size", [width, height])
            if manifest_point is not None:
                manifest_point = np.asarray(manifest_point, dtype=np.float64).reshape(2)
                manifest_point = manifest_point / np.maximum(np.asarray(manifest_size, dtype=np.float64), 1.0)
        elif manifest_item is not None:
            manifest_point = np.asarray(manifest_item, dtype=np.float64).reshape(2)
            if np.max(np.abs(manifest_point)) > 1.5:
                manifest_point = manifest_point / np.maximum(np.asarray([width, height], dtype=np.float64), 1.0)

        # Validation/test must also use a coarse point with a controlled
        # perturbation. Evaluating with coarse == ground truth would measure
        # the trivial crop-centre case instead of detector-to-refiner error.
        if manifest_point is not None:
            coarse_norm = np.clip(manifest_point, 0.0, 1.0)
            if self.train and self.noise_std > 0.0:
                coarse_norm = np.clip(coarse_norm + rng.normal(0.0, self.noise_std * 0.25, 2), 0.02, 0.98)
        elif self.noise_std > 0.0:
            if rng.random() < 0.15:
                coarse_norm = np.array([record.x_norm, record.y_norm], dtype=np.float64)
            else:
                coarse_norm = np.clip(
                    np.array([record.x_norm, record.y_norm]) + rng.normal(0.0, self.noise_std, 2),
                    0.02, 0.98,
                )
            if self.train:
                image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
                image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
                image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.90, 1.10)))
        else:
            coarse_norm = np.array([record.x_norm, record.y_norm], dtype=np.float64)
        coarse = coarse_norm * np.array([width, height], dtype=np.float64)
        crop, box = _crop_square(image, coarse, self.roi_fraction)
        target_rel = (gt - np.array([box.left, box.top])) / box.side
        target_rel = np.clip(target_rel, 0.0, 1.0)
        meta = torch.tensor([
            box.left, box.top, box.side, width, height,
            gt[0], gt[1], coarse[0], coarse[1],
        ], dtype=torch.float32)
        target = torch.tensor(target_rel, dtype=torch.float32)
        return _prepare_image(crop), target, meta, record.image_path


class ROIRefiner(nn.Module):
    """Spatial ROI localizer; no fire/no-fire classification head."""

    def __init__(self, backbone="mobilenetv4_conv_medium", pretrained=True,
                 heatmap_size=28, temperature=0.07):
        super().__init__()
        self.backbone_name = backbone
        self.heatmap_size = int(heatmap_size)
        self.temperature = float(temperature)
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        info = getattr(self.backbone, "feature_info", None)
        if info:
            last = info[-1]
            channels = int(last["num_chs"] if isinstance(last, dict) else last.num_chs)
        else:
            with torch.no_grad():
                channels = int(self.backbone.forward_features(torch.zeros(1, 3, 224, 224)).shape[1])
        self.neck = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(256, 1, 1)

    def _soft_argmax(self, logits):
        b, _, h, w = logits.shape
        probs = F.softmax(logits.flatten(1) / self.temperature, dim=1).view(b, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device)
        return torch.stack([
            (probs * xs.view(1, 1, w)).sum((1, 2)),
            (probs * ys.view(1, h, 1)).sum((1, 2)),
        ], dim=1)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.neck(features)
        features = F.interpolate(features, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)
        logits = self.heatmap(features)
        return {"coord": self._soft_argmax(logits), "heatmap_logits": logits}


def load_backbone_from_checkpoint(model: ROIRefiner, checkpoint_path: Path):
    """Reuse only backbone weights from the existing detector checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in source.items()
        if key.startswith("backbone.")
    }
    result = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"loaded_backbone={len(backbone_state)} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
    return model


def _gaussian_target(coords, height, width, sigma=1.5):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=coords.device, dtype=coords.dtype),
        torch.arange(width, device=coords.device, dtype=coords.dtype), indexing="ij"
    )
    cx, cy = coords[:, 0] * (width - 1), coords[:, 1] * (height - 1)
    heat = torch.exp(-((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2) / (2 * sigma ** 2))
    return heat[:, None]


def roi_loss(outputs, targets):
    heat_target = _gaussian_target(targets, outputs["heatmap_logits"].shape[-2], outputs["heatmap_logits"].shape[-1])
    heat_loss = F.mse_loss(torch.sigmoid(outputs["heatmap_logits"]), heat_target)
    point_loss = F.smooth_l1_loss(outputs["coord"], targets)
    return {"total": heat_loss + 2.0 * point_loss, "heatmap": heat_loss, "point": point_loss}


@torch.no_grad()
def roi_metrics(outputs, targets, meta):
    pred = outputs["coord"]
    side = meta[:, 2:3]
    pred_px = torch.cat([meta[:, 0:1], meta[:, 1:2]], 1) + pred * side
    gt_px = meta[:, 5:7]
    error = torch.linalg.vector_norm(pred_px - gt_px, dim=1)
    return {
        "mae_px": float(error.mean()),
        "pck10": float((error <= 10).float().mean()),
        "pck25": float((error <= 25).float().mean()),
    }


class ROIRefinerInference:
    """Inference adapter used between an upstream detector and 3D geometry.

    The checkpoint contains only the ROI refiner. The upstream detector keeps
    responsibility for fire/no-fire classification and supplies ``coarse_xy``.
    """

    def __init__(self, checkpoint_path: Path, device: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ROIRefiner(
            backbone=checkpoint.get("backbone", "mobilenetv4_conv_medium"),
            pretrained=False,
            heatmap_size=int(checkpoint.get("heatmap_size", 28)),
            temperature=float(checkpoint.get("temperature", 0.07)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.roi_fraction = float(checkpoint.get("roi_fraction", 0.70))

    @torch.inference_mode()
    def refine(self, image, coarse_xy: Sequence[float]) -> RefinedPoint:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8)[..., :3])
        image = image.convert("RGB")
        crop, box = _crop_square(image, coarse_xy, self.roi_fraction)
        output = self.model(_prepare_image(crop).unsqueeze(0).to(self.device))
        relative = output["coord"][0].detach().cpu().numpy()
        point = np.array([box.left, box.top], dtype=np.float64) + relative * box.side
        probabilities = torch.sigmoid(output["heatmap_logits"])[0, 0]
        confidence = float(probabilities.max().detach().cpu())
        return RefinedPoint(
            point=(float(point[0]), float(point[1])),
            confidence=confidence,
            roi=box,
            heatmap=probabilities.detach().cpu().numpy(),
        )


def run_roi_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, metrics = [], []
    for images, targets, meta, _ in loader:
        images, targets, meta = images.to(device), targets.to(device), meta.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = roi_loss(outputs, targets)
        if training:
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append({k: float(v.detach().cpu()) for k, v in loss.items()})
        metrics.append(roi_metrics(outputs, targets, meta))
    keys = losses[0].keys()
    loss_mean = {k: float(np.mean([v[k] for v in losses])) for k in keys}
    metric_mean = {k: float(np.mean([v[k] for v in metrics])) for k in metrics[0]}
    return loss_mean, metric_mean


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records, stats = load_records(args.labels, args.dataset_root)
    splits = split_records(records, args.seed)
    positives = {name: [r for r in items if r.has_fire] for name, items in splits.items()}
    coarse_manifest = {}
    if args.coarse_manifest:
        coarse_manifest = json.loads(args.coarse_manifest.read_text(encoding="utf-8"))
        def has_coarse(record):
            item = coarse_manifest.get(record.image_path, coarse_manifest.get(Path(record.image_path).name))
            return isinstance(item, dict) and item.get("point") is not None
        available_counts = {key: sum(has_coarse(record) for record in value) for key, value in positives.items()}
        print(f"coarse_manifest_positive_splits={available_counts}")
        # A missed upstream detection cannot be repaired by a point refiner.
        # Exclude those records from ROI regression metrics; their recall is
        # still reported by the detector benchmark as a separate metric.
        positives = {key: [record for record in value if has_coarse(record)] for key, value in positives.items()}
    positive_counts = {key: len(value) for key, value in positives.items()}
    if not positive_counts["train"] or not positive_counts["val"] or not positive_counts["test"]:
        raise ValueError(f"Need positive fire records in all splits, got {positive_counts}")
    print(f"device={device} records={stats} positive_splits={positive_counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        "train": DataLoader(NarrowROIDataset(positives["train"], True, args.repeats, args.roi_fraction, args.noise_std, args.seed, coarse_manifest), args.batch_size, True, num_workers=args.workers),
        "val": DataLoader(NarrowROIDataset(positives["val"], False, 1, args.roi_fraction, args.noise_std, args.seed + 1, coarse_manifest), args.batch_size, False, num_workers=args.workers),
        "test": DataLoader(NarrowROIDataset(positives["test"], False, 1, args.roi_fraction, args.noise_std, args.seed + 2, coarse_manifest), args.batch_size, False, num_workers=args.workers),
    }
    # If an existing checkpoint is supplied, avoid an unnecessary network
    # download: its backbone weights are loaded immediately below.
    use_pretrained = not args.no_pretrained and not bool(args.init_checkpoint)
    model = ROIRefiner(pretrained=use_pretrained).to(device)
    if args.init_checkpoint:
        load_backbone_from_checkpoint(model, args.init_checkpoint)
    if args.freeze_epochs > 0:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
        train_loss, train_metric = run_roi_epoch(model, loaders["train"], device, optimizer)
        val_loss, val_metric = run_roi_epoch(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric}
        history.append(row)
        print(f"epoch={epoch:03d} train={train_loss['total']:.5f} val={val_loss['total']:.5f} MAE={val_metric['mae_px']:.2f}px PCK25={val_metric['pck25']:.3f}")
        if val_loss["total"] < best:
            best = val_loss["total"]
            torch.save({
                "architecture": ROI_ARCHITECTURE,
                "backbone": model.backbone_name,
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": best,
                "roi_fraction": args.roi_fraction,
                "heatmap_size": model.heatmap_size,
                "temperature": model.temperature,
            }, args.output_dir / "best_roi.pth")
        (args.output_dir / "history_roi.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(args.output_dir / "best_roi.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metric = run_roi_epoch(model, loaders["test"], device)
    print(f"test_loss={test_loss}")
    print(f"test_metric={test_metric}")
    (args.output_dir / "test_metrics_roi.json").write_text(json.dumps({"loss": test_loss, "metric": test_metric}, indent=2), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=root / "fire-detection-from-cctv")
    parser.add_argument("--init-checkpoint", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument("--coarse-manifest", type=Path, default=None, help="JSON coarse points produced by an upstream detector")
    parser.add_argument("--output-dir", type=Path, default=root / "fire-model-data" / "week6_roi")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--workers", type=int, default=0); parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--freeze-epochs", type=int, default=5); parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--roi-fraction", type=float, default=0.70); parser.add_argument("--noise-std", type=float, default=0.08); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default=None); parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(); train(args)


if __name__ == "__main__": main()
