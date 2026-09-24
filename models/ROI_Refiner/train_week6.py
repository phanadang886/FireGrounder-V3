"""Week 6 training pipeline for fire classification + pixel localisation.

The old notebook trained one globally pooled vector to predict confidence and
coordinates. This script fixes the main problems:

* source train/test folders are kept separate when present;
* duplicate records and unusable Kaggle paths are resolved/deduplicated;
* horizontal flipping updates x_norm together with the image;
* confidence uses BCEWithLogitsLoss;
* coordinate loss is masked to positive samples only;
* a spatial heatmap head preserves location information and uses soft-argmax;
* validation/test report pixel MAE, PCK@10 and PCK@25.

Example (from the project directory):

    python train_week6.py --epochs 30 --device cuda

The new checkpoint has architecture="spatial_heatmap_v2" and is intentionally
not silently loaded by the old GAP-based FireDetector.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
import timm


ARCHITECTURE = "spatial_heatmap_v2"
IMAGE_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class Record:
    image_path: str
    has_fire: int
    x_norm: float
    y_norm: float
    source_split: str
    group: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _source_split(path_text: str) -> str:
    parts = Path(path_text.replace("\\", "/")).parts
    for split in ("train", "val", "test"):
        if split in parts:
            return split
    return "unknown"


def _group_name(path_text: str) -> str:
    """Use the nearest dataset folder as a conservative scene/video group."""
    p = Path(path_text.replace("\\", "/"))
    parts = list(p.parts)
    for marker in ("train", "test", "val"):
        if marker in parts:
            idx = parts.index(marker)
            # Parent class is useful for stratification but not a scene id.
            return "/".join(parts[max(0, idx - 2):idx]) or marker
    return "/".join(parts[-3:-1]) or "unknown"


def _build_image_index(dataset_root: Path) -> Dict[str, Path]:
    """Index image paths by ``train/class/file`` under any dataset nesting."""
    index: Dict[str, Path] = {}
    if not dataset_root.exists() or not dataset_root.is_dir():
        return index
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    split_names = {"train", "test", "val"}
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_extensions:
            continue
        parts = [part.lower() for part in path.parts]
        for index_position, part in enumerate(parts):
            if part not in split_names or index_position + 2 >= len(parts):
                continue
            relative = "/".join(parts[index_position:])
            # A class folder and filename are required; this prevents a
            # random unrelated image with the same filename from matching.
            if relative.count("/") >= 2:
                index.setdefault(relative, path.resolve())
                index.setdefault("img_data/" + relative, path.resolve())
            break
    return index


def resolve_image_path(raw_path: str, dataset_root: Path,
                       image_index: Optional[Dict[str, Path]] = None) -> Optional[Path]:
    """Resolve Kaggle-style paths against arbitrarily nested image datasets."""
    raw = Path(raw_path)
    candidates = [raw]
    normalized = raw_path.replace("\\", "/")
    match = re.search(r"(?:^|/)img_data/(train|test|val)/([^/]+/[^/]+)$", normalized)
    if match:
        split, rel = match.groups()
        if image_index:
            for key in (f"{split}/{rel}", f"img_data/{split}/{rel}"):
                indexed = image_index.get(key.lower())
                if indexed is not None and indexed.is_file():
                    return indexed
        candidates.extend([
            dataset_root / split / rel,
            dataset_root / "img_data" / split / rel,
            dataset_root / "data" / "data" / "img_data" / split / rel,
            dataset_root / "data" / "img_data" / split / rel,
        ])
    candidates.extend([
        dataset_root / raw.name,
        dataset_root / "data" / "data" / "img_data" / raw.name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_records(labels_path: Path, dataset_root: Path) -> Tuple[List[Record], Dict[str, int]]:
    """Load labels, resolve paths and remove duplicate physical images."""
    raw_items = json.loads(labels_path.read_text(encoding="utf-8"))
    records: List[Record] = []
    seen = set()
    stats = Counter()
    image_index = _build_image_index(dataset_root)
    stats["indexed_images"] = len(set(image_index.values()))
    for item in raw_items:
        resolved = resolve_image_path(str(item["image_path"]), dataset_root, image_index)
        if resolved is None:
            stats["unresolved"] += 1
            continue
        key = str(resolved).lower()
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        has_fire = int(item.get("has_fire", 1) > 0.5)
        xy = item.get("p_fire", [0.0, 0.0]) if has_fire else [0.0, 0.0]
        records.append(Record(
            image_path=str(resolved),
            has_fire=has_fire,
            x_norm=float(np.clip(xy[0], 0.0, 1.0)),
            y_norm=float(np.clip(xy[1], 0.0, 1.0)),
            source_split=_source_split(str(item["image_path"])),
            group=_group_name(str(item["image_path"])),
        ))
    stats["loaded"] = len(records)
    stats["fire"] = sum(r.has_fire for r in records)
    stats["no_fire"] = len(records) - stats["fire"]
    return records, dict(stats)


def _stratified_take(records: Sequence[Record], ratio: float, rng: random.Random) -> Tuple[List[Record], List[Record]]:
    by_class: Dict[int, List[Record]] = defaultdict(list)
    for record in records:
        by_class[record.has_fire].append(record)
    selected, remaining = [], []
    for cls_records in by_class.values():
        shuffled = list(cls_records)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * ratio))) if len(shuffled) > 1 else 0
        selected.extend(shuffled[:count])
        remaining.extend(shuffled[count:])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def split_records(records: Sequence[Record], seed: int = 42) -> Dict[str, List[Record]]:
    """Create leakage-safe splits.

    When source train/test metadata exists, all source-test records remain in
    test and only source-train records are split into train/validation. This is
    preferable to mixing an official test set into training. If source folders
    are unavailable, a stratified 70/15/15 split is used.
    """
    rng = random.Random(seed)
    source_train = [r for r in records if r.source_split == "train"]
    source_test = [r for r in records if r.source_split == "test"]
    if source_train and source_test:
        val, train = _stratified_take(source_train, 0.15, rng)
        return {"train": train, "val": val, "test": list(source_test)}

    shuffled = list(records)
    rng.shuffle(shuffled)
    # Stratify through two-stage class-wise selection.
    test, rest = _stratified_take(shuffled, 0.15, rng)
    val, train = _stratified_take(rest, 0.15 / 0.85, rng)
    return {"train": train, "val": val, "test": test}


class FireDataset(Dataset):
    """Image/label dataset with synchronized geometry augmentation."""

    def __init__(self, records: Sequence[Record], train: bool = False,
                 image_size: Tuple[int, int] = IMAGE_SIZE, flip_p: float = 0.3):
        self.records = list(records)
        self.train = train
        self.image_size = image_size
        self.flip_p = flip_p

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        original_size = torch.tensor([image.width, image.height], dtype=torch.float32)
        x, y = record.x_norm, record.y_norm

        # The label is transformed together with the image. Color changes do
        # not alter geometry, while horizontal flipping requires x <- 1-x.
        if self.train and random.random() < self.flip_p:
            image = TF.hflip(image)
            if record.has_fire:
                x = 1.0 - x
        if self.train:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, MEAN, STD)
        target = torch.tensor([float(record.has_fire), x, y], dtype=torch.float32)
        return tensor, target, original_size, record.image_path


class SpatialFireModel(nn.Module):
    """MobileNetV4 feature map + spatial heatmap/coordinate head.

    The coordinate branch never uses global average pooling. It upsamples the
    final backbone map to a 14x14 spatial map, predicts a heatmap and obtains
    (x,y) with soft-argmax. The confidence branch uses log-sum-exp over a
    class activation map instead of a pooled linear regression head.
    """

    def __init__(self, backbone: str = "mobilenetv4_conv_medium", temperature: float = 0.07,
                 pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.temperature = temperature
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        feature_info = getattr(self.backbone, "feature_info", None)
        if feature_info is not None:
            last_info = feature_info[-1]
            channels = int(last_info["num_chs"] if isinstance(last_info, dict) else last_info.num_chs)
        else:
            with torch.no_grad():
                channels = int(self.backbone.forward_features(torch.zeros(1, 3, 224, 224)).shape[1])
        self.spatial_neck = nn.Sequential(
            nn.Conv2d(channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.coord_head = nn.Conv2d(256, 1, 1)
        self.class_head = nn.Conv2d(256, 1, 1)

    def _soft_argmax(self, logits: torch.Tensor) -> torch.Tensor:
        b, _, h, w = logits.shape
        flat = logits.flatten(1) / self.temperature
        probs = F.softmax(flat, dim=1).view(b, h, w)
        xs = torch.linspace(0.0, 1.0, w, device=logits.device)
        ys = torch.linspace(0.0, 1.0, h, device=logits.device)
        x = (probs * xs.view(1, 1, w)).sum(dim=(1, 2))
        y = (probs * ys.view(1, h, 1)).sum(dim=(1, 2))
        return torch.stack([x, y], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone.forward_features(x)
        features = self.spatial_neck(features)
        features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
        coord_logits = self.coord_head(features)
        class_map = self.class_head(features)
        # LogSumExp is a smooth "any fire region" aggregator and retains a
        # spatial activation map, unlike the old GAP -> Linear coordinate head.
        confidence_logit = torch.logsumexp(class_map.flatten(1), dim=1) - math.log(class_map.shape[-1] * class_map.shape[-2])
        return {
            "confidence_logit": confidence_logit,
            "coord": self._soft_argmax(coord_logits),
            "coord_logits": coord_logits,
            "class_map": class_map,
        }


def gaussian_targets(target_xy: torch.Tensor, positive: torch.Tensor, height: int, width: int, sigma: float = 1.5) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=target_xy.device, dtype=target_xy.dtype),
        torch.arange(width, device=target_xy.device, dtype=target_xy.dtype),
        indexing="ij",
    )
    cx = target_xy[:, 0] * max(width - 1, 1)
    cy = target_xy[:, 1] * max(height - 1, 1)
    heat = torch.exp(-((xx.unsqueeze(0) - cx[:, None, None]) ** 2 + (yy.unsqueeze(0) - cy[:, None, None]) ** 2) / (2 * sigma ** 2))
    return heat.unsqueeze(1) * positive[:, None, None, None]


class FireLoss(nn.Module):
    def __init__(self, lambda_coord: float = 5.0, lambda_heatmap: float = 0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_heatmap = lambda_heatmap
        self.confidence = nn.BCEWithLogitsLoss()
        self.coordinate = nn.SmoothL1Loss(reduction="none")

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        positive = targets[:, 0] > 0.5
        conf_loss = self.confidence(outputs["confidence_logit"], targets[:, 0])
        if positive.any():
            coord_loss = self.coordinate(outputs["coord"][positive], targets[positive, 1:]).mean()
        else:
            coord_loss = outputs["coord"].sum() * 0.0
        b, _, h, w = outputs["coord_logits"].shape
        target_heat = gaussian_targets(targets[:, 1:], positive.float(), h, w)
        heat_loss = F.mse_loss(torch.sigmoid(outputs["coord_logits"]), target_heat)
        total = conf_loss + self.lambda_coord * coord_loss + self.lambda_heatmap * heat_loss
        return {"total": total, "confidence": conf_loss, "coordinate": coord_loss, "heatmap": heat_loss}


@torch.no_grad()
def metrics(outputs: Dict[str, torch.Tensor], targets: torch.Tensor, sizes: torch.Tensor) -> Dict[str, float]:
    pred_conf = torch.sigmoid(outputs["confidence_logit"])
    cls_true = targets[:, 0] > 0.5
    cls_pred = pred_conf >= 0.5
    tp = int((cls_pred & cls_true).sum().item())
    fp = int((cls_pred & ~cls_true).sum().item())
    fn = int(((~cls_pred) & cls_true).sum().item())
    tn = int(((~cls_pred) & (~cls_true)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = {
        "accuracy": float((cls_pred == cls_true).float().mean().item()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }
    if not cls_true.any():
        result.update({"mae_px": float("nan"), "pck10": float("nan"), "pck25": float("nan")})
        return result
    pred = outputs["coord"][cls_true]
    truth = targets[cls_true, 1:]
    wh = sizes[cls_true]
    error = torch.sqrt((((pred - truth) * wh) ** 2).sum(dim=1).clamp_min(1e-12))
    result.update({
        "mae_px": float(error.mean().item()),
        "pck10": float((error <= 10).float().mean().item()),
        "pck25": float((error <= 25).float().mean().item()),
    })
    return result


def _mean_dict(values: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(values)
    keys = items[0].keys() if items else []
    return {key: float(np.nanmean([x[key] for x in items])) for key in keys}


def _exact_metrics(confidence, coords, targets, sizes):
    """Compute sample-weighted metrics once per epoch, not once per batch."""
    pred_conf = torch.sigmoid(confidence)
    cls_true = targets[:, 0] > 0.5
    cls_pred = pred_conf >= 0.5
    tp = int((cls_pred & cls_true).sum().item())
    fp = int((cls_pred & ~cls_true).sum().item())
    fn = int(((~cls_pred) & cls_true).sum().item())
    tn = int(((~cls_pred) & (~cls_true)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = {
        "accuracy": float((cls_pred == cls_true).float().mean()),
        "precision": float(precision), "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(1e-12, precision + recall)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
    }
    if not cls_true.any():
        result.update({"mae_px": float("nan"), "pck10": float("nan"), "pck25": float("nan")})
        return result
    error = torch.linalg.vector_norm((coords[cls_true] - targets[cls_true, 1:]) * sizes[cls_true], dim=1)
    result.update({
        "mae_px": float(error.mean()),
        "pck10": float((error <= 10).float().mean()),
        "pck25": float((error <= 25).float().mean()),
    })
    return result


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, confidence_values, coordinate_values, target_values, size_values = [], [], [], [], []
    for images, targets, sizes, _ in loader:
        images, targets, sizes = images.to(device), targets.to(device), sizes.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        # AMP is selected from the device here; callers do not need an amp
        # keyword, which keeps the training API stable on Kaggle and locally.
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
        if training:
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append({key: float(value.detach().cpu()) for key, value in loss_dict.items()})
        confidence_values.append(outputs["confidence_logit"].detach().float().cpu())
        coordinate_values.append(outputs["coord"].detach().float().cpu())
        target_values.append(targets.detach().float().cpu())
        size_values.append(sizes.detach().float().cpu())
    if not losses:
        raise RuntimeError("DataLoader produced no batches")
    return (
        _mean_dict(losses),
        _exact_metrics(
            torch.cat(confidence_values), torch.cat(coordinate_values),
            torch.cat(target_values), torch.cat(size_values)
        ),
    )


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, val_loss: float, split_info, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": ARCHITECTURE,
        "backbone": model.backbone_name,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "image_size": IMAGE_SIZE,
        "split_info": split_info,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=root / "fire-detection-from-cctv")
    parser.add_argument("--output-dir", type=Path, default=root / "fire-model-data" / "week6_spatial")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="one batch train/validation sanity check")
    parser.add_argument("--no-pretrained", action="store_true", help="do not download/use timm weights; useful for smoke tests")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    records, load_stats = load_records(args.labels, args.dataset_root)
    splits = split_records(records, args.seed)
    split_info = {name: {"count": len(items), "fire": sum(r.has_fire for r in items)} for name, items in splits.items()}
    print(f"device={device}")
    print(f"load_stats={load_stats}")
    print(f"split_info={split_info}")
    if not all(split_info[name]["count"] for name in ("train", "val", "test")):
        raise RuntimeError(f"Empty split: {split_info}")

    train_loader = DataLoader(FireDataset(splits["train"], train=True), args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(FireDataset(splits["val"]), args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(FireDataset(splits["test"]), args.batch_size, shuffle=False, num_workers=args.workers)
    model = SpatialFireModel(pretrained=not args.no_pretrained).to(device)
    criterion = FireLoss(lambda_coord=5.0, lambda_heatmap=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, args.epochs), eta_min=args.lr / 100)

    if args.smoke_test:
        train_loss, train_metric = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metric = run_epoch(model, val_loader, criterion, device)
        print(f"smoke_train_loss={train_loss}")
        print(f"smoke_train_metric={train_metric}")
        print(f"smoke_val_loss={val_loss}")
        print(f"smoke_val_metric={val_metric}")
        return

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metric = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metric = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        line = {"epoch": epoch, "train": train_loss, "val": val_loss, "train_metric": train_metric, "val_metric": val_metric}
        history.append(line)
        print(
            f"epoch={epoch:03d} train={train_loss['total']:.5f} val={val_loss['total']:.5f} "
            f"val_mae_px={val_metric['mae_px']:.2f} val_pck10={val_metric['pck10']:.3f} "
            f"val_pck25={val_metric['pck25']:.3f}"
        )
        if val_loss["total"] < best:
            best = val_loss["total"]
            save_checkpoint(args.output_dir / "best_spatial.pth", model, optimizer, scheduler, epoch, best, split_info, args)
        save_checkpoint(args.output_dir / "last_spatial.pth", model, optimizer, scheduler, epoch, val_loss["total"], split_info, args)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    test_loss, test_metric = run_epoch(model, test_loader, criterion, device)
    print(f"test_loss={test_loss}")
    print(f"test_metric={test_metric}")
    (args.output_dir / "test_metrics.json").write_text(json.dumps({"loss": test_loss, "metric": test_metric}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
