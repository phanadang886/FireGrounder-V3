"""
FireGrounder V3 - MobileNetV4 FPN-lite heatmap pipeline.

The default flags below are deliberately disabled. Running this file performs
only dependency discovery and a random-input smoke test. Set RUN_TRAIN or
RUN_EVAL to True manually when you want to run an experiment on Kaggle or on
another machine.

Required alongside this file:
    firegrounder_v3.py
    dataset_labels.json and the referenced images for train/eval
"""

# %%
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from firegrounder_v3 import (  # noqa: E402
    BACKBONE_NAME,
    FIRE_THRESHOLD,
    HEATMAP_SIZE,
    IMG_SIZE,
    MEAN,
    STD,
    FireGrounderV3,
    FireLossV3,
    load_checkpoint_v3,
)


# ----------------------------- user configuration ----------------------------
def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


RUN_TRAIN = env_bool("FIRE_RUN_TRAIN", False)
RUN_EVAL = env_bool("FIRE_RUN_EVAL", False)
RUN_PREVIEW = env_bool("FIRE_RUN_PREVIEW", False)

BACKBONE = BACKBONE_NAME
JSON_PATH = os.environ.get("FIRE_JSON_PATH", "")
IMAGE_ROOT = os.environ.get("FIRE_IMAGE_ROOT", "")
CHECKPOINT_PATH = os.environ.get("FIRE_CHECKPOINT_PATH", "")
OUTPUT_DIR = os.environ.get("FIRE_OUTPUT_DIR", "")

SEED = 42
VAL_RATIO = 0.15
NUM_WORKERS = -1               # -1: 2 on Kaggle, 0 on a normal local run.
BATCH_SIZE = 48
EPOCHS = 50
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 3e-4
PATIENCE = 7
THRESHOLD = FIRE_THRESHOLD
W_CLS = 1.5
W_HEATMAP = 5.0
W_COORD = 1.0
GAUSSIAN_SIGMA = 2.0
SOFTARGMAX_TEMPERATURE = 0.07


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_output_dir() -> Path:
    if OUTPUT_DIR:
        path = Path(OUTPUT_DIR)
    elif Path("/kaggle/working").exists():
        path = Path("/kaggle/working")
    else:
        path = PROJECT_DIR / "v3_outputs"
    return path


def find_json() -> Optional[Path]:
    if JSON_PATH:
        path = Path(JSON_PATH)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates: List[Path] = []
    for root in (Path("/kaggle/input"), PROJECT_DIR, Path.cwd()):
        if root.exists():
            candidates.extend(root.glob("**/dataset_labels.json"))
            candidates.extend(root.glob("**/dataset_labels*.json"))
    candidates = list({path.resolve() for path in candidates if path.exists()})
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("fire_ground_dataset" not in str(path), len(str(path))))
    return candidates[0]


def find_checkpoint() -> Optional[Path]:
    if CHECKPOINT_PATH:
        path = Path(CHECKPOINT_PATH)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates: List[Path] = []
    for root in (Path("/kaggle/input"), get_output_dir(), PROJECT_DIR):
        if root.exists():
            candidates.extend(root.glob("**/best_v3_fpn.pth"))
    candidates = list({path.resolve() for path in candidates if path.exists()})
    return candidates[0] if candidates else None


def build_image_index(roots: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                index.setdefault(path.name.lower(), path)
                index.setdefault(path.as_posix().lower(), path)
    return index


def resolve_image(raw_path: str, index: Dict[str, Path]) -> Path:
    direct = Path(raw_path)
    if direct.exists():
        return direct
    normalized = raw_path.replace("\\", "/").lower()
    if normalized in index:
        return index[normalized]
    basename = Path(normalized).name
    if basename in index:
        return index[basename]
    raise FileNotFoundError(f"Image not found: {raw_path}")


class FireDatasetV3(Dataset):
    """Dataset with coordinate-safe horizontal flipping."""

    def __init__(self, json_path: Path, index: Dict[str, Path], split: str):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        records = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a non-empty list in {json_path}")
        order = list(range(len(records)))
        random.Random(SEED).shuffle(order)
        n_val = max(1, int(len(order) * VAL_RATIO))
        selected = order[:n_val] if split == "val" else order[n_val:]
        if not selected:
            raise ValueError("Training split is empty; reduce VAL_RATIO")
        self.records = [records[i] for i in selected]
        self.index = index
        self.split = split
        self.jitter = transforms.ColorJitter(0.4, 0.4, 0.3, 0.08)
        self.erase = transforms.RandomErasing(0.25, (0.02, 0.15), (0.3, 3.3), "random")
        fire_count = sum(int(item.get("has_fire", 1)) == 1 for item in self.records)
        print("[" + split + "] " + str(len(self.records)) + " samples: fire=" + str(fire_count) + ", no-fire=" + str(len(self.records) - fire_count))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.records[item_index]
        image = Image.open(resolve_image(str(item["image_path"]), self.index)).convert("RGB")
        has_fire = float(item.get("has_fire", 1))
        if has_fire > 0.5:
            x_norm, y_norm = item.get("p_fire", [0.0, 0.0])[:2]
            x_norm = max(0.0, min(1.0, float(x_norm)))
            y_norm = max(0.0, min(1.0, float(y_norm)))
        else:
            x_norm, y_norm = 0.0, 0.0

        if self.split == "train":
            image = self.jitter(image)
            if random.random() < 0.5:
                image = TF.hflip(image)
                if has_fire > 0.5:
                    x_norm = 1.0 - x_norm
            if random.random() < 0.3:
                image = TF.adjust_gamma(image, random.uniform(0.7, 1.4))

        image = TF.resize(image, [IMG_SIZE, IMG_SIZE])
        tensor = TF.to_tensor(image)
        if self.split == "train":
            tensor = self.erase(tensor)
        tensor = TF.normalize(tensor, MEAN, STD)
        target = torch.tensor([has_fire, x_norm, y_norm], dtype=torch.float32)
        return tensor, target


def make_loader(dataset: Dataset, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= BATCH_SIZE,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def get_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    pred_fire = predictions[:, 0] >= THRESHOLD
    true_fire = targets[:, 0] > 0.5
    tp = int((pred_fire & true_fire).sum())
    tn = int((~pred_fire & ~true_fire).sum())
    fp = int((pred_fire & ~true_fire).sum())
    fn = int((~pred_fire & true_fire).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    if true_fire.any():
        errors = torch.linalg.vector_norm(
            (predictions[true_fire, 1:3] - targets[true_fire, 1:3]) * IMG_SIZE,
            dim=1,
        )
        mean_error = float(errors.mean())
        median_error = float(errors.median())
        p90_error = float(torch.quantile(errors, 0.9))
        within_5 = float((errors <= 5).float().mean() * 100)
        within_10 = float((errors <= 10).float().mean() * 100)
        within_15 = float((errors <= 15).float().mean() * 100)
    else:
        mean_error = median_error = p90_error = float("nan")
        within_5 = within_10 = within_15 = float("nan")
    return {
        "accuracy": (tp + tn) / max(len(targets), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
        "mean_error_px": mean_error,
        "median_error_px": median_error,
        "p90_error_px": p90_error,
        "within_5px_pct": within_5,
        "within_10px_pct": within_10,
        "within_15px_pct": within_15,
    }


@torch.no_grad()
def evaluate(model: FireGrounderV3, loader: DataLoader, criterion: FireLossV3,
             device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses = 0.0
    count = 0
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, return_dict=True)
        loss, _ = criterion(outputs, labels)
        losses += float(loss) * len(images)
        count += len(images)
        predictions.append(outputs["pred"].cpu())
        targets.append(labels.cpu())
    return losses / max(count, 1), get_metrics(torch.cat(predictions), torch.cat(targets))


def train_epoch(model: FireGrounderV3, loader: DataLoader, criterion: FireLossV3,
                optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses = 0.0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, return_dict=True)
        loss, _ = criterion(outputs, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses += float(loss) * len(images)
        count += len(images)
    return losses / max(count, 1)


def smoke_test(device: torch.device) -> None:
    print("\n[V3 smoke test]")
    model = FireGrounderV3(BACKBONE, pretrained=False, img_size=IMG_SIZE,
                           heatmap_size=HEATMAP_SIZE,
                           softargmax_temperature=SOFTARGMAX_TEMPERATURE).to(device).eval()
    criterion = FireLossV3(W_CLS, W_HEATMAP, W_COORD, gaussian_sigma=GAUSSIAN_SIGMA)
    images = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    labels = torch.tensor([[1.0, 0.25, 0.75], [0.0, 0.0, 0.0]], device=device)
    with torch.no_grad():
        outputs = model(images, return_dict=True)
        loss, _ = criterion(outputs, labels)
    assert outputs["pred"].shape == (2, 3)
    assert outputs["heatmap_logits"].shape == (2, 1, HEATMAP_SIZE, HEATMAP_SIZE)
    assert torch.isfinite(loss).all()
    assert ((outputs["pred"][:, 1:] >= 0) & (outputs["pred"][:, 1:] <= 1)).all()
    print(f"pred={tuple(outputs['pred'].shape)} heatmap={tuple(outputs['heatmap_logits'].shape)} loss={loss.item():.6f}")
    print("Smoke test passed. No training was run.")


def main() -> None:
    set_seed(SEED)
    device = get_device()
    save_dir = get_output_dir()
    workers = 2 if NUM_WORKERS < 0 and Path("/kaggle").exists() else max(NUM_WORKERS, 0)
    print(f"FireGrounder V3 | device={device} | RUN_TRAIN={RUN_TRAIN} | RUN_EVAL={RUN_EVAL}")
    smoke_test(device)

    json_path = find_json()
    if json_path is None:
        print("No dataset_labels JSON found; smoke test only.")
        return
    records = json.loads(json_path.read_text(encoding="utf-8"))
    print("Dataset JSON name=" + json_path.name + " samples=" + str(len(records)))
    if not RUN_TRAIN and not RUN_EVAL:
        print("RUN_TRAIN and RUN_EVAL are False; no data loader or training was run.")
        return

    roots = [json_path.parent, Path("/kaggle/input"), PROJECT_DIR]
    if IMAGE_ROOT:
        roots.insert(0, Path(IMAGE_ROOT))
    index = build_image_index(roots)
    train_set = FireDatasetV3(json_path, index, "train")
    val_set = FireDatasetV3(json_path, index, "val")
    train_loader = make_loader(train_set, True, workers)
    val_loader = make_loader(val_set, False, workers)
    criterion = FireLossV3(W_CLS, W_HEATMAP, W_COORD, gaussian_sigma=GAUSSIAN_SIGMA)
    model: Optional[FireGrounderV3] = None
    best_path = save_dir / "best_v3_fpn.pth"

    if RUN_TRAIN:
        save_dir.mkdir(parents=True, exist_ok=True)
        model = FireGrounderV3(BACKBONE, pretrained=True, img_size=IMG_SIZE,
                               heatmap_size=HEATMAP_SIZE,
                               softargmax_temperature=SOFTARGMAX_TEMPERATURE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-6)
        best_loss = float("inf")
        wait = 0
        history: List[Dict[str, float]] = []
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
            improved = val_loss < best_loss
            print(f"epoch={epoch:02d} train={train_loss:.5f} val={val_loss:.5f} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"F1={metrics['f1']:.3f} FP={int(metrics['fp'])} FN={int(metrics['fn'])}")
            if improved:
                best_loss = val_loss
                wait = 0
                torch.save({
                    "model_version": FireGrounderV3.model_version,
                    "model_state_dict": model.state_dict(),
                    "backbone": BACKBONE,
                    "img_size": IMG_SIZE,
                    "heatmap_size": HEATMAP_SIZE,
                    "hidden_channels": model.hidden_channels,
                    "softargmax_temperature": model.softargmax_temperature,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "history": history,
                }, best_path)
            else:
                wait += 1
                if wait >= PATIENCE:
                    print("Early stopping reached.")
                    break
        (save_dir / "train_history_v3.json").write_text(
            json.dumps(history, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(f"Saved checkpoint: {best_path}")

    if RUN_EVAL:
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = best_path if RUN_TRAIN else find_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("RUN_EVAL=True but no V3 checkpoint was found")
        model = load_checkpoint_v3(str(checkpoint), device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        report = {"checkpoint": str(checkpoint), "dataset": str(json_path),
                  "resolution": IMG_SIZE, "threshold": THRESHOLD,
                  "val_loss": val_loss, "metrics": metrics}
        report_path = save_dir / "v3_eval_metrics.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
        print(f"Evaluation saved: {report_path}")

    if RUN_PREVIEW:
        save_dir.mkdir(parents=True, exist_ok=True)
        if model is None:
            raise RuntimeError("RUN_PREVIEW=True requires RUN_TRAIN or RUN_EVAL")
        import matplotlib.pyplot as plt

        count = min(12, len(val_set))
        indices = random.Random(999).sample(range(len(val_set)), count)
        columns = 4
        rows = max(1, (count + columns - 1) // columns)
        figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
        axes = np.asarray(axes).reshape(-1)
        inverse = transforms.Normalize(
            mean=[-mean / std for mean, std in zip(MEAN, STD)],
            std=[1.0 / std for std in STD],
        )
        model.eval()
        for axis, index in zip(axes, indices):
            tensor, target = val_set[index]
            prediction = model(tensor.unsqueeze(0).to(device))[0].cpu().numpy()
            axis.imshow(inverse(tensor).permute(1, 2, 0).numpy().clip(0, 1))
            if target[0] > 0.5:
                axis.scatter(target[1] * IMG_SIZE, target[2] * IMG_SIZE,
                             c="lime", s=90, label="GT")
            if prediction[0] >= THRESHOLD:
                axis.scatter(prediction[1] * IMG_SIZE, prediction[2] * IMG_SIZE,
                             c="red", marker="x", s=110, label="V3")
            axis.set_title(f"p={prediction[0]:.2f}")
            axis.axis("off")
            if target[0] > 0.5 or prediction[0] >= THRESHOLD:
                axis.legend(fontsize=7, loc="upper right")
        for axis in axes[count:]:
            axis.axis("off")
        figure.tight_layout()
        preview_path = save_dir / "v3_val_preview.png"
        figure.savefig(preview_path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
