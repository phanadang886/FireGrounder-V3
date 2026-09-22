"""
label_sam3.py
=============
Tu dong tao nhan (x_norm, y_norm) cho tung anh trong dataset
bang SAM 2/3 + Adaptive Baseline algorithm.

Chay tren Kaggle:
    python label_sam3.py --img_dir /kaggle/input/fire/images \
                         --out_dir /kaggle/working/dataset \
                         --sam_ckpt /kaggle/input/sam2/sam2.1_hiera_large.pt \
                         --sam_cfg  sam2.1_hiera_l.yaml

Output:
    dataset/
        images/          <-- symlink hoac copy anh goc
        annotations.json <-- [{"image": "xx.jpg", "x_norm": 0.42, "y_norm": 0.81}, ...]
        train_annotations.json
        val_annotations.json
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Adaptive Baseline: tim diem day lua tu binary mask
# --------------------------------------------------------------------------
def adaptive_baseline(mask: np.ndarray,
                       alpha: float = 0.35,
                       jump_thresh: float = 1.8) -> tuple[float, float] | None:
    """
    Tim toa do chuan hoa cua diem tiep xuc day lua voi san.

    Thuat toan:
      1. Giu lai phan duoi (alpha * height) cua mask de loai than lua
      2. Quyet liet loc cac pixel co gradient doc qua cao (vach dung lua)
      3. Lay X trung binh va Y cao nhat cua vung con lai

    Args:
        mask         : uint8 ndarray (H, W), 0 hoac 255
        alpha        : Ty le giu lai tu day mask (default 35%)
        jump_thresh  : Nguong gradient doc de loc vach lua (default 1.8)

    Returns:
        (x_norm, y_norm) hoac None neu mask qua nho/khong hop le
    """
    ys, xs = np.where(mask > 0)
    if len(ys) < 50:
        return None

    g_ymin, g_ymax = int(ys.min()), int(ys.max())
    if g_ymax == g_ymin:
        return None

    # Giu phan duoi
    y_cut  = g_ymax - alpha * (g_ymax - g_ymin)
    bot    = ys >= y_cut
    bys, bxs = ys[bot], xs[bot]

    if len(bys) < 10:
        return None

    # Loc jump doc
    order = np.argsort(bxs)
    cx, cy = [bxs[order[0]]], [bys[order[0]]]
    for i in order[1:]:
        dy = abs(bys[i] - cy[-1])
        dx = abs(bxs[i] - cx[-1]) + 1e-6
        if dy / dx <= jump_thresh:
            cx.append(bxs[i])
            cy.append(bys[i])

    if len(cy) < 5:
        return None

    H, W = mask.shape
    x_norm = float(np.mean(cx)) / W
    y_norm = float(np.max(cy))  / H
    return round(x_norm, 6), round(y_norm, 6)


# --------------------------------------------------------------------------
# SAM3 labeling
# --------------------------------------------------------------------------
def label_with_sam(img_dir: str,
                    out_dir: str,
                    sam_ckpt: str,
                    sam_cfg: str,
                    device: str = "cuda",
                    alpha: float = 0.35) -> list[dict]:
    """
    Chay SAM2/3 tren toan bo anh, trich diem day lua, luu JSON.
    """
    # Import SAM2 (phai cai truoc: pip install sam2)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(
        build_sam2(sam_cfg, sam_ckpt, device=device)
    )

    out_dir  = Path(out_dir)
    img_out  = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    paths  = [p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts]
    print(f"[INFO] Found {len(paths)} images in {img_dir}")

    annotations = []
    skipped     = 0

    for i, img_path in enumerate(paths):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W   = img_np.shape[:2]

        # Prompt: diem 70% chieu cao anh (vung co lua)
        pt  = np.array([[W // 2, int(H * 0.70)]])
        lbl = np.array([1])

        predictor.set_image(img_np)
        masks, scores, _ = predictor.predict(
            point_coords=pt,
            point_labels=lbl,
            multimask_output=True,
        )

        # Chon mask tot nhat
        best  = int(scores.argmax())
        mask  = (masks[best] * 255).astype(np.uint8)

        result = adaptive_baseline(mask, alpha=alpha)
        if result is None:
            skipped += 1
            continue

        x_norm, y_norm = result
        fname = img_path.name

        # Sao chep anh sang output
        shutil.copy2(img_path, img_out / fname)

        annotations.append({
            "image"    : fname,
            "x_norm"   : x_norm,
            "y_norm"   : y_norm,
            "sam_score": round(float(scores[best]), 4),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(paths)}] OK={len(annotations)}  Skip={skipped}")

    print(f"[INFO] Labeled: {len(annotations)} | Skipped: {skipped}")
    return annotations


def split_and_save(annotations: list[dict],
                    out_dir: str,
                    val_ratio: float = 0.15,
                    seed: int = 42):
    """Chia train/val va luu ra JSON."""
    random.seed(seed)
    data = annotations.copy()
    random.shuffle(data)

    n_val   = max(1, int(len(data) * val_ratio))
    val_set = data[:n_val]
    trn_set = data[n_val:]

    out = Path(out_dir)
    with open(out / "annotations.json",       "w") as f: json.dump(data,    f, indent=2)
    with open(out / "train_annotations.json", "w") as f: json.dump(trn_set, f, indent=2)
    with open(out / "val_annotations.json",   "w") as f: json.dump(val_set, f, indent=2)

    print(f"[INFO] Saved  train={len(trn_set)}  val={len(val_set)}  -> {out_dir}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",  required=True,  help="Thu muc anh goc")
    ap.add_argument("--out_dir",  default="dataset", help="Thu muc output")
    ap.add_argument("--sam_ckpt", required=True,  help="Duong dan sam2.1_*.pt")
    ap.add_argument("--sam_cfg",  required=True,  help="Ten config yaml (vd: sam2.1_hiera_l.yaml)")
    ap.add_argument("--device",   default="cuda")
    ap.add_argument("--alpha",    type=float, default=0.35)
    ap.add_argument("--val_ratio",type=float, default=0.15)
    args = ap.parse_args()

    anns = label_with_sam(args.img_dir, args.out_dir,
                           args.sam_ckpt, args.sam_cfg,
                           args.device, args.alpha)
    split_and_save(anns, args.out_dir, args.val_ratio)
