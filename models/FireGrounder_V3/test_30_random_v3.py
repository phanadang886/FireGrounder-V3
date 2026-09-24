"""
Script chạy thử nghiệm 30 ảnh ngẫu nhiên trong thư mục 'fire-samples' bằng mô hình FireGrounder V3.
- Tự động nhận diện thư mục dù bạn chạy từ thư mục gốc hay thư mục con FireGrounder-V3-main.
- Load checkpoint: best_v3_fpn.pth
- Sử dụng ngưỡng tùy chọn: --threshold 0.10 (hoặc 0.05)
- Vẽ tọa độ gốc lửa (Fire Base Contact Point) lên ảnh và lưu thành ảnh preview.
"""

import os
import sys
import random
import argparse
from pathlib import Path

# Thiết lập stdout UTF-8 cho Windows console
sys.stdout.reconfigure(encoding="utf-8")

# ─── 1. Tự động tìm đường dẫn linh hoạt ───────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 1. Tìm thư mục fire-samples
candidate_sample_dirs = [
    PROJECT_ROOT / "datasets" / "fire-samples",
    SCRIPT_DIR / "fire-samples",
    PROJECT_ROOT / "fire-samples",
    Path.cwd() / "datasets" / "fire-samples",
]

SAMPLES_DIR = None
for p in candidate_sample_dirs:
    if p.exists() and p.is_dir():
        SAMPLES_DIR = p
        break

if SAMPLES_DIR is None:
    raise FileNotFoundError("❌ Không tìm thấy thư mục 'fire-samples' trong datasets/fire-samples!")

# 2. Tìm file firegrounder_v3.py
CODE_DIR = SCRIPT_DIR
if not (CODE_DIR / "firegrounder_v3.py").exists():
    for base in [SCRIPT_DIR, PROJECT_ROOT / "models" / "FireGrounder_V3", Path.cwd()]:
        if (base / "firegrounder_v3.py").exists():
            CODE_DIR = base
            break

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 3. Tìm checkpoint V3
candidate_ckpts = [
    SCRIPT_DIR / "best_v3_fpn.pth",
    PROJECT_ROOT / "models" / "FireGrounder_V3" / "best_v3_fpn.pth",
    SCRIPT_DIR / "v3_outputs" / "best_v3_fpn.pth",
]

CKPT_PATH = None
for p in candidate_ckpts:
    if p.exists():
        CKPT_PATH = p
        break

if CKPT_PATH is None:
    raise FileNotFoundError("❌ Không tìm thấy checkpoint V3 'best_v3_fpn.pth'!")


# ─── 2. Import thư viện AI ───────────────────────────────────────────────────
try:
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from firegrounder_v3 import FireGrounderV3, MEAN, STD
except ModuleNotFoundError as e:
    print("\n" + "=" * 80)
    print("❌ LỖI THIẾU THƯ VIỆN PYTORCH / TIMM:")
    print(f"   {e}")
    print("\n👉 NGUYÊN NHÂN: Terminal đang chạy bằng Python mặc định của Windows thay vì Conda!")
    print("👉 CÁCH KHẮC PHỤC (chọn 1 trong 2 cách):")
    print("   Cách 1: Kích hoạt môi trường conda trước:")
    print("           conda activate firegrounder")
    print(f"           python {Path(__file__).name} --threshold 0.10")
    print("   Cách 2: Chạy trực tiếp bằng python của Conda:")
    print(f"           & \"C:\\Users\\TOTIN\\miniconda3\\envs\\firegrounder\\python.exe\" {Path(__file__).name} --threshold 0.10")
    print("=" * 80 + "\n")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm 30 ảnh ngẫu nhiên với FireGrounder V3")
    parser.add_argument("--threshold", type=float, default=0.05, help="Ngưỡng phân loại lửa (mặc định: 0.05)")
    parser.add_argument("--num-samples", type=int, default=30, help="Số lượng ảnh ngẫu nhiên (mặc định: 30)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None để ngẫu nhiên mỗi lần)")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ matplotlib trực tiếp")
    return parser.parse_args()

args = parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
FIRE_THRESHOLD = args.threshold
NUM_SAMPLES = args.num_samples
SEED = args.seed

# ─── 3. Tải mô hình V3 ───────────────────────────────────────────────────────
print("=" * 82)
print(f"🚀 Thiết bị chạy       : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU                 : {torch.cuda.get_device_name(0)}")
print(f"📂 Thư mục ảnh mẫu     : {SAMPLES_DIR}")
print(f"📦 File Checkpoint V3  : {CKPT_PATH}")
print(f"🎯 Ngưỡng phát hiện    : {FIRE_THRESHOLD}")
print("=" * 82)

model = FireGrounderV3(pretrained=False).to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Đã tải thành công mô hình FireGrounder V3 (Epoch {checkpoint.get('epoch', '?')})\n")

# ─── 4. Lấy ngẫu nhiên 30 ảnh ────────────────────────────────────────────────
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
all_images = [p for p in SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]

if len(all_images) == 0:
    raise RuntimeError(f"❌ Không tìm thấy file ảnh hợp lệ nào trong {SAMPLES_DIR}")

rng = random.Random(SEED)
num_to_pick = min(NUM_SAMPLES, len(all_images))
chosen_images = rng.sample(all_images, num_to_pick)

print(f"🎲 Đã chọn ngẫu nhiên {len(chosen_images)} / {len(all_images)} ảnh để chạy thử nghiệm:")
print("-" * 82)
print(f"{'STT':<4} | {'Tên file ảnh':<36} | {'p_fire':<8} | {'Kết quả':<10} | {'Gốc lửa (x, y)':<18}")
print("-" * 82)

results = []

# ─── 5. Suy luận từng ảnh ─────────────────────────────────────────────────────
with torch.no_grad():
    for idx, img_path in enumerate(chosen_images, start=1):
        orig_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = orig_img.size

        # Preprocessing: resize 256x256 và chuẩn hóa ImageNet
        resized_img = TF.resize(orig_img, [IMG_SIZE, IMG_SIZE])
        tensor = TF.normalize(TF.to_tensor(resized_img), MEAN, STD).unsqueeze(0).to(DEVICE)

        # Forward qua model V3
        out = model(tensor)
        if isinstance(out, dict):
            out = out["pred"]
        pred = out[0].cpu().numpy()

        p_fire = float(pred[0])
        x_norm = float(pred[1])
        y_norm = float(pred[2])
        is_fire = p_fire >= FIRE_THRESHOLD

        # Tọa độ trên ảnh gốc
        px_orig = int(x_norm * w_orig)
        py_orig = int(y_norm * h_orig)

        status_str = "🔥 CÓ LỬA" if is_fire else "🌿 Không"
        coord_str = f"({px_orig:4d}, {py_orig:4d}) px" if is_fire else "       -      "

        print(f"#{idx:02d} | {img_path.name[:36]:<36} | {p_fire*100:6.2f}% | {status_str:<10} | {coord_str}")

        results.append({
            "path": img_path,
            "orig_img": orig_img,
            "p_fire": p_fire,
            "is_fire": is_fire,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "px_orig": px_orig,
            "py_orig": py_orig,
        })

print("-" * 82)

num_fire = sum(1 for r in results if r["is_fire"])
num_no_fire = len(results) - num_fire
print(f"📊 Kết quả tổng: {num_fire} ảnh báo CÓ LỬA 🔥 | {num_no_fire} ảnh báo KHÔNG LỬA 🌿")

# ─── 6. Vẽ lưới trực quan hóa (mỗi hàng 2 ảnh to rõ nét) ─────────────────────
cols = 2
rows = (len(results) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 6.0))
axes = np.array(axes).flatten()

for i, res in enumerate(results):
    ax = axes[i]
    ax.imshow(res["orig_img"])

    title_color = "#D32F2F" if res["is_fire"] else "#2E7D32"
    status_label = "[FIRE]" if res["is_fire"] else "[NO FIRE]"

    if res["is_fire"]:
        ax.scatter(
            res["px_orig"], res["py_orig"],
            c="yellow", marker="o", s=180, edgecolors="black", linewidths=1.5, zorder=5
        )
        ax.scatter(
            res["px_orig"], res["py_orig"],
            c="red", marker="x", s=220, linewidths=3.0, zorder=6,
            label=f"Base ({res['px_orig']},{res['py_orig']})"
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85, facecolor="white")

    title_text = f"#{i+1:02d} {res['path'].name[:20]}\n{status_label} (p={res['p_fire']*100:.1f}%)"
    ax.set_title(title_text, fontsize=9.5, fontweight="bold", color=title_color, pad=6)
    ax.axis("off")

for j in range(len(results), len(axes)):
    axes[j].axis("off")

plt.tight_layout()

# Lưu ảnh kết quả
save_dir = CKPT_PATH.parent / "previews" if (CKPT_PATH.parent / "previews").exists() else CKPT_PATH.parent
save_img_path = save_dir / "preview_30_fire_samples.png"
plt.savefig(save_img_path, dpi=130, bbox_inches="tight")
print(f"\n🖼️ Đã lưu lưới ảnh kết quả trực quan ra: {save_img_path}")


if args.show:
    plt.show()
else:
    plt.close()

print("🎉 Hoàn tất kiểm thử 30 ảnh!")
