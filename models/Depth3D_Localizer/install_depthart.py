"""
install_depthart.py — Script cài đặt DepthART từ GitHub + tải checkpoint
=========================================================================

Chạy một lần để setup:
    python install_depthart.py

Lưu ý:
- DepthART yêu cầu biên dịch CUDA Selective Scan extension → cần NVIDIA GPU
- Trên CPU-only: dùng Depth Anything V2 thay thế (không cần script này)
- Trên Kaggle GPU: chạy script này trong notebook trước khi dùng prefer="depthart"
"""

import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).parent
REPO_DIR = MODULE_DIR / "DepthART"
CKPT_DIR = REPO_DIR / "checkpoints"

def run(cmd: list[str], cwd=None):
    print(f"\n▶ {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại: {' '.join(cmd)}")

def install():
    print("=" * 60)
    print("  DepthART Installation Script")
    print("=" * 60)

    # 1. Clone repo
    if not REPO_DIR.exists():
        print("\n📥 Clone DepthART repo...")
        run(["git", "clone", "https://github.com/xuefeng-cvr/DepthART",
             str(REPO_DIR)])
    else:
        print(f"✅ Repo đã tồn tại: {REPO_DIR}")

    # 2. Install dependencies
    print("\n📦 Cài đặt dependencies cần thiết...")
    # Repo gốc dùng environment.yml; cài trực tiếp các package cốt lõi
    run([sys.executable, "-m", "pip", "install", "-q", "timm>=0.6.12", "einops>=0.8.0"])

    # 3. Tải checkpoint MobileNetV4-S (CPU-friendly)
    print("\n⬇️  Tải checkpoint depthart_relative_mnv4s_224.pth từ HuggingFace...")
    try:
        run([
            sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"
        ])
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="Fengxue93/DepthART",
            filename="relative/depthart_relative_mnv4s_224.pth",
            local_dir=str(CKPT_DIR),
        )
        print(f"✅ Đã tải checkpoint: {ckpt_path}")
    except Exception as e:
        print(f"⚠️  Không tải được checkpoint tự động: {e}")
        print("   Hãy tải thủ công từ: https://huggingface.co/Fengxue93/DepthART")

    # 4. Build Selective Scan extension (chỉ trên Linux/CUDA)
    import platform
    if platform.system() == "Linux":
        print("\n🔧 Build PyTorch Selective Scan extension...")
        setup_sh = REPO_DIR / "deploy" / "pc" / "setup.sh"
        if setup_sh.exists():
            try:
                run(["bash", str(setup_sh), "pytorch-only"], cwd=str(REPO_DIR))
                print("✅ Build extension thành công!")
            except RuntimeError as e:
                print(f"⚠️  Build thất bại (có thể do không có CUDA): {e}")
                print("   Fallback về Depth Anything V2 sẽ được dùng tự động.")
        else:
            print("⚠️  Không tìm thấy setup.sh")
    else:
        print(f"\n⚠️  OS = {platform.system()}: Selective Scan extension chỉ hỗ trợ Linux.")
        print("   DepthART sẽ dùng Depth Anything V2 làm fallback.")

    print("\n" + "=" * 60)
    print("  Cài đặt hoàn tất!")
    print("  → Dùng prefer='depthart' nếu extension build thành công")
    print("  → Dùng prefer='dav2' trên CPU / Windows")
    print("=" * 60)


if __name__ == "__main__":
    install()
