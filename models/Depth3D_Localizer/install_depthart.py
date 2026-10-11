"""
install_depthart.py — Cài DepthART repo + tải checkpoint
==========================================================

Cài repo + checkpoint RELATIVE (mặc định):
    python install_depthart.py

Cài thêm checkpoint METRIC (depth tính bằng mét — cần cho xyz_metric):
    python install_depthart.py --metric indoor --encoder s

Tải nhiều checkpoint:
    python install_depthart.py --all

────────────────────────────────────────────────────────────────────────────
⚠️  Về CUDA Selective Scan extension
────────────────────────────────────────────────────────────────────────────
DepthART dùng kiến trúc Mamba (TinyViM). Trên **CPU** repo đã có sẵn
`selective_scan_reference()` — chạy được nhưng chậm hơn.

Vì vậy:
- Windows / CPU-only  → chạy ĐƯỢC với encoder MNV4-S (khuyến nghị)
- Linux + CUDA        → có thể build extension để nhanh hơn (encoder S/B/L)
- Metric model (S/B/L) → dùng TinyViM nên CHẬM trên CPU. Chấp nhận được
                         cho ảnh tĩnh, không phù hợp realtime trên CPU.
"""

import argparse
import hashlib
import io
import subprocess
import sys
from pathlib import Path

# Windows console mặc định dùng cp1252 → không in được tiếng Việt.
# Bọc stdout để script chạy được trên mọi máy Windows.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

MODULE_DIR = Path(__file__).parent
REPO_DIR = MODULE_DIR / "DepthART"
CKPT_DIR = REPO_DIR / "checkpoints"
HF_REPO = "Fengxue93/DepthART"
HF_BASE = "https://huggingface.co/Fengxue93/DepthART"

# ── Checkpoint ───────────────────────────────────────────────────────────────
# relative: (hf_filename, local_subdir)
RELATIVE_CKPTS = {
    "mnv4s_224": "relative/depthart_relative_mnv4s_224.pth",
    "mnv4s_448": "relative/depthart_relative_mnv4s_448.pth",
    "mnv4m_224": "relative/depthart_relative_mnv4m_224.pth",
    "s_224":     "relative/depthart_relative_s_224.pth",
    "b_224":     "relative/depthart_relative_b_224.pth",
}
DEFAULT_RELATIVE = "mnv4s_224"   # CPU-friendly, không cần CUDA

# metric: chỉ TinyViM S/B/L @448, theo domain
METRIC_CKPTS = {
    ("indoor", "s"):  "metric/depthart_metric_indoor_s_448.pth",
    ("indoor", "b"):  "metric/depthart_metric_indoor_b_448.pth",
    ("indoor", "l"):  "metric/depthart_metric_indoor_l_448.pth",
    ("outdoor", "s"): "metric/depthart_metric_outdoor_s_448.pth",
    ("outdoor", "b"): "metric/depthart_metric_outdoor_b_448.pth",
    ("outdoor", "l"): "metric/depthart_metric_outdoor_l_448.pth",
}


def run(cmd: list[str], cwd=None):
    print(f"\n> {' '.join(str(c) for c in cmd)}")
    ret = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại: {' '.join(str(c) for c in cmd)}")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def expected_checksum(rel_name: str) -> str | None:
    """Đọc CHECKSUMS.sha256 từ repo DepthART (nếu có)."""
    sums = REPO_DIR / "CHECKSUMS.sha256"
    if not sums.exists():
        return None
    key = f"checkpoints/{rel_name}"
    for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == key:
            return parts[0]
    return None


def clone_repo() -> None:
    if (REPO_DIR / "relative").is_dir():
        print(f"  Repo da ton tai: {REPO_DIR}")
        return
    if REPO_DIR.exists():
        print(f"  [canh bao] {REPO_DIR} ton tai nhe khong dung cau truc.")
        print("           Xoa thu muc do va chay lai de clone sach.")
        return
    print("\n[1/3] Clone DepthART repo...")
    run(["git", "clone", "https://github.com/xuefeng-cvr/DepthART", str(REPO_DIR)])


def install_deps() -> None:
    print("\n[2/3] Cai dependencies...")
    run([sys.executable, "-m", "pip", "install", "-q",
         "timm>=0.6.12", "einops>=0.8.0", "huggingface_hub>=0.23.0",
         "opencv-python>=4.8.0", "numpy>=1.24.0", "pillow>=10.0.0"])


def download_checkpoint(rel_name: str, verify: bool = True) -> Path | None:
    """Tải 1 checkpoint từ HuggingFace, lưu vào DepthART/checkpoints/."""
    from huggingface_hub import hf_hub_download

    target = CKPT_DIR / rel_name
    if target.exists():
        print(f"  [skip] Da co: {target.name}")
    else:
        print(f"  [get ] {rel_name}")
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=rel_name,
                local_dir=str(CKPT_DIR),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] Khong tai duoc {rel_name}: {e}")
            print(f"         Tai thu cong: {HF_BASE}")
            return None

    if not target.exists():
        print(f"  [FAIL] Khong thay file sau khi tai: {target}")
        return None

    size_mb = target.stat().st_size / 1e6
    print(f"  [ok  ] {target.name}  ({size_mb:.1f} MB)")

    if verify:
        want = expected_checksum(rel_name)
        if want:
            got = sha256(target)
            if got.lower() == want.lower():
                print(f"  [hash] SHA256 khop")
            else:
                print(f"  [hash] !! SHA256 KHONG KHOP")
                print(f"         expected {want}")
                print(f"         got      {got}")
                print(f"         File co the hong hoac khong day du.")
                return target
        else:
            print("  [hash] (khong co trong CHECKSUMS.sha256 — bo qua)")
    return target


def build_cuda_ext() -> None:
    """Build Selective Scan CUDA extension (chỉ Linux + CUDA có lợi)."""
    import platform
    print("\n[3/3] Selective Scan CUDA extension...")
    if platform.system() != "Linux":
        print(f"  OS = {platform.system()}: extension chi ho tro Linux.")
        print("  CPU se dung reference fallback (chay duoc, cham hon mot chut).")
        print("  → Dung encoder MNV4-S de chay nhanh nhat tren CPU.")
        return
    setup_sh = REPO_DIR / "deploy" / "pc" / "setup.sh"
    if not setup_sh.exists():
        print(f"  Khong tim thay {setup_sh}")
        return
    try:
        run(["bash", str(setup_sh), "pytorch-only"], cwd=str(REPO_DIR))
        print("  Build thanh cong.")
    except RuntimeError as e:
        print(f"  Build that bai (thuong do khong co CUDA): {e}")
        print("  Se dung reference fallback.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cai DepthART + tai checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--metric", choices=("indoor", "outdoor"), default=None,
                   help="tai them checkpoint METRIC (depth don vi met)")
    p.add_argument("--encoder", choices=("s", "b", "l"), default="s",
                   help="encoder cho checkpoint metric (mac dinh: s)")
    p.add_argument("--relative", default=DEFAULT_RELATIVE,
                   choices=list(RELATIVE_CKPTS),
                   help=f"checkpoint relative (mac dinh: {DEFAULT_RELATIVE})")
    p.add_argument("--all", action="store_true",
                   help="tai nhieu checkpoint (bo qua --metric/--relative)")
    p.add_argument("--no-build", action="store_true",
                   help="bo qua buoc build CUDA extension")
    p.add_argument("--no-verify", action="store_true",
                   help="bo qua kiem tra SHA256")
    args = p.parse_args()

    print("=" * 62)
    print("  DEPTHART INSTALLER")
    print("=" * 62)

    clone_repo()
    install_deps()

    print("\n[3/4] Tai checkpoint...")
    targets: list[str] = []
    if args.all:
        targets += list(RELATIVE_CKPTS.values())
        targets += list(METRIC_CKPTS.values())
    else:
        targets.append(RELATIVE_CKPTS[args.relative])
        if args.metric:
            targets.append(METRIC_CKPTS[(args.metric, args.encoder)])

    ok_paths = []
    for rel in targets:
        if download_checkpoint(rel, verify=not args.no_verify) is not None:
            ok_paths.append(rel)

    if not args.no_build:
        build_cuda_ext()

    print("\n" + "=" * 62)
    print("  HOAN TAT")
    print("=" * 62)
    print(f"  Repo     : {REPO_DIR}")
    print(f"  Checkpoint tai duoc: {len(ok_paths)}/{len(targets)}")
    for rel in targets:
        mark = "OK " if rel in ok_paths else "-- "
        print(f"    [{mark}] {rel}")
    print()
    print("  Su dung:")
    print("    # Relative depth (zone, xyz_normalized)")
    print("    from depth_model import DepthBackend")
    print("    b = DepthBackend(prefer='depthart'); b.load()")
    if any(t.startswith("metric/") for t in ok_paths):
        print()
        print("    # Metric depth (xyz_metric — can intrinsics that)")
        print("    from metric_depth import MetricDepthBackend")
        print("    from localizer_3d import CameraIntrinsics")
        print("    intr = CameraIntrinsics.load('intrinsics.json')")
        print("    m = MetricDepthBackend(intrinsics=intr, domain='indoor')")
        print("    m.load()")
    print()
    print(f"  Tai thu cong: {HF_BASE}")
    return 0 if ok_paths else 1


if __name__ == "__main__":
    sys.exit(main())
