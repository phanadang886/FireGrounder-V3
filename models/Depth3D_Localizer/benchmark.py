# -*- coding: utf-8 -*-
"""
benchmark.py — Đo hiệu năng pipeline Depth3D
============================================

Đo trên ảnh thật trong `datasets/` để có con số thay vì ước lượng.

Usage:
    python benchmark.py                          # relative depth, ảnh từ fire-samples
    python benchmark.py --images <thư mục> --num 30
    python benchmark.py --depth metric --intrinsics intrinsics.json
    python benchmark.py --depth none            # chỉ đo YOLO detector
    python benchmark.py --synthetic              # ảnh giả, không cần dataset

Máy target của dự án: Intel i5-1145G7, 16GB RAM, Intel Iris Xe (KHÔNG CUDA).
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from localizer_3d import CameraIntrinsics  # noqa: E402
from pipeline import (  # noqa: E402
    DEFAULT_WEIGHTS, FireDetector, FirePipeline3D, PipelineConfig, read_image,
)

REPO_ROOT = _HERE.parent.parent

# Thư mục ảnh lửa mặc định (nằm trong .gitignore nhưng có sẵn trên máy dev)
DEFAULT_IMAGE_DIRS = [
    REPO_ROOT / "datasets" / "fire-samples",
    REPO_ROOT / "datasets" / "dataset_fire_pose" / "images",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def gather_images(dirs: list[Path], num: int) -> list[Path]:
    """Gom ảnh từ các thư mục (bỏ qua thư mục không tồn tại)."""
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
                if len(out) >= num:
                    return out
    return out


def synthetic_frames(num: int, size: tuple[int, int] = (640, 480)) -> list[np.ndarray]:
    """
    Ảnh giả có cấu trúc không gian để depth model không bị "lười".
    Một ảnh đồng nhất sẽ cho depth map hằng số → số đo sai.
    """
    rng = np.random.default_rng(0)
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = []
    for i in range(num):
        img = np.zeros((H, W, 3), np.uint8)
        img[..., 0] = (yy / H * 200).astype(np.uint8)
        img[..., 1] = (xx / W * 200).astype(np.uint8)
        img[..., 2] = 90
        for _ in range(14):
            cx, cy = rng.integers(0, W), rng.integers(0, H)
            r = int(rng.integers(20, 90))
            cv2.circle(img, (int(cx), int(cy)), r,
                       (int(rng.integers(60, 255)), 90, 40), -1)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        frames.append(img)
    return frames


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def report(name: str, times: list[float], unit: str = "ms") -> None:
    if not times:
        print(f"  {name:<14} (khong co mau)")
        return
    mean = statistics.fmean(times)
    print(f"  {name:<14} mean={mean:8.2f}{unit}  "
          f"p50={percentile(times, 50):7.2f}  "
          f"p95={percentile(times, 95):7.2f}  "
          f"min={min(times):7.2f}  max={max(times):7.2f}")


def parse_args() -> argparse.Namespace:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    p = argparse.ArgumentParser(
        description="Đo hiệu năng pipeline Depth3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--images", default=None,
                   help="thư mục ảnh (mặc định: datasets/fire-samples)")
    p.add_argument("--num", type=int, default=20, help="số ảnh dùng để đo")
    p.add_argument("--warmup", type=int, default=2, help="số frame bỏ qua (warmup)")
    p.add_argument("--synthetic", action="store_true",
                   help="dùng ảnh giả thay vì dataset thật")

    p.add_argument("--depth", choices=("relative", "metric", "none"),
                   default="relative")
    p.add_argument("--prefer", default="auto",
                   choices=("auto", "depthart", "dav2", "dav1"))
    p.add_argument("--domain", choices=("indoor", "outdoor"), default="indoor")
    p.add_argument("--encoder", choices=("s", "b", "l"), default="s")
    p.add_argument("--intrinsics", default=None)
    p.add_argument("--no-detector", action="store_true",
                   help="bỏ YOLO, chỉ đo depth+localize")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", default=None)
    p.add_argument("--keypoints", default="0.5,0.5",
                   help="keypoint tương đối (0-1) để đo localize. Mặc định giữa ảnh")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Chuẩn bị ảnh ─────────────────────────────────────────────────
    if args.synthetic:
        frames = synthetic_frames(args.num + args.warmup)
        src_desc = f"synthetic {frames[0].shape[1]}x{frames[0].shape[0]}"
    else:
        dirs = ([Path(args.images)] if args.images else DEFAULT_IMAGE_DIRS)
        paths = gather_images(dirs, args.num + args.warmup)
        if not paths:
            print("LỖI: không tìm thấy ảnh nào.")
            print("  → Dùng --synthetic để benchmark không cần dataset,")
            print("    hoặc --images <thư mục>")
            return 1
        frames = []
        for p in paths:
            try:
                frames.append(read_image(p))
            except Exception:  # noqa: BLE001
                continue
        src_desc = f"{len(frames)} ảnh từ {dirs[0].name}"
        # Cắt về số ảnh yêu cầu
        frames = frames[: args.num + args.warmup]

    print("=" * 72)
    print("  BENCHMARK — FireGrounder 3D pipeline")
    print("=" * 72)

    # Thông tin môi trường
    try:
        import torch
        torch_v = torch.__version__
        cuda = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    except Exception:  # noqa: BLE001
        torch_v, cuda, dev_name = "?", False, "?"

    import platform
    print(f"  Python      : {platform.python_version()}  ({platform.system()})")
    print(f"  PyTorch     : {torch_v}")
    print(f"  Thiết bị    : {dev_name}   CUDA={cuda}")
    print(f"  OpenCV      : {cv2.__version__}")
    print(f"  Nguồn ảnh   : {src_desc}")
    print()

    # ── Dựng pipeline ─────────────────────────────────────────────────
    intr: Optional[CameraIntrinsics] = None
    if args.depth == "metric":
        if args.intrinsics:
            intr = CameraIntrinsics.load(args.intrinsics)
        else:
            h, w = frames[0].shape[:2]
            intr = CameraIntrinsics.from_fov(width=w, height=h, fov_h_deg=90.0)
            print("  [warn] Không có --intrinsics → dùng ước lượng FOV=90°")
            print("         (đủ để đo THỜI GIAN, không đủ để đo độ chính xác)")
            print()

    detector = None
    if not args.no_detector:
        detector = FireDetector(
            weights=args.weights, conf=args.conf,
            imgsz=args.imgsz, device=args.device,
        )

    cfg = PipelineConfig(
        depth_mode=args.depth, prefer=args.prefer,
        domain=args.domain, encoder=args.encoder, intrinsics=intr,
    )
    pipe = FirePipeline3D(cfg, detector)

    print("[đang nạp model]...")
    t0 = time.perf_counter()
    try:
        pipe.warmup()
    except Exception as e:  # noqa: BLE001
        print(f"LỖI nạp model: {e}")
        return 1
    load_s = time.perf_counter() - t0
    print(f"[ok] nạp xong sau {load_s:.1f}s")
    if pipe.depth_model_name != "disabled":
        print(f"[ok] Depth: {pipe.depth_model_name}")
    if detector is not None:
        print(f"[ok] YOLO  : {detector.weights.name} @ {detector.device}")
    print()

    # ── Chạy ──────────────────────────────────────────────────────────
    kp_frac = [float(v) for v in args.keypoints.split(",")]
    timings: dict[str, list[float]] = {
        "detect": [], "depth": [], "localize": [], "annotate": [], "TOTAL": [],
    }
    depths: list[float] = []
    zones: dict[str, int] = {}

    n_warm = args.warmup
    total_frames = len(frames)
    print(f"[đang chạy] {total_frames} frame (bỏ qua {n_warm} frame warmup)...")

    for i, frame in enumerate(frames):
        H, W = frame.shape[:2]
        kx = int(W * kp_frac[0])
        ky = int(H * kp_frac[1]) if len(kp_frac) > 1 else int(H * kp_frac[0])

        # Đo riêng phần ngoài pipeline (đọc ảnh + decode không tính)
        t_frame = time.perf_counter()
        res = pipe.process(frame, fixed_keypoints=[(kx, ky)])
        elapsed = (time.perf_counter() - t_frame) * 1000

        if i < n_warm:
            continue

        timings["detect"].append(res.timings["detect"])
        timings["depth"].append(res.timings["depth"])
        timings["localize"].append(res.timings["localize"])
        timings["annotate"].append(res.timings["annotate"])
        timings["TOTAL"].append(elapsed)
        for loc in res.locations:
            zones[loc.zone] = zones.get(loc.zone, 0) + 1
            if loc.depth_m is not None:
                depths.append(loc.depth_m)
        if (i - n_warm + 1) % 10 == 0:
            print(f"  ... {i - n_warm + 1}/{total_frames - n_warm}")

    # ── Báo cáo ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  KẾT QUẢ")
    print("=" * 72)
    report("detect", timings["detect"])
    report("depth", timings["depth"])
    report("localize", timings["localize"])
    report("annotate", timings["annotate"])
    print("  " + "-" * 68)
    report("TOTAL", timings["TOTAL"])
    print()

    fps = 1000.0 / max(statistics.fmean(timings["TOTAL"]), 1e-9)
    print(f"  FPS trung bình      : {fps:.2f}")
    print(f"  Thời gian nạp model  : {load_s:.1f}s")
    if depths:
        print(f"  Depth (mét)          : min={min(depths):.2f}  "
              f"median={statistics.median(depths):.2f}  max={max(depths):.2f}")
    if zones:
        print(f"  Phân bố zone        : "
              + "  ".join(f"{k}={v}" for k, v in sorted(zones.items())))

    # ── Nhận xét ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  NHẬN XÉT")
    print("=" * 72)
    depth_mean = statistics.fmean(timings["depth"]) if timings["depth"] else 0.0
    total_mean = statistics.fmean(timings["TOTAL"]) if timings["TOTAL"] else 0.0
    pct_depth = 100.0 * depth_mean / max(total_mean, 1e-9)

    print(f"  Depth chiếm {pct_depth:.0f}% tổng thời gian "
          f"({depth_mean:.0f}ms / {total_mean:.0f}ms)")
    print()
    if args.depth == "metric":
        print("  Metric depth dùng TinyViM (Mamba) → CHẬM trên CPU.")
        print("  Không phù hợp realtime. Gợi ý:")
        print("    - Dùng --depth relative cho realtime (zone vẫn hoạt động)")
        print("    - Dùng metric chỉ khi thật sự cần tọa độ mét")
        print("    - Hoặc build CUDA Selective Scan extension trên Linux/GPU")
    elif args.depth == "none":
        print("  Đang đo CHỈ detector (không có depth).")
    else:
        print("  Relative depth chạy được realtime-ish trên CPU.")
        if fps >= 10:
            print(f"  {fps:.1f} FPS — đủ cho video giám sát ở độ phân giải giảm.")
        else:
            print(f"  {fps:.1f} FPS — dưới ngưỡng realtime 10 FPS.")
            print("  Gợi ý: giảm --imgsz của YOLO, dùng camera độ phân giải thấp,")
            print("         hoặc chạy song song 2 luồng (YOLO thread + depth thread).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())