# -*- coding: utf-8 -*-
"""
metric_depth.py — DepthART METRIC depth backend (depth tính bằng mét)
=========================================================================

Khác biệt so với `depth_model.py` (relative depth):

| | Relative depth (`depth_model.py`) | Metric depth (`metric_depth.py`) |
|---|---|---|
| Output | affine-invariant, giá trị CAO = gần | **đơn vị MÉT** trên trục Z |
| Cần intrinsics | không | **BẮT BUỘC** (model có camera conditioning) |
| Cho `xyz_metric` | ❌ không (thiếu scale) | ✅ có |
| Checkpoint | `checkpoints/relative/*.pth` | `checkpoints/metric/*.pth` |

⚠️  Metric depth KHÔNG phải "relative + hệ số". Model nhận intrinsics làm
    input (camera-aware) và trực tiếp xuất khoảng cách mét. Dùng sai
    intrinsics → sai số mét theo tỉ lệ, KHÔNG phải lỗi affine.

────────────────────────────────────────────────────────────────────────────
Cài đặt
────────────────────────────────────────────────────────────────────────────
    python install_depthart.py --metric indoor --encoder s

Sử dụng
────────────────────────────────────────────────────────────────────────────
    from metric_depth import MetricDepthBackend
    from localizer_3d import CameraIntrinsics, FireLocalizer3D

    intr = CameraIntrinsics.load("intrinsics.json")   # từ calibrate_camera.py
    backend = MetricDepthBackend(domain="indoor", encoder="s", intrinsics=intr)
    backend.load()

    depth_m = backend.infer_path("fire.jpg")          # (H, W) float32, ĐƠN VỊ MÉT
    #    depth_map CAO = GẦN (xem ghi chú quy ước bên dưới)

    loc = FireLocalizer3D().localize(
        depth_m, kx=320, ky=240, img_w=640, img_h=480,
        intrinsics=intr, metric_depth=True,
    )
    print(loc.xyz_metric)     # (X, Y, Z) mét

────────────────────────────────────────────────────────────────────────────
⚠️  HAI QUY ƯỚC NGƯỢC NHAU — ĐỌC KỸ TRƯỚC KHI DÙNG
────────────────────────────────────────────────────────────────────────────
1. `depth_model.py` (relative): giá trị CAO = GẦN  (disparity-like)
2. `metric_depth.py` (metric) : giá trị THẤP = GẦN (khoảng cách thật, tăng dần
   khi ra xa). Giá trị = khoảng cách tâm camera theo trục Z, tính bằng mét.

`FireLocalizer3D._sample_depth()` lấy MEDIAN trong patch — hoạt động đúng cho
cả hai. Nhưng nếu bạn tự visualize depth map, hãy tự đảo chiều:
    hiển thị metric  →  colorize(max_depth - depth_m)
    hiển thị relative→  colorize(depth)      (đã đúng sẵn)

`MetricDepthBackend.infer_pil()` trả về **mét thật**. Hàm
`infer_disparity_like()` cung cấp bản đã đảo chiều để nhất quán với
`depth_model.DepthBackend` khi bạn muốn chạy chung một visualization code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Thư mục chứa DepthART repo (mặc định: models/Depth3D_Localizer/DepthART)
_HERE = Path(__file__).resolve().parent

# Encoder của metric DepthART: chỉ TinyViM S/B/L (MobileNetV4 chỉ có relative)
_METRIC_ENCODERS = ("S", "B", "L")

# Kích thước input mặc định theo domain (khớp infer_image.py chính thức)
_DEFAULT_SIZE = {
    "indoor": (640, 480),    # (width, height)
    "outdoor": (448, 448),
}

# max_depth theo domain (khớp metric/model.py load_model)
_MAX_DEPTH = {"indoor": 10.0, "outdoor": 80.0}


def _find_metric_repo(extra_paths: list[Path] | None = None) -> Path | None:
    """Tìm thư mục DepthART có `metric/model.py`."""
    candidates = [
        _HERE / "DepthART",
        _HERE.parent / "DepthART",
        _HERE.parent.parent / "DepthART",
        Path.cwd() / "DepthART",
    ]
    if extra_paths:
        candidates = list(extra_paths) + candidates
    for p in candidates:
        if p.is_dir() and (p / "metric" / "model.py").exists():
            return p
    return None


def _checkpoint_path(repo_root: Path, domain: str, encoder: str) -> Path:
    """Đường dẫn checkpoint metric theo đúng naming convention của DepthART."""
    stem = f"depthart_metric_{domain}_{encoder.lower()}_448.pth"
    return repo_root / "checkpoints" / "metric" / stem


class MetricDepthBackend:
    """
    DepthART metric depth — output đơn vị MÉT.

    Model là camera-aware: intrinsics được đưa vào model (qua
    `DenseCameraEmbedder`) chứ không chỉ dùng ở bước back-projection.
    Vì vậy truyền sai intrinsics sẽ cho kết quả sai NGAY ở depth map,
    không chỉ sai ở xyz_metric.

    Parameters
    ----------
    intrinsics    : CameraIntrinsics — BẮT BUỘC, phải khớp camera thật.
                    Có thể là ước lượng FOV (kém chính xác hơn) hoặc từ calibration.
    domain        : "indoor" (max_depth 10m) | "outdoor" (max_depth 80m)
    encoder       : "S" | "B" | "L"  (TinyViM — metric chỉ hỗ trợ TinyViM)
    checkpoint    : đường dẫn .pth — mặc định suy ra từ domain+encoder
    depthart_repo : thư mục DepthART — mặc định tự tìm
    device        : "cpu" | "cuda" — mặc định tự chọn
    """

    def __init__(
        self,
        intrinsics,
        domain: str = "indoor",
        encoder: str = "s",
        checkpoint: Optional[str | Path] = None,
        depthart_repo: Optional[str | Path] = None,
        device: Optional[str] = None,
        use_optimized_scan: bool = True,
    ):
        if domain not in _MAX_DEPTH:
            raise ValueError(
                f"domain không hợp lệ: {domain!r}. Chọn: {list(_MAX_DEPTH)}"
            )
        enc = str(encoder).upper()
        if enc not in _METRIC_ENCODERS:
            raise ValueError(
                f"metric DepthART chỉ hỗ trợ TinyViM encoder {list(_METRIC_ENCODERS)}, "
                f"không có MobileNetV4. Nhận: {encoder!r}"
            )
        if intrinsics is None:
            raise ValueError(
                "MetricDepthBackend BẮT BUỘC cần CameraIntrinsics.\n"
                "  → Nạp từ calibration: CameraIntrinsics.load('intrinsics.json')\n"
                "  → Hoặc ước lượng:   CameraIntrinsics.from_fov(w, h, fov_deg=90)"
            )

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.intrinsics = intrinsics
        self.domain = domain
        self.encoder = enc
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.depthart_repo = Path(depthart_repo) if depthart_repo else None
        self.use_optimized_scan = use_optimized_scan

        self.max_depth = _MAX_DEPTH[domain]
        self.model_name = "unloaded"
        self._model = None
        self._scan_mode = "reference"

    # ── Load ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Tải model + checkpoint. Raise nếu thiếu."""
        if self._model is not None:
            return

        repo = self._resolve_repo()
        ckpt = self._resolve_checkpoint(repo)
        metric_dir = repo / "metric"

        # metric/ dùng import tương đối phẳng (from network import ...),
        # nên phải thêm chính thư mục metric/ vào sys.path.
        if str(metric_dir) not in sys.path:
            sys.path.insert(0, str(metric_dir))

        self._enable_optimized_scan(repo)

        try:
            from common import make_K, preprocess  # type: ignore
            from model import load_model           # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Không import được DepthART metric API từ {metric_dir}.\n"
                f"  Lỗi: {e}\n"
                f"  → Kiểm tra: repo đúng commit? đã cài timm, einops?"
            ) from e

        logger.info(
            f"[MetricDepthART] Load domain={self.domain} encoder={self.encoder} "
            f"device={self.device}"
        )
        model = load_model(str(ckpt), self.encoder, self.domain, self.device)
        model.eval()
        self._model = model

        # Khi ảnh đưa vào model bị resize, DepthART tự scale K theo.
        # Ta dựng K từ intrinsics ở kích thước ảnh gốc rồi để
        # preprocess() lo việc scale — nên cache K theo kích thước gốc.
        self._make_K = make_K
        self._preprocess = preprocess

        self.model_name = (
            f"DepthART-METRIC-{self.domain}-{self.encoder} @448px [{self.device.upper()}]"
        )
        print(f"[MetricDepthBackend] Loaded: {self.model_name}  scan={self._scan_mode}")

    def _resolve_repo(self) -> Path:
        repo = self._find_repo()
        if repo is None:
            raise ImportError(
                "Không tìm thấy DepthART repo (thư mục có metric/model.py).\n"
                "  → Chạy: python install_depthart.py --metric indoor --encoder s"
            )
        self.depthart_repo = repo
        return repo

    def _find_repo(self) -> Path | None:
        extra = [self.depthart_repo] if self.depthart_repo else None
        return _find_metric_repo(extra)

    def _resolve_checkpoint(self, repo: Path) -> Path:
        ckpt = self.checkpoint or _checkpoint_path(repo, self.domain, self.encoder)
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"Checkpoint metric DepthART không tồn tại:\n  {ckpt}\n"
                f"  → Chạy: python install_depthart.py "
                f"--metric {self.domain} --encoder {self.encoder.lower()}"
            )
        self.checkpoint = ckpt
        return ckpt

    def _enable_optimized_scan(self, repo: Path) -> None:
        """
        Thử bật CUDA Selective Scan extension.
        Chỉ có tác dụng trên Linux+CUDA; trên CPU dùng reference fallback.
        """
        if not self.use_optimized_scan:
            self._scan_mode = "disabled"
            return
        ext_root = repo / "deploy" / "shared" / "selective_scan"
        if str(ext_root) not in sys.path:
            sys.path.insert(0, str(ext_root))
        try:
            import depthart_selective_scan_cuda  # noqa: F401
            from depthart_selective_scan import install_depthart
            from network import tvimblock  # type: ignore

            install_depthart(tvimblock)
            self._scan_mode = "optimized_cuda"
        except Exception as e:  # noqa: BLE001 — thiếu ext là chuyện bình thường
            logger.info(f"[MetricDepthART] Selective Scan CUDA ext không dùng được: {e}")
            self._scan_mode = "reference"

    # ── Inference ────────────────────────────────────────────────────────

    def infer_pil(self, pil_img: Image.Image) -> np.ndarray:
        """
        → np.ndarray (H, W) float32, **đơn vị MÉT** (Z-axis distance).

        Giá trị THẤP = gần camera, giá trị CAO = xa. Đây là quy ước
        ngược với relative depth — xem module docstring.
        """
        self.load()

        import torch
        import torch.nn.functional as F

        img = pil_img.convert("RGB")
        orig_w, orig_h = img.size
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Intrinsics phải theo kích thước ảnh gốc; preprocess() tự scale K.
        K = self._make_K(
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.cx, self.intrinsics.cy,
        )

        tw, th = _DEFAULT_SIZE[self.domain]
        # Nếu ảnh nhỏ hơn target, đưa lên tối thiểu để không mất chi tiết
        scale = max(tw / orig_w, th / orig_h)
        tw = max(tw, int(round(orig_w * scale)))
        th = max(th, int(round(orig_h * scale)))

        tensor, K_t = self._preprocess(bgr, K, tw, th)
        tensor = tensor.to(self.device)
        K_t = K_t.to(self.device)

        with torch.inference_mode():
            prediction = self._model(tensor, K_t)
            prediction = F.interpolate(
                prediction[:, None], (orig_h, orig_w),
                mode="bilinear", align_corners=True,
            )[0, 0]

        depth = prediction.float().cpu().numpy().astype(np.float32)

        # Depth lọn (NaN/inf/âm) → thay bằng max_depth hợp lệ
        depth = np.nan_to_num(depth, nan=self.max_depth,
                              posinf=self.max_depth, neginf=self.max_depth)
        depth = np.clip(depth, 0.0, self.max_depth)
        return depth

    def infer_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """→ depth map (H, W) mét từ OpenCV BGR frame."""
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.infer_pil(pil)

    def infer_path(self, img_path: str | Path) -> np.ndarray:
        """→ depth map (H, W) mét từ file ảnh."""
        return self.infer_pil(Image.open(str(img_path)).convert("RGB"))

    def infer_disparity_like(self, img) -> np.ndarray:
        """
        Trả về depth đã ĐẢO CHIỀU để khớp quy ước của `depth_model.py`
        (cao = gần). Chỉ dùng cho visualization / code dùng chung.
        Không dùng để tính xyz_metric — phải dùng `infer_pil()`.
        """
        d = self.infer_pil(img)
        return (self.max_depth - d).astype(np.float32)

    # ── Helpers ──────────────────────────────────────────────────────────

    def colorize(self, depth: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Depth mét → BGR để hiển thị. Màu SÁNG = GẦN, tối = XA.
        (Ngược với mặc định colorize của depth_model.py — vì ở đây
         giá trị thấp = gần, phải đảo trước khi colormap.)
        """
        inv = float(self.max_depth) - depth
        d_min, d_max = float(inv.min()), float(inv.max())
        if d_max - d_min < 1e-6:
            norm = np.zeros_like(inv, dtype=np.uint8)
        else:
            norm = ((inv - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    def describe(self) -> str:
        """Chuỗi mô tả cấu hình (để log)."""
        i = self.intrinsics
        return (
            f"DepthART-METRIC-{self.domain}/{self.encoder} @448 "
            f"[{self.device}] max_depth={self.max_depth}m "
            f"fx={i.fx:.1f} fy={i.fy:.1f} cx={i.cx:.1f} cy={i.cy:.1f} "
            f"@ {i.width}x{i.height} (source={i.source}) scan={self._scan_mode}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    sys.path.insert(0, str(_HERE))
    from localizer_3d import CameraIntrinsics

    p = argparse.ArgumentParser(
        description="DepthART metric depth (đơn vị mét) — 1 ảnh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--image", required=True)
    p.add_argument("--domain", choices=("indoor", "outdoor"), default="indoor")
    p.add_argument("--encoder", choices=("s", "b", "l"), default="s")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out-depth", default=None, help="lưu .npy")
    p.add_argument("--out-viz", default=None, help="lưu ảnh colorize .png")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--intrinsics", help="file JSON từ calibrate_camera.py")
    g.add_argument("--fov", type=float, default=None,
                   help="FOV ngang (độ) để ước lượng intrinsics — kém chính xác")
    p.add_argument("--img-size", nargs=2, type=int, metavar=("W", "H"), default=None,
                   help="kích thước ảnh để ước lượng intrinsics khi dùng --fov")

    args = p.parse_args()

    if args.intrinsics:
        intr = CameraIntrinsics.load(args.intrinsics)
    else:
        img = Image.open(args.image)
        w, h = args.img_size if args.img_size else img.size
        intr = CameraIntrinsics.from_fov(width=w, height=h, fov_h_deg=args.fov or 90.0)
        print(f"[warn] Dùng intrinsics ƯỚC LƯỢNG (FOV={args.fov or 90.0}°) — "
              f"sai số mét có thể lớn. Nên calibrate: python calibrate_camera.py")

    backend = MetricDepthBackend(
        intrinsics=intr,
        domain=args.domain,
        encoder=args.encoder,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(backend.describe())
    depth = backend.infer_path(args.image)
    print(f"depth: shape={depth.shape} dtype={depth.dtype} "
          f"min={depth.min():.3f} max={depth.max():.3f} mean={depth.mean():.3f} (mét)")

    if args.out_depth:
        out = Path(args.out_depth)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, depth)
        print(f"saved {out}")
    if args.out_viz:
        out = Path(args.out_viz)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), backend.colorize(depth))
        print(f"saved {out}")


if __name__ == "__main__":
    _main()