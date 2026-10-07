"""
depth_model.py — DepthART primary backend + fallback chain
===========================================================

Backend priority (default prefer="depthart"):
  1. DepthART MobileNetV4-S @ 224   ← CPU-friendly, no CUDA ext needed
  2. DepthART MobileNetV4-M @ 224   ← slightly larger, still CPU-ok
  3. DepthART TinyViM-S @ 224       ← có fallback CPU, nhưng chậm hơn
  4. Depth Anything V2 Small        ← fallback nếu DepthART repo chưa cài
  5. Depth Anything V1 Small        ← last resort

Tất cả backends trả về:
    depth_map : np.ndarray (H, W) float32
                Giá trị LỚN hơn = GẦN hơn camera (affine-invariant relative depth)
    model_name: str — tên model đang dùng

Setup DepthART:
    python install_depthart.py      ← clone repo + tải checkpoint từ HuggingFace
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Hằng số DepthART
# ─────────────────────────────────────────────────────────────────────────────

# Tên encoder trong DepthART repo (khớp với API load_relative_model)
_DEPTHART_ENCODERS = {
    # encoder_key : (api_name, ckpt_stem, requires_cuda_ext, cpu_friendly)
    "MNV4-S":          ("MNV4-S",          "mnv4s",       False, True),   # MobileNetV4-S
    "MNV4-M":          ("MNV4-M",          "mnv4m",       False, True),   # MobileNetV4-M
    "MNV4-M-SLIM-SPF": ("MNV4-M-SLIM-SPF", "mnv4m_slim",  False, True),   # slim SPF
    "S":               ("S",               "s",           True,  False),  # TinyViM-S
    "B":               ("B",               "b",           True,  False),  # TinyViM-B
    "L":               ("L",               "l",           True,  False),  # TinyViM-L
}

# Thứ tự thử tự động theo device
_CPU_ENCODER_PRIORITY = ["MNV4-S", "MNV4-M", "S"]
_GPU_ENCODER_PRIORITY = ["S", "MNV4-S", "MNV4-M", "B"]


def _find_depthart_repo(extra_paths: list[Path] | None = None) -> Path | None:
    """Tìm thư mục gốc DepthART repo (phải chứa thư mục relative/)."""
    here = Path(__file__).parent
    candidates = [
        here / "DepthART",
        here.parent / "DepthART",
        here.parent.parent / "DepthART",
        Path.cwd() / "DepthART",
    ]
    if extra_paths:
        candidates = list(extra_paths) + candidates

    for p in candidates:
        if p.is_dir() and (p / "relative").is_dir():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1: DepthART
# ─────────────────────────────────────────────────────────────────────────────

def _load_depthart_variant(
    repo_root: Path,
    encoder_key: str,
    resolution: int,
    checkpoint: Path | None,
    device: str,
) -> tuple[Callable, str]:
    """
    Load một variant cụ thể của DepthART.
    Raise RuntimeError / ImportError nếu thất bại.
    """
    import torch
    import torchvision.transforms as T

    api_name, ckpt_stem, needs_cuda_ext, cpu_friendly = _DEPTHART_ENCODERS[encoder_key]

    # Cảnh báo nếu dùng TinyViM trên CPU
    if needs_cuda_ext and device == "cpu":
        logger.warning(
            f"[DepthART] TinyViM encoder '{encoder_key}' chạy chậm trên CPU. "
            f"Khuyến nghị dùng MobileNetV4-S/M cho CPU."
        )

    # Tìm checkpoint
    if checkpoint is None:
        # DepthART lưu theo pattern: depthart_relative_<stem>_<res>.pth
        ckpt_name = f"depthart_relative_{ckpt_stem}_{resolution}.pth"
        checkpoint = repo_root / "checkpoints" / "relative" / ckpt_name

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint DepthART '{encoder_key}' không tìm thấy tại:\n  {checkpoint}\n"
            f"  → Chạy: python install_depthart.py  (hoặc tải thủ công từ HuggingFace)"
        )

    # Thêm repo/relative vào sys.path
    relative_path = repo_root / "relative"
    if str(relative_path) not in sys.path:
        sys.path.insert(0, str(relative_path))

    try:
        from models import load_relative_model  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Không import được DepthART API từ {relative_path}.\n"
            f"  Lỗi: {e}\n"
            f"  → Kiểm tra repo có đúng cấu trúc và dependencies đã cài (timm, einops)"
        ) from e

    logger.info(f"[DepthART] Load model: encoder={api_name}, resolution={resolution}")
    model, _ = load_relative_model(api_name, str(checkpoint), device)
    model.eval()

    # Preprocessing theo DepthART official
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def infer(pil_img: Image.Image) -> np.ndarray:
        orig_w, orig_h = pil_img.size
        # Resize về resolution×resolution bằng PIL (như script chính thức)
        rgb = np.array(pil_img.convert("RGB").resize((resolution, resolution), Image.BICUBIC), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(tensor)
            # Resize về kích thước ảnh gốc
            import torch.nn.functional as F
            depth = F.interpolate(
                prediction, (orig_h, orig_w), mode="bilinear", align_corners=True
            )[0, 0].float().cpu().numpy()
        return depth

    display_name = f"DepthART-{encoder_key} @ {resolution}px  [{device.upper()}]"
    return infer, display_name


def _try_load_depthart(
    repo_root: Optional[str | Path] = None,
    encoder: Optional[str] = None,
    resolution: int = 224,
    checkpoint: Optional[str | Path] = None,
    device: str = "cpu",
) -> tuple[Callable, str]:
    """
    Tải DepthART — tự chọn encoder phù hợp theo device nếu encoder=None.

    Thứ tự thử trên CPU: MNV4-S → MNV4-M → S
    Thứ tự thử trên GPU: S → MNV4-S → MNV4-M → B

    Returns (infer_fn, model_name) hoặc raise.
    """
    # Tìm repo
    extra = [Path(repo_root)] if repo_root else None
    repo = _find_depthart_repo(extra)
    if repo is None:
        raise ImportError(
            "Không tìm thấy DepthART repo.\n"
            "  → Chạy: python install_depthart.py  để clone + tải checkpoint."
        )

    logger.info(f"[DepthART] Repo: {repo}")

    # Xác định thứ tự encoder sẽ thử
    if encoder is not None:
        # Người dùng chỉ định cụ thể
        encoder_key = encoder.upper()
        if encoder_key not in _DEPTHART_ENCODERS:
            raise ValueError(f"Encoder không hợp lệ: {encoder}. Chọn trong: {list(_DEPTHART_ENCODERS)}")
        priority = [encoder_key]
    else:
        import torch
        is_gpu = device == "cuda" and torch.cuda.is_available()
        priority = _GPU_ENCODER_PRIORITY if is_gpu else _CPU_ENCODER_PRIORITY

    last_err = None
    for enc_key in priority:
        try:
            logger.info(f"[DepthART] Thử encoder: {enc_key} ...")
            infer_fn, name = _load_depthart_variant(
                repo, enc_key, resolution,
                Path(checkpoint) if checkpoint else None,
                device,
            )
            logger.info(f"[DepthART] ✅ Loaded: {name}")
            return infer_fn, name
        except (FileNotFoundError, ImportError, RuntimeError) as e:
            logger.warning(f"[DepthART] ⚠ {enc_key} skipped: {e}")
            last_err = e

    raise RuntimeError(
        f"Không load được bất kỳ DepthART encoder nào trên {device}.\n"
        f"  Lỗi cuối: {last_err}\n"
        f"  → Chạy python install_depthart.py để tải checkpoint."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 2: Depth Anything V2 (HuggingFace, fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_depth_anything_v2(device: str = "cpu") -> tuple[Callable, str]:
    """Depth Anything V2 Small — fallback khi DepthART chưa được cài."""
    import torch
    from transformers import pipeline as hf_pipeline

    logger.info("[Fallback] Load Depth Anything V2 Small từ HuggingFace...")
    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if (device == "cuda" and torch.cuda.is_available()) else -1,
    )

    def infer(pil_img: Image.Image) -> np.ndarray:
        result = pipe(pil_img.convert("RGB"))
        # Lấy tensor depth thay vì ảnh colorized
        depth_tensor = result["predicted_depth"]
        return depth_tensor.squeeze().cpu().numpy().astype(np.float32)

    return infer, "Depth-Anything-V2-Small [fallback]"


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 3: Depth Anything V1 (last resort)
# ─────────────────────────────────────────────────────────────────────────────

def _try_load_depth_anything_v1(device: str = "cpu") -> tuple[Callable, str]:
    import torch
    from transformers import pipeline as hf_pipeline

    logger.info("[Fallback] Load Depth Anything V1 Small...")
    pipe = hf_pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-small-hf",
        device=0 if (device == "cuda" and torch.cuda.is_available()) else -1,
    )

    def infer(pil_img: Image.Image) -> np.ndarray:
        result = pipe(pil_img.convert("RGB"))
        depth_tensor = result["predicted_depth"]
        return depth_tensor.squeeze().cpu().numpy().astype(np.float32)

    return infer, "Depth-Anything-V1-Small [fallback]"


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

class DepthBackend:
    """
    Backend depth estimation với DepthART là primary.

    Sử dụng:
        # Mặc định: dùng DepthART, tự chọn encoder phù hợp
        backend = DepthBackend()
        backend.load()
        depth_map = backend.infer_pil(pil_image)   # → np.ndarray (H, W)

    Tham số prefer:
        "depthart"  — DepthART only (mặc định), raise nếu không có
        "auto"      — DepthART first, fallback về DAv2, DAv1
        "dav2"      — Depth Anything V2 only
        "dav1"      — Depth Anything V1 only

    Encoder DepthART (tham số depthart_encoder):
        None            — tự chọn: MNV4-S (CPU) hoặc S (GPU)
        "MNV4-S"        — MobileNetV4-S, CPU-friendly (~2M params)
        "MNV4-M"        — MobileNetV4-M, lớn hơn chút
        "MNV4-M-SLIM-SPF" — slim SPF decoder
        "S","B","L"     — TinyViM S/B/L, có fallback CPU nhưng chậm
    """

    def __init__(
        self,
        prefer: str = "depthart",
        depthart_repo: Optional[str | Path] = None,
        depthart_encoder: Optional[str] = None,   # None = tự chọn
        depthart_resolution: int = 224,
        depthart_checkpoint: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._prefer = prefer
        self._infer_fn: Optional[Callable] = None
        self.model_name = "unloaded"

        self._depthart_kwargs = dict(
            repo_root=depthart_repo,
            encoder=depthart_encoder,
            resolution=depthart_resolution,
            checkpoint=depthart_checkpoint,
            device=device,
        )

    def load(self) -> None:
        """Tải model (lazy). Gọi lần đầu sẽ tải checkpoint và khởi tạo model."""
        if self._infer_fn is not None:
            return

        prefer = self._prefer

        if prefer == "depthart":
            # Chỉ thử DepthART, không fallback
            fn, name = _try_load_depthart(**self._depthart_kwargs)
            self._infer_fn, self.model_name = fn, name

        elif prefer == "auto":
            # DepthART → DAv2 → DAv1
            for loader, label in [
                (lambda: _try_load_depthart(**self._depthart_kwargs), "DepthART"),
                (lambda: _try_load_depth_anything_v2(self.device),    "DAv2"),
                (lambda: _try_load_depth_anything_v1(self.device),    "DAv1"),
            ]:
                try:
                    self._infer_fn, self.model_name = loader()
                    break
                except Exception as e:
                    logger.warning(f"[auto] {label} skip: {e}")
            else:
                raise RuntimeError("Không load được bất kỳ backend nào.")

        elif prefer == "dav2":
            self._infer_fn, self.model_name = _try_load_depth_anything_v2(self.device)

        elif prefer == "dav1":
            self._infer_fn, self.model_name = _try_load_depth_anything_v1(self.device)

        else:
            raise ValueError(f"prefer không hợp lệ: '{prefer}'. Dùng: depthart | auto | dav2 | dav1")

        print(f"[DepthBackend] Loaded: {self.model_name}  ({self.device})")

    def infer_pil(self, pil_img: Image.Image) -> np.ndarray:
        """
        Inference từ PIL Image.
        Trả về depth map (H, W) float32.
        Giá trị CAO = điểm GẦN camera.
        """
        self.load()
        return self._infer_fn(pil_img)

    def infer_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Inference từ OpenCV BGR frame."""
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.infer_pil(pil)

    def infer_path(self, img_path: str | Path) -> np.ndarray:
        """Inference từ file path."""
        return self.infer_pil(Image.open(str(img_path)).convert("RGB"))

    def colorize(self, depth: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Depth map float → BGR image để hiển thị.
        Màu sáng (vàng) = GẦN,  màu tối (tím) = XA.
        """
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            norm = np.zeros_like(depth, dtype=np.uint8)
        else:
            norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    @property
    def is_loaded(self) -> bool:
        return self._infer_fn is not None
