"""
localizer_3d.py — Fire 3D Localization từ pixel (kx, ky) + depth map
======================================================================

Input:
    - keypoint (kx, ky)  từ FireBase-YOLO26 (pixel coordinate)
    - depth_map (H, W)   từ DepthBackend (relative depth: giá trị cao = gần)
    - intrinsics         CameraIntrinsics (hiện chưa dùng)

Output:
    FireLocation3D với:
    - depth_value      : giá trị relative depth tại điểm (kx, ky)
    - depth_normalized : depth chuẩn hóa về [0, 1], 1.0 = gần nhất
    - xyz_normalized   : (X_n, Y_n, Z_n) — đặc trưng vị trí tương đối
                         X, Y: vị trí pixel chuẩn hóa [-1, 1]
                         Z: 1 - depth_normalized (0 = gần, 1 = xa)
    - xyz_metric       : None (chưa triển khai, cần metric depth model)
    - zone             : "NEAR" / "MID" / "FAR" dựa theo phần trăm depth
    - metadata         : dict thêm thông tin

⚠️  LƯU Ý: xyz_normalized và zone là đặc trưng TƯƠNG ĐỐI với từng ảnh.
    Không phải tọa độ 3D thực trong không gian vật lý.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────
#  Camera Intrinsics
# ─────────────────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """
    Thông số nội tại camera.
    
    Nếu không biết, có thể ước lượng từ FOV:
        fx ≈ W / (2 * tan(fov_h / 2))
        fy ≈ H / (2 * tan(fov_v / 2))
        cx ≈ W / 2,  cy ≈ H / 2
    """
    fx: float          # tiêu cự theo trục X (pixel)
    fy: float          # tiêu cự theo trục Y (pixel)
    cx: float          # principal point X (pixel)
    cy: float          # principal point Y (pixel)
    width: int         # image width
    height: int        # image height

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        fov_h_deg: float = 90.0,
    ) -> "CameraIntrinsics":
        """
        Ước lượng intrinsics từ field-of-view ngang (degrees).
        Mặc định FOV=90° — phù hợp webcam phổ thông.
        """
        import math
        fx = width / (2.0 * math.tan(math.radians(fov_h_deg / 2.0)))
        fy = fx  # assume square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    @classmethod
    def from_image_shape(cls, h: int, w: int) -> "CameraIntrinsics":
        """Ước lượng nhanh với FOV=90° mặc định."""
        return cls.from_fov(width=w, height=h, fov_h_deg=90.0)


# ─────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────

@dataclass
class FireLocation3D:
    """Kết quả định vị 3D của điểm BASE (gốc lửa)."""

    # Pixel input
    pixel_x: int
    pixel_y: int

    # Depth info
    depth_value: float          # giá trị raw từ model
    depth_normalized: float     # [0, 1], 1.0 = gần nhất
    depth_percentile: float     # bao nhiêu % điểm ảnh xa hơn điểm này

    # 3D coordinates (normalized, không đơn vị)
    #   X: phương ngang (dương = phải)
    #   Y: phương dọc  (dương = xuống)
    #   Z: phương sâu  (dương = xa camera)
    xyz_normalized: Tuple[float, float, float]

    # 3D coordinates (metric) — CHƯA TRIỂN KHAI
    # ⚠️  None: relative depth không thể chuyển thành tọa độ mét chính xác
    #     Cần dùng metric depth model + intrinsics thật
    xyz_metric: Optional[Tuple[float, float, float]] = None

    # Distance estimate (tỉ lệ)
    relative_distance: float = 0.0  # 0 = gần nhất, 1 = xa nhất

    # Zone label
    zone: str = "UNKNOWN"    # "NEAR" | "MID" | "FAR"

    # Extra info
    model_name: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pixel": {"x": self.pixel_x, "y": self.pixel_y},
            "depth_value": float(self.depth_value),
            "depth_normalized": float(self.depth_normalized),
            "depth_percentile": float(self.depth_percentile),
            "xyz_normalized": list(self.xyz_normalized),
            "xyz_metric": list(self.xyz_metric) if self.xyz_metric else None,
            "relative_distance": float(self.relative_distance),
            "zone": self.zone,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        xn, yn, zn = self.xyz_normalized
        zone_emoji = {"NEAR": "🔴", "MID": "🟡", "FAR": "🟢"}.get(self.zone, "⚪")
        lines = [
            f"🔥 FireBase 3D Location @ pixel({self.pixel_x}, {self.pixel_y})",
            f"   Depth  : {self.depth_value:.4f} (raw)  →  normalized {self.depth_normalized:.3f}",
            f"   Zone   : {zone_emoji} {self.zone}  (sâu hơn {self.depth_percentile:.1f}% scene)",
            f"   XYZ(n) : ({xn:+.4f}, {yn:+.4f}, {zn:.4f})  — relative position feature",
        ]
        if self.xyz_metric:
            xm, ym, zm = self.xyz_metric
            lines.append(f"   XYZ(m) : ({xm:+.3f} m, {ym:+.3f} m, {zm:.3f} m)  — metric")
        else:
            lines.append(f"   XYZ(m) : Not available (cần metric depth model)")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
#  Core Localizer
# ─────────────────────────────────────────────────────────

class FireLocalizer3D:
    """
    Chuyển keypoint 2D + depth map → tọa độ 3D.

    Ví dụ nhanh:
    ```python
    from localizer_3d import FireLocalizer3D
    loc = FireLocalizer3D()
    result = loc.localize(depth_map, kx=320, ky=240, img_w=640, img_h=480)
    print(result)
    ```
    """

    def __init__(
        self,
        near_thresh: float = 0.70,   # depth_normalized > này → NEAR
        far_thresh: float = 0.30,    # depth_normalized < này → FAR
        patch_radius: int = 3,       # lấy trung bình patch xung quanh keypoint
    ):
        self.near_thresh = near_thresh
        self.far_thresh = far_thresh
        self.patch_radius = patch_radius

    def _sample_depth(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Lấy median depth trong patch r×r xung quanh (x, y)."""
        H, W = depth_map.shape[:2]
        r = self.patch_radius
        y0 = max(0, y - r); y1 = min(H, y + r + 1)
        x0 = max(0, x - r); x1 = min(W, x + r + 1)
        patch = depth_map[y0:y1, x0:x1]
        return float(np.median(patch))

    def localize(
        self,
        depth_map: np.ndarray,
        kx: int,
        ky: int,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        model_name: str = "",
    ) -> FireLocation3D:
        """
        Tính tọa độ 3D của điểm (kx, ky).

        Parameters
        ----------
        depth_map   : (H, W) float32 từ DepthBackend, giá trị lớn = gần
        kx, ky      : pixel coordinate của FireBase keypoint
        img_w, img_h: kích thước ảnh gốc (nếu khác depth_map)
        intrinsics  : CameraIntrinsics (hiện chưa dùng)
        model_name  : tên backend đang dùng

        Returns
        -------
        FireLocation3D
        """
        H_d, W_d = depth_map.shape[:2]

        # Scale pixel về kích thước depth map nếu khác nhau
        if img_w is not None and img_h is not None:
            scale_x = W_d / img_w
            scale_y = H_d / img_h
            dx = int(round(kx * scale_x))
            dy = int(round(ky * scale_y))
        else:
            dx, dy = kx, ky

        dx = np.clip(dx, 0, W_d - 1)
        dy = np.clip(dy, 0, H_d - 1)

        # Lấy depth tại điểm
        raw_depth = self._sample_depth(depth_map, dx, dy)

        # Normalize depth về [0, 1]
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        eps = 1e-8
        depth_norm = float((raw_depth - d_min) / (d_max - d_min + eps))

        # Percentile: bao nhiêu % điểm CÓ depth NHỎ HƠN (= xa hơn) điểm này
        depth_percentile = float(np.mean(depth_map < raw_depth) * 100.0)

        # Relative distance: 0 = gần nhất (depth_norm=1), 1 = xa nhất (depth_norm=0)
        relative_dist = 1.0 - depth_norm

        # Zone label
        if depth_norm >= self.near_thresh:
            zone = "NEAR"
        elif depth_norm <= self.far_thresh:
            zone = "FAR"
        else:
            zone = "MID"

        # Normalized XYZ (không đơn vị)
        # Z: đảo ngược depth_norm → Z lớn = xa camera
        # X, Y: tính từ tỉ lệ pixel so với center
        Z_n = relative_dist  # 0 = gần, 1 = xa
        X_n = (kx - (img_w or W_d) / 2.0) / ((img_w or W_d) / 2.0)  # [-1, 1]
        Y_n = (ky - (img_h or H_d) / 2.0) / ((img_h or H_d) / 2.0)  # [-1, 1]

        # Metric XYZ (CHƯA TRIỂN KHAI — cần metric depth model)
        # ⚠️  WARNING: Relative depth không thể chuyển thành tọa độ mét chính xác
        # Để xuất XYZ mét, cần:
        #   1. Dùng DepthART metric indoor/outdoor model (448px)
        #   2. Camera intrinsics thật (fx, fy, cx, cy) tương ứng ảnh gốc
        #   3. Depth output từ model đó đã có đơn vị mét
        # Công thức: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy
        xyz_metric = None

        return FireLocation3D(
            pixel_x=kx,
            pixel_y=ky,
            depth_value=raw_depth,
            depth_normalized=depth_norm,
            depth_percentile=depth_percentile,
            xyz_normalized=(X_n, Y_n, Z_n),
            xyz_metric=xyz_metric,
            relative_distance=relative_dist,
            zone=zone,
            model_name=model_name,
        )

    def localize_batch(
        self,
        depth_map: np.ndarray,
        keypoints: list[tuple[int, int]],
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        model_name: str = "",
    ) -> list[FireLocation3D]:
        """Localize nhiều keypoint cùng lúc."""
        return [
            self.localize(depth_map, kx, ky, img_w, img_h, intrinsics, model_name)
            for kx, ky in keypoints
        ]
