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
    - xyz_metric       : (X, Y, Z) tính bằng mét — CHỈ có khi depth_map là
                         METRIC depth (đơn vị mét) VÀ có CameraIntrinsics thật
    - zone             : "NEAR" / "MID" / "FAR" dựa theo phần trăm depth
    - metadata         : dict thêm thông tin

⚠️  LƯU Ý: xyz_normalized và zone là đặc trưng TƯƠNG ĐỐI với từng ảnh.
    Không phải tọa độ 3D thực trong không gian vật lý.

─────────────────────────────────────────────────────────────────────────────
Hai chế độ depth
─────────────────────────────────────────────────────────────────────────────
1. RELATIVE depth (DepthART relative, Depth Anything V1/V2)
   → `localize(..., metric_depth=False)` (mặc định)
   → xyz_metric = None. Chỉ có xyz_normalized + zone.

2. METRIC depth (DepthART metric indoor/outdoor — xem metric_depth.py)
   → `localize(..., metric_depth=True, intrinsics=intr)`
   → xyz_metric = (X, Y, Z) đơn vị mét, tính bằng:
        Z = depth_value           (mét, theo trục quang)
        X = (kx - cx) * Z / fx
        Y = (ky - cy) * Z / fy
   → Lưu ý: Y dương = hướng xuống trong ảnh (đúng với quy ước pixel).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
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

    ⚠️  `from_fov` chỉ là ƯỚC LƯỢNG — đủ tốt cho zone/xyz_normalized,
        KHÔNG đủ cho xyz_metric chính xác. Dùng `calibrate_camera.py`
        để lấy intrinsics thật.
    """
    fx: float          # tiêu cự theo trục X (pixel)
    fy: float          # tiêu cự theo trục Y (pixel)
    cx: float          # principal point X (pixel)
    cy: float          # principal point Y (pixel)
    width: int         # image width
    height: int        # image height

    # Nguồn gốc intrinsics — dùng để cảnh báo khi đang ước lượng
    source: str = "fov_estimate"
    fov_h_deg: Optional[float] = None
    reprojection_error: Optional[float] = None   # RMSE (pixel), chỉ có khi calibrate

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

        ⚠️  Đây là ƯỚC LƯỢNG, không phải calibration thật.
        """
        fx = width / (2.0 * math.tan(math.radians(fov_h_deg / 2.0)))
        fy = fx  # assume square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            source="fov_estimate",
            fov_h_deg=float(fov_h_deg),
        )

    @classmethod
    def from_image_shape(cls, h: int, w: int) -> "CameraIntrinsics":
        """Ước lượng nhanh với FOV=90° mặc định."""
        return cls.from_fov(width=w, height=h, fov_h_deg=90.0)

    @classmethod
    def from_values(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        source: str = "manual",
        reprojection_error: Optional[float] = None,
    ) -> "CameraIntrinsics":
        """Tạo intrinsics từ 4 giá trị đã biết (ví dụ: từ calibration tool)."""
        return cls(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=int(width), height=int(height),
            source=source, reprojection_error=reprojection_error,
        )

    # ── Matrix & serialization ────────────────────────────────────────────

    def to_matrix(self) -> np.ndarray:
        """Trả về ma trận K 3x3."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            "fx": float(self.fx), "fy": float(self.fy),
            "cx": float(self.cx), "cy": float(self.cy),
            "width": int(self.width), "height": int(self.height),
            "source": self.source,
            "fov_h_deg": self.fov_h_deg,
            "reprojection_error": self.reprojection_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(d["fx"]), fy=float(d["fy"]),
            cx=float(d["cx"]), cy=float(d["cy"]),
            width=int(d["width"]), height=int(d["height"]),
            source=str(d.get("source", "manual")),
            fov_h_deg=(float(d["fov_h_deg"]) if d.get("fov_h_deg") is not None else None),
            reprojection_error=(
                float(d["reprojection_error"])
                if d.get("reprojection_error") is not None else None
            ),
        )

    def save(self, path: str | Path) -> Path:
        """Lưu intrinsics ra file JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        """Đọc intrinsics từ file JSON (xem `calibrate_camera.py --output`)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file intrinsics: {p}\n"
                f"  → Chạy: python calibrate_camera.py --images <thư mục ảnh bàn cờ>"
            )
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def scaled(self, new_width: int, new_height: int) -> "CameraIntrinsics":
        """
        Scale intrinsics sang kích thước ảnh khác (giữ nguyên FOV).
        Dùng khi ảnh bị resize trước khi đưa vào model.
        """
        sx = new_width / self.width
        sy = new_height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            width=int(new_width), height=int(new_height),
            source=self.source, fov_h_deg=self.fov_h_deg,
            reprojection_error=self.reprojection_error,
        )

    def to_image_size(self, img_w: int, img_h: int) -> "CameraIntrinsics":
        """Alias của `scaled()` cho trường hợp chỉ khác kích thước."""
        return self.scaled(img_w, img_h)

    @property
    def is_calibrated(self) -> bool:
        """True nếu intrinsics đến từ calibration thật (không phải ước lượng)."""
        return self.source not in ("fov_estimate", "", None)

    # ── Pinhole projection ───────────────────────────────────────────────

    def pixel_to_ray(self, kx: float, ky: float) -> Tuple[float, float, float]:
        """
        Đổi pixel → tia camera (unnormalized) theo mô hình pinhole.

        Quy ước trục: X phải, Y xuống, Z tiến (ra xa camera) — khớp với
        hệ tọa độ ảnh, nên ray luôn có Z = 1.
        """
        return ((kx - self.cx) / self.fx, (ky - self.cy) / self.fy, 1.0)

    def pixel_to_metric_xyz(
        self, kx: float, ky: float, depth_m: float
    ) -> Tuple[float, float, float]:
        """
        Back-project pixel + metric depth → tọa độ 3D đơn vị mét.

        Parameters
        ----------
        kx, ky   : pixel coordinates (ảnh mà intrinsics này áp dụng)
        depth_m  : Z theo trục quang, đơn vị mét (metric depth model)

        Returns
        -------
        (X, Y, Z) mét — X phải, Y xuống, Z ra xa camera.
        """
        if depth_m is None or not np.isfinite(depth_m):
            raise ValueError(f"depth_m không hợp lệ: {depth_m}")
        rx, ry, rz = self.pixel_to_ray(kx, ky)
        # ray đã chuẩn hóa theo Z=1 nên nhân thẳng depth_m
        return (rx * depth_m, ry * depth_m, depth_m)


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

    # 3D coordinates (metric) — chỉ có khi dùng METRIC depth + intrinsics
    # ⚠️  None khi depth_map là RELATIVE (affine-invariant):
    #     không thể chuyển thành mét chính xác, cần scale/shift thật.
    xyz_metric: Optional[Tuple[float, float, float]] = None

    # Khoảng cách theo trục quang, đơn vị mét (chỉ khi metric depth)
    depth_m: Optional[float] = None

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
            "depth_m": float(self.depth_m) if self.depth_m is not None else None,
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
            lines.append("   XYZ(m) : Not available (cần metric depth model)")
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
        metric_depth: bool = False,
    ) -> FireLocation3D:
        """
        Tính tọa độ 3D của điểm (kx, ky).

        Parameters
        ----------
        depth_map   : (H, W) float32 từ DepthBackend, giá trị lớn = gần
        kx, ky      : pixel coordinate của FireBase keypoint
        img_w, img_h: kích thước ảnh gốc (nếu khác depth_map)
        intrinsics  : CameraIntrinsics — BẮT BUỘC khi metric_depth=True
        model_name  : tên backend đang dùng
        metric_depth: True nếu depth_map đơn vị MÉT (DepthART metric).
                      Khi đó xyz_metric / depth_m sẽ được tính.
                      False (mặc định) = relative depth → xyz_metric = None.

        Returns
        -------
        FireLocation3D
        """
        depth_map = np.asarray(depth_map, dtype=np.float32)
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

        # Percentile: xếp hạng của điểm này trong cảnh, đặt theo quy ước
        # RELATIVE (cao = gần). Với metric depth sẽ được đảo ở dưới.
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

        # ── Metric XYZ ────────────────────────────────────────────────────
        # Cần: (1) depth đơn vị mét, (2) intrinsics thật.
        # Công thức pinhole: X = (u - cx)·Z/fx,  Y = (v - cy)·Z/fy,  Z = depth
        xyz_metric = None
        depth_m = None
        if metric_depth:
            if intrinsics is None:
                raise ValueError(
                    "metric_depth=True yêu cầu CameraIntrinsics.\n"
                    "  → Nạp từ file: CameraIntrinsics.load('intrinsics.json')\n"
                    "  → Hoặc ước lượng: CameraIntrinsics.from_fov(img_w, img_h, fov_deg)"
                )
            if raw_depth <= 0:
                raise ValueError(
                    f"Metric depth không hợp lệ tại pixel({kx}, {ky}): {raw_depth}\n"
                    "  → Depth map metric phải > 0 (khoảng cách tới camera, đơn vị mét)."
                )

            # ⚠️  ĐẢO CHIỀU — điểm dễ sai nhất khi làm việc với metric depth.
            #
            # Relative depth : giá trị CAO = GẦN  (disparity-like)
            # Metric depth   : giá trị THẤP = GẦN (khoảng cách thật, tính bằng mét)
            #
            # `depth_norm` và `zone` ở trên được tính theo quy ước RELATIVE.
            # Nếu không đảo, điểm GẦN nhất sẽ bị gán zone FAR và ngược lại —
            # tức mọi cảnh báo khoảng cách bị lật ngược 100%.
            depth_norm = 1.0 - depth_norm
            depth_percentile = 100.0 - depth_percentile
            relative_dist = 1.0 - depth_norm
            if depth_norm >= self.near_thresh:
                zone = "NEAR"
            elif depth_norm <= self.far_thresh:
                zone = "FAR"
            else:
                zone = "MID"

            # Intrinsics phải khớp với ảnh mà depth_map được lấy
            intr = intrinsics
            if (intr.width, intr.height) != (W_d, H_d):
                intr = intr.scaled(W_d, H_d)
            # depth_map đã resize về (W_d, H_d) → lấy tại toạ độ đã scale
            xyz_metric = intr.pixel_to_metric_xyz(float(dx), float(dy), float(raw_depth))
            depth_m = float(raw_depth)

        return FireLocation3D(
            pixel_x=kx,
            pixel_y=ky,
            depth_value=raw_depth,
            depth_normalized=depth_norm,
            depth_percentile=depth_percentile,
            xyz_normalized=(X_n, Y_n, Z_n),
            xyz_metric=xyz_metric,
            depth_m=depth_m,
            relative_distance=relative_dist,
            zone=zone,
            model_name=model_name,
            metadata=(
                {"intrinsics_source": intrinsics.source} if intrinsics is not None else {}
            ),
        )

    def localize_batch(
        self,
        depth_map: np.ndarray,
        keypoints: list[tuple[int, int]],
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        model_name: str = "",
        metric_depth: bool = False,
    ) -> list[FireLocation3D]:
        """Localize nhiều keypoint cùng lúc."""
        return [
            self.localize(
                depth_map, kx, ky, img_w, img_h, intrinsics, model_name,
                metric_depth=metric_depth,
            )
            for kx, ky in keypoints
        ]
