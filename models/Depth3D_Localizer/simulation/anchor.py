"""Pixel trên ảnh RGB → toạ độ 3D gốc lửa, kèm **đối chứng độc lập bằng mesh**.

Không import ``torch`` — chạy được khi RAM chật (xem `AGENTS.md` → lỗi ``OpenBLAS``).
Phần chấm DepthART nằm ở ``run_eval.py``, tách khỏi file này cố ý.

Ba tầng toạ độ — đừng lẫn
-------------------------
| Tầng | Nguồn | Sai số so với thực | Vai trò |
|------|-------|--------------------|---------|
| **GT** | depth GT của dataset × intrinsics | **0** — đây là định nghĩa | điểm neo lửa (ground truth) |
| **MESH** | tia từ camera đập vào bề mặt mesh | **~2 mm** | **bắt lỗi code**, không phải để đo |
| **PRED** | DepthART đoán từ ảnh | **~312 mm** | **thứ cần đo** (ở `run_eval.py`) |

Tầng GT và MESH lệch nhau ~2 mm ⇒ bộ máy hình học đúng. Nhờ đó mọi lệch lạc ở tầng
PRED là lỗi của model, **không phải lỗi của mình** — đó là câu biện hộ phương pháp
luận cho báo cáo. Không có con số 2 mm thì 312 mm vô nghĩa.

Vì sao MESH không dùng làm GT
------------------------------
Depth GT sinh ra **từ chính mesh đó**, nên hai thứ không độc lập hoàn toàn. Nhưng
depth GT là giá trị dataset đóng gói và là thứ mọi metric hiện có (`0.3119 m`) đã
neo vào → đổi sang mesh sẽ làm số cũ so sánh được. Mesh chỉ dùng để **chứng minh**
rằng phép back-project không đặt điểm ra ngoài phòng.

Công thức
---------
::

    Z0      = depth_gt[v, u]                                  (mét)
    P_cam   = ((u-cx)·Z0/fx, (v-cy)·Z0/fy, Z0)                (hệ OpenCV: X phải, Y xuống, Z xa)
    P_world = R @ P_cam + t

Cả ba đều có sẵn trong ``utils`` — file này không tự viết lại toán, để không có
"hai bản của cùng một công thức" (nguồn lỗi hồi 2026-10-09, xem CHANGELOG).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import utils  # noqa: E402
import mesh_io  # noqa: E402

#: Gốc dữ liệu Replica preprocessed. Ổ **D** — CẤM ghi sang ổ C (`SIM_CONTRACT.md` mục 3).
DEFAULT_ROOT = mesh_io.replica_root()

#: Nơi neo anchor mặc định. Cũng trên ổ D.
DEFAULT_ANCHOR_DIR = Path(r"D:\replica_sim\anchors")

#: ⚠️ **ĐÃ BỊ THAY THẾ làm bộ phân loại sản xuất** — còn giữ để tái hiện quá trình
#: tìm lỗi (xem `plane_residual()`). Ngưỡng ``max−min`` trong cửa sổ 11×11, mét.
#: Nó nhầm **pixel tâm** (delta 0,0 mm) thành cạnh che chỉ vì tường nghiêng.
EDGE_SPAN_M = 0.10

#: Ngưỡng phân loại **cạnh che** đang dùng: **phần dư** sau khi khớp mặt phẳng
#: least-squares trên cửa sổ 11×11, tính bằng mét. Chốt bằng đo 8 scene × 400 điểm
#: (ngưỡng 5 cm) — xem `plane_residual()`.
#:
#: Vì sao không dùng ``max−min``: nó không phân biệt được **bước nhảy** (silhouette,
#: thứ làm mesh cross-check vô nghĩa) với **độ dốc** (tường nghiêng, hoàn toàn lành).
#: Phần dư thì có: độ dốc bị mô hình phẳng hấp thụ hết.
EDGE_RESID_M = 0.05


# ── 1. Đọc pose / depth ─────────────────────────────────────────────────


def load_pose(traj_lines: Sequence[str], frame: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dòng ``frame`` của ``traj.txt`` → ``(R, t)`` hệ OpenCV.

    ⚠️ ``opengl_to_opencv`` **lấy từ dataset, không đoán**. Bản preprocessed đã ở
    hệ OpenCV ⇒ ``False``. Đặt sai ⇒ toạ độ world lệch trung vị ~6 m (đã gặp
    2026-10-09). Hằng số nằm trong ``utils.REPLICA_CAM`` để chỉ có **một** nơi khai báo.
    """
    if not 0 <= frame < len(traj_lines):
        raise IndexError(f"frame {frame} ngoài khoảng [0, {len(traj_lines)})")
    return utils.parse_pose_line(
        traj_lines[frame],
        opengl_to_opencv=bool(utils.REPLICA_CAM["opengl_to_opencv"]),
    )


def rgb_path(root: Path, scene: str, frame: int) -> Path:
    return Path(root) / scene / "results" / f"frame{frame:06d}.jpg"


def depth_path(root: Path, scene: str, frame: int) -> Path:
    return Path(root) / scene / "results" / f"depth{frame:06d}.png"


# ── 2. Kết quả một điểm neo ─────────────────────────────────────────────


class Anchor(NamedTuple):
    """Một điểm neo lửa: (scene, frame, u, v) → toạ độ 3D + biên bản đối chứng."""

    scene: str
    frame: int
    u: float
    v: float
    valid: bool
    reason: str

    depth_gt_m: float                    # Z0 — depth GT tại pixel
    t_hit_m: float                       # khoảng cách theo trục quang tới mesh
    xyz_cam_gt: Tuple[float, float, float]
    xyz_world_gt: Tuple[float, float, float]
    xyz_world_mesh: Tuple[float, float, float]
    delta_m: float                       # |P_world_gt − P_world_mesh|

    def to_dict(self) -> Dict[str, Any]:
        """Schema JSON ổn định — `run_eval.py` và `fire_sprite.py` đọc sau này."""
        return {
            "schema": "firegrounder.anchor.v1",
            "scene": self.scene,
            "frame": self.frame,
            "pixel_uv": [round(self.u, 4), round(self.v, 4)],
            "valid": self.valid,
            "reason": self.reason,
            "depth_gt_m": None if not math.isfinite(self.depth_gt_m) else round(self.depth_gt_m, 6),
            "t_hit_m": None if not math.isfinite(self.t_hit_m) else round(self.t_hit_m, 6),
            "xyz_cam_gt": [round(x, 6) for x in self.xyz_cam_gt],
            "xyz_world_gt": [round(x, 6) for x in self.xyz_world_gt],
            "xyz_world_mesh": [None if not math.isfinite(x) else round(x, 6)
                               for x in self.xyz_world_mesh],
            "delta_m": None if not math.isfinite(self.delta_m) else round(self.delta_m, 6),
            "units": "meter",
            "axis_convention": "opencv: +X right, +Y down, +Z away from camera",
        }


# ── 3. Máy tính anchor ──────────────────────────────────────────────────


class AnchorComputer:
    """Giữ mesh + raycaster (đắt, ~10 s lần đầu) rồi trả anchor cho từng pixel (rẻ).

    Parameters
    ----------
    root : Path
        Gốc ``Replica`` (chứa ``<scene>_mesh.ply``, ``<scene>/traj.txt``, ``cam_params.json``).
    scene : str
        Ví dụ ``room0`` / ``office0``.
    load_mesh : bool
        ``False`` để chạy không cần mesh (chỉ GT) — hữu ích khi file mesh thiếu.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT, scene: str = "room0",
                 load_mesh: bool = True, verbose: bool = True):
        self.root = Path(root)
        self.scene = scene
        self.verbose = verbose
        self.intr = utils.replica_intrinsics(self.root)
        self.traj = (self.root / scene / "traj.txt").read_text().splitlines()
        self._depth_cache: Dict[int, np.ndarray] = {}
        self._pose_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.raycaster: Optional[mesh_io.Raycaster] = None
        self.mesh: Optional[mesh_io.PlyMesh] = None
        if load_mesh:
            self._load_mesh()

    # -- mesh -------------------------------------------------------------
    def _load_mesh(self) -> None:
        ply = self.root / f"{self.scene}_mesh.ply"
        if not ply.exists():
            raise FileNotFoundError(f"khong thay mesh: {ply}")
        t0 = time.time()
        self.mesh = mesh_io.read_ply_mesh(ply)
        if self.verbose:
            print(f"[{self.scene}] mesh {self.mesh.n_vertices:,} dinh, "
                  f"{self.mesh.n_faces:,} mat tu giac — {time.time()-t0:.1f}s")
        t0 = time.time()
        self.raycaster = mesh_io.Raycaster(self.mesh)
        if self.verbose:
            print(f"[{self.scene}] dang cai raycast — {time.time()-t0:.1f}s")

    # -- per-frame data (lazy) --------------------------------------------
    def depth(self, frame: int) -> np.ndarray:
        if frame not in self._depth_cache:
            p = depth_path(self.root, self.scene, frame)
            if not p.exists():
                raise FileNotFoundError(f"khong thay depth: {p}")
            self._depth_cache[frame] = utils.load_depth_map(
                p, scale=self.intr["png_depth_scale"]
            )
        return self._depth_cache[frame]

    def pose(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        if frame not in self._pose_cache:
            self._pose_cache[frame] = load_pose(self.traj, frame)
        return self._pose_cache[frame]

    # -- core -------------------------------------------------------------
    def anchor(self, u: float, v: float, frame: int) -> Anchor:
        """Pixel ``(u, v)`` → `Anchor`. **Không bao giờ ném** cho pixel hợp lệ về mặt
        chỉ số; lỗi dữ liệu (depth NaN, tia lọt, ngoài ảnh) hiện ra ở ``valid=False``.
        """
        d = self.depth(frame)
        H, W = d.shape
        ui, vi = int(math.floor(u)), int(math.floor(v))

        bad = Anchor(self.scene, frame, u, v, False, "", math.nan, math.nan,
                     (math.nan,) * 3, (math.nan,) * 3, (math.nan,) * 3, math.nan)

        if not (0 <= ui < W and 0 <= vi < H):
            return bad._replace(reason=f"ngoai anh ({W}x{H})")
        z0 = float(d[vi, ui])
        if not math.isfinite(z0) or z0 <= 0:
            return bad._replace(reason="depth GT khong hop le (NaN/0)")

        p_cam = utils.pixel_to_cam((u, v), z0, self.intr)
        R, t = self.pose(frame)
        p_world = utils.cam_to_world(p_cam, R, t)

        t_hit, xyz_mesh_w = (math.nan, (math.nan,) * 3)
        delta = math.nan
        reason = ""
        if self.raycaster is not None:
            t_hit_arr, xyz_arr = self.raycaster.cast_cam_rays([(u, v)], self.intr, R, t)
            t_hit = float(t_hit_arr[0])
            xyz_mesh_w = tuple(float(x) for x in xyz_arr[0])
            if not math.isfinite(t_hit):
                reason = "tia khong cham mesh"
            else:
                delta = float(np.linalg.norm(np.asarray(p_world) - np.asarray(xyz_mesh_w)))
        else:
            reason = "khong tai mesh (load_mesh=False)"

        return Anchor(self.scene, frame, u, v, True, reason, z0, t_hit,
                      tuple(float(x) for x in p_cam), tuple(float(x) for x in p_world),
                      xyz_mesh_w, delta)

    def anchors(self, pixels: Iterable[Sequence[float]], frame: int) -> List[Anchor]:
        return [self.anchor(float(u), float(v), frame) for u, v in pixels]


# ── 4. Nghịch đảo: world → pixel (để tự kiểm + đặt sprite) ──────────────


def world_to_pixel(xyz_world: Sequence[float], R: np.ndarray, t: np.ndarray,
                   intr: Dict[str, Any]) -> Tuple[float, float, float]:
    """Điểm world → ``(u, v, Z)`` hệ camera. ``Z <= 0`` ⇒ điểm **sau lưng** camera."""
    p_cam = np.asarray(R, dtype=float).T @ (np.asarray(xyz_world, dtype=float) - np.asarray(t, dtype=float))
    z = float(p_cam[2])
    if z <= 0:
        return math.nan, math.nan, z
    u = float(intr["fx"]) * p_cam[0] / z + float(intr["cx"])
    v = float(intr["fy"]) * p_cam[1] / z + float(intr["cy"])
    return u, v, z


def pixel_height(height_m: float, depth_m: float, intr: Dict[str, Any]) -> float:
    """Vật cao ``height_m`` mét ở cách ``depth_m`` mét → chiếm bao nhiêu pixel dọc.

    ``h_px = height_m · fy / Z``. Cần cho `fire_sprite.py` (Phase 2): lửa giả phải
    có kích thước ảnh **đúng tỉ lệ** với khoảng cách 3D, nếu không thì sai số đo
    được là lỗi của chỗ dán ảnh chứ không phải của DepthART.
    """
    return float(height_m) * float(intr["fy"]) / float(depth_m)


# ── 5. Ghi / đọc JSON ───────────────────────────────────────────────────


def save_anchors(anchors: Sequence[Anchor], out: Path | str) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "firegrounder.anchor_set.v1",
        "generated_by": "simulation/anchor.py",
        "n": len(anchors),
        "n_valid": int(sum(1 for a in anchors if a.valid)),
        "intrinsics_source": None,
        "items": [a.to_dict() for a in anchors],
    }
    if anchors:
        payload["scene"] = anchors[0].scene
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_anchors(path: Path | str) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "firegrounder.anchor_set.v1":
        raise ValueError(f"{path}: schema lạ {raw.get('schema')!r} — không phải anchor_set.v1")
    return raw["items"]


# ── 5b. Phân loại cạnh che (occlusion edge) ─────────────────────────────


def occlusion_edge_span(depth: np.ndarray, u: float, v: float, half: int = 5) -> float:
    """``max − min`` depth trong cửa sổ (2·half+1)² quanh pixel. Trả ``nan`` nếu cửa
    sổ vượt ra ngoài ảnh.

    .. deprecated:: 2026-10-16
        **Đừng dùng làm bộ phân loại** — dùng `plane_residual()`. Hàm này chỉ còn để
        tái hiện quá trình tìm lỗi. Nó nhầm pixel tâm (delta 0,0 mm) thành cạnh che,
        và ``nan`` (pixel sát border) bị gom nhầm vào nhóm "sạch".

    Vì sao phải có hàm này: mesh là lưới tam giác rời rạc, nên **đúng tại** ranh giới
    hai bề mặt thì tia qua tâm pixel có thể lọt ngay sát mép vật gần và đập vào nền
    phía sau. Khi đó ``t_hit`` nhảy tới bề mặt xa — **không phải lỗi code**, mà là
    giới hạn của việc dùng mesh làm chuẩn.

    **Cửa sổ 11×11 là BẮT BUỘC — 7×7 bỏ lọt cạnh che.** Đây là phát hiện đắt nhất
    của cả phiên, và nó **đảo ngược** kết luận trung gian của chính tôi: thấy outlier
    41–49 mm còn sót ở ngưỡng 10 cm, tôi định hạ ngưỡng xuống 5 cm. Nhưng mổ điểm
    ``(300,300)`` room0 thì 7×7 chỉ thấy span **2,9 cm** mà 11×11 thấy **25,9 cm** —
    mép thật nằm **cách tâm 5 pixel**, ngoài tầm với của 7×7.

    ::

        depth 9×9 quanh (300,300) room0 f0, đơn vị mét:
          2.042  2.046  2.050  …  2.075     ← gradient MƯỢT, 7×7 nhìn vào đây và
          2.044  2.048  2.052  …  2.077        kết luận "bề mặt liên tục"
          2.046  2.051  2.055  …  1.840     ← 2 pixel LỘT 24 cm, nằm ngoài 7×7
        ⇒ mesh đập 1,827 m trong khi GT 2,061 m, delta 261 mm. Đúng là cạnh che,
          nhưng 7×7 xếp nhầm nó vào nhóm sạch.

    Quét **đồng thời 2 trục (cửa sổ × ngưỡng)**, 8 scene × 400 điểm, seed 0, frame 0.
    Tiêu chí: worst-scene interior MAX < 25 mm, và trong các cấu hình đạt được thì
    giữ nhiều điểm sạch nhất. Ô đậm = chốt.

    | cửa sổ | ngưỡng | % điểm sạch | interior med | p95 | worst-scene MAX | >25 mm | >50 mm |
    |---|---|---|---|---|---|---|---|
    | 7×7 | 5 cm | 85,1 % | 1,83 mm | 3,95 | 9,1 mm | 0 | 0 |
    | 7×7 | 10 cm | 95,0 % | 2,01 mm | 5,68 | **49,3 mm** | 2 | 0 |
    | **11×11** | **10 cm** | **87,7 %** | **1,93 mm** | **4,57** | **14,5 mm** | **0** | **0** |
    | 11×11 | 15 cm | 92,5 % | 2,00 mm | 5,50 | **49,3 mm** | 2 | 0 |
    | 15×15 | 15 cm | 87,8 % | 1,96 mm | 4,91 | 14,5 mm | 0 | 0 |
    | 15×15 | 30 cm | 94,6 % | 2,02 mm | 6,06 | **257,9 mm** | 4 | 2 |

    Ba kết luận:

    1. **Ngưỡng 0,10 ban đầu ĐÚNG; thủ phạm là ``half=3``.** Muốn sạch, 7×7 buộc
       phải hạ xuống 5 cm (mất 10 % dữ liệu, còn 85,1 %); 11×11 ở **đúng ngưỡng cũ**
       đã sạch mà vẫn giữ 87,7 %. Nếu chỉ quét một trục rồi chốt, tôi đã ghi
       "0,05 là số đo ra" vào docstring — **số liệu đúng kết luận, sai nguyên nhân.**
       Bài học: khi một tham số cho kết quả xấu, quét cả tham số *bên cạnh* trước
       khi đổ lỗi cho nó.
    2. **Cả hai điểm đã mổ đều bị bắt ở cấu hình chốt:** (300,300) span 25,9 cm và
       (364,592) span 79,0 cm — đều > 10 cm. Ở 7×7 thì (300,300) **trượt**.
    3. **15×15 không đáng:** cùng 87,8 % sạch, cùng MAX 14,5 mm, nhưng cửa sổ to hơn
       ⇒ chậm hơn và dễ gộp mép của vật xa vào label.

    Tỉ lệ cạnh che vẫn **rất không đều giữa scene** ⇒ cổng "dưới 5 %" từng đặt là số
    bịa, sẽ chặn oan. Điều phép kiểm cần thật ra chỉ là **đủ lực** ⇒ cổng:
    "còn ≥ 250 điểm sạch trong 400" (cấu hình chốt cho ~351).

    **Hệ quả cho Phase 2:** pixel neo lửa không nên đặt lên cạnh che — ở đó mesh
    cross-check mất nghĩa, còn depth GT **vẫn dùng được** (depth GT là giá trị
    dataset đóng gói, không suy ra từ lưới). Viewer tô cam những pixel này.
    """
    H, W = depth.shape
    ui, vi = int(math.floor(u)), int(math.floor(v))
    if not (half <= ui < W - half and half <= vi < H - half):
        return float("nan")
    win = depth[vi - half:vi + half + 1, ui - half:ui + half + 1]
    fin = win[np.isfinite(win)]
    if fin.size == 0:
        return float("nan")
    return float(fin.max() - fin.min())


# ── 5c. Bộ phân loại cạnh che ĐANG DÙNG ─────────────────────────────────


#: Thiết kế ma trận ``[dx, dy, 1]`` cho từng cỡ cửa sổ — dựng một lần, dùng lại.
_PLANE_DESIGN: Dict[int, np.ndarray] = {}


def _plane_design(half: int) -> np.ndarray:
    if half not in _PLANE_DESIGN:
        gy, gx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
        _PLANE_DESIGN[half] = np.stack(
            [gx.ravel(), gy.ravel(), np.ones(gx.size)], 1)
    return _PLANE_DESIGN[half]


def plane_residual(depth: np.ndarray, u: float, v: float, half: int = 5) -> float:
    """``max − min`` của **phần dư** sau khi khớp mặt phẳng least-squares trên cửa
    sổ (2·half+1)² — số đo **bước nhảy** cục bộ. Trả ``nan`` nếu cửa sổ vượt ra
    ngoài ảnh hoặc có dưới 10 điểm hợp lệ.

    Vì sao hàm này tồn tại (và vì sao ``occlusion_edge_span`` bị thay thế)
    ---------------------------------------------------------------------
    ``max−min`` chỉ đo "cửa sổ này chênh nhau bao nhiêu", **không phân biệt được
    hai nguyên nhân hoàn toàn khác nhau**:

    * **tường nghiêng** — depth đổi dần, bề mặt liền mạch → mesh cross-check **hợp lệ**
    * **silhouette** — hai bề mặt chồng nhau → mesh cross-check **vô nghĩa**

    Khớp một mặt phẳng ``z = a·dx + b·dy + c`` bằng least-squares (numpy thuần,
    không cần scipy) rồi đo phần dư: **độ dốc bị ``a, b`` hấp thụ hết**, chỉ còn
    bước nhảy lọt qua.

    Ba điểm đã mổ xác (room0 f0) — khoảng cách tới ngưỡng là cả vấn đề:

    | pixel | delta GT↔mesh | ``max−min`` 11×11 | **phần dư** | bản chất |
    |---|---|---|---|---|
    | (600,340) tâm | **0,0 mm** | 14,1 cm → **nhầm EDGE** | **0,1 cm** ✅ | tường nghiêng ở 4,4 m |
    | (300,300) | 261 mm | 25,9 cm | **25,8 cm** ✅ | bước nhảy 24 cm, cách 5 px |
    | (364,592) | 837 mm | 79,0 cm | **77,2 cm** ✅ | silhouette |

    ``max−min`` tách sạch/không-sạch bằng tỉ lệ **25,9 / 14,1 = 1,8×** — sát ngưỡng
    nên tất nhiên lật. Phần dư tách bằng **25,8 / 0,1 = 258×**.

    So sánh trên 8 scene × 400 điểm (n=3200, seed 0, frame 0). "phi" = điểm tốt
    (delta < 10 mm) bị vứt oan; "sót" = điểm delta > 25 mm không bị bắt:

    | bộ phân loại | ngưỡng | % điểm sạch | interior MAX | phi | sót |
    |---|---|---|---|---|---|
    | ``max−min`` | 10 cm | 87,7 % | 14,5 mm | **335** | 0 |
    | phần dư | 1 cm | 86,2 % | 20,0 mm | 400 | 0 |
    | phần dư | 2 cm | 90,4 % | 20,0 mm | 272 | 0 |
    | phần dư | 3 cm | 92,3 % | 20,0 mm | 218 | 0 |
    | **phần dư** | **5 cm** | **93,6 %** | **20,0 mm** | **182** | **0** |
    | phần dư | 10 cm | 94,8 % | 41,4 mm | 148 | **1** ❌ |

    Chốt **5 cm**: giữ thêm 5,9 % dữ liệu so với ``max−min``, giảm 46 % điểm tốt bị
    vứt oan, interior MAX vẫn 20,0 mm (< gate 25 mm). 10 cm tuy giữ nhiều điểm hơn
    nhưng **sót 1 điểm > 25 mm** — đúng loại gate này sinh ra để bắt.

    Ba điều phải nhớ khi dùng
    ------------------------
    1. **``nan`` là lớp THỨ BA**, không phải "sạch". Pixel sát border không phân loại
       được ⇒ mesh cross-check **không đáng tin** ở đó, nhưng depth GT vẫn dùng được.
       Gom nhầm nó vào nhóm sạch sẽ làm gate interior-MAX fail oan **và** che bug thật.
    2. **Vẫn là heuristik, không phải lá chắn.** Phần dư to có thể do texture/lưới
       thô chứ không chỉ do silhouette.
    3. **Depth GT không liên quan gì tới chuyện này.** GT là giá trị dataset đóng gói;
       chỉ có *mesh cross-check* mất nghĩa tại cạnh che.

    **Hệ quả cho Phase 2:** pixel neo lửa không nên đặt lên cạnh che. Viewer tô cam
    những pixel này và in riêng cảnh báo cho lớp ``nan``.
    """
    H, W = depth.shape
    ui, vi = int(math.floor(u)), int(math.floor(v))
    if not (half <= ui < W - half and half <= vi < H - half):
        return float("nan")
    win = depth[vi - half:vi + half + 1, ui - half:ui + half + 1].ravel()
    good = np.isfinite(win)
    if good.sum() < 10:
        return float("nan")
    A = _plane_design(half)[good]
    y = win[good]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(r.max() - r.min())


def classify_anchor(depth: np.ndarray, u: float, v: float) -> str:
    """``"clean"`` / ``"edge"`` / ``"unknown"`` — lớp thứ ba là **bắt buộc**, xem
    ghi chú trong `plane_residual()`. Trả ``"unknown`` cho pixel sát border nơi cửa
    sổ phân loại không với tới.
    """
    r = plane_residual(depth, u, v)
    if not math.isfinite(r):
        return "unknown"
    return "edge" if r > EDGE_RESID_M else "clean"


# ── 6. Tự kiểm ──────────────────────────────────────────────────────────


def _selftest(root: Path, scene: str = "room0", frame: int = 0, n_rand: int = 400) -> int:
    """Kiểm chứng hình học bằng **nguồn độc lập** (mesh) — không tin nhau nội bộ.

    Điểm mấu chốt rút từ lỗi 2026-10-09: một phép kiểm chỉ so 2 đoạn code dùng chung
    ``R`` sẽ **tự thỏa mãn** (Case A ra 0.0000 m dù quy ước sai). Ở đây mỗi phép kiểm
    phải có một phía không đi qua phía kia.
    """
    import open3d as o3d  # noqa: F401  — đảm bảo mesh_io dùng được

    fails = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    ac = AnchorComputer(root, scene, verbose=True)
    intr = ac.intr
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    # (a) pixel chính giữa quang trục → X = Y = 0, Z = depth GT
    a = ac.anchor(cx, cy, frame)
    check("pixel tam → X_cam = Y_cam = 0",
          a.valid and abs(a.xyz_cam_gt[0]) < 1e-9 and abs(a.xyz_cam_gt[1]) < 1e-9,
          f"xyz_cam={tuple(round(x, 9) for x in a.xyz_cam_gt)}")
    check("Z_cam == depth GT", a.valid and abs(a.xyz_cam_gt[2] - a.depth_gt_m) < 1e-12,
          f"{a.xyz_cam_gt[2]:.6f} vs {a.depth_gt_m:.6f}")

    # (b) round-trip world → pixel (phép nghịch KHÔNG đi qua depth GT nữa)
    R, t = ac.pose(frame)
    u2, v2, _ = world_to_pixel(a.xyz_world_gt, R, t, intr)
    check("round-trip the pixel sai so < 1e-9 px",
          abs(u2 - cx) < 1e-9 and abs(v2 - cy) < 1e-9,
          f"duoc ({u2:.12f}, {v2:.12f})")

    # (c) đối chứng ĐỘC LẬP bằng mesh — phép kiểm không tự thỏa mãn
    d = ac.depth(frame)
    ys, xs = np.nonzero(np.isfinite(d))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ys), size=min(n_rand, len(ys)), replace=False)
    res = ac.anchors([(float(xs[i]), float(ys[i])) for i in sel], frame)
    ok = [r for r in res if r.valid and math.isfinite(r.delta_m)]
    deltas = np.array([r.delta_m for r in ok])
    check("toan bo pixel ngau nhien co GT + mesh hit",
          len(ok) >= 0.98 * len(res), f"{len(ok)}/{len(res)}")
    print(f"  --- delta |GT - MESH| toan bo (n={len(deltas)}): "
          f"median {np.median(deltas)*1000:6.2f} mm  "
          f"p95 {np.percentile(deltas, 95)*1000:6.2f} mm  "
          f"max {deltas.max()*1000:7.2f} mm")

    # (c2) tach theo CANH CHE — day moi la phan kiem bat duoc bug quy uoc truc.
    # Bug he thong se lam sai **moi** diem, ke ca diem tren be mat lien tuc; con
    # outlier o canh che la tinh chat cua du lieu (xem doc `plane_residual`).
    #
    # BA lop, khong phai hai. Ban dau viet ``is_edge = finite & (span > ngam)`` rồi
    # lay ``~is_edge`` lam "sach" — sai o cho ``nan`` (pixel sat bo anh) bi gom vao
    # nhom sach, vua lam gate MAX fail oan, vua che mat bug that.
    resid = np.array([plane_residual(d, r.u, r.v) for r in ok])
    known = np.isfinite(resid)
    is_edge = known & (resid > EDGE_RESID_M)
    is_inner = known & (resid <= EDGE_RESID_M)
    is_unk = ~known
    inner, edge = deltas[is_inner], deltas[is_edge]
    print(f"  --- phan loai bang PHAN DU MAT PHANG (ngam {EDGE_RESID_M*100:.0f} cm, "
          f"cua so 11x11): {int(is_inner.sum())} sach, {int(is_edge.sum())} canh che "
          f"({is_edge.mean()*100:.1f}%), {int(is_unk.sum())} khong phan loai duoc")
    for lbl, arr in (("be mat lien tuc  ", inner), ("canh che         ", edge)):
        if len(arr):
            print(f"  delta[{lbl}] (n={len(arr):3d}): median {np.median(arr)*1000:6.2f} mm  "
                  f"p95 {np.percentile(arr, 95)*1000:6.2f} mm  max {arr.max()*1000:7.2f} mm")
    if is_unk.any():
        print(f"  delta[khong phan loai ] (n={int(is_unk.sum()):3d}): "
              f"max {deltas[is_unk].max()*1000:7.2f} mm  (KHONG tinh vao gate)")
    # Lop 'unknown' phai HIEN, khong im lang — neu khong thi pixel sat bo anh
    # tu dong thanh 'sach' va gate gia mau la qua.
    check("lop khong phan loai duoc duoc tach rieng (khong gom vao sach)",
          len(inner) + len(edge) + int(is_unk.sum()) == len(deltas),
          f"{len(inner)} + {len(edge)} + {int(is_unk.sum())} = {len(deltas)}")

    med_i = float(np.median(inner)) if len(inner) else float("inf")
    p95_i = float(np.percentile(inner, 95)) if len(inner) else float("inf")
    max_i = float(inner.max()) if len(inner) else float("inf")
    # Ngưỡng theo **bộ phân loại phần dư @ 5 cm** (bảng trong docstring
    # plane_residual, 8 scene gộp, n=3000 diem sach): interior MAX worst-scene
    # 20,0 mm · 0 diem > 25 mm · 0 diem soc.
    # Lich su: max-min 7x7 → p95<50mm, max<250mm (rong, bug 20cm van lot);
    #          max-min 11x11 → p95<10mm, max<25mm (dung nhung 'phi' 335 diem tot).
    check("median delta < 10 mm (mesh doc lap khop GT, diem lien tuc)",
          med_i < 0.010, f"{med_i*1000:.2f} mm")
    check("p95 delta < 10 mm (diem lien tuc)", p95_i < 0.010, f"{p95_i*1000:.2f} mm")
    check("khong co diem lien tuc nao lech > 25 mm (bug quy uoc truc lech hang met)",
          max_i < 0.025, f"max {max_i*1000:.1f} mm")
    check("khong con diem lien tuc nao > 50 mm",
          int((inner > 0.050).sum()) == 0, f"{int((inner > 0.050).sum())} diem")
    # Cổng "cạnh che < 5%" cũ là số BỊA — tỉ lệ thật dao động mạnh theo scene.
    # Điều phép kiểm cần là đủ LỰC, không phải ít điểm bị loại.
    # 8 scene @ phan du 5 cm: 93,6 % sach ⇒ ~374/400; gate 250 de du margin.
    check("con du >= 250/400 diem sach de phep kiem co luc (8 scene: 93,6 % sach)",
          len(inner) >= 250,
          f"{len(inner)}/{len(deltas)} sach = {len(inner) / max(1, len(deltas)) * 100:.1f}%")
    # 'Phi' = diem TOT (delta<10mm) bi gan nham EDGE. Day la chi so CHAT LUONG bo
    # phan loai, khong phai chi so hinh hoc — do 8 scene: 182/3200 (5,7 %).
    waste = int((is_edge & (deltas < 0.010)).sum())
    check("bo phan loai khong vuot diem tot qua 10% (chi so 'phi')",
          waste <= 0.10 * len(deltas),
          f"{waste}/{len(deltas)} = {waste / max(1, len(deltas)) * 100:.1f}% bi vot oan")

    # (d) điểm neo phải nằm trong phòng, không bay lên trần / sau lưng camera
    cam_pos = np.asarray(t, dtype=float)
    dists = np.array([np.linalg.norm(np.asarray(r.xyz_world_gt) - cam_pos) for r in ok])
    check("moi GT nam cach camera 0.1–12 m (khong loai bo)",
          bool(((dists > 0.1) & (dists < 12.0)).all()),
          f"min {dists.min():.3f} max {dists.max():.3f} m")

    # (e) pixel hỏng → valid False, khong nem exception
    bad = ac.anchor(cx, cy, frame)  # hợp lệ, dùng làm mốc
    oob = ac.anchor(-5.0, 10.0, frame)
    check("pixel ngoai anh → valid=False", not oob.valid, f"reason={oob.reason!r}")
    zero_uv = ac.anchor(0.0, 0.0, frame)
    check("pixel goc khong bi crash", zero_uv.valid or not math.isfinite(d[0, 0]),
          f"valid={zero_uv.valid} reason={zero_uv.reason!r}")
    check("Anchor.to_dict co du truong", set(bad.to_dict()) >= {
        "schema", "scene", "frame", "pixel_uv", "depth_gt_m",
        "xyz_world_gt", "xyz_world_mesh", "delta_m"}, "")

    # (f) pixel_height — thang đo cho sprite lửa
    h_px = pixel_height(0.30, 3.0, intr)
    check("cong thuc h_px = h·fy/Z", abs(h_px - 0.30 * fy / 3.0) < 1e-12,
          f"lua 30 cm cach 3 m → {h_px:.1f} px")

    # (g) JSON round-trip
    tmp = DEFAULT_ANCHOR_DIR / f"_selftest_{scene}.json"
    save_anchors([bad, oob], tmp)
    back = load_anchors(tmp)
    # so sanh co dung sai — to_dict() lam tron 6 le so, dung '==' se FAIL oan
    rt = back[0]["xyz_world_gt"]
    close = len(rt) == 3 and all(abs(a - b) <= 5e-7 for a, b in zip(rt, bad.xyz_world_gt))
    check("JSON round-trip giu toa do (trong 5e-7 m)",
          len(back) == 2 and close,
          f"tmp={tmp}  {rt}")
    tmp.unlink()

    print(f"\n{'TAT CA PASS' if fails == 0 else f'{fails} FAIL'}")
    return 0 if fails == 0 else 1


# ── 7. CLI ──────────────────────────────────────────────────────────────


def _parse_uv(tok: str) -> Tuple[float, float]:
    parts = tok.replace(";", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"pixel phai la 'u,v', gap {tok!r}")
    return float(parts[0]), float(parts[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pixel → toạ độ 3D anchor + đối chứng mesh")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--pixels", nargs="*", type=_parse_uv, default=None,
                    help="danh sach 'u,v' (pixel, co the la float)")
    ap.add_argument("--out", default=None, help="ghi JSON (mac dinh <anchors>/<scene>_f<frame>.json)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        return _selftest(Path(args.root), scene=args.scene, frame=args.frame)

    if not args.pixels:
        ap.print_help()
        return 2

    ac = AnchorComputer(Path(args.root), args.scene)
    res = ac.anchors(args.pixels, args.frame)
    print(f"\n{'u':>8s} {'v':>8s} {'Z_GT(m)':>9s} {'t_hit(m)':>9s} "
          f"{'X':>8s} {'Y':>8s} {'Z':>8s} {'delta(mm)':>10s}  valid")
    for r in res:
        w = r.xyz_world_gt
        dl = f"{r.delta_m*1000:10.2f}" if math.isfinite(r.delta_m) else f"{'-':>10s}"
        zz = f"{r.depth_gt_m:9.3f}" if math.isfinite(r.depth_gt_m) else f"{'-':>9s}"
        th = f"{r.t_hit_m:9.3f}" if math.isfinite(r.t_hit_m) else f"{'-':>9s}"
        ws = f"{w[0]:8.3f} {w[1]:8.3f} {w[2]:8.3f}" if r.valid else f"{'-':>8s} {'-':>8s} {'-':>8s}"
        print(f"{r.u:8.1f} {r.v:8.1f} {zz} {th} {ws} {dl}  {r.valid} {r.reason}")

    out = Path(args.out) if args.out else DEFAULT_ANCHOR_DIR / f"{args.scene}_f{args.frame:06d}.json"
    save_anchors(res, out)
    print(f"\nDa ghi {len(res)} anchor → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())