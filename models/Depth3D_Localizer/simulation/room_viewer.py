"""Trình mô phỏng tương tác: **click pixel trên ảnh RGB → thấy điểm 3D trong phòng**.

Hai cửa sổ chạy cạnh nhau trong cùng một tiến trình
-----------------------------------------------
| Cửa sổ | Nội dung | Vai trò |
|--------|----------|---------|
| RGB (OpenCV) | ảnh ``frame%06d.jpg`` gốc, **photorealistic** | bạn bấm vào đây — đây là thứ duy nhất vào đường đo |
| 3D (Open3D) | mesh màu đã giảm + marker + đoạn nối | bạn nhìn thấy lệch lạc ở đâu trong phòng |

Mỗi cú click sinh **hai** điểm:

* **Xanh lá** = ``P_GT`` — pixel back-project bằng **depth GT** × intrinsics.
  Sai số so với thực **định nghĩa = 0**; đây là điểm neo lửa thật.
* **Đỏ** = ``P_MESH`` — tia từ camera đập vào **mesh**. Lệch ~2 mm.
* **Vàng** = đoạn nối hai điểm. **Không thấy thanh vàng ⇒ hình học đúng; thấy thanh
  dài ⇒ có bug ở code, không phải ở model.**

Ba tầng số — đừng lẫn (chi tiết trong ``anchor.py``)
----------------------------------------------------
``GT`` (0 m, định nghĩa) · ``MESH`` (~2 mm, **máy bắt lỗi code**) ·
``PRED`` DepthART (~312 mm, **thứ cần đo**, chấm ở ``run_eval.py``).

Vì sao viewer **không import torch**
------------------------------------
Máy dev RAM chật; ``import torch`` khi chỉ còn ~1 GB trống thì OpenBLAS abort cứng
cả tiến trình (``try/except`` không bắt được — xem ``AGENTS.md`` → ``OpenBLAS error``).
Viewer chỉ cần numpy + open3d + cv2, mở được mọi lúc. DepthART chấm riêng offline.

Phím
----
::

    click trai   dat anchor          s   luu JSON (o D:)
    u           hoan tai cuoi        c   xoa tat ca
    [ / ]       frame truoc / sau    q / ESC   thoat

Quy ước trục: ``+X`` phải, ``+Y`` **xuống**, ``+Z`` ra xa camera (OpenCV).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import anchor as A  # noqa: E402
import mesh_io  # noqa: E402

IMG_WIN = "Replica RGB - click de dat anchor"

#: Màu (RGB 0..1) — xanh lá = GT, đỏ = mesh, vàng = đoạn nối.
COL_GT = (0.10, 0.85, 0.20)
COL_MESH = (0.95, 0.15, 0.10)
COL_LINK = (1.00, 0.85, 0.00)

#: Pixel bơm tự động ở chế độ ``--smoke``. Có sẵn 2 điểm **cạnh che** đã đo
#: (364,592) và (108,645) để ép đường đi delta lớn — nếu chỉ bấm điểm đẹp thì
#: smoke test không bao giờ chạm nhánh xử lý outlier.
SMOKE_PIXELS: Tuple[Tuple[float, float], ...] = (
    (600.0, 340.0), (364.0, 592.0), (900.0, 200.0), (108.0, 645.0),
)


# ── 0. Luật ổ đĩa ───────────────────────────────────────────────────────


def assert_drive_d(path: Path | str, what: str) -> Path:
    """Chặn mọi ghi ra ổ khác **D** (``AGENTS.md`: ổ C chỉ còn ~0,8 GB).

    Đường tương đối (không có ``drive``) được phép — ``cwd`` hiện tại là D.
    """
    p = Path(path)
    drv = (p.drive or "").upper()
    if drv and drv != "D:":
        raise SystemExit(
            f"{what} tro toi {p} — o {drv} CAM ghi (quy tac AGENTS.md). "
            "Chuyen ve D:\\replica_sim\\."
        )
    return p


# ── 1. Toán quay: trụ nối hai điểm ──────────────────────────────────────


def rot_align_z(direction: Sequence[float]) -> np.ndarray:
    """Ma trận quay đưa trục ``+Z`` về ``direction`` (Rodrigues, không cần scipy).

    Cần vì ``o3d...create_cylinder`` dựng trụ dọc theo ``+Z`` quanh gốc.
    """
    d = np.asarray(direction, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.eye(3)
    d = d / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, d))
    if s < 1e-9:  # song song (cung chieu hoac nguoc)
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ── 2. Mô hình (không cần cửa sổ — selftest chạy được headless) ──────────


class RoomViewer:
    """Giữ ``AnchorComputer`` + danh sách anchor + hình học marker.

    Tách làm hai lớp có chủ đích:
    * **mô hình** (class này, không window) — ``--selftest`` kiểm được;
    * **cửa sổ** (``run``) — cần người thật nhìn, chỉ smoke test được một phần.
    """

    def __init__(self, scene: str = "room0", frame: int = 0,
                 root: Path | str | None = None,
                 decimate: int = 300_000, verbose: bool = True):
        self.ac = A.AnchorComputer(root or A.DEFAULT_ROOT, scene, verbose=verbose)
        self.root = self.ac.root
        self.scene = scene
        self.frame = frame
        self.decimate = decimate
        self.verbose = verbose

        self.anchors: List[A.Anchor] = []
        self.pending: List[Tuple[float, float]] = []
        self.status: str = ""
        self.quit = False

        self.n_frames = len(self.ac.traj)
        self._marker_names: List[str] = []
        self._marker_objs: List[Any] = []
        self._img: Optional[np.ndarray] = None
        self.viz = None          # o3d.visualization.Visualizer | None
        self._o3d = None
        self._cv2 = None

    # -- dữ liệu ----------------------------------------------------------
    def depth(self) -> np.ndarray:
        return self.ac.depth(self.frame)

    def edge_resid(self, u: float, v: float) -> float:
        return A.plane_residual(self.depth(), u, v)

    def is_edge(self, u: float, v: float) -> bool:
        r = self.edge_resid(u, v)
        return bool(math.isfinite(r) and r > A.EDGE_RESID_M)

    def is_unclassified(self, u: float, v: float) -> bool:
        """``nan`` = cửa sổ vượt ra ngoài ảnh → **không phân loại được**.

        Phải tách thành lớp thứ ba, không được gom vào "sạch": điểm border vẫn có
        thể là cạnh che, và nếu gom nhầm thì nó vừa làm gate interior-MAX fail oan,
        vừa che mất bug thật nếu có.
        """
        return not math.isfinite(self.edge_resid(u, v))

    # -- thao tác ---------------------------------------------------------
    def add_pixel(self, u: float, v: float) -> A.Anchor:
        an = self.ac.anchor(u, v, self.frame)
        self.anchors.append(an)
        resid = self.edge_resid(an.u, an.v)
        edge = bool(math.isfinite(resid) and resid > A.EDGE_RESID_M)
        idx = len(self.anchors) - 1

        if not an.valid:
            self.status = f"[#{idx}] ({an.u:.0f},{an.v:.0f}) KHONG hop le: {an.reason}"
            print(f"  {self.status}")
            return an

        dl = an.delta_m * 1000 if math.isfinite(an.delta_m) else float("nan")
        self.status = (f"[#{idx}] ({an.u:.0f},{an.v:.0f}) f{self.frame}  "
                       f"Z_GT={an.depth_gt_m:.3f} m  t_hit={an.t_hit_m:.3f} m  "
                       f"delta={dl:.1f} mm")
        print(f"  {self.status}")
        print(f"      world GT = ({an.xyz_world_gt[0]:+.3f}, {an.xyz_world_gt[1]:+.3f}, "
              f"{an.xyz_world_gt[2]:+.3f}) m")
        if edge:
            print(f"      ^ CANH CHE (phan du mat phang {resid*100:.1f} cm > "
                  f"{A.EDGE_RESID_M*100:.0f} cm): mesh cross-check VO NGHIA o day — "
                  f"GT van dung duoc.")
            print("        → khong dat neo lua tren pixel nay (Phase 2).")
        elif not math.isfinite(resid):
            print("      ? KHONG PHAN LOAI DUOC (cua so vuot ngoai anh) — do mesh o day "
                  "khong tin duoc, GT van dung.")
        elif math.isfinite(an.delta_m) and an.delta_m > 0.05:
            print(f"      ! delta {dl:.0f} mm lon hon ngam dinh 50 mm — kiem tra lai.")
        return an

    def undo(self) -> None:
        if self.anchors:
            gone = self.anchors.pop()
            self.status = f"hoan: bo anchor ({gone.u:.0f},{gone.v:.0f}) f{gone.frame}"
            print(f"  {self.status}")

    def clear(self) -> None:
        n = len(self.anchors)
        self.anchors.clear()
        self.status = f"xoa {n} anchor"
        print(f"  {self.status}")

    def change_frame(self, delta: int) -> None:
        nf = max(0, min(self.n_frames - 1, self.frame + delta))
        if nf == self.frame:
            return
        self.frame = nf
        self._img = None            # buoc tai lai anh
        self.status = f"frame {nf}/{self.n_frames-1}"
        print(f"  {self.status}")

    def save(self, out: Path | str | None = None) -> Path:
        out = assert_drive_d(
            Path(out) if out else A.DEFAULT_ANCHOR_DIR / f"{self.scene}_interactive.json",
            "file anchor",
        )
        p = A.save_anchors(self.anchors, out)
        nv = sum(1 for a in self.anchors if a.valid)
        self.status = f"da luu {len(self.anchors)} anchor ({nv} hop le) → {p}"
        print(f"  {self.status}")
        return p

    # -- hình học marker --------------------------------------------------
    def _sphere(self, center: Sequence[float], radius: float, color: Tuple[float, float, float]):
        g = self._o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
        g.translate(np.asarray(center, dtype=float))
        g.compute_vertex_normals()
        g.paint_uniform_color(color)
        return g

    def _cylinder(self, a: Sequence[float], b: Sequence[float],
                  radius: float, color: Tuple[float, float, float]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            return None
        g = self._o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=10, split=1)
        g.rotate(rot_align_z(d / length), center=(0.0, 0.0, 0.0))
        g.translate((a + b) / 2.0)
        g.compute_vertex_normals()
        g.paint_uniform_color(color)
        return g

    def marker_geoms(self) -> List[Tuple[str, Any]]:
        """Danh sách ``(ten, geometry)`` cho mọi anchor. **Không cần visualizer.**"""
        out: List[Tuple[str, Any]] = []
        for i, an in enumerate(self.anchors):
            if not an.valid or not math.isfinite(an.depth_gt_m):
                continue
            # bán kính tăng theo khoảng cách để điểm xa vẫn nhìn thấy
            r = max(0.018, 0.006 * an.depth_gt_m)
            out.append((f"mk{i}_gt", self._sphere(an.xyz_world_gt, r, COL_GT)))
            if math.isfinite(an.t_hit_m) and all(math.isfinite(x) for x in an.xyz_world_mesh):
                out.append((f"mk{i}_mesh", self._sphere(an.xyz_world_mesh, r, COL_MESH)))
                cyl = self._cylinder(an.xyz_world_gt, an.xyz_world_mesh, r * 0.35, COL_LINK)
                if cyl is not None:
                    out.append((f"mk{i}_link", cyl))
        return out

    def camera_frame_geom(self, length: float = 0.30):
        """Trục toạ độ tại vị trí camera: X đỏ, Y xanh lá, Z xanh dương."""
        R, t = self.ac.pose(self.frame)
        t = np.asarray(t, dtype=float)
        pts = [t, t + R @ np.array([length, 0, 0]),
               t + R @ np.array([0, length, 0]), t + R @ np.array([0, 0, length])]
        ls = self._o3d.geometry.LineSet()
        ls.points = self._o3d.utility.Vector3dVector(np.array(pts))
        ls.lines = self._o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
        ls.colors = self._o3d.utility.Vector3dVector(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))
        return ls

    # -- cửa sổ -----------------------------------------------------------
    def _load_image(self) -> np.ndarray:
        p = A.rgb_path(self.root, self.scene, self.frame)
        img = self._cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"khong doc duoc anh: {p}")
        return img

    def _add(self, g, name: str, dynamic: bool = True, reset_bbox: bool = False) -> None:
        """Visualizer của Open3D xoá theo **đối tượng**, không theo tên ⇒ phải giữ
        tham chiếu. Chỉ geometry ``dynamic`` (marker) mới vào danh sách bị xoá lại;
        mesh phòng và khung camera là cố định.

        ``reset_bbox=True`` BẮT BUỘC cho geometry đầu tiên (mesh phòng): với False
        cửa sổ thật chỉ hiện nền đen (đã đo: 0 % vs 73 % pixel sáng)."""
        ok = self.viz.add_geometry(g, reset_bounding_box=reset_bbox)
        if ok and dynamic:
            self._marker_names.append(name)
            self._marker_objs.append(g)

    def sync_markers(self) -> int:
        if self.viz is None:
            return 0
        for g in self._marker_objs:
            self.viz.remove_geometry(g, reset_bounding_box=False)
        self._marker_objs.clear()
        self._marker_names.clear()
        geoms = self.marker_geoms()
        for name, g in geoms:
            self._add(g, name)
        return len(geoms)

    def _set_view_to_camera(self) -> None:
        """Đặt camera 3D **khớp góc nhìn của ảnh RGB** — để điểm bấm ra đúng chỗ.

        ``front`` = hướng nhìn (OpenCV +Z), ``up`` = −Y (OpenCV +Y chỉ xuống).
        """
        R, t = self.ac.pose(self.frame)
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        front = R @ np.array([0.0, 0.0, 1.0])
        up = R @ np.array([0.0, -1.0, 0.0])
        front = front / np.linalg.norm(front)
        up = up / np.linalg.norm(up)
        ctl = self.viz.get_view_control()
        ctl.set_lookat(t + front * 2.5)     # quay quanh điểm giữa phòng
        ctl.set_front(-front)   # Open3D: front = từ điểm nhìn TỚI camera (ngược hướng nhìn)
        ctl.set_up(up)
        ctl.set_zoom(0.45)

    def open_windows(self) -> None:
        import cv2
        import open3d as o3d
        self._cv2, self._o3d = cv2, o3d

        self._img = self._load_image()
        cv2.namedWindow(IMG_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(IMG_WIN, 1200, 680)
        cv2.setMouseCallback(IMG_WIN, _mouse_cb, self.pending)

        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(window_name=f"Phong 3D — {self.scene}", width=1280, height=800)
        disp = mesh_io.decimate_for_display(self.ac.mesh, self.decimate, verbose=self.verbose)
        self._add(disp, "_room_", dynamic=False, reset_bbox=True)
        self._add(self.camera_frame_geom(), "_camframe_", dynamic=False)
        opt = self.viz.get_render_option()
        opt.background_color = np.array([0.06, 0.06, 0.08])
        for _ in range(4):                  # cho cửa sổ có kích thước trước khi đặt camera
            self.viz.poll_events()
            self.viz.update_renderer()
        self._set_view_to_camera()
        self.sync_markers()

    def render_image(self) -> np.ndarray:
        """Ảnh + vòng marker + HUD. Trả BGR để ``cv2.imshow``."""
        cv2 = self._cv2
        if self._img is None:
            self._img = self._load_image()
        canvas = self._img.copy()

        for i, an in enumerate(self.anchors):
            if an.frame != self.frame or not an.valid:
                continue
            u, v = int(round(an.u)), int(round(an.v))
            edge = self.is_edge(an.u, an.v)
            col = (0, 165, 255) if edge else (0, 220, 0)      # BGR: cam / xanh lá
            cv2.circle(canvas, (u, v), 9, col, 2)
            cv2.circle(canvas, (u, v), 2, col, -1)
            cv2.putText(canvas, str(i), (u + 12, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        H, W = canvas.shape[:2]
        n_here = sum(1 for a in self.anchors if a.frame == self.frame and a.valid)
        lines = [
            f"{self.scene} | frame {self.frame}/{self.n_frames-1} | da dat {len(self.anchors)} diem",
            "CACH DUNG: CLICK TRAI vao anh de dat diem. Xem cua so 3D de thay do lech.",
            "Vong XANH = diem tot | vong CAM = canh che (khong tin mesh o day)",
            "u = bo diem vua dat | c = xoa HET diem | s = luu file | [ ] = doi frame | q = thoat",
        ]
        if n_here == 0 and not self.anchors:
            lines.append(">>> Chua co diem nao. Hay click vao anh. <<<")
        if self.status:
            lines.append(self.status[:110])
        y = 20
        n_lines = len(lines)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (min(W, 760), 10 + 20 * n_lines), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
        for ln in lines:
            cv2.putText(canvas, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        return canvas

    def _drain(self) -> None:
        if not self.pending:
            return
        pix = list(self.pending)
        self.pending.clear()      # xoa TAI CHO: callback chuot giu tham chieu list nay
        for u, v in pix:
            self.add_pixel(float(u), float(v))
        self.sync_markers()

    def pump(self) -> int:
        """Bơm sự kiện cho **cả hai** event loop. Trả mã phím (0 = không).

        Đóng **một trong hai** cửa sổ (bấm X) ⇒ thoát cả hai, không để tiến trình
        treo. OpenCV báo qua ``getWindowProperty(...VISIBLE)``, Open3D qua
        ``poll_events() is False``.
        """
        key = 0
        try:
            self._cv2.imshow(IMG_WIN, self.render_image())
            key = self._cv2.waitKey(1) & 0xFF
            if self._cv2.getWindowProperty(IMG_WIN, self._cv2.WND_PROP_VISIBLE) < 1:
                self.quit = True
        except Exception as e:                        # cửa sổ bị người dùng đóng
            if not self.quit:
                print(f"  [!] Lo OpenCV: {e}")
            self.quit = True
        if self.viz is not None:
            try:
                if not self.viz.poll_events():        # False = nguoi dung bam X
                    self.quit = True
                self.viz.update_renderer()
            except Exception as e:
                print(f"  [!] Lo Open3D: {e}")
                self.quit = True
        return key

    def handle_key(self, key: int) -> None:
        if key in (ord('q'), 27):
            self.quit = True
        elif key == ord('s'):
            self.save()
        elif key == ord('u'):
            self.undo()
            self.sync_markers()
        elif key == ord('c'):
            self.clear()
            self.sync_markers()
        elif key in (ord('['), 81):                  # 81 = Left (cursor)
            self.change_frame(-1)
            self._repoint_camera()
        elif key in (ord(']'), 83):                  # 83 = Right
            self.change_frame(+1)
            self._repoint_camera()

    def _repoint_camera(self) -> None:
        if self.viz is not None:
            self._set_view_to_camera()

    def run(self, smoke_seconds: float = 0.0, out: Path | str | None = None) -> Path:
        t0 = time.time()
        self.open_windows()
        if smoke_seconds > 0:
            self.pending.extend(SMOKE_PIXELS)
            print(f"  [smoke] bom {len(SMOKE_PIXELS)} pixel, chay {smoke_seconds:.0f}s")
        while not self.quit:
            self._drain()
            self.handle_key(self.pump())
            if smoke_seconds > 0 and (time.time() - t0) >= smoke_seconds:
                self.quit = True
        p = self.save(out)
        try:
            self.viz.destroy_window()
        except Exception:
            pass
        self._cv2.destroyAllWindows()
        print(f"  [smoke] xong sau {time.time()-t0:.1f}s — {len(self.anchors)} anchor")
        return p


def _mouse_cb(event, x, y, flags, param):
    """Chỉ ghi pixel vào hàng đợi — **không** tính anchor trong callback.

    Tính anchor (raycast) trong mouse callback của OpenCV là chạy trên luồng sự kiện
    của GUI; chậm thì cửa sổ đơ và Windows báo "Not Responded".
    """
    import cv2
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((float(x), float(y)))


# ── 3. Tự kiểm (headless — không mở cửa sổ) ─────────────────────────────


def _selftest(root: Path, scene: str = "room0", frame: int = 0) -> int:
    """Kiểm **mô hình** — phần cửa sổ không tự kiểm được, chỉ ``--smoke``."""
    fails = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    import open3d as o3d
    v = RoomViewer(scene=scene, frame=frame, root=root, verbose=False)
    v._o3d = o3d                       # cho phep dung _sphere/_cylinder khong can window

    intr = v.ac.intr
    check("doc intrinsics tu dataset", intr["source"] != "estimate",
          f"nguon={intr['source']} fx={intr['fx']}")

    # (a) rot_align_z — trụ nối phải chỉ đúng từ GT tới mesh
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        got = rot_align_z(d) @ np.array([0.0, 0.0, 1.0])
        errs.append(float(np.linalg.norm(got - d)))
    check("rot_align dung: +Z → moi huong (n=200)", max(errs) < 1e-9,
          f"loi lon nhat {max(errs):.2e}")
    check("rot_align la ma tran truc giao",
          float(np.abs(np.linalg.det(rot_align_z([0.3, -0.5, 1.0]))) - 1.0) < 1e-9, "")

    # (b) marker hình học. Anchor **tâm** có delta ~0 mm nên ``_cylinder()`` trả None
    #     theo đúng thiết kế (vẽ trụ dài 0 là rác). Vì vậy không được giả định
    #     "3 object / anchor" — phải tính số trụ kỳ vọng **từ delta**, rồi đối chiếu
    #     số thực đếm trong ``geoms``. Hai bên đi qua đường khác nhau ⇒ kiểm có nghĩa.
    v.add_pixel(intr["cx"], intr["cy"])          # #0 — delta ~0 mm, KHÔNG có trụ nối
    v.add_pixel(300.0, 300.0)                    # #1 — delta lớn, có trụ nối để nhìn
    geoms = v.marker_geoms()
    gdict = dict(geoms)

    def has_pair(a) -> bool:
        return bool(a.valid and math.isfinite(a.t_hit_m)
                    and all(math.isfinite(x) for x in a.xyz_world_mesh))

    n_pair = sum(1 for a in v.anchors if has_pair(a))
    n_link_expect = sum(1 for a in v.anchors
                        if has_pair(a) and math.isfinite(a.delta_m) and a.delta_m > 1e-4)
    n_link = sum(1 for n in gdict if n.endswith("_link"))
    check("so tru noi = so anchor co delta > 0.1 mm (tinh tu delta, khong dem lai)",
          n_link == n_link_expect and n_link_expect >= 1,
          f"{n_link} thuc vs {n_link_expect} ky vong")
    check("tong object = 2·(anchor co mesh hit) + so tru noi",
          len(geoms) == 2 * n_pair + n_link,
          f"{len(geoms)} = 2·{n_pair} + {n_link}")
    check("ten marker khong trung (quyet dinh khi xoa)",
          len(set(gdict)) == len(geoms), f"{len(set(gdict))} ten")

    # (c) đầu trụ nối đúng hai điểm — kiểm bằng số, không bằng mắt
    li = next((i for i, a in enumerate(v.anchors)
               if has_pair(a) and math.isfinite(a.delta_m) and a.delta_m > 1e-4), None)
    check("co it nhat 1 anchor delta > 0.1 mm de kiem tru noi", li is not None,
          f"delta = {[round(a.delta_m*1000, 1) for a in v.anchors]} mm")
    if li is not None:
        an = v.anchors[li]
        cyl = gdict[f"mk{li}_link"]
        ctr = np.asarray(cyl.get_center(), dtype=float)
        mid = (np.asarray(an.xyz_world_gt) + np.asarray(an.xyz_world_mesh)) / 2.0
        check("tam tru noi = trung diem GT/mesh", float(np.linalg.norm(ctr - mid)) < 1e-3,
              f"lech {np.linalg.norm(ctr - mid)*1000:.3f} mm")
        bb = np.asarray(cyl.get_min_bound()), np.asarray(cyl.get_max_bound())
        diag = float(np.linalg.norm(bb[1] - bb[0]))
        check("chieu dai tru noi = delta (khong roi, khong trung)",
              an.delta_m - 1e-6 <= diag <= an.delta_m * 1.06 + 1e-3,
              f"bb {diag*1000:.1f} mm vs delta {an.delta_m*1000:.1f} mm")

    # (d) Bộ phân loại phải tách **"bước nhảy"** khỏi **"độ dốc"**.
    #     Đây là lỗi đã sửa: `max−min` trên cửa sổ 11×11 gán nhầm **pixel tâm**
    #     (delta 0,0 mm — điểm sạch nhất ảnh) là cạnh che, chỉ vì tường ở 4,4 m
    #     nghiêng 14 cm qua 11 px. max−min tách 14,1 vs 25,9 cm = **1,8×** (sat
    #     ngưỡng, tất nhiên nhầm); phần dư sau khi khớp mặt phẳng least-squares
    #     tách 0,1 vs 25,8 cm = **258×**, vì độ dốc bị mô hình hấp thụ hết.
    if scene == "room0" and frame == 0:
        for pu, pv, lbl, want in (
                (600.0, 340.0, "tam — delta 0.0 mm, tuong nghien", False),
                (300.0, 300.0, "buoc nhay 24 cm cach 5 px", True),
                (364.0, 592.0, "silhouette cheo 79 cm", True)):
            rv = v.edge_resid(pu, pv)
            check(f"phan loai dung ({pu:.0f},{pv:.0f}) → "
                  f"{'CANH CHE' if want else 'SACH'} [{lbl}]",
                  v.is_edge(pu, pv) == want, f"phan du {rv*100:.1f} cm")
        r_c = v.edge_resid(intr["cx"], intr["cy"])
        r_e = v.edge_resid(364.0, 592.0)
        # Không chỉ "đúng nhãn" — phải **xa ngưỡng cả hai phía**. Điểm sát ngưỡng
        # thì đổi scene/frame là lật nhãn, và selftest sẽ không báo trước.
        check("ca hai cach deu xa ngam ≥3× (khong phai diem sat ngam)",
              r_c < A.EDGE_RESID_M / 3.0 and r_e > A.EDGE_RESID_M * 3.0,
              f"tam {r_c*100:.2f} cm · ngam {A.EDGE_RESID_M*100:.0f} cm · "
              f"canh che {r_e*100:.1f} cm")
        check("diem sat bo anh → nan = KHONG PHAN LOAI DUOC, khong goi la sach",
              v.is_unclassified(2.0, 2.0) and not v.is_edge(2.0, 2.0),
              f"phan du tai (2,2) = {v.edge_resid(2.0, 2.0)}")

    # (e) Raycast BẮT BUỘC dùng mesh gốc.
    #
    # Hai kiểm tôi viết trước đó đều **vô nghĩa** và đã bỏ:
    #   · ``mesh.n_faces == mesh.n_faces``  → tautology (đúng cái bẫy AGENTS.md ghi).
    #   · "mesh gốc vs mesh giảm phải cho kết quả KHÁC nhau" → đo thật: ở 200k tam
    #     giác, lệch lớn nhất so với mesh 1,9 triệu tam giác chỉ **0,6 mm** trên 600
    #     tia. Gate đó không phân biệt được gì, chỉ ngẫu nhiên FAIL.
    #
    # Kiểm CÓ lực, và độc lập với Open3D: **đếm số mặt thẳng từ header PLY bằng tay**
    # (chuỗi ascii trước ``end_header``). Đường này không đi qua `mesh_io` lẫn
    # `open3d`, nên nếu một trong hai âm thầm giảm mesh thì hai số sẽ lệch nhau.
    # Con số này cũng chính là bằng chứng số học byte đã khoá ngày 2026-10-16
    # (953.647 mặt × 17 byte + 954.492 × 27 + 293 = đúng kích thước file).
    with open(v.ac.mesh.path, "rb") as fh:
        raw = fh.read(1 << 16)
    cut = raw.index(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:cut].decode("ascii", "replace")
    n_face_hdr = int([ln.split()[-1] for ln in hdr.splitlines()
                      if ln.startswith("element face")][0])
    check("so mat mesh goc = header PLY doc tay (khong qua Open3D/mesh_io)",
          v.ac.mesh.n_faces == n_face_hdr,
          f"mesh_io {v.ac.mesh.n_faces:,} vs header {n_face_hdr:,}")

    rc0, nf0 = v.ac.raycaster, v.ac.mesh.n_faces
    full = mesh_io.quad_to_triangle_faces(v.ac.mesh.faces)
    disp = mesh_io.decimate_for_display(v.ac.mesh, 200_000, verbose=False)
    check("display mesh nhe hon mesh raycast", len(disp.triangles) < len(full),
          f"{len(disp.triangles):,} < {len(full):,} tam giac")
    check("dung display mesh KHONG lam doi mesh/ raycaster goc",
          v.ac.raycaster is rc0 and v.ac.mesh.n_faces == nf0 == n_face_hdr,
          f"{v.ac.mesh.n_faces:,} mat giu nguyen")

    # Ghi lại THẬT để agent sau khỏi nghi ngờ (không phải gate): decimate 1,9M → 200k
    # chỉ lệch 0,6 mm nên **an toàn cho hiển thị**; nhưng gate ở trên là số mặt, vì
    # "kết quả gần nhau" không chứng minh được gì — nó đúng với cả mesh sai.
    dv = np.asarray(disp.vertices, dtype=np.float64)
    dt = np.asarray(disp.triangles, dtype=np.int32)
    low = mesh_io.PlyMesh(
        xyz=dv, rgb=np.zeros((len(dv), 3)), normals=np.zeros((len(dv), 3)),
        # gia dinh tu giac: trung dinh thu 3 ⇒ 1 tam giac suy bien (khong bao gio cham)
        faces=np.stack([dt[:, 0], dt[:, 1], dt[:, 2], dt[:, 2]], 1),
        n_vertices=len(dv), n_faces=len(dt), header_bytes=0, path="<decimated>")
    d = v.depth()
    ys, xs = np.nonzero(np.isfinite(d))
    rng2 = np.random.default_rng(1)
    sel2 = rng2.choice(len(ys), size=min(600, len(ys)), replace=False)
    uv2 = np.stack([xs[sel2], ys[sel2]], 1)
    R, t = v.ac.pose(v.frame)
    t_full, _ = v.ac.raycaster.cast_cam_rays(uv2, intr, R, t)
    t_low, _ = mesh_io.Raycaster(low).cast_cam_rays(uv2, intr, R, t)
    m = np.isfinite(t_full) & np.isfinite(t_low)
    if m.any():
        print(f"  [info] decimate 200k: lech lon nhat so voi mesh goc = "
              f"{np.abs(t_full[m] - t_low[m]).max()*1000:.2f} mm tren {int(m.sum())} tia "
              f"→ an toan de HINH THI, khong dung de DO")
    check("median |t_hit(mesh goc) - depth GT| < 10 mm (duong do khong bi hong)",
          float(np.median(np.abs(t_full[m] - d[ys[sel2], xs[sel2]][m]))) * 1000 < 10.0,
          f"{np.median(np.abs(t_full[m] - d[ys[sel2], xs[sel2]][m]))*1000:.2f} mm")

    # (f) undo / clear
    n0 = len(v.anchors)
    v.undo()
    check("undo bo dung 1 anchor", len(v.anchors) == n0 - 1, f"{n0} → {len(v.anchors)}")
    v.clear()
    check("clear ve 0", len(v.anchors) == 0 and len(v.marker_geoms()) == 0, "")

    # (f2) đường CHUỘT: callback giữ tham chiếu list lúc đăng ký. Nếu _drain tạo list
    #      mới (self.pending = []) thì mọi click SAU click đầu rơi vào list mồ côi
    #      (bug user gặp: bấm 'c' xong không đặt lại được).
    import cv2
    held = v.pending                       # đúng thứ mà setMouseCallback giữ
    got = []
    for rnd in range(3):
        _mouse_cb(cv2.EVENT_LBUTTONDOWN, 600 + rnd * 10, 340, 0, held)
        v._drain()
        got.append(len(v.anchors))
    check("click chuot lien tiep deu vao anchor (3 lan)", got == [1, 2, 3], f"so anchor: {got}")
    check("pending giu nguyen DOI TUONG list (callback khong bi mo coi)",
          v.pending is held, "")
    v.clear()
    _mouse_cb(cv2.EVENT_LBUTTONDOWN, 650, 340, 0, held)
    v._drain()
    check("sau 'c' xoa van dat lai duoc", len(v.anchors) == 1, f"{len(v.anchors)} anchor")
    v.clear()

    # (g) đổi frame không làm hỏng anchor đã lưu (chúng ở hệ world)
    v.add_pixel(intr["cx"], intr["cy"])
    kept = v.anchors[0]
    v.change_frame(+1)
    check("doi frame giu anchor cu", len(v.anchors) == 1 and v.anchors[0] is kept,
          f"frame={v.frame}/{v.n_frames-1}")
    check("anchor cu van doc duoc bang frame moi (he world)",
          all(math.isfinite(x) for x in v.anchors[0].xyz_world_gt), "")

    # (h) luật ổ đĩa + JSON round-trip
    try:
        assert_drive_d(Path(r"C:\Users\x\anchors.json"), "test")
        check("chan ghi ra o C", False, "khong nem exception")
    except SystemExit as e:
        check("chan ghi ra o C", True, str(e).splitlines()[0][:60])
    v.add_pixel(300.0, 300.0)
    p = assert_drive_d(A.DEFAULT_ANCHOR_DIR / f"_selftest_viewer_{scene}.json", "selftest")
    A.save_anchors(v.anchors, p)
    back = A.load_anchors(p)
    check("JSON round-trip qua viewer → load_anchors", len(back) == len(v.anchors),
          f"{len(back)} item, {p}")
    p.unlink()

    print(f"\n{'TAT CA PASS' if fails == 0 else f'{fails} FAIL'}")
    return 0 if fails == 0 else 1


# ── 4. CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="kiem mo hinh headless (khong mo cua so)")
    ap.add_argument("--smoke", type=float, default=0.0, metavar="GIAY",
                    help="mo cua so that, tu bam vai pixel, rong 'giay' giay roi thoat "
                         "— de chung minh GUI dung khong can tay nguoi")
    ap.add_argument("--root", default=str(A.DEFAULT_ROOT))
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--decimate", type=int, default=300_000,
                    help="so tam giac toi da cho mesh hien thi (0 = giu nguyen)")
    ap.add_argument("--out", default=None,
                    help="duong dan JSON (mac dinh D:\\replica_sim\\anchors\\<scene>_interactive.json)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    if args.selftest:
        return _selftest(root, scene=args.scene, frame=args.frame)

    v = RoomViewer(scene=args.scene, frame=args.frame, root=root,
                   decimate=(10**9 if args.decimate <= 0 else args.decimate))
    print("\n" + "=" * 66)
    print("  CACH DUNG (2 cua so se mo ra)")
    print("=" * 66)
    print("  1. CUA SO ANH (ben trai): CLICK TRAI vao 1 diem bat ky tren anh.")
    print("     -> terminal in ra: do sau that, do sau mesh, do lech (mm).")
    print("  2. CUA SO 3D (ben phai): thay phong 3D. Cham xanh la = diem that (GT),")
    print("     cham do = diem do tia mesh. Hai cham cang gan nhau = cang dung.")
    print("     Xoay: keo chuot trai | Phong to: lan chuot | Dich: Ctrl + keo.")
    print("  3. PHIM (phai bam khi dang chon CUA SO ANH, khong phai terminal):")
    print("       u = bo diem vua dat      c = xoa HET diem de dat lai")
    print("       s = luu ra file JSON     [ ] = lui / tien 1 frame")
    print("       q = thoat (tu dong luu)")
    print("  Vong CAM tren anh = canh che: mesh khong dung duoc o do, dung dat diem o do.")
    print("=" * 66 + "\n")
    v.run(smoke_seconds=args.smoke, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
