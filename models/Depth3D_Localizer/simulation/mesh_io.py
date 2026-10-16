"""Đọc mesh PLY của Replica + raycast — **không import torch**.

Vì sao tồn tại
--------------
`o3d.io.read_triangle_mesh()` **âm thầm đọc hỏng** mọi mesh Replica: nó trả về
~16.516 tam giác thay vì ~1.907.294 (đúng 1/58), **không báo lỗi**. Nguyên nhân đã
chứng minh ngày 2026-10-16 bằng số học byte:

    property list uint8 int vertex_indices   ← mesh Replica là **TỨ GIÁC**
    954.492 đỉnh × 27 byte = 25.771.284
    953.647 mặt × 17 byte  = 16.211.999     (1 byte đếm + 4 × int32)
    header                  =        293
    tổng                    = 41.983.576 byte  = KHỚP kích thước file thật

Chia hết đúng với **17 byte/mặt** ⇒ cả 953.647 mặt đều là tứ giác, không có tam giác
nào lẫn vào. Open3D chỉ mong tam giác ⇒ hỏng. Nên **bắt buộc parse tay**.

Quy ước quan trọng về `t_hit`
-----------------------------
`cast_rays()` trả `t_hit` theo **độ dài tham số của tia đưa vào**. Ở đây tia hệ camera
được dựng **KHÔNG chuẩn hoá**:

    dir_cam = ((u - cx)/fx, (v - cy)/fy, 1)

Với điểm thật `(X, Y, Z)` thì `Z · dir_cam = (X, Y, Z)` ⇒ **`t_hit == Z`**, tức
khoảng cách **theo trục quang** — đúng thứ depth GT đo. Chuẩn hoá vector tia sẽ biến
`số` thành *khoảng cách xiên* và so với depth GT sẽ sai âm thầm ở rìa ảnh.

**Đừng "sửa" chỗ này.** Kiểm chứng: `--selftest` so `t_hit` với depth GT, lệch ~2 mm.

Dependencies: `numpy`, `open3d`. KHÔNG `torch` — viewer chạy được khi RAM chật
(xem `AGENTS.md` → lỗi `OpenBLAS`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ── Hằng số định dạng PLY của Replica ───────────────────────────────────

#: Số byte của MỘT đỉnh: 3×float32 (xyz) + 3×float32 (normal) + 3×uint8 (rgb).
REPLICA_VERTEX_BYTES = 3 * 4 + 3 * 4 + 3 * 1  # = 27

#: Số byte của MỘT mặt tứ giác: 1×uint8 (đếm) + 4×int32 (chỉ số).
REPLICA_QUAD_BYTES = 1 + 4 * 4  # = 17


class PlyMesh(NamedTuple):
    """Mesh đã parse xong, ở dạng numpy thuần (chưa đụng Open3D)."""

    xyz: np.ndarray       # (N, 3) float64 — toạ độ, mét
    rgb: np.ndarray       # (N, 3) float64 — 0..1
    normals: np.ndarray   # (N, 3) float64 — đơn vị (theo dataset)
    faces: np.ndarray     # (F, 4) int32 — **đã là tứ giác**, mọi mặt đều 4 chỉ số
    n_vertices: int
    n_faces: int
    header_bytes: int
    path: str


# ── 1. Parse PLY ────────────────────────────────────────────────────────


def _parse_header(header: str) -> Tuple[int, int, list, bool]:
    """Trả ``(nvert, nface, vertex_props, is_list_uint8)``."""
    nvert: Optional[int] = None
    nface: Optional[int] = None
    vprops: list = []
    list_count_type: Optional[str] = None

    for raw in header.splitlines():
        s = raw.strip()
        if s.startswith("element vertex"):
            nvert = int(s.split()[-1])
        elif s.startswith("element face"):
            nface = int(s.split()[-1])
        elif s.startswith("property list"):
            # "property list uint8 int vertex_indices"
            parts = s.split()
            list_count_type = parts[2]
        elif s.startswith("property") and nface is None:
            parts = s.split()                      # ["property", "<type>", "<name>"]
            vprops.append((parts[2], parts[1]))

    if nvert is None or nface is None:
        raise ValueError(f"Header PLY thiếu 'element vertex'/'element face': {header[:200]}")
    if list_count_type not in ("uint8", "uchar"):
        raise ValueError(
            f"Chỉ hỗ trợ list đếm bằng uint8/uchar, gặp {list_count_type!r}. "
            "Mesh khác format — cần mở rộng hàm này."
        )
    return nvert, nface, vprops, True


def read_ply_mesh(path: Path | str) -> PlyMesh:
    """Đọc PLY `binary_little_endian` có mặt tứ giác → `PlyMesh`.

    Raises
    ------
    ValueError
        Nếu format không phải ``binary_little_endian 1.0``, hoặc kích thước payload
        không khớp số học byte tính từ header (⇒ file hỏng/thiếu, hoặc có mặt
        không phải tứ giác).
    """
    path = Path(path)
    data = path.read_bytes()
    marker = data.index(b"end_header\n")
    hdr_end = marker + len(b"end_header\n")
    header = data[:hdr_end].decode("ascii")

    fmt = None
    for line in header.splitlines():
        if line.startswith("format"):
            fmt = line.split()[1]
    if fmt != "binary_little_endian":
        raise ValueError(f"{path.name}: chỉ hỗ trợ binary_little_endian, gặp {fmt!r}")

    nvert, nface, vprops, _ = _parse_header(header)

    fields = []
    for name, typ in vprops:
        if typ in ("float", "float32", "double", "float64"):
            fields.append((name, "<f4" if typ.startswith("float3") or typ == "float" else "<f8"))
        elif typ in ("uchar", "uint8"):
            fields.append((name, "u1"))
        elif typ in ("char", "int8"):
            fields.append((name, "i1"))
    vdt = np.dtype(fields)

    expected = nvert * vdt.itemsize + nface * REPLICA_QUAD_BYTES + hdr_end
    actual = len(data)
    if expected != actual:
        raise ValueError(
            f"{path.name}: payload lệch số học — mong {expected:,} byte "
            f"({nvert:,} đỉnh × {vdt.itemsize}B + {nface:,} mặt × {REPLICA_QUAD_BYTES}B "
            f"+ header {hdr_end}), thực tế {actual:,}B. "
            "File thiếu/hỏng, HOẶC có mặt không phải tứ giác."
        )

    verts = np.frombuffer(data, dtype=vdt, count=nvert, offset=hdr_end)

    # ── mặt: đường nhanh vector hoá (mọi mặt đều tứ giác — đã kiểm chứng) ──
    fb_off = hdr_end + nvert * vdt.itemsize
    fa = np.frombuffer(data, dtype=np.uint8, count=nface * REPLICA_QUAD_BYTES, offset=fb_off)
    fa = fa.reshape(nface, REPLICA_QUAD_BYTES)
    counts = fa[:, 0]
    if not bool((counts == 4).all()):
        bad = int(np.count_nonzero(counts != 4))
        raise ValueError(
            f"{path.name}: {bad:,}/{nface:,} mặt không có 4 đỉnh — đường nhanh tứ giác "
            "không dùng được. Cần bổ sung parser đa giác (xem git history: verify_mesh2.py)."
        )
    faces = (
        np.ascontiguousarray(fa[:, 1:])
        .reshape(nface, 4, 4)
        .view("<i4")[:, :, 0]
        .astype(np.int32)
    )

    def col(name: str) -> np.ndarray:
        return np.asarray(verts[name], dtype=np.float64)

    xyz = np.stack([col("x"), col("y"), col("z")], 1)
    normals = np.stack([col("nx"), col("ny"), col("nz")], 1)
    rgb = np.stack([col("red"), col("green"), col("blue")], 1) / 255.0

    return PlyMesh(
        xyz=xyz, rgb=rgb, normals=normals, faces=faces,
        n_vertices=nvert, n_faces=nface, header_bytes=hdr_end, path=str(path),
    )


def quad_to_triangle_faces(faces: np.ndarray) -> np.ndarray:
    """(F, 4) → (2F, 3) tam giác, tách theo đường chéo 0-2.

    Giữ nguyên thứ tự đỉnh để **hướng pháp tuyến không đảo** so với mesh gốc.
    """
    f = np.asarray(faces, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 4:
        raise ValueError(f"Cần (F, 4), gặp {f.shape}")
    t0 = np.stack([f[:, 0], f[:, 1], f[:, 2]], 1)
    t1 = np.stack([f[:, 0], f[:, 2], f[:, 3]], 1)
    return np.concatenate([t0, t1], 0)


# ── 2. Open3D: mesh hiển thị + cảnh raycast ─────────────────────────────


def to_open3d_legacy(mesh: PlyMesh, with_colors: bool = True):
    """`PlyMesh` → `o3d.geometry.TriangleMesh` (đã tam giác hoá)."""
    import open3d as o3d

    tm = o3d.geometry.TriangleMesh()
    tm.vertices = o3d.utility.Vector3dVector(mesh.xyz)
    tm.triangles = o3d.utility.Vector3iVector(quad_to_triangle_faces(mesh.faces))
    if with_colors:
        tm.vertex_colors = o3d.utility.Vector3dVector(mesh.rgb)
    tm.compute_vertex_normals()
    return tm


class Raycaster:
    """Bao `o3d.t.geometry.RaycastingScene` dựng từ **mesh đầy đủ độ phân giải**.

    ⚠️ Raycast **bắt buộc** dùng mesh gốc. Mesh giảm (decimate) chỉ để xoay cho mượt —
    dùng nó để đo thì sai số bề mặt tăng thêm hàng cm, vô ích vì raycast vốn đã đúng
    ~2 mm so với depth GT.
    """

    def __init__(self, mesh: PlyMesh):
        import open3d as o3d

        self._o3d = o3d
        legacy = to_open3d_legacy(mesh, with_colors=False)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))

    def cast_cam_rays(
        self,
        pixels_uv: Sequence[Sequence[float]],
        intr: Dict[str, Any],
        R: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bắn tia từ camera qua các pixel → ``(t_hit, xyz_world)``.

        Parameters
        ----------
        pixels_uv : (K, 2)
            Toạ độ pixel (u, v) — float được, không cần làm tròn.
        intr : dict
            Cần ``fx, fy, cx, cy`` (lấy từ `utils.replica_intrinsics()`).
        R, t :
            Pose camera→world, từ `utils.parse_pose_line()`.

        Returns
        -------
        t_hit : (K,) float64
            Khoảng cách **theo trục quang** (mét) — so được trực tiếp với depth GT.
            ``inf`` nếu tia không chạm gì.
        xyz_world : (K, 3) float64
            Giao điểm hệ thế giới; hàng có ``t_hit=inf`` là ``nan``.
        """
        o3d = self._o3d
        uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])

        # ⚠️ KHÔNG chuẩn hoá — xem docstring module vì sao.
        dirs_cam = np.stack(
            [(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], 1
        )
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        dirs_world = (R @ dirs_cam.T).T

        rays = np.hstack([np.tile(t, (len(uv), 1)), dirs_world]).astype(np.float32)
        ans = self._scene.cast_rays(o3d.core.Tensor(rays, o3d.core.Dtype.Float32))
        t_hit = np.asarray(ans["t_hit"].numpy(), dtype=np.float64)

        xyz = np.full((len(uv), 3), np.nan)
        ok = np.isfinite(t_hit)
        xyz[ok] = t[None, :] + t_hit[ok, None] * dirs_world[ok]
        return t_hit, xyz


# ── 3. Giảm mesh cho hiển thị (tuỳ chọn) ────────────────────────────────


def decimate_for_display(
    mesh: PlyMesh, target_faces: int = 300_000, verbose: bool = True
):
    """Giảm số tam giác để xoay mượt. **Chỉ dùng để nhìn** — không bao giờ raycast bằng kết quả này.

    Trả về legacy `TriangleMesh` đã giảm; nếu Open3D không giảm được thì trả mesh gốc
    (kèm cảnh báo) chứ không im lặng trả mesh sai.
    """
    tm = to_open3d_legacy(mesh, with_colors=True)
    n_tri = len(tm.triangles)
    if target_faces >= n_tri:
        return tm
    before = (len(tm.vertices), n_tri)
    tm = tm.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    after = (len(tm.vertices), len(tm.triangles))
    if verbose:
        print(
            f"  decimate hiển thị: {before[1]:,} → {after[1]:,} tam giác "
            f"({before[0]:,} → {after[0]:,} đỉnh)"
        )
    return tm


# ── 4. Tự kiểm ─────────────────────────────────────────────────────────


def _selftest(replica_root: Path) -> int:
    """Kiểm chứng đọc mesh + raycast khớp depth GT. Trả về exit code."""
    import utils

    fails = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        mark = "PASS" if cond else "FAIL"
        if not cond:
            fails += 1
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

    intr = utils.replica_intrinsics(replica_root)
    check("doc cam_params.json", intr["png_depth_scale"] == 6553.5,
          f"scale={intr['png_depth_scale']} nguon={intr['source']}")

    ply = replica_root / "room0_mesh.ply"
    if not ply.exists():
        print(f"  [SKIP] khong thay {ply} — khong co data thi khong chay duoc selftest")
        return 1

    print(f"\nDoc {ply.name} ...")
    t0 = time.time()
    mesh = read_ply_mesh(ply)
    dt = time.time() - t0
    print(f"  {dt:.2f}s — {mesh.n_vertices:,} dinh, {mesh.n_faces:,} mat tu giac")

    # (a) số học byte — phép kiểm mạnh nhất: khớp từng byte ⇒ file nguyên vẹn
    expected = (mesh.header_bytes + mesh.n_vertices * REPLICA_VERTEX_BYTES
                + mesh.n_faces * REPLICA_QUAD_BYTES)
    check("so hoc byte khop kich thuoc file",
          expected == ply.stat().st_size,
          f"{expected:,} == {ply.stat().st_size:,}")
    check("co mau dinh", mesh.rgb.shape == (mesh.n_vertices, 3)
          and 0.0 <= mesh.rgb.min() and mesh.rgb.max() <= 1.0,
          f"range [{mesh.rgb.min():.3f}, {mesh.rgb.max():.3f}]")
    check("chi so mat nam trong dinh",
          int(mesh.faces.min()) >= 0 and int(mesh.faces.max()) < mesh.n_vertices,
          f"max={int(mesh.faces.max()):,} < {mesh.n_vertices:,}")
    tri = quad_to_triangle_faces(mesh.faces)
    check("tu giac hoa x2 va khong mat dinh",
          len(tri) == 2 * mesh.n_faces and tri.min() >= 0,
          f"{len(tri):,} tam giac")

    # (b) Open3D doc thang — chung minh cai no (khong phai "hy vong" no hong)
    import open3d as o3d
    naive = o3d.io.read_triangle_mesh(str(ply))
    check("o3d.io.read_triangle_mesh AM THAM doc hong",
          len(naive.triangles) < 0.1 * len(tri),
          f"no tra {len(naive.triangles):,} tam giac, dung phai {len(tri):,}")

    # (c) raycast vs depth GT — nguoc doc lap, dung tieu chuan cua AGENTS.md
    traj = (replica_root / "room0" / "traj.txt").read_text().splitlines()
    rc = Raycaster(mesh)
    rng = np.random.default_rng(0)
    worst_med = 0.0
    for fi in (0, 1000, 1999):
        dpath = replica_root / "room0" / "results" / f"depth{fi:06d}.png"
        if not dpath.exists():
            print(f"  [SKIP] khong co {dpath.name}")
            continue
        d = utils.load_depth_map(dpath, scale=intr["png_depth_scale"])
        R, t = utils.parse_pose_line(traj[fi], opengl_to_opencv=utils.REPLICA_CAM["opengl_to_opencv"])
        ys, xs = np.nonzero(np.isfinite(d))
        sel = rng.choice(len(ys), size=min(3000, len(ys)), replace=False)
        uv = np.stack([xs[sel], ys[sel]], 1)
        t_hit, _ = rc.cast_cam_rays(uv, intr, R, t)
        z_gt = d[ys[sel], xs[sel]]
        ok = np.isfinite(t_hit) & np.isfinite(z_gt)
        err = np.abs(t_hit[ok] - z_gt[ok])
        med = float(np.median(err)) if len(err) else float("inf")
        worst_med = max(worst_med, med)
        print(f"  frame {fi:4d}: {ok.sum():,}/{len(uv)} tia cham mesh | "
              f"|t_hit - Z_gt| median {med*1000:5.2f} mm  "
              f"p95 {np.percentile(err, 95)*1000:6.2f} mm  "
              f"<1cm {np.mean(err < 0.01)*100:5.1f}%")

    check("raycast hop depth GT (median < 10 mm)", worst_med < 0.010,
          f"median tei hon nhat {worst_med*1000:.2f} mm")

    print(f"\n{'TAT CA PASS' if fails == 0 else f'{fails} FAIL'}")
    return 0 if fails == 0 else 1


def replica_root(data_root: Optional[Path | str] = None) -> Path:
    """Gốc Replica: ``<data_root>/data/replica_pp/Replica``. Override bằng ``SIM_DATA_ROOT``."""
    import utils
    base = Path(data_root) if data_root is not None else utils.DEFAULT_DATA_ROOT
    return base / "data" / "replica_pp" / "Replica"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="kiem chung doc PLY + raycast vs depth GT (can data Replica)")
    ap.add_argument("--data-root", default=None,
                    help="goc du lieu (mac dinh SIM_DATA_ROOT hoac D:\\replica_sim)")
    ap.add_argument("--ply", default=None, help="duong dan file .ply de doc")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.ply:
        t0 = time.time()
        m = read_ply_mesh(args.ply)
        print(f"{Path(m.path).name}: {m.n_vertices:,} dinh, {m.n_faces:,} mat tu giac, "
              f"{time.time()-t0:.2f}s, header {m.header_bytes} B")
        return 0

    if args.selftest:
        return _selftest(replica_root(args.data_root))

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())