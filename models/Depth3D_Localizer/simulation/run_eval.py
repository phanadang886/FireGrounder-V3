"""
`run_eval.py` — Chạy DepthART trên các điểm test và ghi sai số từng mẫu.

**Phase 4 của `SIM_CONTRACT.md` mục 11.**

## Script này làm gì

Với mỗi điểm test (1 pixel trong 1 ảnh):

```
đọc depth_pred tại pixel  →  pixel_to_cam  →  cam_to_world
                          →  so với xyz_gt_world trong sample JSON
                          →  ghi 1 dòng CSV
```

## 3 case

| Case | Depth dùng để dự đoán | Mục đích |
|---|---|---|
| **A** | **depth GT** đọc lại từ đĩa | Kiểm tra **toán** pinhole + đổi hệ. MAE phải < 1 cm |
| **B_clean** | DepthART, pose chuẩn | Chấm riêng model depth |
| **B_noisy** | DepthART, pose nhiễu 2 cm / 0.5° | **Số gần thực tế — đây là số để báo cáo** |

⛔ **GATE:** Case A MAE ≥ 1 cm ⇒ **dừng**, sửa `utils.py` / `sample_points.py`.
Số Case B khi gate chưa pass là vô nghĩa (sai do toán, không phải do model).

## Vì sao Case A phải đọc lại từ đĩa

Case A **không** lấy `depth_gt_m` từ JSON, mà đọc lại `depth_path` và lấy depth tại pixel,
đồng thời **đọc lại `traj.txt`** từ scene. Nếu lấy từ JSON thì nó so với chính nó — vô nghĩa,
và sai sót quy ước quay (OpenGL vs OpenCV) sẽ không bao giờ bị phát hiện.

## ⚠️ Căn depth relative — phải đo, đừng tin

Model relative cho depth **affine-invariant**: `d_metric = s · d_rel + b` (theo docstring
`depth_model.py`). Nhưng nếu model thực ra trả **disparity** (1/z) thì phải căn kiểu
`s / d_rel + b`. Chọn sai thì sai số ra to.

→ Script hỗ trợ `--align affine|disparity` và **in ra sai số depth của cả hai** cho mỗi
frame, để quyết định dựa trên số liệu chứ không dựa trên giả định.

## Dùng

```bash
# Case A — phải < 1 cm
python run_eval.py --case A --samples D:\\replica_sim\\samples \\
    --data-dir D:\\replica_sim\\data\\replica_pp --source replica_pp \\
    --output D:\\replica_sim\\results\\case_A

# Case B_clean
python run_eval.py --case B --pose-mode clean --depth-model relative \\
    --samples D:\\replica_sim\\samples --data-dir D:\\replica_sim\\data\\replica_pp \\
    --source replica_pp --output D:\\replica_sim\\results\\case_B_clean

# Case B_noisy — số để báo cáo
python run_eval.py --case B --pose-mode noisy --seed 42 \\
    --samples ... --output D:\\replica_sim\\results\\case_B_noisy
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import (
    add_pose_noise,
    align_scale_shift,
    cam_to_world,
    check_disk_space,
    load_depth_map,
    load_module_backend,
    parse_pose_line,
    pixel_to_cam,
)

# ── Hằng số ─────────────────────────────────────────────────────────────

#: ⛔ GATE: Case A phải nhỏ hơn ngưỡng này (mét) thì mới tin số Case B.
GATE_MAE_A_M = 0.01

#: 18 cột — theo danh sách ở `SIM_CONTRACT.md` mục 5.3 (phần Phase 4 ghi "17 cột" là
#: đếm sai; danh sách là chuẩn và được giữ nguyên).
CSV_COLUMNS = [
    "sample_id", "source", "scene", "frame", "region",
    "depth_gt", "depth_pred", "depth_err",
    "gt_x", "gt_y", "gt_z", "pred_x", "pred_y", "pred_z",
    "err", "err_x", "err_y", "err_z",
]

#: Cấu hình nguồn dữ liệu — phải khớp `sample_points.py` `SOURCE_CFG`.
SOURCE_CFG: Dict[str, Dict[str, Any]] = {
    # traj.txt da o he OpenCV (do bang mesh that) - KHONG dao cot 1,2
    "replica_pp": {"pose_file": "traj.txt", "opengl_to_opencv": False},
    "nyuv2": {"pose_file": None, "opengl_to_opencv": False},
}


def _die(msg: str) -> None:
    print(f"\nLỖI: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def _fmt_case(case: str, pose_mode: str) -> str:
    return case if case == "A" else f"B_{pose_mode}"


# ── Nạp sample ──────────────────────────────────────────────────────────


def _load_samples(samples_dir: Path) -> List[Dict[str, Any]]:
    """Đọc mọi JSON trong thư mục, validate tối thiểu. Raise nếu rỗng."""
    if not samples_dir.is_dir():
        _die(f"Không tìm thấy thư mục samples: {samples_dir}\n"
             f"    → Chạy `sample_points.py` trước.")
    files = sorted(samples_dir.glob("*.json"))
    if not files:
        _die(f"Thư mục samples rỗng: {samples_dir}\n"
             f"    → `sample_points.py` chưa sinh ra file nào. Kiểm tra lại --data-dir.")

    out: List[Dict[str, Any]] = []
    required = ("sample_id", "pixel", "depth_gt_m", "xyz_gt_world", "intrinsics",
                "frame_path", "depth_path", "depth_scale")
    skipped = 0
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"  ! {f.name}: JSON hỏng ({e}) — bỏ qua")
            skipped += 1
            continue
        missing = [k for k in required if k not in d]
        if missing:
            print(f"  ! {f.name}: thiếu trường {missing} — bỏ qua")
            skipped += 1
            continue
        out.append(d)
    if skipped:
        print(f"  (bỏ qua {skipped}/{len(files)} file JSON không hợp lệ)")
    if not out:
        _die(f"Không còn sample hợp lệ nào trong {samples_dir}.")
    return out


def _pose_from_source(
    data_dir: Path,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Đọc lại pose từ `traj.txt` của scene — **nguồn độc lập với sample JSON**.

    Nhờ vậy Case A mới thật sự kiểm được quy ước trục. Trả `(None, None)` nếu
    nguồn không có file pose (NYUv2) — khi đó dùng pose trong JSON và sẽ cảnh báo.
    """
    pf = cfg["pose_file"]
    if pf is None:
        return None, None
    p = data_dir / sample["scene"] / pf
    if not p.is_file():
        return None, None
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    fi = int(sample["frame"])
    if fi >= len(lines):
        return None, None
    return parse_pose_line(lines[fi], opengl_to_opencv=cfg["opengl_to_opencv"])


# ── Cache theo frame ────────────────────────────────────────────────────


class FrameCache:
    """Cache depth GT, ảnh RGB và depth dự đoán **theo frame**.

    Nhiều điểm test có thể nằm cùng 1 frame (tối đa 5 theo luật mục 5.2). Không cache
    thì cùng 1 ảnh bị đọc/infer lại nhiều lần — DepthART ~73 ms/ảnh, 200 mẫu / 5 mẫu mỗi
    frame = 40 ảnh, không cache sẽ tốn gấp 5 lần.
    """

    def __init__(self, backend: Any = None, align: str = "affine",
                 verbose: bool = True) -> None:
        self.backend = backend
        self.align = align
        self.verbose = verbose
        self._gt: Dict[str, np.ndarray] = {}
        self._pred: Dict[str, np.ndarray] = {}
        self._align_info: Dict[str, Dict[str, float]] = {}

    def depth_gt(self, sample: Dict[str, Any]) -> np.ndarray:
        key = sample["depth_path"]
        if key not in self._gt:
            self._gt[key] = load_depth_map(
                key, scale=sample.get("depth_scale"))
        return self._gt[key]

    def depth_pred(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Depth **đã căn về mét** cho 1 frame. Gọi model đúng 1 lần mỗi frame."""
        key = sample["frame_path"]
        if key in self._pred:
            return self._pred[key], self._align_info[key]

        if self.backend is None:
            raise RuntimeError("Case B cần backend depth — khởi tạo FrameCache(backend=...)")
        if not Path(key).is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {key}")

        t0 = time.perf_counter()
        pred = np.asarray(self.backend.infer_path(key), dtype=np.float32)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        gt = self.depth_gt(sample)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Depth dự đoán {pred.shape} khác depth GT {gt.shape} cho {key}\n"
                f"    → Sai kích thước ảnh. Kiểm tra backend có resize về kích thước gốc không."
            )

        info: Dict[str, float] = {"infer_ms": dt_ms}
        if self.backend.model_name and "metric" in str(self.backend.model_name).lower():
            # Model metric đã có đơn vị mét → KHÔNG căn
            aligned = pred
            info["mode"] = 0.0
        else:
            aligned, info = self._align(pred, gt, info)
        aligned = np.asarray(aligned, dtype=np.float32)
        aligned[~np.isfinite(aligned)] = np.nan
        aligned[aligned <= 0.0] = np.nan

        self._pred[key] = aligned
        self._align_info[key] = info
        return aligned, info

    def _align(self, pred: np.ndarray, gt: np.ndarray,
               info: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Căn depth relative → mét, đồng thời **đo cả hai giả thuyết** để so sánh."""
        out: Optional[np.ndarray] = None
        best: Optional[Dict[str, float]] = None
        for mode in (self.align, "disparity" if self.align == "affine" else "affine"):
            try:
                if mode == "affine":
                    ss = align_scale_shift(pred, gt)
                    cand = ss.scale * pred + ss.shift
                else:
                    # d_metric = s / d_rel + b — dùng khi model trả disparity (1/z)
                    inv = np.where(np.abs(pred) > 1e-8, 1.0 / pred, np.nan)
                    ss = align_scale_shift(inv, gt)
                    cand = ss.scale * inv + ss.shift
                err = float(np.nanmean(np.abs(cand - gt)))
            except ValueError as e:
                info[f"{mode}_err"] = float("nan")
                if self.verbose:
                    print(f"    (căn kiểu {mode} thất bại: {e})")
                continue
            info[f"{mode}_err"] = err
            if mode == self.align:
                out = cand
                best = info
        if out is None:
            raise ValueError("Không căn được depth theo bất kỳ chế độ nào.")
        best["mode"] = 1.0 if self.align == "affine" else 2.0
        return out, best


# ── Tính 1 dòng CSV ─────────────────────────────────────────────────────


def _eval_one(sample: Dict[str, Any], depth_pred_map: np.ndarray,
              R: Optional[np.ndarray], t: Optional[np.ndarray]
              ) -> Optional[Dict[str, Any]]:
    """So toạ độ 3D dự đoán với đáp án chuẩn. Trả `None` nếu depth không hợp lệ."""
    u, v = sample["pixel"]
    try:
        d = float(depth_pred_map[v, u])
    except IndexError:
        return None
    if not np.isfinite(d) or d <= 0.0:
        return None

    intr = sample["intrinsics"]
    xyz_cam = pixel_to_cam((u, v), d, intr)
    xyz_w = (cam_to_world(xyz_cam, R, t) if R is not None
             else np.asarray(xyz_cam, dtype=float))
    gt = np.asarray(sample["xyz_gt_world"], dtype=float)
    diff = xyz_w - gt
    return {
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "scene": sample["scene"],
        "frame": sample["frame"],
        "region": sample["region"],
        "depth_gt": float(sample["depth_gt_m"]),
        "depth_pred": d,
        "depth_err": d - float(sample["depth_gt_m"]),
        "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2],
        "pred_x": xyz_w[0], "pred_y": xyz_w[1], "pred_z": xyz_w[2],
        "err": float(np.linalg.norm(diff)),
        "err_x": diff[0], "err_y": diff[1], "err_z": diff[2],
    }


# ── Vòng lặp chính ──────────────────────────────────────────────────────


def run(samples: List[Dict[str, Any]], data_dir: Path, source: str,
        case: str, pose_mode: str, seed: int, cache: FrameCache,
        out_csv: Path, tag: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Chạy 1 case, ghi CSV, trả thống kê tóm tắt."""
    cfg = SOURCE_CFG[source]
    total = len(samples) if max_rows is None else min(max_rows, len(samples))
    n_skip = 0
    errs: List[float] = []
    err_regions: List[str] = []      # đi cặp với errs — KHÔNG suy lại từ samples
    depth_errs: List[float] = []
    reported: set = set()            # frame đã báo cáo căn rồi
    align_report: List[str] = []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, s in enumerate(samples[:total], start=1):
            try:
                if case == "A":
                    # Đọc lại depth từ ĐĨA, không lấy từ JSON (tránh tự so với chính mình)
                    dmap = cache.depth_gt(s)
                else:
                    dmap, info = cache.depth_pred(s)
                    # mỗi frame chỉ báo cáo căn 1 lần (nhiều mẫu chung 1 frame)
                    if s["frame_path"] not in reported:
                        reported.add(s["frame_path"])
                        if len(align_report) < 8:
                            name = Path(s["frame_path"]).name
                            if "affine_err" in info or "disparity_err" in info:
                                align_report.append(
                                    f"    {name}: "
                                    f"affine_err={info.get('affine_err', float('nan')):.3f} m  "
                                    f"disparity_err={info.get('disparity_err', float('nan')):.3f} m")
                            else:
                                align_report.append(
                                    f"    {name}: model metric — không căn (depth đã là mét)")
                R, t = _pose_from_source(data_dir, s, cfg)
                if R is None:
                    pose = s.get("pose")
                    if pose is not None:
                        R = np.asarray(pose["R"], dtype=float)
                        t = np.asarray(pose["t"], dtype=float)
                        if case == "A" and n_skip == 0:
                            print("  ! Nguồn không có traj.txt — dùng pose trong JSON. "
                                  "Case A KHÔNG còn kiểm được quy ước trục!")
                    else:
                        R = np.eye(3)
                        t = np.zeros(3)
                if pose_mode == "noisy":
                    # seed theo frame ⇒ tái lập được, nhưng mỗi frame nhiễu khác nhau
                    R, t = add_pose_noise(R, t, seed=seed + int(s["frame"]))

                row = _eval_one(s, dmap, R, t)
            except (FileNotFoundError, ValueError, KeyError) as e:
                print(f"  ! {s.get('sample_id', i)}: {e}")
                n_skip += 1
                continue

            if row is None:
                n_skip += 1
                continue
            writer.writerow(row)
            errs.append(row["err"])
            err_regions.append(row["region"])
            depth_errs.append(row["depth_err"])
            print(f"[{tag}] {row['sample_id']} ({i}/{total}): "
                  f"depth_err={row['depth_err']:+.3f}m err={row['err']:.3f}m")

    mae = float(np.mean(errs)) if errs else float("nan")

    def _region_stat(reg: str) -> Dict[str, Any]:
        sel = [e for e, r in zip(errs, err_regions) if r == reg]
        return {"n": len(sel),
                "mae": float(np.mean(sel)) if sel else None}

    stats = {
        "case": _fmt_case(case, pose_mode),
        "n_written": len(errs),
        "n_skipped": n_skip,
        "mae_m": mae,
        "median_m": float(np.median(errs)) if errs else float("nan"),
        "p95_m": float(np.percentile(errs, 95)) if errs else float("nan"),
        "depth_mae_m": float(np.mean(np.abs(depth_errs))) if depth_errs else float("nan"),
        "by_region": {r: _region_stat(r) for r in ("center", "border")},
        "csv": str(out_csv),
    }
    return stats, align_report


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(
        description="Chấm DepthART + toán 2D→3D trên các điểm test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python run_eval.py --case A --samples D:\\replica_sim\\samples \\\n"
            "      --data-dir D:\\replica_sim\\data\\replica_pp --output D:\\replica_sim\\results\\case_A\n"
            "  python run_eval.py --case B --pose-mode noisy --depth-model relative \\\n"
            "      --samples ... --output D:\\replica_sim\\results\\case_B_noisy\n\n"
            f"GATE: Case A MAE phải < {GATE_MAE_A_M} m, nếu không thì số Case B vô nghĩa.\n"
        ),
    )
    p.add_argument("--case", choices=["A", "B"], required=True,
                   help="A = dùng depth GT (kiểm tra toán); B = dùng DepthART.")
    p.add_argument("--samples", required=True, help="Thư mục JSON từ sample_points.py.")
    p.add_argument("--data-dir", required=True,
                   help="Thư mục chứa scene (dùng để đọc lại traj.txt trong Case A).")
    p.add_argument("--source", default="replica_pp", choices=sorted(SOURCE_CFG))
    p.add_argument("--output", required=True, help="Thư mục ghi per_sample.csv.")
    p.add_argument("--pose-mode", choices=["clean", "noisy"], default="clean",
                   help="Chỉ dùng cho Case B (mặc định: clean).")
    p.add_argument("--depth-model", default="relative",
                   choices=["relative", "metric"],
                   help="relative = DepthART affine-invariant; metric = đơn vị mét. "
                        "(Mặc định: relative)")
    p.add_argument("--depthart-encoder", default=None,
                   help="Encoder DepthART (ví dụ mnv4s cho relative, s/b/l cho metric).")
    p.add_argument("--align", choices=["affine", "disparity"], default="affine",
                   help="Cách căn depth relative. Mặc định affine. Script vẫn in "
                        "sai số của cả hai để so sánh.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed cho nhiễu pose (mặc định: 42).")
    p.add_argument("--limit", type=int, default=None,
                   help="Chỉ chạy N mẫu đầu (để thử nhanh).")
    p.add_argument("--data-root", default=None, help="Gốc để check ổ đĩa.")
    args = p.parse_args(argv)

    samples = _load_samples(Path(args.samples))
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        _die(f"Không tìm thấy --data-dir: {data_dir}")

    try:
        info = check_disk_space(min_c_gb=0.5, min_d_gb=1.0,
                                data_root=args.data_root or data_dir)
        print(f"Ổ đĩa: {info}")
    except RuntimeError as e:
        _die(str(e))

    backend = None
    if args.case == "B":
        print(f"\nĐang nạp DepthART ({args.depth_model})…")
        print("  ⚠ Nếu lỗi OpenBLAS: đóng Chrome/Docker để giải phóng RAM rồi chạy lại.")
        try:
            if args.depth_model == "metric":
                # Backend metric BẮT BUỘC cần intrinsics (là input của model).
                # Lấy từ sample đầu tiên — mọi sample cùng nguồn dùng chung 1 bộ camera
                # (Replica: cam_params.json của dataset; đã verify khớp yaml).
                # NOTE: phải chèn sys.path TRƯỚC khi import (load_module_backend
                # làm việc này, nhưng import ở đây chạy trước nó).
                import sys as _sys
                _parent = str(Path(__file__).resolve().parent.parent)
                if _parent not in _sys.path:
                    _sys.path.insert(0, _parent)
                from localizer_3d import CameraIntrinsics  # lazy: chỉ khi cần metric
                intr = CameraIntrinsics.from_dict(samples[0]["intrinsics"])
                print(f"  intrinsics cho model metric: fx={intr.fx}, fy={intr.fy}, "
                      f"cx={intr.cx}, cy={intr.cy}, {intr.width}x{intr.height}")
                backend = load_module_backend(
                    "metric", intrinsics=intr, load=True,
                    **({"encoder": args.depthart_encoder}
                       if args.depthart_encoder else {}),
                )
            else:
                backend = load_module_backend(
                    args.depth_model, load=True,
                    **({"depthart_encoder": args.depthart_encoder}
                       if args.depthart_encoder else {}),
                )
        except Exception as e:  # noqa: BLE001
            _die(f"Không nạp được backend depth: {type(e).__name__}: {e}\n"
                 f"    → Nếu chưa tải checkpoint: "
                 f"`python models/Depth3D_Localizer/install_depthart.py`")
        print(f"  model: {backend.model_name}")

    cache = FrameCache(backend=backend, align=args.align)
    tag = _fmt_case(args.case, args.pose_mode)
    out_csv = Path(args.output) / "per_sample.csv"
    print(f"\n{'=' * 66}\nCASE {tag}\n{'=' * 66}")

    stats, align_report = run(samples, data_dir, args.source, args.case,
                              args.pose_mode, args.seed, cache, out_csv, tag,
                              max_rows=args.limit)

    if align_report:
        print("\nSai số depth sau khi căn (để chọn đúng chế độ --align):")
        print("\n".join(align_report))
        if any("affine_err=" in a for a in align_report):
            print("  → Chế độ affine và disparity đều được đo; chọn chế độ có err nhỏ hơn.")
        else:
            print("  → Model metric: depth đã là mét, không căn scale/shift.")

    print(f"\n{'=' * 66}")
    print(f"KẾT QUẢ CASE {tag}")
    print(f"{'=' * 66}")
    print(f"  ghi {stats['n_written']} dòng vào {stats['csv']}")
    if stats["n_skipped"]:
        print(f"  ⚠ bỏ qua {stats['n_skipped']} mẫu (xem dòng '!' ở trên)")
    print(f"  MAE vị trí 3D : {stats['mae_m']:.4f} m")
    print(f"  Median        : {stats['median_m']:.4f} m")
    print(f"  P95           : {stats['p95_m']:.4f} m")
    print(f"  MAE depth     : {stats['depth_mae_m']:.4f} m")
    for k, v in stats["by_region"].items():
        if v["n"]:
            print(f"    - {k:<8} n={v['n']:<4} MAE={v['mae']:.4f} m")

    if args.case == "A":
        print()
        if stats["mae_m"] < GATE_MAE_A_M:
            print(f"✅ GATE PASS: Case A MAE {stats['mae_m'] * 100:.3f} cm "
                  f"< {GATE_MAE_A_M * 100:.0f} cm → tin được số Case B.")
        else:
            print(f"⛔ GATE FAIL: Case A MAE {stats['mae_m'] * 100:.2f} cm "
                  f">= {GATE_MAE_A_M * 100:.0f} cm.")
            print("   → DỪNG. Sửa `utils.py` / `sample_points.py` (pinhole, pose, depth scale).")
            print("   → Số Case B lúc này là SAI DO TOÁN, không phải do model. Đừng báo cáo.")
            return 2
    elif args.pose_mode == "noisy":
        print("\n💡 Đây là số gần thực tế (pose nhiễu) — đây là số nên báo cáo.")
    else:
        print("\n💡 Số này dùng pose chuẩn, KHÔNG phải thực tế. "
              "Hãy chạy --pose-mode noisy để có số để báo cáo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
