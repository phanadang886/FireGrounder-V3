"""
`eval_metrics.py` — Tổng hợp metric từ `per_sample.csv` của các case.

**Phase 5 của `SIM_CONTRACT.md` mục 11.**

## Script này làm gì

Đọc CSV do `run_eval.py` ghi ra, tính metric chuẩn và xuất `metrics_summary.json`.

## ⛔ Quy tắc số liệu (mục 0)

- **Chỉ** ghi ra số **thật sự tính được** từ CSV. Thiếu dữ liệu → `null` + lý do, không đoán.
- Nếu **thiếu Case A**, hoặc Case A ≥ 1 cm → in **cảnh báo lớn**, vì số Case B lúc đó là
  sai do toán chứ không phải do model.
- Không gộp số của các case khác nhau vào cùng một bảng.

## Dùng

```bash
python eval_metrics.py --results-dir D:\\replica_sim\\results \\
    --output D:\\replica_sim\\results\\metrics_summary.json

python eval_metrics.py --results-dir ... --output ... --plot
python eval_metrics.py --results-dir ... --output ... --cases case_A case_B_noisy
```
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import check_disk_space

#: Ngưỡng GATE cho Case A (m). Khớp `run_eval.GATE_MAE_A_M`.
GATE_MAE_A_M = 0.01

#: Khoảng cách dùng để tách nhóm (mét) — khớp contract mục 5.4.
DIST_BINS = [(0.0, 2.0, "0-2m"), (2.0, 4.0, "2-4m"), (4.0, 6.0, "4-6m")]

#: Dưới ngưỡng này thì cảnh báo "nhóm quá nhỏ, không kết luận được".
MIN_N_FOR_CONCLUSION = 20

FLOAT_COLS = ("depth_gt", "depth_pred", "depth_err", "gt_x", "gt_y", "gt_z",
              "pred_x", "pred_y", "pred_z", "err", "err_x", "err_y", "err_z")


def _die(msg: str) -> None:
    print(f"\nLỖI: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


# ── Đọc CSV ─────────────────────────────────────────────────────────────


def read_case_csv(path: Path) -> List[Dict[str, Any]]:
    """Đọc `per_sample.csv` thành list dict (số đã ép float). Raise nếu hỏng."""
    if not path.is_file():
        _die(f"Không tìm thấy CSV: {path}\n"
             f"    → Chạy `run_eval.py` trước, hoặc chỉ định --results-dir đúng.")
    with path.open("r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            _die(f"CSV rỗng hoặc không có header: {path}")
        missing = [c for c in ("depth_gt", "depth_pred", "err", "region") if c not in rdr.fieldnames]
        if missing:
            _die(f"CSV {path} thiếu cột bắt buộc: {missing}\n"
                 f"    → Có: {rdr.fieldnames}")
        rows: List[Dict[str, Any]] = []
        bad = 0
        for r in rdr:
            try:
                d: Dict[str, Any] = {k: r.get(k) for k in
                                     ("sample_id", "source", "scene", "frame", "region")}
                for c in FLOAT_COLS:
                    v = r.get(c)
                    d[c] = float(v) if v not in (None, "") else float("nan")
                rows.append(d)
            except (TypeError, ValueError):
                bad += 1
        if bad:
            _warn(f"{path.name}: bỏ qua {bad} dòng không parse được")
    if not rows:
        _die(f"CSV không có dòng dữ liệu nào: {path}")
    return rows


def discover_cases(results_dir: Path,
                   only: Optional[Sequence[str]] = None) -> List[Tuple[str, Path]]:
    """Tìm mọi thư mục con có `per_sample.csv`."""
    if not results_dir.is_dir():
        _die(f"Không tìm thấy thư mục kết quả: {results_dir}")
    found: List[Tuple[str, Path]] = []
    for d in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        csv_p = d / "per_sample.csv"
        if csv_p.is_file():
            found.append((d.name, csv_p))
    if only:
        want = {c.lower() for c in only}
        found = [(n, p) for n, p in found if n.lower() in want]
        missing = want - {n.lower() for n, _ in found}
        if missing:
            _warn(f"không có case nào tên: {sorted(missing)} — có sẵn: "
                  f"{[n for n, _ in found]}")
    if not found:
        _die(f"Không tìm thấy case nào có per_sample.csv trong {results_dir}")
    return found


# ── Tính metric ─────────────────────────────────────────────────────────


def _stats(e: np.ndarray) -> Dict[str, Any]:
    """MAE / median / p95 / rmse / max trên 1 mảng sai số. Trả `null` nếu rỗng."""
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, "mae": None, "median": None, "p95": None,
                "rmse": None, "max": None}
    return {
        "n": int(e.size),
        "mae": float(np.mean(np.abs(e))),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "max": float(np.max(e)),
    }


def depth_metrics(rows: Sequence[Dict[str, Any]], note: str) -> Dict[str, Any]:
    """Metric depth chuẩn (AbsRel, δ<1.25^k) — cùng định nghĩa với literature."""
    gt = np.array([r["depth_gt"] for r in rows], dtype=float)
    pr = np.array([r["depth_pred"] for r in rows], dtype=float)
    ok = np.isfinite(gt) & np.isfinite(pr) & (gt > 0) & (pr > 0)
    if not ok.any():
        return {"n": 0, "mae": None, "rmse": None, "absrel": None,
                "delta_125": None, "delta_125_2": None, "delta_125_3": None,
                "note": f"{note} — KHÔNG có mẫu depth hợp lệ"}
    gt, pr = gt[ok], pr[ok]
    d = np.abs(pr - gt)
    ratio = np.maximum(pr / gt, gt / pr)
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(d)),
        "rmse": float(np.sqrt(np.mean((pr - gt) ** 2))),
        "absrel": float(np.mean(d / gt)),
        "delta_125": float(np.mean(ratio < 1.25)),
        "delta_125_2": float(np.mean(ratio < 1.25 ** 2)),
        "delta_125_3": float(np.mean(ratio < 1.25 ** 3)),
        "note": note,
    }


def position_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Metric vị trí 3D + tách theo trục / khoảng cách / vùng ảnh."""
    err = np.array([r["err"] for r in rows], dtype=float)
    base = _stats(err)
    finite = np.isfinite(err)
    n = int(finite.sum())

    out: Dict[str, Any] = dict(base)
    # tỉ lệ dưới ngưỡng — quy ước lấy từ báo cáo tuần 8
    out["le_0_25m"] = float(np.mean(err[finite] < 0.25)) if n else None
    out["le_0_50m"] = float(np.mean(err[finite] < 0.50)) if n else None

    # theo trục
    axis: Dict[str, Any] = {}
    for k, c in (("mae_x", "err_x"), ("mae_y", "err_y"), ("mae_z", "err_z")):
        a = np.array([r[c] for r in rows], dtype=float)
        a = a[np.isfinite(a)]
        axis[k] = float(np.mean(np.abs(a))) if a.size else None
    axis["note"] = ("X ngang, Y dọc, Z tia ngắm — trục Z kỳ vọng chiếm phần lớn "
                    "vì nhiễu depth tăng theo khoảng cách")
    out["by_axis"] = axis

    # theo khoảng cách (dựa trên depth GT)
    dist: Dict[str, Any] = {}
    for lo, hi, name in DIST_BINS:
        sel = [r for r in rows if lo <= r["depth_gt"] < hi]
        st = _stats(np.array([r["err"] for r in sel], dtype=float))
        dist[name] = {"n": st["n"], "mae": st["mae"], "median": st["median"],
                      "p95": st["p95"]}
        if st["n"] and st["n"] < MIN_N_FOR_CONCLUSION:
            dist[name]["warning"] = (
                f"n={st['n']} < {MIN_N_FOR_CONCLUSION}: KHÔNG kết luận được cho nhóm này")
    out["by_distance"] = dist

    # theo vùng ảnh
    reg: Dict[str, Any] = {}
    for name in ("center", "border"):
        sel = [r for r in rows if r["region"] == name]
        st = _stats(np.array([r["err"] for r in sel], dtype=float))
        reg[name] = {"n": st["n"], "mae": st["mae"], "median": st["median"],
                     "p95": st["p95"]}
        if st["n"] and st["n"] < MIN_N_FOR_CONCLUSION:
            reg[name]["warning"] = (
                f"n={st['n']} < {MIN_N_FOR_CONCLUSION}: so sánh center vs border chưa đáng tin")
    unknown = sorted({r["region"] for r in rows} - {"center", "border"})
    if unknown:
        reg["_unknown_regions"] = unknown
    if reg["center"]["n"] and reg["border"]["n"]:
        c, b = reg["center"]["mae"], reg["border"]["mae"]
        if c is not None and b is not None:
            out["region_mae_ratio"] = (c / b) if b > 0 else None
            out["region_note"] = (
                ">1 nghĩa là sai số ở GIỮA ảnh lớn hơn rìa ảnh. Lý thuyết (contract mục 1) "
                "dự đoán NGƯỢC lại (rìa xa tâm nên bị khuếch đại). Nếu quan sát được điều này "
                "thì sai số depth đang lấn át yếu tố khuếch đại hình học — cần thêm mẫu "
                "để kết luận.")
    out["by_region"] = reg
    return out


def evaluate_case(name: str, csv_path: Path) -> Dict[str, Any]:
    rows = read_case_csv(csv_path)
    name_l = name.lower()
    if "metric" in name_l:
        note = ("model metric: depth đã là mét, so trực tiếp, KHÔNG căn scale/shift")
    elif "b_" in name_l:
        note = "depth relative đã căn scale+shift trước khi so"
    else:
        note = ("Case A: depth_pred lấy từ depth GT — dùng để kiểm tra toán, "
                "KHÔNG phải đánh giá model")
    return {
        "case": name,
        "csv": str(csv_path),
        "num_samples": len(rows),
        "num_scenes": len({r["scene"] for r in rows}),
        "num_frames": len({(r["scene"], r["frame"]) for r in rows}),
        "sources": sorted({r["source"] for r in rows if r["source"]}),
        "depth": depth_metrics(rows, note),
        "position_3d": position_metrics(rows),
    }


# ── GATE ────────────────────────────────────────────────────────────────


def check_gate(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Case A thiếu hoặc ≥ 1 cm → số Case B không đáng tin."""
    a = summary.get("case_A")
    if a is None:
        return {"status": "MISSING",
                "message": ("⛔ KHÔNG CÓ Case A. Số Case B lúc này KHÔNG đáng tin — "
                            "chưa biết toán 2D→3D có đúng không. Chạy `--case A` trước.")}
    mae = a["position_3d"].get("mae")
    if mae is None:
        return {"status": "NO_MAE",
                "message": "⛔ Case A không tính được MAE — kiểm tra lại CSV Case A."}
    ok = mae < GATE_MAE_A_M
    return {
        "status": "PASS" if ok else "FAIL",
        "case_a_mae_m": mae,
        "threshold_m": GATE_MAE_A_M,
        "message": (f"✅ GATE PASS — Case A MAE {mae * 100:.3f} cm < "
                    f"{GATE_MAE_A_M * 100:.0f} cm: số Case B đáng tin."
                    if ok else
                    f"⛔ GATE FAIL — Case A MAE {mae * 100:.2f} cm ≥ "
                    f"{GATE_MAE_A_M * 100:.0f} cm. Số Case B là SAI DO TOÁN, không phải do "
                    f"model. Sửa `utils.py` / `sample_points.py` rồi chạy lại. KHÔNG báo cáo."),
    }


# ── Vẽ biểu đồ ──────────────────────────────────────────────────────────


def plot_all(summary: Dict[str, Any], out_png: Path) -> bool:
    """Vẽ biểu đồ tổng hợp. Trả False nếu thiếu matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _warn("thiếu matplotlib — bỏ qua --plot "
              "(cài: `pip install matplotlib`)")
        return False

    cases = {k: v for k, v in summary.items() if k.startswith("case_")}
    if not cases:
        _warn("không có case nào để vẽ")
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = list(cases)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    # (1) MAE theo khoảng cách
    ax = axes[0]
    w = 0.8 / len(names)
    for i, c in enumerate(names):
        bd = cases[c]["position_3d"].get("by_distance", {})
        xs = [b for b in ("0-2m", "2-4m", "4-6m") if bd.get(b, {}).get("mae") is not None]
        ys = [bd[b]["mae"] for b in xs]
        ax.bar(np.arange(len(xs)) + i * w, ys, width=w, label=c, color=colors[i % 5])
    ax.set_xticks(np.arange(3) + 0.4 - w / 2)
    ax.set_xticklabels(["0-2m", "2-4m", "4-6m"])
    ax.set_ylabel("MAE vị trí (m)")
    ax.set_title("MAE theo khoảng cách")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (2) center vs border
    ax = axes[1]
    xs = np.arange(2)
    for i, c in enumerate(names):
        br = cases[c]["position_3d"].get("by_region", {})
        ys = [br.get("center", {}).get("mae"), br.get("border", {}).get("mae")]
        ax.bar(xs + i * w, [y if y is not None else 0 for y in ys],
               width=w, label=c, color=colors[i % 5])
    ax.set_xticks(xs + 0.4 - w / 2)
    ax.set_xticklabels(["center", "border"])
    ax.set_ylabel("MAE vị trí (m)")
    ax.set_title("center vs border")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (3) phân phối sai số
    ax = axes[2]
    for i, c in enumerate(names):
        st = cases[c]["position_3d"]
        vals = [st.get(k) for k in ("mae", "median", "p95")]
        if all(v is not None for v in vals):
            ax.plot([0, 1, 2], vals, "o-", label=c, color=colors[i % 5])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["MAE", "Median", "P95"])
    ax.set_ylabel("sai số (m)")
    ax.set_title("MAE / Median / P95")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Đánh giá DepthART + toán 2D→3D — Replica preprocessed", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  + biểu đồ: {out_png}")
    return True


# ── In báo cáo ──────────────────────────────────────────────────────────


def print_table(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("BẢNG TỔNG HỢP")
    print("=" * 78)
    hdr = f"{'case':<16}{'n':>6}{'MAE':>9}{'Median':>9}{'P95':>9}{'<0.25m':>9}{'AbsRel':>9}"
    print(hdr)
    print("-" * 78)
    for k, v in summary.items():
        if not k.startswith("case_"):
            continue
        p, d = v["position_3d"], v["depth"]
        def f(x, w=9, p_=3):
            return f"{x:>{w}.{p_}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
        print(f"{k:<16}{p['n']:>6}{f(p.get('mae'))}{f(p.get('median'))}"
              f"{f(p.get('p95'))}{f(p.get('le_0_25m'))}{f(d.get('absrel'))}")

    print("\n" + "-" * 78)
    print("TÁCH THEO KHOẢNG CÁCH (MAE m)")
    print(f"{'case':<16}" + "".join(f"{nm:>14}" for _, _, nm in DIST_BINS))
    for k, v in summary.items():
        if not k.startswith("case_"):
            continue
        bd = v["position_3d"].get("by_distance", {})
        row = f"{k:<16}"
        for _, _, nm in DIST_BINS:
            e = bd.get(nm, {})
            mae, n = e.get("mae"), e.get("n", 0)
            row += f"{('%.3f(n=%d)' % (mae, n) if mae is not None else '—'):>14}"
        print(row)

    print("\n" + "-" * 78)
    print("TÁCH THEO VÙNG ẢNH (MAE m)")
    print(f"{'case':<16}{'center':>16}{'border':>16}{'tỉ lệ':>10}")
    for k, v in summary.items():
        if not k.startswith("case_"):
            continue
        br = v["position_3d"].get("by_region", {})
        c, b = br.get("center", {}), br.get("border", {})
        cs = ("%.3f(n=%d)" % (c["mae"], c["n"])) if c.get("mae") is not None else "—"
        bs = ("%.3f(n=%d)" % (b["mae"], b["n"])) if b.get("mae") is not None else "—"
        ratio = v["position_3d"].get("region_mae_ratio")
        rs = ("%.2f" % ratio) if ratio is not None else "—"
        print(f"{k:<16}{cs:>16}{bs:>16}{rs:>10}")
        for gname, g in (("center", c), ("border", b)):
            if g.get("warning"):
                _warn(f"{k} / {gname}: {g['warning']}")

    # cảnh báo nhóm quá nhỏ
    small = [k for k, v in summary.items() if k.startswith("case_")
             and v["position_3d"]["n"] < MIN_N_FOR_CONCLUSION]
    if small:
        print()
        _warn(f"n < {MIN_N_FOR_CONCLUSION} ở: {small} — số này chỉ để kiểm tra máy, "
              f"CHƯA đủ để kết luận.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(
        description="Tổng hợp metric từ per_sample.csv của các case.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python eval_metrics.py --results-dir D:\\replica_sim\\results \\\n"
            "      --output D:\\replica_sim\\results\\metrics_summary.json --plot\n\n"
            f"GATE: Case A phải < {GATE_MAE_A_M} m, nếu không số Case B là sai do toán.\n"
        ),
    )
    p.add_argument("--results-dir", required=True,
                   help="Thư mục chứa các thư mục case, mỗi case có per_sample.csv.")
    p.add_argument("--output", required=True, help="Đường dẫn metrics_summary.json.")
    p.add_argument("--cases", nargs="*", default=None,
                   help="Chỉ tính các case này (mặc định: tất cả tìm thấy).")
    p.add_argument("--plot", action="store_true", help="Vẽ biểu đồ tổng hợp.")
    p.add_argument("--plot-output", default=None,
                   help="Ảnh biểu đồ (mặc định: <output>.png).")
    p.add_argument("--data-root", default=None, help="Gốc để check ổ đĩa.")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    try:
        check_disk_space(min_c_gb=0.5, min_d_gb=0.5,
                         data_root=args.data_root or results_dir)
    except RuntimeError as e:
        _warn(str(e).split("\n")[0])

    cases = discover_cases(results_dir, args.cases)
    print(f"Tìm thấy {len(cases)} case: {', '.join(n for n, _ in cases)}")

    summary: Dict[str, Any] = {
        "metadata": {
            "generated_by": "eval_metrics.py",
            "dataset": "Replica preprocessed (NICE-SLAM)",
            "note": ("Mọi số ở đây được tính từ per_sample.csv — không có số nào "
                     "được đoán hay điền tay."),
        }
    }
    for name, csv_p in cases:
        print(f"  [{name}] đọc {csv_p.name} …")
        summary[name] = evaluate_case(name, csv_p)

    summary["gate"] = check_gate(summary)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(summary)
    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    print(summary["gate"]["message"])
    print(f"\n✓ Ghi {out}")
    if args.plot:
        png = Path(args.plot_output) if args.plot_output else out.with_suffix(".png")
        if not plot_all(summary, png):
            pass
    # exit 1 nếu gate fail — để script khác bắt được
    return 0 if summary["gate"]["status"] in ("PASS",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
