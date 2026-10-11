"""
`sample_points.py` — Chọn điểm test và tính sẵn đáp án chuẩn.

**Phase 3 của `SIM_CONTRACT.md` mục 11.**

## Script này làm gì

Simulator chấm điểm theo "điểm test" = **1 pixel trong 1 ảnh**. Script này làm
**đề thi + đáp án**: chọn pixel nào sẽ được chấm, rồi tính sẵn toạ độ 3D chuẩn
từ depth GT. `run_eval.py` sau đó hỏi DepthART "depth tại pixel này là bao nhiêu?",
quy ra 3D, và so với đáp án đã lưu.

## Quy tắc chọn điểm (mục 5.2 contract)

| Quy tắc              | Vì sao |
|----------------------|--------|
| depth ∈ [1.5, 6.0] m | sát camera quá thì sai số tương đối lớn; quá xa thì ảnh vô dùng |
| cách rìa ảnh ≥ 10 px | pixel sát mép hay lỗi, mà sai số ở đó bị nhân lên 2–2.5 lần |
| tối đa **5 điểm/frame** | không thì dồn vào 1 căn phòng → số tổng bị thiên lệch theo phòng |
| trộn đều center/border | làm lộ hiện tượng khuếch đại sai số ở rìa ảnh (mục 1 contract) |
| seed cố định           | chạy lại ra **cùng kết quả**, không "đổi số" mỗi lần đo |

## ⚠️ Ba chỗ dễ sai — đã xác minh từ source

1. **Hệ số depth Replica là 6553.5, KHÔNG phải 1000.** Đọc sai ⇒ mọi số sai 6.55 lần.
2. **Pose `traj.txt` lưu theo quy ước OpenGL** → phải đảo cột 1,2 về hệ OpenCV.
3. **Ảnh là `results/frame%06d.jpg`, depth là `results/depth%06d.png`** — cùng thư mục,
   zero-pad 6 số. Không phải `rgb/` + `depth/` tách riêng như một số tài liệu mô tả.

Nguồn: `cvg/nice-slam` → `src/utils/datasets.py` (class `Replica`) và
`configs/Replica/replica.yaml`.

## Dùng

```bash
# Xem trước sẽ tạo ra gì (không ghi file)
python sample_points.py --data-dir D:\\replica_sim\\data\\replica_pp \\
    --source replica_pp --num-samples 200 --dry-run

# Chạy thật
python sample_points.py --data-dir D:\\replica_sim\\data\\replica_pp \\
    --source replica_pp --num-samples 200 --output D:\\replica_sim\\samples --seed 42

# Vẽ chấm lên ảnh để KIỂM BẰNG MẮT
python sample_points.py --data-dir ... --source replica_pp \\
    --num-samples 200 --output D:\\replica_sim\\samples --visualize
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import (
    MILLIMETER_PNG_DEPTH_SCALE,
    REPLICA_PNG_DEPTH_SCALE,
    cam_to_world,
    check_disk_space,
    load_depth_map,
    parse_pose_line,
    pixel_to_cam,
    replica_intrinsics,
)

# ── Quy tắc chọn điểm (mục 5.2 contract) ────────────────────────────────

DEPTH_MIN_M = 1.5
DEPTH_MAX_M = 6.0
MARGIN_PX = 10          # cách rìa ảnh tối thiểu
MAX_PER_FRAME = 5       # tối đa số điểm trên 1 frame
CAND_PER_FRAME = 40     # số ứng viên random mỗi frame trước khi lọc

#: Scene có sẵn trong `Replica.zip`. Tên KHÔNG có gạch dưới.
REPLICA_SCENES = ("office0", "office1", "office2", "office3", "office4",
                  "room0", "room1", "room2")

# ── Cấu hình theo nguồn dữ liệu ─────────────────────────────────────────

SOURCE_CFG: Dict[str, Dict[str, Any]] = {
    "replica_pp": {
        "rgb_pattern": "results/frame{idx:06d}.jpg",
        "depth_pattern": "results/depth{idx:06d}.png",
        "pose_file": "traj.txt",
        "depth_scale": REPLICA_PNG_DEPTH_SCALE,
        "opengl_to_opencv": False,  # do bang mesh: traj.txt da OpenCV
        "intrinsics": replica_intrinsics,   # hàm, vì dataset không có file
    },
    "nyuv2": {
        "rgb_pattern": "rgb/{idx:06d}.png",
        "depth_pattern": "depth/{idx:06d}.png",
        "pose_file": None,                 # không có → pose đơn vị
        "depth_scale": MILLIMETER_PNG_DEPTH_SCALE,
        "opengl_to_opencv": False,
        "intrinsics": "intrinsics.txt",    # đường dẫn trong scene
    },
}


def _die(msg: str) -> None:
    """In lỗi ra stderr rồi thoát — không để lỗi im lặng."""
    print(f"\nLỖI: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def _region_of(u: float, v: float, w: int, h: int) -> str:
    """Chia vùng ảnh: `center` nếu nằm trong nửa giữa, ngược lại `border`.

    Tách vùng là bắt buộc: sai số vị trí bị nhân với
    `√(1 + ((u−cx)/fx)² + ((v−cy)/fy)²)` — ở rìa ảnh hệ số này tới 2–2.5 lần.
    Chấm gộp một số sẽ che mất hiện tượng này.
    """
    if (w / 4.0 <= u <= 3.0 * w / 4.0) and (h / 4.0 <= v <= 3.0 * h / 4.0):
        return "center"
    return "border"


def _list_frames(scene_dir: Path, cfg: Dict[str, Any]) -> List[int]:
    """Liệt kê chỉ số frame có **đủ** ảnh + depth. Trả về [] nếu thiếu."""
    rgb_tpl = cfg["rgb_pattern"]
    depth_tpl = cfg["depth_pattern"]
    idxs: List[int] = []
    i = 0
    # dò tới khi cả ảnh và depth đều không còn nữa
    while True:
        rp, dp = scene_dir / rgb_tpl.format(idx=i), scene_dir / depth_tpl.format(idx=i)
        if not rp.is_file() and not dp.is_file():
            break
        if rp.is_file() and dp.is_file():
            idxs.append(i)
        i += 1
    return idxs


def _load_pose(scene_dir: Path, cfg: Dict[str, Any], frame: int
               ) -> Tuple[Optional[Any], Optional[Any]]:
    """Đọc pose của 1 frame → `(R, t)` ở hệ OpenCV, hoặc `(None, None)` nếu không có."""
    pf = cfg["pose_file"]
    if pf is None:
        return None, None
    p = scene_dir / pf
    if not p.is_file():
        return None, None
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    if frame >= len(lines):
        return None, None
    return parse_pose_line(lines[frame], opengl_to_opencv=cfg["opengl_to_opencv"])


def _pick_candidates(depth: np.ndarray, rng, w: int, h: int
                     ) -> List[Tuple[int, int, float]]:
    """Chọn ứng viên trong 1 frame theo luật depth + lề + vùng ảnh."""
    out: List[Tuple[int, int, float]] = []
    seen: set = set()
    for _ in range(CAND_PER_FRAME):
        u = int(rng.integers(MARGIN_PX, w - MARGIN_PX))
        v = int(rng.integers(MARGIN_PX, h - MARGIN_PX))
        if (u, v) in seen:
            continue
        d = float(depth[v, u])
        if not np.isfinite(d) or not (DEPTH_MIN_M <= d <= DEPTH_MAX_M):
            continue
        seen.add((u, v))
        out.append((u, v, d))
        if len(out) >= MAX_PER_FRAME * 3:
            break
    return out


def _balance_regions(cands: List[Tuple[int, int, float]], w: int, h: int
                     ) -> List[Tuple[int, int, float, str]]:
    """Chọn tối đa `MAX_PER_FRAME` điểm, **ưu tiên cân bằng** center và border.

    Không cân bằng thì `by_region` trong metrics sẽ lệch mạnh và không kết luận được
    gì về hiện tượng khuếch đại sai số ở rìa ảnh.
    """
    with_r = [(u, v, d, _region_of(u, v, w, h)) for u, v, d in cands]
    center = [i for i, c in enumerate(with_r) if c[3] == "center"]
    border = [i for i, c in enumerate(with_r) if c[3] == "border"]
    # Xen kẽ center/border, mỗi lượt lấy 1 điểm, nhóm nào hết thì lấy nhóm kia.
    # Với 5 điểm không thể đạt đúng 50/50 (2/3 hoặc 3/2) — nên cứ xen kẹp để tự
    # cân bằng, thay vì cắt cứng 2 center + 3 border (lệch 40%).
    picked_idx: List[int] = []
    ci = bi = 0
    for turn in range(MAX_PER_FRAME):
        order = (center, border) if turn % 2 == 0 else (border, center)
        for pool in order:
            nxt = ci if pool is center else bi
            if nxt < len(pool):
                picked_idx.append(pool[nxt])
                if pool is center:
                    ci += 1
                else:
                    bi += 1
                break
        else:
            break  # cả hai nhóm đã cạn
    return [with_r[i] for i in picked_idx[:MAX_PER_FRAME]]


def sample_one_scene(
    scene_dir: Path,
    scene: str,
    source: str,
    cfg: Dict[str, Any],
    rng,
    max_samples: int,
    intr: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Sinh sample JSON dict cho 1 scene (đã tính sẵn đáp án 3D).

    `intr` cho phép truyền intrinsics đã đọc 1 lần từ `cam_params.json` — Replica dùng
    chung 1 bộ camera cho mọi scene, đọc lại 8 lần là thừa.
    """
    frames = _list_frames(scene_dir, cfg)
    if not frames:
        print(f"  - {scene}: KHÔNG tìm thấy frame nào khớp mẫu "
              f"'{cfg['rgb_pattern']}' / '{cfg['depth_pattern']}'")
        return []

    # intrinsics
    if intr is None:
        if callable(cfg["intrinsics"]):
            intr = cfg["intrinsics"]()
        else:
            from utils import parse_intrinsics_txt
            intr = parse_intrinsics_txt(scene_dir / cfg["intrinsics"])
    w, h = int(intr["width"]), int(intr["height"])
    scale = cfg["depth_scale"]

    out: List[Dict[str, Any]] = []
    for fi in frames:
        if len(out) >= max_samples:
            break
        rgb_p = scene_dir / cfg["rgb_pattern"].format(idx=fi)
        depth_p = scene_dir / cfg["depth_pattern"].format(idx=fi)
        try:
            depth = load_depth_map(depth_p, scale=scale)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ! {scene}/frame{fi:06d}: bỏ qua — {e}")
            continue

        dh, dw = depth.shape
        if (dh, dw) != (h, w):
            # depth là nguồn chân lý cho hình học; nhưng intrinsics theo ảnh màu.
            # Trộn lẫn kích thước là sai số hệ thống → phải báo, không âm thầm dùng.
            print(f"  ! {scene}/frame{fi:06d}: depth {dw}×{dh} KHÔNG khớp "
                  f"intrinsics {w}×{h} — bỏ qua frame này.")
            continue

        R, t = _load_pose(scene_dir, cfg, fi)
        cands = _balance_regions(_pick_candidates(depth, rng, dw, dh), dw, dh)
        for u, v, d, region in cands:
            if len(out) >= max_samples:
                break
            xyz_cam = pixel_to_cam((u, v), d, intr)
            xyz_world = (cam_to_world(xyz_cam, R, t) if R is not None
                         else np.asarray(xyz_cam, dtype=float))
            out.append({
                "sample_id": f"{source}_{scene}_{len(out):04d}",
                "source": source,
                "scene": scene,
                "frame": fi,
                "frame_path": str(rgb_p),
                "depth_path": str(depth_p),
                "pixel": [int(u), int(v)],
                "region": region,
                "depth_gt_m": float(d),
                "xyz_gt_cam": [float(x) for x in xyz_cam],
                "xyz_gt_world": [float(x) for x in xyz_world],
                "intrinsics": {k: intr[k] for k in
                               ("fx", "fy", "cx", "cy", "width", "height")},
                "depth_scale": float(scale),
                "pose_convention": "opencv",
                "pose": ({"R": [[float(x) for x in row] for row in R],
                          "t": [float(x) for x in t]} if R is not None else None),
            })
    return out


def visualize(samples: Sequence[Dict[str, Any]], out_dir: Path,
              max_images: int = 12) -> int:
    """Vẽ chấm lên ảnh RGB để **kiểm bằng mắt**. Trả về số ảnh đã vẽ."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    by_frame: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_frame.setdefault(s["frame_path"], []).append(s)

    n = 0
    for fp, group in list(by_frame.items())[:max_images]:
        img = cv2.imread(fp)
        if img is None:
            print(f"  ! không đọc được ảnh: {fp}")
            continue
        for s in group:
            u, v = s["pixel"]
            color = (0, 255, 0) if s["region"] == "center" else (0, 165, 255)
            cv2.circle(img, (u, v), 9, color, 2)
            cv2.putText(img, f"{s['depth_gt_m']:.2f}m", (u + 12, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        name = f"prev_{Path(fp).stem}_{s['scene']}.jpg"
        cv2.imwrite(str(out_dir / name), img)
        print(f"  + {name}  ({len(group)} điểm)")
        n += 1
    return n


def report(samples: Sequence[Dict[str, Any]], by_scene: Dict[str, int]) -> None:
    """In phân bố — phải **cân bằng** hoặc metric `by_region` vô nghĩa."""
    if not samples:
        print("\n⚠ KHÔNG tạo được sample nào.")
        return
    reg = {"center": 0, "border": 0}
    for s in samples:
        reg[s["region"]] = reg.get(s["region"], 0) + 1
    d = np.array([s["depth_gt_m"] for s in samples], dtype=float)
    frames = {s["frame_path"] for s in samples}
    print("\n" + "=" * 66)
    print("PHÂN BỐ SAMPLE")
    print("=" * 66)
    print(f"  tổng số sample   : {len(samples)}")
    print(f"  số frame dùng     : {len(frames)}")
    print(f"  trung bình/frame   : {len(samples) / max(len(frames), 1):.2f} "
          f"(giới hạn {MAX_PER_FRAME})")
    print(f"  center / border   : {reg['center']} / {reg['border']} "
          f"({reg['center'] / len(samples) * 100:.0f}% center)")
    print(f"  depth min/median/max: {d.min():.2f} / {np.median(d):.2f} / {d.max():.2f} m")
    for sc, n in sorted(by_scene.items()):
        print(f"    - {sc:<12} {n}")
    ratio = min(reg.values()) / max(max(reg.values()), 1)
    print(f"\n  cân bằng vùng ảnh  : {'OK' if ratio >= 0.5 else '⚠ LỆCH — by_region sẽ vô nghĩa'}"
          f" (tỉ lệ nhỏ/lớn = {ratio:.2f})")
    if reg["border"] == 0:
        print("  ⚠ KHÔNG có điểm 'border' nào — hiện tượng khuếch đại ở rìa ảnh sẽ bị bỏ sót.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    p = argparse.ArgumentParser(
        description="Chọn điểm test + tính sẵn đáp án 3D chuẩn từ depth GT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python sample_points.py --data-dir D:\\replica_sim\\data\\replica_pp \\\n"
            "      --source replica_pp --num-samples 200 --output D:\\replica_sim\\samples \\\n"
            "      --seed 42 --visualize\n\n"
            "Luật chọn điểm: depth 1.5-6.0 m, lề >= 10 px, <= 5 điểm/frame,\n"
            "cân bằng center/border, seed cố định (chạy lại ra cùng kết quả).\n"
        ),
    )
    p.add_argument("--data-dir", required=True,
                   help="Thư mục chứa các scene (ví dụ D:\\replica_sim\\data\\replica_pp).")
    p.add_argument("--source", default="replica_pp", choices=sorted(SOURCE_CFG),
                   help="Nguồn dữ liệu (mặc định: replica_pp).")
    p.add_argument("--scenes", default=None,
                   help="Chỉ lấy các scene này, phân tách bằng dấu phẩy. "
                        "Mặc định: tất cả scene tìm thấy.")
    p.add_argument("--num-samples", type=int, default=200,
                   help="Tổng số điểm test cần (mặc định: 200).")
    p.add_argument("--output", default=None,
                   help="Thư mục ghi JSON. Bắt buộc trừ khi --dry-run.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed cố định để chạy lại ra cùng kết quả (mặc định: 42).")
    p.add_argument("--data-root", default=None,
                   help="Gốc dữ liệu để check ổ đĩa (mặc định D:\\replica_sim).")
    p.add_argument("--visualize", action="store_true",
                   help="Vẽ chấm lên ảnh để kiểm bằng mắt.")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ báo cáo phân bố, KHÔNG ghi file.")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        _die(f"Không tìm thấy thư mục dữ liệu: {data_dir}\n"
             f"    → Chạy `download_dataset.py` trước, hoặc kiểm tra lại --data-dir.")
    if args.num_samples <= 0:
        _die(f"--num-samples phải > 0, nhận {args.num_samples}")
    if not args.dry_run and not args.output:
        _die("Thiếu --output (hoặc dùng --dry-run để chỉ xem phân bố).")

    # Ổ đĩa — chặn trước khi ghi hàng loạt file.
    # ⚠️ Ngưỡng KHÁC với `download_dataset.py`: đây là I/O **nhẹ** (vài chục file JSON
    # trên ổ D), không phải tải/giải nén 11 GB. Ổ C thấp chỉ nguy hiểm khi ta sắp
    # làm I/O nặn; với thao tác nhẹ thì 0.5 GB vẫn an toàn. Xem contract mục 3.
    try:
        info = check_disk_space(min_c_gb=0.5, min_d_gb=1.0,
                                data_root=args.data_root or data_dir)
        print(f"Ổ đĩa: {info}")
    except RuntimeError as e:
        _die(str(e))

    cfg = SOURCE_CFG[args.source]
    available = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if not available:
        _die(f"Không có thư mục scene nào trong {data_dir}\n"
             f"    → Replica bản NICE-SLAM có: {', '.join(REPLICA_SCENES)}\n"
             f"    → Lưu ý: zip giải nén tạo thêm cấp `Replica\\`, nên --data-dir có thể phải là\n"
             f"      `{data_dir}\\Replica` thay vì `{data_dir}`.")
    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = available
    bad = [s for s in scenes if s not in available]
    if bad:
        _die(f"Scene không tồn tại trong {data_dir}: {', '.join(bad)}\n"
             f"    → Scene ĐANG CÓ: {', '.join(available)}\n"
             f"    → Nếu thấy `Replica` trong danh sách, hãy thêm `\\Replica` vào --data-dir.")

    print(f"\nNguồn   : {args.source}")
    print(f"Data    : {data_dir}")
    print(f"Scene   : {', '.join(scenes)}")
    print(f"Cần     : {args.num_samples} điểm, seed={args.seed}")

    rng = np.random.default_rng(args.seed)

    all_samples: List[Dict[str, Any]] = []
    by_scene: Dict[str, int] = {}

    # Replica dùng CHUNG 1 bộ camera cho mọi scene → đọc cam_params.json 1 lần.
    # Hệ số depth cũng lấy từ đó, không hardcode.
    intr_cache = None
    scale = cfg["depth_scale"]
    if args.source == "replica_pp":
        intr_cache = replica_intrinsics(data_dir)
        scale = intr_cache["png_depth_scale"]
        print(f"Camera (từ dataset): {intr_cache['width']}×{intr_cache['height']}, "
              f"fx={intr_cache['fx']}, fy={intr_cache['fy']}, "
              f"cx={intr_cache['cx']}, cy={intr_cache['cy']}")
        print(f"  nguồn: {intr_cache['source']}")
        print(f"  png_depth_scale: {scale}")
        if scale != cfg["depth_scale"]:
            print(f"  ⚠ dataset ghi scale={scale}, khác hằng số "
                  f"{cfg['depth_scale']} — dùng giá trị của dataset.")

    # chia đều quota cho mỗi scene để không dồn hết vào phòng đầu tiên
    per_scene = max(1, args.num_samples // len(scenes) + 1)
    for sc in scenes:
        print(f"\n[{sc}]")
        got = sample_one_scene(data_dir / sc, sc, args.source, cfg, rng,
                               per_scene, intr=intr_cache)
        if intr_cache is not None and scale != cfg["depth_scale"]:
            # đảm bảo sample ghi đúng hệ số đã dùng
            for g in got:
                g["depth_scale"] = float(scale)
        by_scene[sc] = len(got)
        all_samples.extend(got)
        print(f"  → {len(got)} điểm")
        if len(all_samples) >= args.num_samples:
            break
    all_samples = all_samples[: args.num_samples]

    report(all_samples, by_scene)
    if not all_samples:
        _die("Không sinh được sample nào — kiểm tra lại cấu trúc thư mục và depth scale.")

    if args.dry_run:
        print("\n[dry-run] Không ghi file.")
        return 0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for s in all_samples:
        sid = f"{s['source']}_{s['scene']}_{s['sample_id'].rsplit('_', 1)[-1]}"
        (out / f"{sid}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Ghi {len(all_samples)} file JSON vào {out}")

    if args.visualize:
        prev = out / "preview"
        print("\nẢnh preview (xanh = center, cam = border):")
        n = visualize(all_samples, prev)
        print(f"  → {n} ảnh trong {prev}")
        print("  HÃY MỞ XEM: chấm phải nằm đúng trên vật thể trong ảnh.")

    print(f"\nĐỌC KẾT QUẢ: {len(all_samples)} điểm. Tiếp theo là Phase 4 (run_eval.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
