# -*- coding: utf-8 -*-
"""
calibrate_camera.py — Lấy camera intrinsics THẬT (fx, fy, cx, cy)
================================================================

Vì sao cần?
-----------
`CameraIntrinsics.from_fov()` chỉ là ƯỚC LƯỢNG. Metric DepthART dùng
intrinsics làm input của model (camera conditioning), nên sai số intrinsics
→ sai số mét ngay ở depth map. Muốn `xyz_metric` đáng tin thì phải calibrate.

Cách dùng (bàn cờ chessboard)
------------------------------
1. In bàn cờ (VD 9x6 ô vuông, mỗi ô 30mm) → kích thước thật: 27cm x 18cm
2. Dán lên bề mặt phẳng cứng
3. Chụp 15-30 ảnh bàn cờ từ nhiều góc / khoảng cách, bao gồm cả góc nghiêng
4. Chạy:
       python calibrate_camera.py --images calib_frames/ --output intrinsics.json

Với camera cố định (webcam): 15-25 ảnh là đủ.
Với camera handheld: 25-40 ảnh, nhiều góc nghiêng hơn.

Yêu cầu chất lượng ảnh
-----------------------
✅ Bàn cờ chiếm ≥ 1/3 khung hình
✅ Nét, không mờ, không chuyển động
✅ Ánh sáng đều, tránh phản chiếu chói
✅ Đủ nét ở CẢ bốn góc bàn cờ
❌ Tránh nền rối, vật cản che bàn cờ

Kết quả
-------
File JSON: {fx, fy, cx, cy, width, height, source, reprojection_error}
Dùng được với `CameraIntrinsics.load()`.

Nếu không thể chụp bàn cờ (ví dụ camera cố định không tay cầm), dùng
`--from-fov` để lưu intrinsics ước lượng — nhớ đánh dấu source là
"fov_estimate" để pipeline cảnh báo khi dùng cho metric depth.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from localizer_3d import CameraIntrinsics  # noqa: E402

# Pattern bàn cờ phổ biến.
#
# ⚠️  QUY ƯỚC: (cols, rows) là số GÓC TRONG (inner corners), đúng như
#     cv2.findChessboardCorners yêu cầu. Một bàn cờ "9x6" ở đây là bàn cờ
#     10x7 ô vuông. Nếu đếm sai, detect sẽ luôn fail.
#
#     Sai số hay gặp: bàn cờ 7x5 ô vuông → phải dùng pattern "6x4".
CHESSBOARD_PATTERNS = {
    "9x6": (9, 6),      # bàn cờ 10x7 ô vuông
    "9x4": (9, 4),      # bàn cờ 10x5 ô vuông
    "8x6": (8, 6),      # bàn cờ 9x7 ô vuông
    "7x5": (7, 5),      # bàn cờ 8x6 ô vuông
    "6x4": (6, 4),      # bàn cờ 7x5 ô vuông
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ─────────────────────────────────────────────────────────────────────────────
#  Detection
# ─────────────────────────────────────────────────────────────────────────────

def find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """
    Tìm góc trong của bàn cờ.
    Dùng SB (structured borders) thay vì classic corner — chịu được
    góc nghiêng và focus kém tốt hơn nhiều.
    """
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if ok:
        return True, corners

    # Fallback: classic + subpixel refinement
    ok, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        return True, corners
    return False, None


def collect_corner_observations(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]], list[Path]]:
    """
    Duyệt ảnh, tìm góc trong, gom về object/image point lists.
    Trả về (objpoints, imgpoints, image_sizes, accepted_paths).
    """
    cols, rows = pattern_size
    obj_template = np.zeros((rows * cols, 3), np.float32)
    obj_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_sizes: list[tuple[int, int]] = []
    accepted: list[Path] = []

    for i, path in enumerate(image_paths, 1):
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            if verbose:
                print(f"  [{i:3d}/{len(image_paths)}] {path.name}: KHONG DOC DUOC")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_chessboard_corners(gray, pattern_size)
        if ok:
            objpoints.append(obj_template.copy())
            imgpoints.append(corners.astype(np.float32))
            image_sizes.append((img.shape[1], img.shape[0]))
            accepted.append(path)
            if verbose:
                print(f"  [{i:3d}/{len(image_paths)}] {path.name}: OK")
        elif verbose:
            print(f"  [{i:3d}/{len(image_paths)}] {path.name}: khong tim thay ban cua")

    return objpoints, imgpoints, image_sizes, accepted


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[CameraIntrinsics, np.ndarray, np.ndarray]:
    """
    Chạy calibrateCamera.

    Returns (intrinsics, rvecs, tvecs, dist)
    """
    width, height = image_size

    # OpenCV yêu cầu: objectPoints = Point3f, imagePoints = Point2f.
    # Ép kiểu ở đây để hàm chấp nhận được mọi dtype float32/float64,
    # tránh lỗi khó hiểu "Unsupported format or combination of formats".
    objpoints = [np.ascontiguousarray(o, dtype=np.float32) for o in objpoints]
    imgpoints = [np.ascontiguousarray(i, dtype=np.float32) for i in imgpoints]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    intrinsics = CameraIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=int(width), height=int(height),
        source="chessboard",
        reprojection_error=float(rms),
    )
    return intrinsics, rvecs, tvecs, dist


def per_view_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """
    RMSE (pixel) của từng ảnh — giúp phát hiện ảnh nhiễu cần loại.

    ⚠️  `dist` PHẢI là distortion ước lượng từ calibrateCamera.
        Truyền None sẽ tính sai cho mọi camera có distortion (ảnh ở rìa khung
        bị cong → RMSE tăng giả → ảnh tốt bị loại nhầm).
    """
    if dist is None:
        dist = np.zeros((1, 5), np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(1, -1)

    errors = []
    for obj, img_pts, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1)))))
    return np.asarray(errors)


def auto_drop_outliers(
    errors: np.ndarray,
    image_paths: list[Path],
    threshold: float = 2.0,
    min_keep: int = 8,
) -> list[int]:
    """
    Chọn các ảnh có RMSE < threshold (hoặc trung vị nếu phần lớn ảnh tệ).
    Trả về danh sách index được giữ.
    """
    med = float(np.median(errors))
    limit = min(threshold, med * 1.5)
    keep = [i for i, e in enumerate(errors) if e <= limit]
    if len(keep) < min_keep:
        # Không đủ ảnh tốt — giữ min_keep ảnh tốt nhất
        order = np.argsort(errors)
        keep = sorted(int(i) for i in order[:min_keep])
    return keep


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_on_image(img: np.ndarray, corners: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray:
    """Vẽ bàn cờ lên ảnh để kiểm tra thủ công."""
    out = img.copy()
    cv2.drawChessboardCorners(out, pattern_size, corners, 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(
        description="Calibrate camera intrinsics bằng chessboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[:1500],
    )
    p.add_argument("--images", help="thư mục chứa ảnh bàn cờ (hoặc 1 file glob)")
    p.add_argument("--pattern", default="9x6",
                   help=f"kích thước bàn cờ = số GÓC TRONG (inner corners). "
                        f"Có sẵn: {list(CHESSBOARD_PATTERNS)}")
    p.add_argument("--square-size", type=float, default=30.0,
                   help="cạnh 1 ô vuông (mm). Chỉ dùng để in/ghi chú — "
                        "không ảnh hưởng intrinsics")
    p.add_argument("--output", default="intrinsics.json", help="file JSON kết quả")
    p.add_argument("--from-fov", type=float, default=None,
                   help="bỏ qua calibration, chỉ lưu intrinsics ước lượng từ FOV (độ)")
    p.add_argument("--img-size", nargs=2, type=int, metavar=("W", "H"),
                   help="kích thước ảnh cho --from-fov")
    p.add_argument("--drop-outliers", action="store_true",
                   help="tự loại ảnh có RMSE cao rồi calibrate lại")
    p.add_argument("--preview", default=None,
                   help="thư mục lưu ảnh vẽ góc trong để kiểm tra thủ công")
    p.add_argument("--min-images", type=int, default=8,
                   help="số ảnh tối thiểu để calibrate")
    return p.parse_args()


def cmd_from_fov(args: argparse.Namespace) -> int:
    if not args.img_size:
        print("Loi: --from-fov can --img-size W H")
        return 1
    w, h = args.img_size
    intr = CameraIntrinsics.from_fov(width=w, height=h, fov_h_deg=args.from_fov)
    out = Path(args.output)
    intr.save(out)
    print(f"Da luu intrinsics UOC LUONG vao {out}")
    print(f"  fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.cx:.2f} cy={intr.cy:.2f} "
          f"@ {intr.width}x{intr.height}")
    print("  [canh bao] day KHONG phai calibration that.")
    print("  Metric DepthART se cho sai so do voi intrinsics uoc luong.")
    return 0


def main() -> int:
    args = parse_args()

    if args.from_fov is not None:
        return cmd_from_fov(args)

    if not args.images:
        print("Loi: thieu --images (thu muc anh ban cua) hoac --from-fov")
        return 1

    if args.pattern not in CHESSBOARD_PATTERNS:
        print(f"Loi: pattern '{args.pattern}' khong hop le. "
              f"Dung mot trong: {list(CHESSBOARD_PATTERNS)}")
        return 1
    pattern_size = CHESSBOARD_PATTERNS[args.pattern]
    cols, rows = pattern_size

    # Gom danh sach anh
    p = Path(args.images)
    if p.is_dir():
        paths = sorted(
            q for q in p.rglob("*")
            if q.suffix.lower() in IMAGE_EXTS
        )
    else:
        paths = [Path(q) for q in sorted(glob.glob(args.images))]
        paths = [q for q in paths if q.suffix.lower() in IMAGE_EXTS]

    if not paths:
        print(f"Loi: khong tim thay anh nao trong '{args.images}'")
        return 1

    print("=" * 66)
    print("  CAMERA INTRINSICS CALIBRATION (chessboard)")
    print("=" * 66)
    print(f"  Pattern   : {cols}x{rows} goc trong "
          f"(= ban cuo {cols + 1}x{rows + 1} o vuong, {args.square_size:g} mm/o)")
    print(f"  So anh    : {len(paths)}")
    print()

    objpoints, imgpoints, image_sizes, accepted = collect_corner_observations(
        paths, pattern_size
    )

    n_ok = len(accepted)
    print()
    if n_ok < args.min_images:
        print(f"LOI: chi tim thay bàn cờ tren {n_ok}/{len(paths)} anh "
              f"(can it nhat {args.min_images}).")
        print()
        print("Nguyen nhan thuong gap:")
        print(f"  - SAI SO O: dang dung --pattern {args.pattern} = "
              f"{cols}x{rows} GOC TRONG,")
        print(f"    tuong duong ban cuo {cols + 1}x{rows + 1} o vuong.")
        print("    Neu ban cuo that khac so, sua --pattern cho dung.")
        print("  - Ban qua nho / qua xa so voi camera")
        print("  - Anh mo nhieu hoac nham nhieu")
        print("  - Ban qua khong phang hoac khong du sang o mot phia")
        print()
        print(f"Thu tu detect (theo file, dung de so sanh):")
        for q in paths:
            print(f"    {q.name}")
        return 1

    if n_ok < len(paths):
        print(f"  [canh bao] {len(paths) - n_ok} anh khong detect duoc ban cua")
    print()

    # Calibrate lần đầu
    img_size = image_sizes[0]
    sizes = {s for s in image_sizes}
    if len(sizes) > 1:
        print(f"  [canh bao] anh khac kich thuoc nhau ({sizes}) — dung kich thuoc dau tien")
        print(f"           nen resize tat ca anh ve cung kich thuoc truoc khi chup")
        print()

    intr, rvecs, tvecs, dist = calibrate(objpoints, imgpoints, img_size)
    errs = per_view_errors(objpoints, imgpoints, rvecs, tvecs,
                           intr.to_matrix(), dist)
    print(f"  RMSE tong the (toan cac anh): {intr.reprojection_error:.4f} px")
    print(f"  Distortion uoc luong: "
          f"k1={dist[0,0]:.4f} k2={dist[0,1]:.4f} p1={dist[0,2]:.5f} "
          f"p2={dist[0,3]:.5f} k3={dist[0,4]:.4f}")

    # Loại ảnh nhiễu nếu được yêu cầu
    if args.drop_outliers:
        keep = auto_drop_outliers(errs, accepted)
        if len(keep) < n_ok:
            print(f"  Tu loai {n_ok - len(keep)} anh nhieu (RMSE cao)")
            objpoints = [objpoints[i] for i in keep]
            imgpoints = [imgpoints[i] for i in keep]
            image_sizes = [image_sizes[i] for i in keep]
            accepted = [accepted[i] for i in keep]
            errs = errs[keep]
            intr, rvecs, tvecs, dist = calibrate(
                objpoints, imgpoints, image_sizes[0]
            )
            print(f"  RMSE sau khi loai: {intr.reprojection_error:.4f} px "
                  f"({len(accepted)} anh)")
        else:
            print("  Khong co anh nao duoc loai (RMSE deu tot)")

    # Preview
    if args.preview:
        prev_dir = Path(args.preview)
        prev_dir.mkdir(parents=True, exist_ok=True)
        for path, corners in zip(accepted, imgpoints):
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imwrite(str(prev_dir / f"detected_{path.stem}.jpg"),
                        draw_on_image(img, corners, pattern_size))
        print(f"  Anh kiem tra ban cua: {prev_dir}")

    # Xuất kết quả
    out = Path(args.output)
    intr.save(out)

    print()
    print("=" * 66)
    print("  KET QUA CALIBRATION")
    print("=" * 66)
    print(f"  fx  = {intr.fx:10.4f}   px")
    print(f"  fy  = {intr.fy:10.4f}   px")
    print(f"  cx  = {intr.cx:10.4f}   px")
    print(f"  cy  = {intr.cy:10.4f}   px")
    print(f"  size= {intr.width} x {intr.height}")
    print(f"  FOV_h = {2 * np.degrees(np.arctan(intr.width / (2 * intr.fx))):.2f} deg")
    print(f"  RMSE = {intr.reprojection_error:.4f} px  tren {len(accepted)} anh")
    print()
    print(f"  Da luu: {out}")

    # Đánh giá chất lượng
    rms = intr.reprojection_error
    print()
    if rms < 0.5:
        print("  Chat luong: RAT TOT  — dung cho metric depth")
    elif rms < 1.0:
        print("  Chat luong: TOT — du dung cho metric depth")
    elif rms < 2.0:
        print("  Chat luong: KHA — du cho zone/xyz_normalized,")
        print("              metric depth co sai so metre")
    else:
        print("  Chat luong: KEM — can chup lai:")
        print("    - Anh so 'mot' nen ro (bo anh rat xa/near)")
        print("    - Bàn cờ phai chiem >= 1/3 khung hinh")
        print("    - Goc nghieng vua phai (khong quá 45 do)")
        print("    - Dung --drop-outliers de loai anh nhieu")

    # Kiểm tra tính hợp lệ của focal length (trường hợp suy biến)
    fov_h = 2 * np.degrees(np.arctan(intr.width / (2 * intr.fx)))
    if fov_h < 20 or fov_h > 140:
        print()
        print("  [CANH BAO] Focal length khong hop le:")
        print(f"    fx={intr.fx:.1f} px → FOV_h={fov_h:.1f}°")
        print("    Camera thuong co FOV_h khoang 40°-110°.")
        print("    Nguyen nhan: cac anh chup qua MON PHANG (chi doi vi tri/rot nhe),")
        print("    khong co bien do thi sau nao → focal length khong xac dinh duoc.")
        print("    Cach sua: chup o nhieu GOC NGHIENG khac nhau (20°-45°)")
        print("             va o nhieu khoang cach khac nhau.")
        print("    Hien tai: chi dung cho zone/xyz_normalized,")
        print("              CHUA dung cho metric depth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())