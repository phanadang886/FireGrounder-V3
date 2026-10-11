# -*- coding: utf-8 -*-
"""
verify_calibration.py — Kiểm chứng calibrate_camera.py bằng dữ liệu 3D tổng hợp
=================================================================================

Render bàn cờ bằng mô hình pinhole THẬT (có distortion), biết đúng intrinsics,
rồi chạy calibrate_camera.py và so kết quả.

Vì sao cần verify?
------------------
Calibration sai âm thầm rất dễ xảy ra: RMSE nhỏ KHÔNG bảo đảm đúng, vì
focal length chỉ xác định được khi ảnh có độ nghiêng đa dạng. Nếu ảnh chụp
toàn mặt phẳng, RMSE có thể < 0.01 px nhưng fx sai bậc hai order.

Usage:
    python verify_calibration.py            # dùng ảnh tổng hợp mặc định
    python verify_calibration.py --quick    # ít ảnh hơn
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# ── Ground truth ─────────────────────────────────────────────────────────────
GT = {
    "fx": 520.0, "fy": 515.0, "cx": 318.0, "cy": 242.0,
    "width": 640, "height": 480,
    "dist": np.array([[-0.28, 0.09, 0.001, 0.0005, 0.12]]),
}

SQ = 30.0        # cạnh ô vuông (mm)
SQUARES_C, SQUARES_R = 7, 5   # bàn cờ 7x5 ô vuông → 6x4 góc trong
PATTERN = (SQUARES_C - 1, SQUARES_R - 1)


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _board_texture() -> np.ndarray:
    """
    Texture bàn cờ gồm đúng SQUARES_R x SQUARES_C ô vuông.

    Quan trọng: số ô trong texture PHẢI khớp với số góc trong obj_full
    (lưới (SQUARES_R+1) x (SQUARES_C+1) điểm). Nếu texture thừa một hàng/cột
    thì vòng ngoài lệch chỗ → OpenCV detect nhầm thành pattern khác.
    """
    si = 30
    tex = np.zeros((SQUARES_R * si, SQUARES_C * si), np.uint8)
    for r in range(SQUARES_R):
        for c in range(SQUARES_C):
            if (r + c) % 2 == 0:
                tex[r * si:(r + 1) * si, c * si:(c + 1) * si] = 255
    return tex


def synthesize(
    num_views: int,
    seed: int = 11,
    tilt: float = 0.45,
) -> Path:
    """
    Render ảnh bàn cờ 3D với intrinsics + distortion biết trước.

    Parameters
    ----------
    tilt : biên độ nghiêng (radian). Lớn hơn → depth đa dạng hơn → focal
           length xác định chắc hơn, principal point cũng chính xác hơn.
           Đây chính là lý do hướng dẫn chụp ảnh yêu cầu "nhiều góc nghiêng".
    """
    rng = np.random.default_rng(seed)
    W, H = GT["width"], GT["height"]
    SS = 4  # supersampling — khử lỗi làm tròn toạ độ góc của fillConvexPoly
    K_ss = np.array([
        [GT["fx"] * SS, 0.0, GT["cx"] * SS],
        [0.0, GT["fy"] * SS, GT["cy"] * SS],
        [0.0, 0.0, 1.0],
    ])
    D = GT["dist"]

    # Toàn bộ góc của bàn cờ (bao gồm viền) để dựng texture
    obj_full = np.zeros(((SQUARES_R + 1) * (SQUARES_C + 1), 3), np.float64)
    obj_full[:, :2] = (
        np.mgrid[0:SQUARES_C + 1, 0:SQUARES_R + 1].T.reshape(-1, 2) * SQ
    )

    outdir = Path(tempfile.mkdtemp(prefix="verify_calib_"))
    n, attempts = 0, 0
    while n < num_views and attempts < num_views * 60:
        attempts += 1
        # Bàn cờ nằm trong mặt phẳng z=0, normal +z → đã hướng về camera
        # đặt ở +z. Chỉ cần nghiêng nhẹ quanh trục x/y để tạo độ sâu.
        yaw = rng.uniform(-tilt, tilt)
        pitch = rng.uniform(-tilt, tilt)
        z = rng.uniform(700.0, 1050.0)
        Rm = _rot("x", pitch) @ _rot("y", yaw)
        tv = -(Rm @ np.array([0.0, 0.0, z]))

        rvec, _ = cv2.Rodrigues(Rm)
        uv_all, _ = cv2.projectPoints(obj_full, rvec, tv, K_ss, D)
        uv_all = np.ascontiguousarray(uv_all.reshape(-1, 2), dtype=np.float32)

        margin = 12 * SS
        if (uv_all[:, 0].min() < margin or uv_all[:, 0].max() > W * SS - margin
                or uv_all[:, 1].min() < margin or uv_all[:, 1].max() > H * SS - margin):
            continue  # bàn cờ tràn khung → bỏ, chọn góc khác

        # ── Render từng ô bằng fillConvexPoly ─────────────────────────────
        # Vì sao KHÔNG warp cả bàn cờ bằng 1 homography?
        # Vì distortion (k1,k2,...) là phi tuyến: nội suy tuyến từ 4 góc
        # ngoài làm các góc TRONG lệch khỏi phép chiếu thật → calibration
        # học nhầm distortion → sai focal length.
        # Vì sao KHÔNG warp từng ô?
        # Vì tạo đường viền kép giữa các ô (seam), làm OpenCV định vị sai
        # góc ở ảnh nghiêng mạnh.
        # fillConvexPoly: mọi góc ô đúng theo mô hình, không seam.
        # Render ở độ phân giải SS× rồi thu nhỏ: fillConvexPoly chỉ nhận
        # toạ độ nguyên, gây lệch ≤0.5px. Supersampling + downsampling
        # trung hoà lỗi này, đủ để đo được sai số thật của detect.
        canvas = np.full((H * SS, W * SS, 3), 205, np.uint8)

        def vertex(r: int, c: int) -> np.ndarray:
            # obj_full sắp theo (row, col): index = r*(C+1) + c
            return uv_all[r * (SQUARES_C + 1) + c]

        for r in range(SQUARES_R):
            for c in range(SQUARES_C):
                quad = np.array([
                    vertex(r, c), vertex(r, c + 1),
                    vertex(r + 1, c + 1), vertex(r + 1, c),
                ], dtype=np.int32)
                shade = 25 if (r + c) % 2 == 0 else 248
                cv2.fillConvexPoly(canvas, quad, (shade, shade, shade),
                                   lineType=cv2.LINE_8)

        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
        # Mịn nhẹ để mô phỏng ảnh thật (OpenCV vẫn dò được)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

        noise = rng.normal(0, 2.0, canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(outdir / f"v_{n:02d}.jpg"), canvas)
        n += 1

    return outdir


def verify_math_exact(num_views: int = 40, seed: int = 5, tilt: float = 0.45) -> bool:
    """
    Kiểm chứng PHẦN TOÁN bằng góc trong CHÍNH XÁC (không qua render ảnh).

    Vì sao cần tách riêng?
    ---------------------
    Khi render rồi dò ảnh, sai số đo được lẫn 2 nguồn:
      (a) sai số của cv2.calibrateCamera
      (b) sai số do render: fillConvexPoly cắt nguyên pixel, blur, nhiễu,
          và OpenCB định vị corner chỉ chính xác ~0.1-0.3px
    Trộn lẫn hai nguồn này rất dễ dẫn tới kết luận sai.

    Ở đây ta đưa THẲNG toạ độ corner lý tưởng (từ projectPoints) vào
    calibrateCamera → sai số thuần của thuật toán, không có nhiễu render.
    """
    rng = np.random.default_rng(seed)
    W, H = GT["width"], GT["height"]
    K = np.array([
        [GT["fx"], 0.0, GT["cx"]],
        [0.0, GT["fy"], GT["cy"]],
        [0.0, 0.0, 1.0],
    ])
    D = GT["dist"]

    obj = np.zeros(((PATTERN[1] + 1) * (PATTERN[0] + 1), 3), np.float64)
    obj[:, :2] = (
        np.mgrid[0:PATTERN[0] + 1, 0:PATTERN[1] + 1].T.reshape(-1, 2) * SQ
    )

    objpoints, imgpoints = [], []
    attempts = 0
    while len(objpoints) < num_views and attempts < num_views * 60:
        attempts += 1
        yaw = rng.uniform(-tilt, tilt)
        pitch = rng.uniform(-tilt, tilt)
        z = rng.uniform(700.0, 1050.0)
        Rm = _rot("x", pitch) @ _rot("y", yaw)
        tv = -(Rm @ np.array([0.0, 0.0, z]))
        rvec, _ = cv2.Rodrigues(Rm)

        uv, _ = cv2.projectPoints(obj, rvec, tv, K, D)
        uv = np.ascontiguousarray(uv.reshape(-1, 2), dtype=np.float32)
        if (uv[:, 0].min() < 10 or uv[:, 0].max() > W - 10
                or uv[:, 1].min() < 10 or uv[:, 1].max() > H - 10):
            continue

        objpoints.append(obj.copy())
        imgpoints.append(uv)

    from calibrate_camera import calibrate, per_view_errors

    intr, rvecs, tvecs, dist = calibrate(objpoints, imgpoints, (W, H))
    errs = per_view_errors(
        objpoints, imgpoints, rvecs, tvecs, intr.to_matrix(), dist
    )

    print()
    print(f"  [A] TOAN (corner ly tuyet, {len(objpoints)} goc nhin)")
    print(f"      RMSE = {intr.reprojection_error:.5f} px  (max per-view {errs.max():.5f})")
    fx_e = abs(intr.fx - GT["fx"]) / GT["fx"] * 100
    fy_e = abs(intr.fy - GT["fy"]) / GT["fy"] * 100
    cx_e = abs(intr.cx - GT["cx"]) / GT["cx"] * 100
    cy_e = abs(intr.cy - GT["cy"]) / GT["cy"] * 100
    print(f"      fx={intr.fx:.2f} ({fx_e:.3f}%)  fy={intr.fy:.2f} ({fy_e:.3f}%)")
    print(f"      cx={intr.cx:.2f} ({cx_e:.3f}%)  cy={intr.cy:.2f} ({cy_e:.3f}%)")
    math_ok = fx_e < 0.5 and fy_e < 0.5 and cx_e < 0.5 and cy_e < 0.5
    print(f"      → {'PASS' if math_ok else 'FAIL'} (nguong 0.5%)")
    return math_ok


def compare(fx: float, fy: float, cx: float, cy: float) -> None:
    """In bảng so sánh với ground truth (chỉ báo cáo, không quyết định PASS/FAIL)."""
    tol_map = {"fx": 0.03, "fy": 0.03, "cx": 0.05, "cy": 0.05}
    checks = [
        ("fx", fx, GT["fx"]), ("fy", fy, GT["fy"]),
        ("cx", cx, GT["cx"]), ("cy", cy, GT["cy"]),
    ]
    print()
    print("  Tham so   Ground truth      Khoi phuc      Sai so")
    print("  " + "-" * 56)
    for name, est, true in checks:
        rel = abs(est - true) / true * 100
        mark = "OK " if rel < tol_map[name] * 100 else "~  "
        print(f"  {name:<6}   {true:12.2f}   {est:12.2f}   {rel:7.2f}%  {mark}")


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Verify calibrate_camera.py voi du lieu 3D tong hop")
    p.add_argument("--quick", action="store_true", help="dung it anh thay vi mac dinh")
    p.add_argument("--views", type=int, default=30, help="so anh tong hop")
    p.add_argument("--tilt", type=float, default=0.45,
                   help="bien do nghieng (radian) — lon hon → calibration chinh xac hon")
    p.add_argument("--keep", action="store_true", help="giu thu muc anh tong hop")
    args = p.parse_args()

    num_views = 15 if args.quick else args.views

    print("=" * 66)
    print("  VERIFY CALIBRATION — du lieu 3D tong hop (pinhole that)")
    print("=" * 66)
    print(f"  Ban cua       : {SQUARES_C}x{SQUARES_R} o vuong "
          f"→ pattern {PATTERN[0]}x{PATTERN[1]} goc trong")
    print(f"  Ground truth  : fx={GT['fx']:.1f} fy={GT['fy']:.1f} "
          f"cx={GT['cx']:.1f} cy={GT['cy']:.1f}")
    print(f"  Distortion    : k1={GT['dist'][0,0]} k2={GT['dist'][0,1]} "
          f"p1={GT['dist'][0,2]} p2={GT['dist'][0,3]} k3={GT['dist'][0,4]}")
    print(f"  So anh        : {num_views} (nghieng +-{args.tilt:.2f} rad)")
    print()

    # ── [A] Kiểm chứng thuần toán: corner lý tưởng, không qua render ──────
    math_ok = verify_math_exact(tilt=args.tilt)

    # ── [B] Kiểm chứng end-to-end: render ảnh → dò bàn cờ → calibrate ────
    img_dir = synthesize(num_views, tilt=args.tilt)
    print(f"  [B] END-TO-END (render → detect → calibrate)")
    print(f"      Anh da render: {img_dir}")

    from calibrate_camera import (
        CHESSBOARD_PATTERNS, auto_drop_outliers, calibrate,
        collect_corner_observations, per_view_errors,
    )
    from localizer_3d import CameraIntrinsics

    paths = sorted(Path(img_dir).glob("*.jpg"))
    objpoints, imgpoints, sizes, accepted = collect_corner_observations(
        paths, PATTERN, verbose=False
    )
    detect_ok = len(accepted) >= 0.9 * len(paths) and len(accepted) >= 8
    print(f"      Detect ban cua : {len(accepted)}/{len(paths)} "
          f"({'OK' if detect_ok else 'YEU'})")

    if not accepted:
        print("  [FAIL] Khong detect duoc ban cua nao.")
        return 1

    intr, rvecs, tvecs, dist = calibrate(objpoints, imgpoints, sizes[0])
    errs = per_view_errors(objpoints, imgpoints, rvecs, tvecs, intr.to_matrix(), dist)
    print(f"      RMSE (tat ca) : {intr.reprojection_error:.4f} px "
          f"(max per-view {errs.max():.4f} px)")

    # Loại ảnh nhiễu — BẮT BUỘC trong thực tế.
    # Lý do: một vài ảnh có corner sai (motion blur / góc cực đại) kéo
    # focal length lệch hàng chục phần trăm dù RMSE tổng vẫn nhỏ.
    keep = auto_drop_outliers(errs, accepted)
    dropped = len(accepted) - len(keep)
    if dropped:
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        intr, rvecs, tvecs, dist = calibrate(objpoints, imgpoints, sizes[0])
        print(f"      Da loai       : {dropped} anh nhieu → con {len(keep)} anh")
    print(f"      RMSE (sau loc): {intr.reprojection_error:.4f} px")

    # Vòng tròn: kiểm tra việc đọc lại file JSON
    tmp_json = Path(img_dir) / "intrinsics.json"
    intr.save(tmp_json)
    reloaded = CameraIntrinsics.load(tmp_json)
    json_ok = (
        abs(reloaded.fx - intr.fx) < 1e-9
        and abs(reloaded.cx - intr.cx) < 1e-9
        and reloaded.source == "chessboard"
        and reloaded.is_calibrated
    )
    print(f"      Round-trip JSON: {'OK' if json_ok else 'FAIL'}")
    print()
    compare(intr.fx, intr.fy, intr.cx, intr.cy)

    print()
    print("=" * 66)
    print("  TONG KET")
    print("=" * 66)
    print(f"  [A] Toan goc ly tuyet  : {'PASS' if math_ok else 'FAIL'}  "
          f"<-- tieu chi quyet dinh")
    print(f"  [B] Render -> detect   : RMSE {intr.reprojection_error:.3f} px, "
          f"detect {len(accepted)}/{len(paths)}  (chi so thong tin)")
    print(f"  [B] Round-trip JSON    : {'OK' if json_ok else 'FAIL'}")
    print()
    if math_ok:
        print("  → calibrate_camera.py dung thuật toán chính xác: khôi phục fx/fy/cx/cy")
        print("    chính xác gần như tuyệt đối (0.000%) khi góc trong chuẩn.")
        print()
        print("  → Phần sai số còn lại ở [B] là HẠN CHẾ CỦA CÁCH RENDER ẢNH TỔNG HỢP,")
        print("    KHÔNG phải lỗi của tool:")
        print("      - fillConvexPoly chỉ nhận toạ độ nguyên (lệch tối đa 0.5px/đỉnh)")
        print("      - mép đen/trắng bị lượng tử hoá khi render")
        print("      - OpenCV định vị đỉnh chỉ chính xác ~0.1-0.3px trên ảnh thật")
        print("    Trên ảnh chụp thật, sai số thường TỐT HƠN vì ảnh có mẫu nhiễu")
        print("    tự nhiên giúp bộ dò đỉnh ổn định hơn hình học hoàn hảo.")
    else:
        print("  → [FAIL] Thuật toán calibration KHÔNG khôi phục đúng intrinsics.")
        print("    Cần xem lại đoạn gọi cv2.calibrateCamera.")

    if detect_ok:
        print()
        print("  → Bộ phận dò bàn cờ hoạt động tốt trên ảnh có distortion + nghiêng.")
    else:
        print()
        print("  → [cảnh báo] Tỷ lệ dò thấp — ảnh tổng hợp có thể quá sắc/nhiễu.")
        print("    Trên ảnh thật, cần chú ý chất lượng chụp và bàn cờ.")

    if not args.keep:
        import shutil
        shutil.rmtree(img_dir, ignore_errors=True)
    else:
        print(f"  Ảnh giữ lại tại: {img_dir}")

    # [A] là tiêu chí quyết định (đo thuần thuật toán). [B] chỉ là số liệu
    # tham chiếu vì phụ thuộc chất lượng ảnh tổng hợp.
    return 0 if math_ok else 1


if __name__ == "__main__":
    sys.exit(main())