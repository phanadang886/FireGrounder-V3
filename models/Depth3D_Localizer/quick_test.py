# -*- coding: utf-8 -*-
"""
quick_test.py - Quick test cho Depth3D_Localizer (không cần model/GPU)
=======================================================================

Chạy: python quick_test.py

Phạm vi kiểm thử:
  - localizer_3d: zone, batch, xyz_normalized, xyz_metric, pinhole
  - CameraIntrinsics: from_fov, scale, save/load JSON
  - HỢP ĐỒNG metric vs relative depth (đảo chiều zone) — mục dễ sai nhất
  - metric_depth / calibrate_camera / pipeline: import + validate (không tải model)

Cố tình KHÔNG test phần cần model nặng (DepthART, YOLO) — dùng benchmark.py.
"""
import io
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

import numpy as np

# Test localizer_3d.py không cần GPU/model
sys.path.insert(0, str(Path(__file__).parent))
from localizer_3d import CameraIntrinsics, FireLocalizer3D, FireLocation3D

PASSED = 0


def check(cond: bool, msg: str):
    global PASSED
    assert cond, msg
    PASSED += 1
    print(f"  [PASS] {msg}")


print("=" * 60)
print("  Depth3D_Localizer — Quick Unit Test")
print("=" * 60)

# --- Test 1: Basic localization (relative depth) ---
print("\n[Test 1] Basic localization voi depth_map gia")
H, W = 480, 640

# Giả lập depth map: vùng giữa gần, rìa xa
depth_map = np.zeros((H, W), dtype=np.float32)
for i in range(H):
    for j in range(W):
        dist_center = np.sqrt((i - H / 2) ** 2 + (j - W / 2) ** 2)
        depth_map[i, j] = 1.0 - dist_center / np.sqrt((H / 2) ** 2 + (W / 2) ** 2)

localizer = FireLocalizer3D(near_thresh=0.70, far_thresh=0.30)

loc_near = localizer.localize(depth_map, kx=320, ky=240, img_w=W, img_h=H, model_name="mock")
print(f"\n  Diem giua (320, 240): zone={loc_near.zone}, depth_norm={loc_near.depth_normalized:.3f}")
check(loc_near.zone == "NEAR", "Diem giua -> NEAR")

loc_far = localizer.localize(depth_map, kx=0, ky=0, img_w=W, img_h=H, model_name="mock")
print(f"  Diem goc (0, 0): zone={loc_far.zone}, depth_norm={loc_far.depth_normalized:.3f}")
check(loc_far.zone == "FAR", "Diem goc -> FAR")

# --- Test 2: CameraIntrinsics tu FOV ---
print("\n[Test 2] CameraIntrinsics tu FOV")
intr = CameraIntrinsics.from_fov(width=640, height=480, fov_h_deg=90.0)
print(f"  fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.cx}, cy={intr.cy}")
check(abs(intr.fx - 320.0) < 1.0, "fx ~ 320 (dung cho FOV=90 do, W=640)")
check(not intr.is_calibrated, "from_fov -> is_calibrated = False")
check(intr.source == "fov_estimate", "from_fov -> source='fov_estimate'")

# --- Test 3: xyz_metric CHI co khi metric_depth=True ---
print("\n[Test 3] xyz_metric — chi co voi METRIC depth + intrinsics")
loc_rel = localizer.localize(
    depth_map, kx=400, ky=300, img_w=W, img_h=H, intrinsics=intr, model_name="mock"
)
check(loc_rel.xyz_metric is None,
      "RELATIVE depth -> xyz_metric = None (khong du scale de doi sang met)")

# Tạo depth map METRIC đơn vị mét
print("\n[Test 4] METRIC depth -> xyz_metric don vi MET")
mH, mW = 100, 100
metric_map = np.tile(np.linspace(1.0, 9.0, mH).reshape(mH, 1), (1, mW)).astype(np.float32)
# QUAN TRỌNG: metric depth giá trị THẤP = GẦN (ngược relative depth)
m_intr = CameraIntrinsics.from_fov(width=mW, height=mH, fov_h_deg=90.0)

loc_m_near = localizer.localize(
    metric_map, kx=50, ky=5, img_w=mW, img_h=mH,
    intrinsics=m_intr, metric_depth=True, model_name="mock",
)
print(f"  Diem gan (y=5):  Z={loc_m_near.depth_m:.2f}m zone={loc_m_near.zone} "
      f"xyz={tuple(round(v, 2) for v in loc_m_near.xyz_metric)}")
check(loc_m_near.zone == "NEAR",
      "metric depth: diem GAN (1m) -> zone NEAR (dung quy uoc)")

loc_m_far = localizer.localize(
    metric_map, kx=50, ky=95, img_w=mW, img_h=mH,
    intrinsics=m_intr, metric_depth=True, model_name="mock",
)
print(f"  Diem xa  (y=95): Z={loc_m_far.depth_m:.2f}m zone={loc_m_far.zone} "
      f"xyz={tuple(round(v, 2) for v in loc_m_far.xyz_metric)}")
check(loc_m_far.zone == "FAR",
      "metric depth: diem XA (9m) -> zone FAR (dung quy uoc)")

check(loc_m_near.depth_m < loc_m_far.depth_m,
      "metric: depth_m cua diem GAN < depth_m cua diem XA")
check(loc_m_far.xyz_metric is not None, "metric depth -> xyz_metric khac None")

# Điểm CHÍNH GIỮA ảnh (50,50) → X,Y phải ~ 0
loc_center = localizer.localize(
    metric_map, kx=50, ky=50, img_w=mW, img_h=mH,
    intrinsics=m_intr, metric_depth=True, model_name="mock",
)
check(abs(loc_center.xyz_metric[0]) < 1e-6 and abs(loc_center.xyz_metric[1]) < 1e-6,
      "metric: diem CHINH GUA -> X,Y ~ 0")
check(loc_center.zone == "MID", "metric: diem chinh giua -> zone MID")

# Dấu của X, Y theo quy ước ảnh: X tăng sang phải, Y tăng xuống dưới
check(loc_m_near.xyz_metric[1] < 0,
      "metric: diem o PHAI TREN -> Y < 0 (dương = hướng xuống)")
check(loc_m_far.xyz_metric[1] > 0,
      "metric: diem o PHAI DUOI -> Y > 0")
# X tính theo khoảng cách tới tâm: cùng X pixel (50) → X = 0
check(abs(loc_m_near.xyz_metric[0]) < 1e-6,
      "metric: cùng cột pixel (x=50 = cx) -> X = 0")

# --- Test 5: validate input metric_depth ---
print("\n[Test 5] Validate input metric_depth")
try:
    localizer.localize(metric_map, kx=50, ky=50, img_w=mW, img_h=mH, metric_depth=True)
    check(False, "phai raise ValueError khi thieu intrinsics")
except ValueError as e:
    check("CameraIntrinsics" in str(e),
          "metric_depth=True thieu intrinsics -> ValueError ro rang")

try:
    localizer.localize(
        np.zeros((10, 10), np.float32), kx=5, ky=5,
        intrinsics=CameraIntrinsics.from_fov(10, 10), metric_depth=True,
    )
    check(False, "phai raise ValueError khi depth <= 0")
except ValueError:
    check(True, "metric depth <= 0 -> ValueError (khong chia 0)")

# --- Test 6: Batch localize ---
print("\n[Test 6] Batch localize")
kpts = [(320, 240), (100, 100), (600, 400)]
locs = localizer.localize_batch(depth_map, kpts, img_w=W, img_h=H)
check(len(locs) == 3, "batch 3 keypoints -> 3 ket qua")
print(f"  Zone: {[l.zone for l in locs]}")

locs_m = localizer.localize_batch(
    metric_map, [(50, 5), (50, 95)], img_w=mW, img_h=mH,
    intrinsics=m_intr, metric_depth=True,
)
check([l.zone for l in locs_m] == ["NEAR", "FAR"],
      "localize_batch + metric_depth -> [NEAR, FAR]")

# --- Test 7: Output formats ---
print("\n[Test 7] Output formats")
d = loc_near.to_dict()
check("zone" in d and "xyz_normalized" in d, "to_dict() co 'zone' va 'xyz_normalized'")
check("depth_m" in d and "xyz_metric" in d, "to_dict() co 'depth_m' va 'xyz_metric'")
print(f"  JSON keys: {list(d.keys())}")
print(f"\n  Str output:\n{loc_m_near}")

# --- Test 8: Intrinsics save/load + scale + pinhole ---
print("\n[Test 8] CameraIntrinsics save/load/scale/pinhole")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "intr.json"
    intr.save(p)
    back = CameraIntrinsics.load(p)
    check(abs(back.fx - intr.fx) < 1e-9 and abs(back.cx - intr.cx) < 1e-9,
          "save/load JSON giu nguyen gia tri fx/cx")

scaled = intr.scaled(1280, 960)
check(abs(scaled.fx - intr.fx * 2) < 1e-6, "scaled(2x) -> fx nhan 2")
check(abs(scaled.cx - intr.cx * 2) < 1e-6, "scaled(2x) -> cx nhan 2")
check(scaled.width == 1280 and scaled.height == 960,
      "scaled(2x) -> luu kich thuoc moi")

ray = intr.pixel_to_ray(320.0, 240.0)
check(abs(ray[0]) < 1e-9 and abs(ray[1]) < 1e-9 and abs(ray[2] - 1) < 1e-9,
      "pixel_to_ray(principal point) -> (0,0,1)")

xyz = intr.pixel_to_metric_xyz(640.0, 240.0, 2.0)
check(xyz[0] > 0 and abs(xyz[2] - 2.0) < 1e-9,
      "pixel_to_metric_xyz: Z=depth, X>0 khi pixel o ben phai")

xyz_left = intr.pixel_to_metric_xyz(0.0, 240.0, 2.0)
check(xyz_left[0] < 0, "pixel_to_metric_xyz: X<0 khi pixel o ben trai")

# --- Test 9: Import metric_depth + validate args ---
print("\n[Test 9] Import module metric_depth + validate args")
try:
    import metric_depth
    check(hasattr(metric_depth, "MetricDepthBackend"),
          "metric_depth.MetricDepthBackend import duoc")

    try:
        metric_depth.MetricDepthBackend(intrinsics=None)
        check(False, "phai raise khi thieu intrinsics")
    except ValueError:
        check(True, "MetricDepthBackend thieu intrinsics -> ValueError")

    try:
        metric_depth.MetricDepthBackend(intrinsics=intr, domain="banana")
        check(False, "phai raise khi domain sai")
    except ValueError:
        check(True, "MetricDepthBackend domain sai -> ValueError")

    try:
        # Metric DepthART chỉ hỗ trợ TinyViM S/B/L, không có MobileNetV4
        metric_depth.MetricDepthBackend(intrinsics=intr, encoder="MNV4-S")
        check(False, "phai raise khi encoder khong ho tro")
    except ValueError:
        check(True, "MetricDepthBackend encoder khong ho tro -> ValueError")
except ImportError as e:
    check(False, f"import metric_depth that bai: {e}")

# --- Test 10: Import calibrate_camera / pipeline ---
print("\n[Test 10] Import module calibrate_camera / pipeline")
try:
    import calibrate_camera
    check("9x6" in calibrate_camera.CHESSBOARD_PATTERNS,
          "calibrate_camera co bang CHESSBOARD_PATTERNS")
    # OpenCV dùng số GÓC TRONG, không phải số ô vuông
    cols, rows = calibrate_camera.CHESSBOARD_PATTERNS["9x6"]
    check(cols == 9 and rows == 6,
          "pattern '9x6' = 9x6 GOC TRONG (dung chuan OpenCV)")
except ImportError as e:
    check(False, f"import calibrate_camera that bai: {e}")

try:
    import pipeline
    check(hasattr(pipeline, "FirePipeline3D"), "pipeline.FirePipeline3D import duoc")
    kp = pipeline.parse_keypoints("112,150 224,160")
    check(kp == [(112, 150), (224, 160)],
          "parse_keypoints('x,y x,y') tra ve danh sach diem")
    check(pipeline.parse_keypoints(None) is None, "parse_keypoints(None) -> None")
except ImportError as e:
    check(False, f"import pipeline that bai: {e}")

# --- Summary ---
print("\n" + "=" * 60)
print(f"  PASSED {PASSED} assertions")
print("  [OK] localizer_3d + intrinsics + hop dong metric/relative")
print()
print("  Tiep theo:")
print("    - Do hieu nang:      python benchmark.py --num 20")
print("    - Kiem chung calib:  python verify_calibration.py")
print("    - Chay pipeline:     python pipeline.py --image <anh> --keypoints x,y")
print("    - Test voi camera:   python pipeline.py --source 0 --display")
print("=" * 60)