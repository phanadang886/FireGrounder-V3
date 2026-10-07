# -*- coding: utf-8 -*-
"""
quick_test.py - Quick test for Depth3D_Localizer (no model/GPU needed)
Run: python quick_test.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import sys
import numpy as np
from pathlib import Path

# Test localizer_3d.py không cần GPU/model
sys.path.insert(0, str(Path(__file__).parent))
from localizer_3d import FireLocalizer3D, CameraIntrinsics, FireLocation3D

print("=" * 55)
print("  Depth3D_Localizer — Quick Unit Test")
print("=" * 55)

# --- Test 1: Basic localization ---
print("\n[Test 1] Basic localization với depth_map giả")
H, W = 480, 640

# Giả lập depth map: vùng giữa gần, rìa xa
depth_map = np.zeros((H, W), dtype=np.float32)
for i in range(H):
    for j in range(W):
        dist_center = np.sqrt((i - H/2)**2 + (j - W/2)**2)
        depth_map[i, j] = 1.0 - dist_center / np.sqrt((H/2)**2 + (W/2)**2)

localizer = FireLocalizer3D(near_thresh=0.70, far_thresh=0.30)

# Điểm gần (giữa ảnh)
loc_near = localizer.localize(depth_map, kx=320, ky=240, img_w=W, img_h=H, model_name="mock")
print(f"\n  Điểm giữa (320, 240): zone={loc_near.zone}, depth_norm={loc_near.depth_normalized:.3f}")
assert loc_near.zone == "NEAR", f"Expected NEAR, got {loc_near.zone}"
print("  ✅ PASS: Điểm giữa → NEAR")

# Điểm xa (góc ảnh)
loc_far = localizer.localize(depth_map, kx=0, ky=0, img_w=W, img_h=H, model_name="mock")
print(f"\n  Điểm góc (0, 0): zone={loc_far.zone}, depth_norm={loc_far.depth_normalized:.3f}")
assert loc_far.zone == "FAR", f"Expected FAR, got {loc_far.zone}"
print("  ✅ PASS: Điểm góc → FAR")

# --- Test 2: CameraIntrinsics ---
print("\n[Test 2] CameraIntrinsics từ FOV")
intr = CameraIntrinsics.from_fov(width=640, height=480, fov_h_deg=90.0)
print(f"  fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.cx}, cy={intr.cy}")
assert abs(intr.fx - 320.0) < 1.0, "fx sai"
print("  ✅ PASS: fx ≈ 320 (đúng cho FOV=90°, W=640)")

# --- Test 3: Localize với intrinsics --- DISABLED vì xyz_metric chưa triển khai
print("\n[Test 3] xyz_metric — Chưa triển khai (cần metric depth model)")
loc_metric = localizer.localize(
    depth_map, kx=400, ky=300, img_w=W, img_h=H,
    intrinsics=intr, model_name="mock"
)
print(f"  xyz_metric = {loc_metric.xyz_metric}")
assert loc_metric.xyz_metric is None, "xyz_metric phải là None với relative depth"
print("  ✅ PASS: xyz_metric = None (chưa triển khai metric depth)")

# --- Test 4: Batch localize ---
print("\n[Test 4] Batch localize")
kpts = [(320, 240), (100, 100), (600, 400)]
locs = localizer.localize_batch(depth_map, kpts, img_w=W, img_h=H)
assert len(locs) == 3
print(f"  Batch 3 keypoints → {[l.zone for l in locs]}")
print("  ✅ PASS: batch localize OK")

# --- Test 5: to_dict & str ---
print("\n[Test 5] Output formats")
d = loc_near.to_dict()
assert "zone" in d and "xyz_normalized" in d
print(f"  JSON keys: {list(d.keys())}")
print(f"\n  Str output:\n{loc_near}")
print("\n  ✅ PASS: to_dict() và __str__() OK")

# --- Summary ---
print("\n" + "=" * 55)
print("  🎉 Tất cả tests PASSED!")
print("  Module localizer_3d.py hoạt động bình thường.")
print("  Tiếp theo: cài đặt transformers và chạy depth3d_demo.ipynb")
print("=" * 55)
