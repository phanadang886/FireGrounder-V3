# 🤖 FireGrounder-V3 — Agent Collaboration Log

**Dự án:** Hệ thống phát hiện và định vị lửa 3D  
**Ngày tạo:** 2026-10-07  
**Mục đích file:** Ghi lại kiến trúc, quyết định thiết kế, và những gì **không được thay đổi** để các AI agent sau đọc và hiểu맥 context mà không phụ thuộc vào conversation history.

---

## 📋 Kiến trúc tổng quan

```
┌─────────────────────┐
│  Camera / Video     │
│  Input Stream       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      (x, y, bbox)
│  FireBase-YOLO26n   │ ────────────────┐
│  2D Detection       │                 │
│  + BASE keypoint    │                 │
└─────────────────────┘                 │
           │                            │
           │ RGB frame                  │
           ▼                            ▼
┌─────────────────────┐      ┌──────────────────────┐
│  DepthBackend       │      │  FireLocalizer3D     │
│  depth_model.py     │      │  localizer_3d.py     │
│                     │──────▶                      │
│  1. DepthART MNV4-S │depth │  Input: (kx,ky) + depth
│  2. DAv2 fallback   │ map  │  Output: FireLocation3D
└─────────────────────┘      └──────────────────────┘
                                       │
                                       ▼
                             ┌──────────────────────┐
                             │  - depth_normalized  │
                             │  - xyz_normalized    │
                             │  - zone: NEAR/MID/FAR│
                             └──────────────────────┘
```

### Module chính

| Module | File chính | Chức năng | Trạng thái |
|--------|-----------|-----------|-----------|
| **FireBase-YOLO26** | `firebase_yolo26n_pose.yaml` | Phát hiện lửa + keypoint BASE (gốc lửa) | ✅ Đã train, **KHÔNG SỬA** |
| **Depth3D_Localizer** | `depth_model.py`, `localizer_3d.py` | Ước lượng depth + chuyển 2D→3D | 🔧 Đang sửa (xem changelog) |
| **Datasets** | `datasets/dataset_fire_pose/` | Dataset đã gán nhãn | ✅ Hoàn chỉnh, **KHÔNG SỬA** |

---

## 🚫 KHÔNG ĐƯỢC THAY ĐỔI

### 1. FireBase-YOLO26n weights
- **File:** `models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/best.pt`
- **Lý do:** Đã train xong, độ chính xác tốt với dataset hiện tại
- **Cấu trúc:** 1 class (fire), 1 keypoint BASE, backbone YOLO26n-pose
- **Output format:** `results.keypoints.xy[i][0]` = `(kx, ky)` pixel tọa độ BASE

### 2. Dataset structure
- **Path:** `datasets/dataset_fire_pose/`
- **Format:** YOLO pose format, mỗi ảnh có `.txt` tương ứng
- **Keypoint convention:** `[x_norm, y_norm, visibility]` cho BASE point
- **Lý do KHÔNG SỬA:** Đã gán nhãn thủ công, đồng bộ với YOLO weights

### 3. File cấu hình YOLO
- `firebase_yolo26n_pose.yaml`: kiến trúc model
- `fire_base_pose.yaml`: cấu hình dataset
- **Lý do:** Đã match với checkpoint trained

---

## 🔄 CHANGELOG — Depth3D_Localizer

### [2026-10-07] — Sửa backend tích hợp DepthART chính thức — AI Agent #2

**Đã hoàn thành:**
1. ✅ Viết lại `depth_model.py` gọi API chính thức `load_relative_model()` từ `relative/models.py`
2. ✅ Sửa `install_depthart.py` tải `depthart_relative_mnv4s_224.pth` thay vì TinyViM-S
3. ✅ Loại bỏ logic "mét giả" trong `xyz_metric` — giờ trả `None` với ghi chú rõ ràng
4. ✅ Sửa backend DAv2/DAv1 lấy `predicted_depth` tensor thay vì ảnh colorized
5. ✅ Thêm resize depth về kích thước ảnh gốc trong inference
6. ✅ Cập nhật `requirements.txt` thêm `einops>=0.8.0` và `timm>=0.6.12`
7. ✅ Sửa `quick_test.py` phản ánh đúng xyz_metric = None
8. ✅ Cập nhật notebook `depth3d_demo.ipynb` dùng `prefer='auto'`
9. ✅ Đổi `xyz_normalized` docstring: "relative position feature" thay vì "normalized coords"

**Kết quả test:**
- `quick_test.py`: ✅ Tất cả tests PASSED
- Backend fallback: ✅ Load được Depth Anything V2 khi DepthART chưa cài
- Console encoding: ✅ Bỏ emoji ✅ trong print để tránh lỗi Windows cp1252

**Trạng thái hiện tại:**
- Backend có thể chạy ngay với `prefer='auto'` — tự fallback DAv2
- Để dùng DepthART: chạy `python models/Depth3D_Localizer/install_depthart.py`
- xyz_metric đúng là None; cần metric depth model riêng nếu muốn tọa độ mét

**Lưu ý cho agent tiếp:**
- Không có checkpoint MobileNetV4 trong repo local → cần chạy installer hoặc dùng DAv2
- TinyViM-S đã có fallback CPU nhưng chậm hơn MNV4-S đáng kể
- Depth output đã resize về ảnh gốc, không cần xử lý thêm ở pipeline

---

### [2026-10-07] — Kiểm tra và phát hiện lỗi bởi AI Agent #2

**Phát hiện:**
1. ❌ `depth_model.py:125` gọi `from relative.model import build_model` — API không tồn tại trong repo DepthART chính thức
2. ❌ `install_depthart.py` tải `depthart_relative_s_224.pth` (TinyViM-S) thay vì `mnv4s` như README nói
3. ❌ `localizer_3d.py:242-251` — `xyz_metric` đang xuất "mét giả" từ relative depth chuẩn hóa
4. ⚠️  `xyz_normalized` chưa phải tọa độ 3D thực, chỉ là feature vị trí
5. ⚠️  Depth Anything V1/V2 backend lấy `result["depth"]` (ảnh 8-bit) thay vì `result["predicted_depth"]` (tensor float)

**Ngữ cảnh:**
- Máy target: Intel i5-1145G7, 16GB RAM, Intel Iris Xe (không có CUDA)
- PyTorch 2.14.0+cpu
- AI trước khẳng định TinyViM cần CUDA, nhưng repo hiện có `selective_scan_reference()` fallback

**Đã kiểm chứng:**
- Clone repo DepthART chính thức commit `b307c123` (2026-09-11)
- Tải checkpoint `depthart_relative_mnv4s_224.pth` và `depthart_relative_s_224.pth` từ HuggingFace
- Đối chiếu SHA256 với `CHECKSUMS.sha256`
- **Chạy thành công trên CPU:** MNV4-S ~148ms, TinyViM-S ~154ms (224×224 input, 1440×640 output)

**Quyết định sửa:**
- Viết lại `_load_depthart_variant()` gọi API thật: `load_relative_model()` từ `relative/models.py`
- Sửa `install_depthart.py` tải đúng MobileNetV4-S checkpoint
- **Loại bỏ `xyz_metric` tạm thời** — ghi chú rõ cần metric depth model + intrinsics thật
- Sửa DAv1/DAv2 backend lấy `predicted_depth` tensor thay vì ảnh colorized
- Thêm resize depth về kích thước ảnh gốc
- Cập nhật `requirements.txt` thêm `einops` và `timm`

---

## 📐 Thiết kế quan trọng

### Depth output convention
- **DepthART:** giá trị **cao** = điểm **gần** camera (affine-invariant relative depth)
- **Depth Anything V2:** tương tự, output là disparity-like
- **FireLocalizer3D:** `depth_normalized = 1.0` = gần nhất, `0.0` = xa nhất
- **Zone logic:**
  - `depth_normalized > 0.70` → NEAR
  - `0.30 < depth_normalized < 0.70` → MID
  - `depth_normalized < 0.30` → FAR

### Camera intrinsics placeholder
- Hiện dùng FOV 90° ước lượng: `fx = W / (2 * tan(45°))`
- **Chưa có intrinsics thật từ camera deployment**
- Metric depth cần:
  1. Model metric (DepthART indoor/outdoor 448)
  2. Intrinsics thật `(fx, fy, cx, cy)` tương ứng ảnh input
  3. Scale factor hoặc ground truth một điểm để căn chỉnh

---

## 🎯 TODO — Ưu tiên cao

### Ngay lập tức
- [x] Tạo file AGENTS.md này
- [x] Sửa `depth_model.py` tích hợp API DepthART chính thức
- [x] Sửa `install_depthart.py` tải đúng checkpoint MobileNetV4-S
- [x] Sửa `localizer_3d.py` loại bỏ "mét giả"
- [ ] Test `depth3d_demo.ipynb` với backend mới (có thể chạy với DAv2 fallback)

### Sau khi backend stable
- [ ] Đánh giá độ chính xác depth trên ảnh lửa thật
- [ ] Đo hiệu năng pipeline hoàn chỉnh (YOLO + Depth + hiển thị)
- [ ] Viết script video inference end-to-end
- [ ] Thử nghiệm với camera thật, lấy intrinsics

### Optional — nếu cần metric depth
- [ ] Tải DepthART metric indoor/outdoor 448 checkpoint
- [ ] Thêm calibration tool lấy camera intrinsics
- [ ] Triển khai metric depth pipeline riêng

---

## 🧪 Testing checklist

Trước khi commit thay đổi quan trọng, chạy:

```bash
# Unit test localizer (không cần model)
python models/Depth3D_Localizer/quick_test.py

# Load backend với fallback
python -c "import sys; sys.path.insert(0,'models/Depth3D_Localizer'); from depth_model import DepthBackend; b=DepthBackend(prefer='auto'); b.load(); print('OK:',b.model_name)"

# Nếu có ảnh test
# python models/Depth3D_Localizer/depth_model.py --image test.jpg --prefer auto

# YOLO inference (đảm bảo không ảnh hưởng)
# python -c "from ultralytics import YOLO; m=YOLO('models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/best.pt'); print(m.predict('test.jpg')[0].keypoints.shape)"
```

---

## 📦 Dependencies quan trọng

```
torch>=2.1.0          # Core
torchvision>=0.16.0
transformers>=4.40.0  # Depth Anything V1/V2 fallback
timm>=0.6.12          # TinyViM backbone registration
einops>=0.8.0         # DepthART tensor ops
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
huggingface_hub>=0.23.0
ultralytics>=8.3.0    # YOLO inference
```

---

## 🔍 Debug hints

### Lỗi "No module named 'relative.model'"
- **Nguyên nhân:** API cũ không tồn tại
- **Fix:** Dùng `load_relative_model()` từ `relative/models.py`

### DepthART không tải được trên CPU
- Kiểm tra `sys.path` có trỏ đến thư mục chứa `relative/`
- Kiểm tra `einops` và `timm` đã cài
- Thử `prefer='auto'` để fallback DAv2 nếu cần

### Depth map toàn NaN hoặc hằng số
- Kiểm tra ảnh input đã normalize đúng `(RGB - mean) / std`
- Kiểm tra model đang ở `.eval()` mode
- In `depth.min()` và `depth.max()` để debug range

### Zone luôn là NEAR hoặc FAR
- Điều chỉnh `near_thresh` và `far_thresh` trong `FireLocalizer3D()`
- Mặc định 0.70 / 0.30 phù hợp ảnh indoor–outdoor hỗn hợp

---

## 📞 Contact / Handoff

Nếu agent tiếp theo cần context nhanh:

1. Đọc phần **KHÔNG ĐƯỢC THAY ĐỔI** trước
2. Đọc **CHANGELOG** mới nhất
3. Xem **TODO** để biết công việc đang dở
4. Chạy **Testing checklist** để verify trạng thái hiện tại
5. Cập nhật file này với thay đổi của bạn trước khi commit

**Nguyên tắc:** Không xóa changelog cũ; thêm mục mới ở đầu mỗi phần.

---

**Last updated:** 2026-10-07 by AI Agent #2  
**Next action:** Sửa `depth_model.py` tích hợp DepthART API chính thức
