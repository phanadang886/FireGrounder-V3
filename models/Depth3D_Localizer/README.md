# 🔥 Depth3D_Localizer

Module xác định vị trí lửa trong không gian 3D, kết hợp với **FireBase-YOLO26**.

**Primary model:** DepthART (Depth Anything Rethought for Tiny Models) — ACM Multimedia 2026

---

## Kiến trúc pipeline

```
┌─────────────────────┐     (kx, ky)     ┌──────────────────────────┐
│  FireBase-YOLO26    │ ─────────────────▶│   FireLocalizer3D        │
│  (2D Detection)     │                  │   localizer_3d.py        │
└─────────────────────┘                  └──────────┬───────────────┘
                                                    │
┌─────────────────────┐   depth_map (H,W)           │
│  DepthBackend       │ ───────────────────────────▶│
│  depth_model.py     │                             ▼
│  (RELATIVE depth)   │              ┌──────────────────────────────┐
└─────────────────────┘              │  FireLocation3D              │
                                     │  - depth_normalized          │
┌─────────────────────┐              │  - xyz_normalized (X, Y, Z)  │
│ MetricDepthBackend  │ ────────────▶│  - xyz_metric (X, Y, Z) mét  │
│  metric_depth.py    │  depth (mét) │  - depth_m                   │
│  (METRIC depth)     │              │  - zone: NEAR/MID/FAR        │
│  + intrinsics       │              └──────────────────────────────┘
└─────────────────────┘
```

---

## ⚠️ Hai loại depth — ĐỌC TRƯỚC KHI DÙNG

Đây là nguồn lỗi phổ biến nhất. Hai backend cho ra **quy ước NGƯỢC NHAU**:

| | Relative (`depth_model.py`) | Metric (`metric_depth.py`) |
|---|---|---|
| Output | affine-invariant | **đơn vị MÉT** |
| Giá trị cao | = **gần** | = **xa** |
| Cần intrinsics | không | **bắt buộc** |
| Cho `xyz_metric` | ❌ không | ✅ có |
| Checkpoint | `relative/*.pth` | `metric/*.pth` |
| Encoder | MNV4-S/M hoặc TinyViM | chỉ TinyViM S/B/L |
| Tốc độ CPU | ~60-90ms | ~2s (chậm) |

`FireLocalizer3D` **tự xử lý** việc đảo chiều khi bạn truyền `metric_depth=True`.
Nhưng nếu tự visualize depth map, bạn phải tự đảo:

```python
# Relative: colorize depth tăng dần → sáng = gần  ✓ (DepthBackend.colorize)
# Metric:   cần đảo trước                            ✓ (MetricDepthBackend.colorize)
```

---

## Cấu trúc files

```
Depth3D_Localizer/
├── depth_model.py          # DepthBackend — RELATIVE depth (DepthART → DAv2 → DAv1)
├── metric_depth.py         # MetricDepthBackend — METRIC depth (đơn vị mét)
├── localizer_3d.py         # Core 3D logic + CameraIntrinsics + FireLocation3D
├── calibrate_camera.py     # Tool lấy intrinsics THẬT (chessboard)
├── pipeline.py             # End-to-end: YOLO → depth → localize → annotate
├── benchmark.py            # Đo hiệu năng (ms/frame, FPS, p50/p95)
├── verify_calibration.py   # Kiểm chứng calibration bằng dữ liệu 3D tổng hợp
├── quick_test.py           # Unit test, không cần model (~37 assertions)
├── install_depthart.py     # Cài repo + tải checkpoint (có verify SHA256)
├── depth3d_demo.ipynb      # Demo interactive (click để xem tọa độ)
└── requirements.txt
```

---

## Cài đặt

### Bước 1: Packages
```bash
pip install -r requirements.txt
```

### Bước 2: DepthART + checkpoint
```bash
# Relative (mặc định, khuyến nghị) — MobileNetV4-S, chạy tốt trên CPU
python install_depthart.py

# Metric — thêm cho xyz_metric
python install_depthart.py --metric indoor --encoder s

# Nhiều checkpoint
python install_depthart.py --all
```

Script tự verify SHA256 theo `DepthART/CHECKSUMS.sha256`.

### Bước 3 (tuỳ chọn): FireBase-YOLO26 weights
`best.pt` bị `.gitignore` (`*.pt`) nên **không có trong repo**. Copy vào:
```
models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/best.pt
```

Nếu chưa có, chạy pipeline **không cần YOLO** (dùng keypoint cho sẵn):
```bash
python pipeline.py --image fire.jpg --keypoints 112,150
```

---

## Sử dụng

### 1. Nhanh nhất — demo interactive
Mở `depth3d_demo.ipynb`, chạy từng cell. **Click vào ảnh** để xem tọa độ 3D.

### 2. Chạy pipeline (khuyến nghị)
```bash
# Ảnh đơn + relative depth + hiện depth map
python pipeline.py --image fire.jpg --show-depth --out out/

# Video/webcam
python pipeline.py --source video.mp4 --out out/
python pipeline.py --source 0 --display          # webcam, xem trực tiếp

# Metric depth (tọa độ mét) — cần intrinsics
python pipeline.py --image fire.jpg --depth metric \
    --intrinsics intrinsics.json --domain indoor --out out/
```

### 3. Python API
```python
from depth_model import DepthBackend
from localizer_3d import FireLocalizer3D, CameraIntrinsics

# ── Relative: zone + xyz_normalized ──
backend = DepthBackend(prefer="auto")   # DepthART → DAv2 → DAv1
backend.load()
depth = backend.infer_pil(pil_img)      # (H, W) float32, CAO = GẦN

loc = FireLocalizer3D().localize(depth, kx=320, ky=240, img_w=640, img_h=480)
print(loc.zone)             # NEAR / MID / FAR
print(loc.xyz_normalized)   # (X, Y, Z) tương đối
print(loc.xyz_metric)       # None — relative depth không đổi sang mét được
```

```python
# ── Metric: xyz_metric (mét) ──
from metric_depth import MetricDepthBackend

intr = CameraIntrinsics.load("intrinsics.json")   # từ calibrate_camera.py
mb = MetricDepthBackend(intrinsics=intr, domain="indoor", encoder="s")
mb.load()

depth_m = mb.infer_pil(pil_img)      # (H, W) float32, ĐƠN VỊ MÉT

loc = FireLocalizer3D().localize(
    depth_m, kx=320, ky=240, img_w=640, img_h=480,
    intrinsics=intr, metric_depth=True,
)
print(loc.depth_m)        # 3.42  (mét)
print(loc.xyz_metric)     # (+0.55, -0.30, 3.42) — X phải, Y xuống, Z ra xa
```

---

## Camera intrinsics

`xyz_metric` chỉ chính xác khi có intrinsics **thật**.

```bash
# 1. In bàn cờ (pattern = SỐ GÓC TRONG theo chuẩn OpenCV)
#    pattern 9x6  ⟹  bàn cờ 10x7 ô vuông
# 2. Chụp 20-30 ảnh: nhiều góc nghiêng (20°-45°), nhiều khoảng cách
python calibrate_camera.py --images calib_frames/ \
    --pattern 9x6 --drop-outliers --output intrinsics.json
```

**Chưa có bàn cờ?** Dùng ước lượng (kém chính xác hơn nhiều, pipeline sẽ cảnh báo):
```bash
python calibrate_camera.py --from-fov 90 --img-size 640 480 --output intrinsics.json
```

### Kiểm chứng calibration
```bash
python verify_calibration.py
```
Render bàn cờ bằng mô hình pinhole 3D thật (có distortion), biết trước đáp án
đúng, rồi so kết quả. Tách 2 phép kiểm:
- **[A]** thuần thuật toán (corner lý tưởng) — phải khôi phục chính xác 0.000%
- **[B]** end-to-end (render → dò → calibrate) — số liệu tham khảo

> **Quan trọng:** RMSE nhỏ KHÔNG bảo đảm focal length đúng. Nếu ảnh chụp
> toàn mặt phẳng, RMSE có thể < 0.01px nhưng fx sai bậc hai order. Đó là lý
> dụng cần nhiều góc nghiêng. Tool sẽ cảnh báo khi FOV ước tính vô lý
> (ngoài 20°-140°).

---

## Đo hiệu năng

```bash
python benchmark.py --num 20                    # relative depth
python benchmark.py --depth metric --intrinsics intrinsics.json
python benchmark.py --synthetic                 # không cần dataset
```

Kết quả đo trên máy dev (i5-1145G7, 16GB, Iris Xe, **không CUDA**):

| Giai đoạn | Mean | p50 | p95 |
|---|---|---|---|
| detect | 0.04 ms | 0.00 | 0.08 |
| **depth** | **72.8 ms** | 56.7 | 118.1 |
| localize | 0.51 ms | 0.41 | 0.90 |
| annotate | 0.36 ms | 0.34 | 0.49 |
| **TOTAL** | **73.8 ms** | 57.9 | 118.8 |

→ **13.6 FPS** với relative depth (DepthART-MNV4-S).
**Depth chiếm 99% thời gian** — đây là nút thắt, không phải YOLO hay localizer.

Metric depth: **~2.3 s/frame** trên CPU (TinyViM/Mamba reference scan).
Chỉ dùng cho ảnh tĩnh hoặc khi thật sự cần tọa độ mét.

---

## Output `FireLocation3D`

| Field | Ý nghĩa |
|---|---|
| `depth_value` | Giá trị raw từ model |
| `depth_normalized` | [0,1] — 1.0 = gần nhất (đã chuẩn hoá cho cả 2 loại depth) |
| `depth_percentile` | % điểm ảnh gần hơn điểm này |
| `xyz_normalized` | (X, Y, Z) tương đối, không đơn vị |
| `xyz_metric` | (X, Y, Z) **mét** — chỉ khi metric depth + intrinsics |
| `depth_m` | Khoảng cách theo trục Z, **mét** |
| `zone` | `NEAR` / `MID` / `FAR` |
| `relative_distance` | 0 = gần nhất, 1 = xa nhất |

### Quy ước trục
- **X**: dương = sang phải
- **Y**: dương = **xuống dưới** (khớp hệ tọa độ ảnh)
- **Z**: dương = ra xa camera

### Zone thresholds
```
depth_normalized > 0.70  → NEAR
0.30 < depth_normalized < 0.70 → MID
depth_normalized < 0.30 → FAR
```
Tune bằng `FireLocalizer3D(near_thresh=..., far_thresh=...)`.

---

## Backends depth

| Backend | Module | Output | Encoder | CPU | Ghi chú |
|---|---|---|---|---|---|
| **DepthART MNV4-S** | `depth_model` | relative | MNV4-S | ✅ nhanh | **PRIMARY** |
| DepthART MNV4-M | `depth_model` | relative | MNV4-M | ✅ | chính xác hơn |
| DepthART TinyViM-S/B/L | `depth_model` | relative | S/B/L | ⚠️ chậm | có reference fallback |
| Depth Anything V2 Small | `depth_model` | relative | — | ✅ | fallback, không cần setup |
| Depth Anything V1 Small | `depth_model` | relative | — | ✅ | last resort |
| **DepthART metric indoor/outdoor** | `metric_depth` | **mét** | TinyViM S/B/L | ⚠️ rất chậm | cần intrinsics |

`prefer=` cho `DepthBackend`: `depthart` | `auto` | `dav2` | `dav1`

---

## Test

```bash
python quick_test.py          # 37 assertions, không cần model (~1s)
python verify_calibration.py  # kiểm chứng calibration (~30s)
python benchmark.py --num 20  # đo hiệu năng
```

---

## Troubleshooting

### `FileNotFoundError: FireBase-YOLO26 weights`
`best.pt` bị gitignore. Copy vào
`models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/best.pt`,
hoặc dùng `--keypoints x,y` để chạy không cần YOLO.

### `ModuleNotFoundError: ultralytics`
```bash
pip install ultralytics
```

### Zone luôn NEAR hoặc FAR
Tune `near_thresh` / `far_thresh`, hoặc kiểm tra depth map có phải hằng số
không (`print(depth.min(), depth.max())`). Ảnh đồng nhất cho depth map hằng.

### Intrinsics sai làm metric depth sai
Metric model dùng intrinsics làm **input**, không chỉ để back-projection.
Sai intrinsics → sai số mét ngay. Luôn chạy `verify_calibration.py` trước,
và dùng `--drop-outliers` khi calibrate ảnh thật.

### `UnicodeEncodeError` trên Windows
Đã xử lý trong tất cả script (bọc stdout UTF-8). Nếu gặp ở script cũ,
chạy `python -X utf8 <script>`.

### Không tìm thấy bàn cờ trong ảnh calibration
Sai số góc trong là nguyên nhân số 1. `--pattern` nhận **số góc trong**
(9x6 = bàn cờ 10x7 ô vuông). Dùng `--preview <dir>` để xem ảnh nào dò được.