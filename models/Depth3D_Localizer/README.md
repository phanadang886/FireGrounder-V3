# 🔥 Depth3D_Localizer

Module xác định vị trí lửa trong không gian 3D, kết hợp với **FireBase-YOLO26**.

**Primary model**: DepthART (Depth Anything Rethought for Tiny Models) — ACM Multimedia 2026

## Kiến trúc Pipeline

```
┌─────────────────────┐     (kx, ky)     ┌──────────────────────────┐
│  FireBase-YOLO26    │ ─────────────────▶│   FireLocalizer3D        │
│  (2D Detection)     │                  │   localizer_3d.py        │
└─────────────────────┘                  └──────────┬───────────────┘
                                                    │
┌─────────────────────┐   depth_map (H,W)           │
│  DepthBackend       │ ───────────────────────────▶│
│  depth_model.py     │                             ▼
│                     │              ┌──────────────────────────────┐
│  1. DepthART tiny   │              │  FireLocation3D              │
│  2. DAv2 Small ✓    │              │  - depth_normalized          │
│  3. DAv1 Small      │              │  - xyz_normalized (X, Y, Z)  │
└─────────────────────┘              │  - zone: NEAR/MID/FAR        │
                                     └──────────────────────────────┘
```

## Cấu trúc files

```
Depth3D_Localizer/
├── depth_model.py        # Backend depth: DepthART → DAv2 → DAv1 fallback
├── localizer_3d.py       # Core 3D localization logic
├── depth3d_demo.ipynb    # Interactive Jupyter demo (click để xem tọa độ)
├── install_depthart.py   # Script cài DepthART từ GitHub (GPU/Linux)
└── requirements.txt      # Dependencies
```

## Cài đặt nhanh

### Bước 1: Cài packages
```bash
pip install torch torchvision opencv-python matplotlib pillow transformers
```

### Bước 2: Clone DepthART repo + tải checkpoint (bắt buộc)
```bash
python install_depthart.py
```
Script sẽ clone repo từ GitHub và tải checkpoint `depthart_relative_mnv4s_224.pth` từ HuggingFace.

### Trên Kaggle GPU (30h/tuần miễn phí)
```bash
# Trong Kaggle notebook
!python install_depthart.py
# Sau đó dùng depthart_encoder='S' để dùng TinyViM-S (nhanh hơn trên GPU)
```

## Sử dụng

### 1. Demo tương tác (Jupyter)
Mở `depth3d_demo.ipynb`, chạy từng cell. **Click vào ảnh** để xem tọa độ 3D tức thì.

### 2. Script Python thuần
```python
from depth_model import DepthBackend
from localizer_3d import FireLocalizer3D
from PIL import Image

# Load model — DepthART là PRIMARY
backend = DepthBackend(
    prefer="depthart",       # depthart | auto | dav2 | dav1
    depthart_encoder=None,  # None = tự chọn: MNV4-S (CPU) / S (GPU)
)
backend.load()

# Ước lượng depth
pil_img = Image.open("fire_image.jpg")
depth_map = backend.infer_pil(pil_img)

# Localize điểm BASE từ YOLO26
localizer = FireLocalizer3D()
loc = localizer.localize(depth_map, kx=320, ky=240,
                         img_w=640, img_h=480)
print(loc)
# → FireBase 3D Location @ pixel(320, 240)
#    Depth  : 0.6543 (raw)  →  normalized 0.721
#    Zone   : 🔴 NEAR  (sâu hơn 72.1% scene)
#    XYZ(n) : (+0.0000, -0.0000, 0.2790)
```

### 3. Pipeline đầy đủ (YOLO + Depth)
Xem **Cell 8** trong `depth3d_demo.ipynb`.

## Backends Depth

| Backend | prefer= | Encoder | Params | CPU | GPU | Ghi chú |
|---------|---------|---------|--------|-----|-----|---------|
| **DepthART MNV4-S** ✅ | `depthart` | `MNV4-S` | ~11M | ✅ | ✅ | **PRIMARY — CPU-friendly** |
| **DepthART MNV4-M** | `depthart` | `MNV4-M` | ~18M | ✅ | ✅ | CPU ok, chính xác hơn |
| **DepthART TinyViM-S** | `depthart` | `S` | 6M | ❌ | ✅ | GPU only, cần CUDA ext |
| Depth Anything V2 Small | `dav2` | — | 24M | ✅ | ✅ | Fallback, không cần setup |
| Depth Anything V1 Small | `dav1` | — | 25M | ✅ | ✅ | Last resort fallback |

> **DepthART TinyViM** (encoder `S/B/L`) yêu cầu compile Selective Scan CUDA extension (Linux/CUDA). Trên Windows CPU, dùng encoder `MNV4-S` hoặc `MNV4-M`.

## Output `FireLocation3D`

| Field | Ý nghĩa |
|-------|---------|
| `depth_value` | Giá trị raw từ model |
| `depth_normalized` | [0,1] — 1.0 = gần camera nhất |
| `depth_percentile` | % điểm ảnh xa hơn điểm này |
| `xyz_normalized` | (X, Y, Z) tỉ lệ không đơn vị |
| `xyz_metric` | (X, Y, Z) mét — chỉ với metric depth + intrinsics |
| `zone` | `NEAR` / `MID` / `FAR` |
| `relative_distance` | 0=gần nhất, 1=xa nhất |

## Cài đặt DepthART (tuỳ chọn, chỉ GPU Linux)

```bash
python install_depthart.py
```
Sau đó dùng `prefer="depthart"` khi khởi tạo `DepthBackend`.

> **Lưu ý**: DepthART yêu cầu biên dịch CUDA Selective Scan operator.
> Trên Windows/CPU, module tự động fallback về Depth Anything V2.
