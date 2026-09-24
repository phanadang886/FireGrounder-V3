# 🔥 FIRE GROUNDING PROJECT — HANDOFF DOCUMENT
> **Mục đích file này:** Cung cấp toàn bộ context kỹ thuật để một AI model khác (DeepSeek, Claude, GPT, Gemini...) có thể tiếp tục dự án mà KHÔNG mất thông tin. Đọc file này TRƯỚC khi làm bất kỳ điều gì.

**Cập nhật lần cuối:** 2026-09-21  
**Workspace:** `d:\Tài liệu lab HMI\Fire_Segmentation`  
**Nền tảng train:** Kaggle (GPU T4/P100, free tier)  
**Framework:** PyTorch + timm  

---

## 1. BÀI TOÁN ĐANG GIẢI QUYẾT

### Tên bài toán: Real-Time Fire Base Grounding (Định vị gốc lửa 2D)

**Không phải** Fire Detection (phát hiện lửa) hay Fire Segmentation (phân đoạn mask).

**Cụ thể:**
- **Input:** Ảnh RGB từ camera giám sát (CCTV/Webcam)
- **Output:** 3 giá trị `[p_fire, x_norm, y_norm]`
  - `p_fire ∈ [0,1]`: xác suất có lửa
  - `(x_norm, y_norm) ∈ [0,1]²`: tọa độ chuẩn hóa của **điểm chân đáy ngọn lửa tiếp xúc mặt sàn** (fire base contact point)
- **Ứng dụng:** Robot chữa cháy tự hành cần biết chính xác GỐC lửa (không phải đỉnh ngọn lửa) để hướng vòi phun

### Yêu cầu kỹ thuật cứng:
| Tiêu chí | Ngưỡng chấp nhận |
|---|---|
| Latency | < 20ms / frame (> 50 FPS) |
| Model size | < 50MB (triển khai Edge: Jetson Orin) |
| Precision | > 95% (không báo giả) |
| Recall | > 95% (không bỏ sót) |
| MAE (pixel) | < 10px trên ảnh 256×256 |

---

## 2. LỊCH SỬ 4 NOTEBOOK — ĐÃ LÀM GÌ, THÀNH/BẠI, VÀ TẠI SAO

### Notebook 1: SAM 3 Auto-Labeling (Data Engine)
- **File:** `label_sam3.py` (code gốc), chạy trên Kaggle
- **Mục đích:** Dùng SAM 3 (Segment Anything Model 3 của Meta) + thuật toán `adaptive_baseline()` để tự động tạo nhãn `(x_norm, y_norm)` cho 6.500 ảnh
- **Kết quả:** Tạo thành công `dataset_labels (1).json` — tập nhãn pseudo-ground-truth
- **Phân bố:** 4.509 ảnh có lửa (`has_fire: 1`) + 1.991 ảnh không lửa (`has_fire: 0`)
- **Nhận xét:** SAM 3 chính xác nhưng CỰC CHẬM (2-4s/ảnh), chỉ dùng offline làm data engine

### Notebook 2: MobileNetV4 V1 — Single Head (THẤT BẠI)
- **File tham khảo:** `kaggle_cells.py` (code gốc trên Kaggle)
- **Kiến trúc:** `MobileNetV4-Medium → GAP → nn.Linear(1280, 3) → Sigmoid`
- **Loss:** `BCE + 2.0 × SmoothL1`
- **Train:** 30 epoch, LR=1e-4, batch=32, img_size=224
- **Kết quả:** Acc ~99%, MAE ~9.5px trên val (NHƯNG...)
- **❌ THẤT BẠI VÌ:**
  1. **Single Head = Gradient Conflict:** Classification gradient và Regression gradient đánh nhau qua 1 lớp Linear duy nhất
  2. **Augmentation bị lỗi logic:** `RandomHorizontalFlip` lật ảnh nhưng KHÔNG đảo `x_norm = 1.0 - x_norm` → model học sai vị trí
  3. **Không có Early Stopping:** Train 30 epoch cứng → overfitting nặng
  4. **BCE không xử lý được class imbalance:** Ảnh không lửa "dễ" quá, model lười học → báo bừa (FP cực cao)
- **Hệ quả thực tế:** Ảnh người đi lại, quầy thu ngân, hành lang → đều bị báo có lửa + chấm tọa độ giả

### Notebook 3: iFormer-M — Inception Transformer (THẤT BẠI)
- **Kiến trúc:** `iformer_m → GAP → nn.Linear(n, 3) → Sigmoid`
- **Ý tưởng:** Dùng Multi-Head Self-Attention thay CNN để phân biệt lửa vs vật thể màu nóng tốt hơn
- **❌ THẤT BẠI VÌ:**
  1. **Patch tokenization mất spatial resolution:** Transformer chia ảnh thành patch 16×16 rồi pool → mất chi tiết nhỏ
  2. **Recall sụt thê thảm (~84%):** Bỏ sót hoàn toàn lửa nhỏ ở cự ly xa
  3. **Chậm gấp 3 lần MobileNet:** ~28ms/frame, không đáp ứng yêu cầu real-time
  4. **Vẫn giữ Single Head + BCE** → chưa fix root cause từ Notebook 2
- **Checkpoint:** `best_iformer_m.pth` (35MB) — có thể dùng để so sánh benchmark

### Notebook 4: FireGrounder V2 — Dual Head + Focal Loss (THÀNH CÔNG NHẤT)
- **File lịch sử:** `mobilenet-v4-l-n-2.ipynb` (notebook Kaggle V2 đầy đủ). Các file `notebook4_v2_mnv4.py` và `notebook4_cell1.py` → `notebook4_cell7.py` đã được xóa khỏi workspace.
- **Kiến trúc V2 (trong `model.py`):**
  ```
  MobileNetV4-Medium
       ↓
      GAP → vector 1280D
       ↓                    ↓
  cls_head                reg_head
  Linear(1280,256)       Linear(1280,256)
  BatchNorm1d(256)       BatchNorm1d(256)
  GELU                   GELU
  Dropout(0.4)           Dropout(0.2)
  Linear(256,1)→Sigmoid  Linear(256,2)→Sigmoid
       ↓                    ↓
    p_fire              (x_norm, y_norm)
  ```
- **Loss V2:** `1.5 × FocalLoss(α=0.5, γ=2.0) + 3.0 × SmoothL1 (chỉ trên ảnh có lửa)`
- **Augmentation chuẩn:** Coord-safe HFlip (`x_new = 1.0 - x`), RandomErasing, ColorJitter mạnh, Gamma transform
- **Config:** img_size=256, LR=1.5e-4, batch=48, WD=3e-4, patience=7
- **Kết quả BEST (Epoch 11, early stopped):**

| Metric | Giá trị |
|---|---|
| F1-Score | **98.4%** |
| Precision | 97.6% |
| Recall | 99.3% |
| MAE (pixel) | ~13.5px |
| False Positives | 16/1000 ảnh |
| False Negatives | 5/975 ảnh |
| Latency | 10.5 - 15.8ms (GPU T4) |
| Params | 3.8M |
| Checkpoint size | ~35MB |

- **Checkpoint:** `best_mobilenetv4.pth` (34MB) — **đây là model tốt nhất hiện tại**

---

## 3. HẠN CHẾ HIỆN TẠI — VẤN ĐỀ CẦN GIẢI QUYẾT

### 3.1. Giới hạn kiến trúc: GAP Ceiling (QUAN TRỌNG NHẤT)

Đây là **root cause** khiến MAE không thể xuống dưới ~10px:

```
Feature Map (8×8×1280)  →  GAP  →  Vector 1D (1280)  →  Linear  →  (x, y)
                              ↑
                    MẤT TOÀN BỘ THÔNG TIN VỊ TRÍ KHÔNG GIAN
```

**Global Average Pooling nén phẳng spatial information.** Model phải "đoán" tọa độ từ semantic features thay vì biết chính xác vị trí. Đây là giới hạn vật lý của kiến trúc Direct Coordinate Regression, KHÔNG THỂ fix bằng cách tune hyperparameter.

### 3.2. False Positive còn sót: `fire7.png`
- Ảnh cổng trường với thùng rác có bóng đổ mạnh → model nhận nhầm là lửa (confidence 70.2%)
- **Nguyên nhân:** Dataset thiếu "hard negatives" (thùng rác, đèn pha, hoàng hôn, áo cam)

### 3.3. Tọa độ lệch ở góc chụp nghiêng
- Một số ảnh chụp xiên hoặc lửa ở rìa khung hình → tọa độ dự đoán lệch 20-30px
- Liên quan trực tiếp đến GAP Ceiling ở mục 3.1

---

## 4. CẤU TRÚC CODEBASE HIỆN TẠI

```
Fire_Segmentation/
├── model.py                    # ⭐ QUAN TRỌNG NHẤT — chứa FireGrounder (V1), FireGrounderV2 (V2),
│                               #    FocalLoss, FireLoss, FireLossV2, load_checkpoint()
│                               #    + transforms (train_transform, val_transform, val_transform_256)
├── train.py                    # Training loop V1 (dùng FireGrounder + FireLoss)
│                               #    ⚠️ CHƯA cập nhật cho V2, chỉ train được V1
├── inference.py                # Inference pipeline: image/video/webcam
│                               #    Tự nhận diện V1 vs V2 qua isinstance()
│                               #    Vẽ crosshair + circle tại fire base point
├── label_sam3.py               # Script tạo nhãn tự động bằng SAM 3
│                               #    Hàm adaptive_baseline() tìm chân đáy lửa từ mask
├── firegrounder_v3.py          # ⭐ Model V3: FPN-lite + heatmap 64x64
├── notebook4_v3_fpn.py         # ⭐ Pipeline V3: smoke test, train, eval, preview
├── inference_v3.py             # Inference V3: image/video/webcam
├── kaggle_cells.py             # Code gốc Notebook 2+3 (tham khảo, KHÔNG dùng nữa)
│
├── dataset_labels (1).json     # Dataset legacy khác (1.034 entries), không dùng cho V3 chính
│                               #    Format: {"image_path": "...", "has_fire": 0|1, "p_fire": [x, y]}
├── fire_ground_dataset/         # Nhãn HomeFire dùng trực tiếp cho V3
│   ├── dataset_labels.json      # 6.500 entries: 4.509 fire, 1.991 no-fire
│   └── stream_cache.jsonl       # Cache resume SAM3; V3 không đọc
├── best_mobilenetv4.pth        # ⭐ Checkpoint V2 tốt nhất (34MB) — DÙNG CÁI NÀY
├── best_iformer_m.pth          # Checkpoint iFormer-M (để benchmark so sánh)
├── best.pth                    # Checkpoint V1 (để benchmark so sánh)
├── last.pth                    # Checkpoint cuối cùng của lần train gần nhất
├── train_history.json          # Lịch sử 30 epoch V1: {epoch, train, val, acc, mae}
│
└── requirements.txt            # torch>=2.2, timm>=1.0.3, opencv-python, Pillow, numpy
```

### Cách load model hiện tại:
```python
from model import load_checkpoint

# Tự nhận diện V1 hay V2 từ state_dict
model = load_checkpoint("best_mobilenetv4.pth", device="cuda")

# Inference
from inference import predict
from PIL import Image
p_fire, x_norm, y_norm = predict(model, Image.open("fire.jpg"), "cuda")
```

### Format dataset `dataset_labels (1).json`:
```json
[
  {"image_path": "/kaggle/input/.../fire001.jpg", "has_fire": 1, "p_fire": [0.432, 0.891]},
  {"image_path": "/kaggle/input/.../room042.jpg", "has_fire": 0, "p_fire": [0.0, 0.0]},
  ...
]
```
- `image_path`: đường dẫn tuyệt đối (trên Kaggle), cần resolve lại nếu chạy local
- `has_fire`: 0 = không lửa, 1 = có lửa
- `p_fire`: `[x_norm, y_norm]` tọa độ gốc lửa chuẩn hóa [0,1]. Với ảnh không lửa = `[0, 0]`

---

## 5. HƯỚNG ĐI TIẾP THEO — ROADMAP ƯU TIÊN

### ⭐ Ưu tiên 1: Heatmap Regression Head (thay thế GAP + Linear)

**Đây là bước ĐỘT PHÁ cần làm ngay.** Mục tiêu: phá vỡ GAP Ceiling, đưa MAE xuống < 5px.

**Ý tưởng:**
```
MobileNetV4-Medium backbone
      ↓
Feature Map (8×8×1280)   ← GIỮ NGUYÊN, KHÔNG qua GAP
      ↓
ConvTranspose2d / PixelShuffle  ← Upsample lên 64×64
      ↓
Conv2d(channels, 1) → Sigmoid
      ↓
Heatmap 64×64   ← Gaussian blob tại vị trí gốc lửa
      ↓
argmax hoặc soft-argmax  ← Trích tọa độ (x, y) từ heatmap
```

**Ground Truth:** Tạo Gaussian heatmap từ tọa độ `(x_norm, y_norm)` đã có:
```python
# Pseudo-code tạo GT heatmap
sigma = 2.0  # pixels trên heatmap 64×64
cx, cy = int(x_norm * 64), int(y_norm * 64)
heatmap[cy, cx] = 1.0
heatmap = gaussian_blur(heatmap, sigma)
```

**Loss đề xuất:** MSE trên heatmap + Classification head riêng (giữ Focal Loss)

**Tham khảo paper:** SimpleBaseline (Xiao et al. 2018), HRNet (Sun et al. 2019) — cả 2 đều dùng heatmap cho human pose, hoàn toàn áp dụng được cho fire base keypoint.

### Ưu tiên 2: YOLOv8n-Pose Pipeline (alternative approach)

Nếu Heatmap Head gặp khó khăn, chuyển sang YOLO:
- Dùng **YOLOv8n-pose** (Ultralytics): detect bounding box ngọn lửa + 1 keypoint (fire base)
- BBox giúp loại trừ false positives (vì cần đúng class "fire" mới có keypoint)
- Nhẹ (~3.2M params), đã có framework sẵn
- **Hạn chế:** Cần convert dataset sang YOLO-pose format (BBox + Keypoint), phải tạo BBox từ mask hoặc thủ công

### Ưu tiên 3: Bổ sung Hard Negatives
- Thu thập thêm 300-500 ảnh: thùng rác ngoài trời, đèn pha, hoàng hôn, áo bảo hộ cam, lò nướng
- Đánh nhãn `has_fire: 0` và thêm vào `dataset_labels.json`
- Retrain sẽ giúp triệt tiêu các ca báo giả như `fire7.png`

---

## 6. BẢNG TỔNG HỢP METRICS CẦN NHỚ

| Model | Notebook | F1 | Precision | Recall | MAE (px) | Latency | Params | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| SAM 3 (GT engine) | NB1 | 98.7% | 98.0% | 99.5% | ~3px | 2500ms | 350M+ | Chỉ dùng offline |
| MNV4 V1 Single-Head | NB2 | 87.1% | 81.2% | 94.0% | ~35.8px | 9.5ms | 3.2M | ❌ FP quá cao |
| iFormer-M | NB3 | 88.5% | 93.5% | 84.1% | ~29.4px | 28.4ms | 19.8M | ❌ Recall quá thấp |
| **MNV4 V2 Dual-Head** | **NB4** | **98.4%** | **97.6%** | **99.3%** | **~13.5px** | **12ms** | **3.8M** | ✅ **Current BEST** |
| Heatmap Head (chưa làm) | NB5? | ? | ? | ? | Mục tiêu <5px | ~15ms | ~4M | 🔜 Cần implement |

---

## 7. LƯU Ý KỸ THUẬT QUAN TRỌNG

### ⚠️ Những sai lầm đã mắc (TRÁNH LẶP LẠI):
1. **KHÔNG bao giờ dùng `RandomHorizontalFlip` mà không đảo tọa độ x.** Nếu flip ảnh, phải `x_new = 1.0 - x_old`. Đây là bug #1 gây ra thất bại Notebook 2.
2. **KHÔNG dùng Single Head** chung cho classification + regression. Gradient conflict sẽ khiến cả 2 task đều kém. Luôn tách riêng.
3. **KHÔNG dùng BCE cho class imbalance.** Focal Loss (γ=2.0) hiệu quả hơn nhiều lần.
4. **KHÔNG train quá 15-20 epoch** với dataset 6.5k ảnh — sẽ overfit. Dùng Early Stopping patience 5-7.
5. **KHÔNG dùng iFormer/ViT nếu cần detect vật thể nhỏ.** Patch tokenization mất spatial resolution.
6. **KHÔNG tin MAE trên tập train.** Chỉ đánh giá trên val set. Và val MAE cũng có thể misleading nếu không test trên ảnh mù (unseen).

### ✅ Những quyết định đã chứng minh đúng:
1. **MobileNetV4-Medium** là backbone tối ưu: nhẹ (3.2M), nhanh (9-12ms), giữ spatial features tốt hơn ViT
2. **Focal Loss** triệt tiêu false positives hiệu quả
3. **Dual Head** giải quyết gradient conflict hoàn toàn
4. **Resolution 256×256** tốt hơn 224×224 cho bài toán này
5. **SAM 3 chỉ nên dùng offline** tạo data, không nhúng real-time
6. **Weight ratio `w_cls=1.5, w_reg=3.0`** cân bằng tốt giữa classification accuracy và localization precision

### 📋 Hyperparameters đã tối ưu cho V2:
```python
BACKBONE     = "mobilenetv4_conv_medium"
IMG_SIZE     = 256
EPOCHS       = 50       # nhưng Early Stop ở epoch ~11
LR           = 1.5e-4
BATCH        = 48
PATIENCE     = 7
W_CLS        = 1.5      # weight classification loss
W_REG        = 3.0      # weight regression loss
WD           = 3e-4     # weight decay
FOCAL_ALPHA  = 0.5
FOCAL_GAMMA  = 2.0
MEAN         = [0.485, 0.456, 0.406]   # ImageNet
STD          = [0.229, 0.224, 0.225]
```

---

## 8. CÁCH CHẠY NHANH

### Train V3 trên Kaggle:
1. Clone repository hoặc upload `firegrounder_v3.py` và `notebook4_v3_fpn.py`.
2. Attach HomeFire images và file `fire_ground_dataset/dataset_labels.json`.
3. Đặt `FIRE_RUN_TRAIN=1`, `FIRE_JSON_PATH` và `FIRE_IMAGE_ROOT`.
4. Bật GPU T4/P100 và chạy `python notebook4_v3_fpn.py`.
5. Checkpoint lưu tại `/kaggle/working/best_v3_fpn.pth`.

Chi tiết đầy đủ cho Kaggle, local và GitHub nằm trong `README.md`.

### Inference local:
```bash
python inference.py --mode image --input fire.jpg --checkpoint best_mobilenetv4.pth
python inference.py --mode webcam --checkpoint best_mobilenetv4.pth
python inference.py --mode video --input fire.mp4 --checkpoint best_mobilenetv4.pth
```

### Test model nhanh:
```python
python model.py  # Smoke test cả V1 và V2
```

---

## 9. TÓM TẮT CHO AI MODEL TIẾP NHẬN

**Bạn đang ở đâu:** Đã có model V2 chạy tốt (F1 98.4%, 12ms). Nhưng MAE còn ~13.5px do giới hạn kiến trúc GAP. V3 đã được viết nhưng chưa train trong workspace này.

**Bạn cần làm gì:** Train và đánh giá **Heatmap Regression Head** V3 trong `notebook4_v3_fpn.py`, sau đó so sánh cùng split với V2. Mục tiêu MAE < 5px là mục tiêu thực nghiệm, không phải kết quả đã đạt.

**Bạn KHÔNG cần làm lại:** Data labeling (đã xong), backbone selection (MNV4 đã chứng minh tối ưu), classification head (Focal Loss đã hoạt động tốt).

**File quan trọng nhất cần đọc:** `firegrounder_v3.py`, `notebook4_v3_fpn.py`, `inference_v3.py`, `README.md` và file này (`HANDOFF.md`).

---

## 10. GIẢI THÍCH STRIDE CHO NGƯỜI MỚI

### 10.1. Stride là gì?

Trong CNN, **stride là bước nhảy của cửa sổ tích chập** khi nó quét qua ảnh. Nếu stride bằng 2, cửa sổ nhảy 2 pixel mỗi lần thay vì 1 pixel. Kết quả thường có kích thước không gian giảm còn một nửa.

Có thể hình dung ảnh 256x256 là bản đồ:

```text
Ảnh gốc              256x256
stride-2             128x128  - vẫn còn nhiều chi tiết
stride-4              64x64   - chi tiết và ngữ cảnh cân bằng
stride-8              32x32   - ngữ cảnh tốt, vị trí thô hơn
stride-16             16x16   - vị trí khá thô
stride-32              8x8    - biết có gì, khó biết chính xác ở đâu
```

**Stride không phải nhiệt độ.** Heatmap trong kế hoạch là bản đồ xác suất vị trí, không cần camera hồng ngoại hay module đo nhiệt.

### 10.2. Tại sao stride quan trọng với bài toán này?

V2 lấy feature map cuối 8x8, sau đó dùng Global Average Pooling (GAP) để biến thành vector 1 chiều. GAP làm mất thông tin vị trí. Model chỉ còn biết ảnh có đặc trưng của lửa, rồi phải đoán tọa độ bằng Linear layer.

Với heatmap, cần giữ feature map 2D. MobileNetV4-Medium đã kiểm tra thực tế với input 256x256 và cho các feature map sau:

```text
out_indices=0: 128x128x32  (stride-2)
out_indices=1:  64x64x48   (stride-4)
out_indices=2:  32x32x80   (stride-8)
out_indices=3:  16x16x160  (stride-16)
out_indices=4:   8x8x960   (stride-32)
```

Stride nhỏ giữ chi tiết tốt hơn nhưng feature nông hơn; stride lớn có ngữ nghĩa mạnh hơn nhưng vị trí thô hơn. Vì vậy phương án khuyến nghị là **fusion stride-4 và stride-8 kiểu FPN-lite**, thay vì chỉ chọn một tầng.

### 10.3. Không được hiểu sai việc upsample

Upsample từ 32x32 lên 64x64 chỉ tạo thêm điểm nội suy, không khôi phục thông tin đã mất. Feature stride-8 có ngữ nghĩa tốt nhưng nên kết hợp với feature stride-4 để lấy lại chi tiết. Heatmap 64x64 cũng không tự động đảm bảo MAE dưới 5px; kết quả phụ thuộc feature, chất lượng nhãn, loss và kiểm thử ảnh chưa thấy.

---

## 11. TỔNG HỢP ĐÁNH GIÁ VÀ QUYẾT ĐỊNH KIẾN TRÚC

### 11.1. Những điểm đã được xác nhận

- V2 trong notebook Kaggle `mobilenet-v4-l-n-2.ipynb` đạt best epoch 11, val loss 0.01995, Precision 97.6%, Recall 99.3%, F1 98.4%, MAE 13.5px.
- Kaggle log ghi **9,092,179 parameters**, vì vậy con số 3.8M trong báo cáo cũ phải được sửa hoặc ghi rõ là của một phiên bản khác.
- V2 chạy khoảng 10.5-15.8ms trên GPU Kaggle; kết quả local/edge phải benchmark lại, không được suy ra trực tiếp từ V100/T4.
- `train_history.json` local có format V1 (`acc`, `mae`), không phải history V2. Không dùng hai file này để so sánh trực tiếp nếu chưa chuẩn hóa cách tính metric.
- Tập `fire-samples` chỉ có 10 ảnh, đủ để smoke test nhưng không đủ để kết luận thống kê về tỷ lệ false positive.

### 11.2. So sánh các quan điểm

```text
Direct regression + GAP:
  dễ triển khai, nhanh, nhưng mất thông tin không gian và đã có dấu hiệu plateau.

Chỉ stride-8:
  nhẹ, có ngữ nghĩa, nhưng upsample không tạo lại chi tiết.

Chỉ stride-4:
  giữ chi tiết, nhưng feature nông dễ phản ứng với vật sáng hoặc màu nóng.

FPN-lite stride-8 + stride-4:
  kết hợp ngữ nghĩa và chi tiết; là lựa chọn cân bằng được khuyến nghị.
```

Kết luận: **giữ MobileNetV4**, không cần tìm backbone mới trước. Thay đổi chính là neck/head để giữ không gian. Heatmap có khả năng cải thiện localization, nhưng mục tiêu MAE <5px là mục tiêu thực nghiệm, không phải lời hứa. Hard negatives vẫn cần làm riêng để xử lý false positive như `fire7.png`.

YOLOv8-Pose chưa nên triển khai ở giai đoạn này vì dataset hiện chỉ có point và class, chưa có bbox. Pseudo-bbox quanh point không phải bbox lửa thật và có thể tạo nhãn lỗi. Chỉ chuyển sang YOLO-Pose sau khi có bbox từ SAM mask hoặc gán nhãn thủ công.

### 11.3. Các tiêu chí chấp nhận V3

Không chấp nhận V3 chỉ vì một con số MAE đẹp trên random split. V3 phải được so sánh cùng seed và cùng val set với V2, đồng thời báo cáo:

1. F1, Precision, Recall, FP, FN ở nhiều threshold.
2. MAE, median error, P90 error trên ảnh có lửa.
3. Tỷ lệ lỗi <=5px, <=10px và <=15px.
4. Kết quả riêng cho fire nhỏ, fire gần rìa ảnh, ảnh góc nghiêng và hard negatives.
5. Latency batch=1 trên GPU và thiết bị mục tiêu.
6. Kết quả trên một test set giữ kín, có người kiểm tra lại nhãn.

---

## 12. PLAN THỰC THI FIREGROUNDER V3

### Bước 0 - Đóng băng baseline

1. Không sửa checkpoint V2.
2. Lưu commit/bản sao của `model.py`, `inference.py` và notebook hiện tại.
3. Chạy baseline V2 trên đúng val split seed=42 và lưu JSON metrics.
4. Benchmark batch=1, warm-up ít nhất 20 frame, đo trung bình và P95 latency.

### Bước 1 - Kiểm tra dữ liệu và nhãn

1. Kiểm tra toàn bộ `image_path`, ảnh lỗi, tọa độ ngoài [0,1].
2. Chia train/val/test theo nguồn hoặc scene nếu có thể, tránh ảnh gần giống nhau ở hai split.
3. Vẽ ít nhất 100 nhãn SAM lên ảnh gốc và kiểm tra thủ công.
4. Ghi riêng các nhãn không chắc chắn. Nếu nhiễu nhãn đã gần 5px thì không đặt mục tiêu V3 dưới 5px một cách cứng nhắc.
5. Giữ biến đổi hình học nhất quán. Nếu dùng resize vuông, phải ghi rõ metric đang tính ở không gian 256x256; tốt hơn là letterbox và chuyển tọa độ đúng khi vẽ ảnh gốc.

### Bước 2 - Tạo model V3 theo FPN-lite

Dùng `features_only=True`, lấy `out_indices=(1, 2, 4)`:

- `f4`: 64x64x48, giữ chi tiết.
- `f8`: 32x32x80, ngữ nghĩa trung gian.
- `f32`: 8x8x960, dùng cho classification head.

Không cần dùng `f32` để dự đoán tọa độ. Tạo lateral 1x1 đưa f4 và f8 về cùng số kênh, upsample f8 lên 64x64, cộng với f4, rồi tạo heatmap một kênh.

### Bước 3 - Sinh target heatmap

Với ảnh có lửa, chuyển `(x_norm,y_norm)` sang tọa độ trên heatmap 64x64 và tạo Gaussian 2D. Bắt đầu thử sigma=2.0; chạy ablation sigma=1, 2, 3. Ảnh không lửa có heatmap toàn số 0 và chỉ tối ưu classification.

### Bước 4 - Loss và tọa độ đầu ra

- Classification: giữ Focal Loss V2.
- Heatmap: thử MSE hoặc BCE-with-logits; không dùng sigmoid trước khi đưa vào `BCEWithLogitsLoss`.
- Tọa độ: dùng soft-argmax trên heatmap logits với temperature, hoặc lấy argmax để đánh giá riêng. Có thể thêm SmoothL1 giữa soft-argmax và GT với trọng số nhỏ sau khi heatmap đã học ổn định.
- Với ảnh không lửa, không tính coordinate loss.

### Bước 5 - Huấn luyện ablation

Chạy tối thiểu các thí nghiệm:

1. V2 baseline.
2. V3 chỉ stride-8.
3. V3 chỉ stride-4.
4. V3 FPN-lite stride-4 + stride-8.
5. FPN-lite + hard negatives nếu dữ liệu đã sẵn sàng.

Giữ cùng seed, split, optimizer và số epoch hợp lý. Dùng early stopping theo validation metric, không chỉ theo tổng loss. Lưu checkpoint gồm config, git/file hash, metric và history.

### Bước 6 - Kiểm tra lỗi và trực quan hóa

Tạo ảnh overlay gồm ảnh gốc, GT point, predicted point, confidence và heatmap. Xem riêng các nhóm lỗi:

- lửa rất nhỏ;
- lửa ở rìa;
- lửa bị khói che;
- vật sáng nhưng không cháy;
- thùng rác/đèn/áo cam;
- ảnh góc nghiêng.

Nếu heatmap có peak ở vật sáng không cháy, đó là vấn đề classification/data, không chỉ là vấn đề tọa độ.

### Bước 7 - Cập nhật inference

`load_checkpoint()` phải nhận diện V3 qua key state dict hoặc metadata `model_version`. `predict()` trả về cùng API `(p_fire,x,y)` để không phá code cũ. Có thể thêm tùy chọn trả heatmap để debug. Chỉ vẽ point khi `p_fire >= threshold`; threshold phải được chọn từ validation/test curve, không mặc định 0.5 một cách mù quáng.

### Bước 8 - Hard negatives

Thu thập và quản lý riêng 300-500 ảnh đại diện cho:

- thùng rác và vật tối có vùng sáng;
- đèn neon, đèn pha, phản xạ màu cam;
- hoàng hôn, ánh nắng gắt;
- áo bảo hộ màu cam;
- bếp, lò, vật nóng nhưng không có ngọn lửa.

Đánh nhãn `has_fire=0`, không dùng tọa độ giả để tính regression. Bảo đảm ảnh hard negative không trùng với test. Sau khi train, kiểm tra Recall fire thật để tránh giảm khả năng phát hiện.

### Bước 9 - Điều kiện chuyển sang YOLO-Pose

Chỉ chuyển sang YOLO-Pose nếu V3 + hard negatives vẫn không đạt mục tiêu vận hành. Trước đó phải có bbox chất lượng: từ mask SAM được kiểm tra hoặc do người gán. Không tạo bbox giả chỉ bằng một điểm rồi dùng nó làm bằng chứng cho detector.

---

## 13. CODE SNIPPETS THAM KHẢO CHO AI THỰC THI

Các đoạn dưới đây là khung tham khảo, không được coi là đã chạy hoàn chỉnh. AI thực thi phải thêm smoke test shape, loss finite và checkpoint loading.

### 13.1. Soft-argmax 2D

```python
class SoftArgmax2D(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        # logits: (B, 1, H, W)
        b, _, h, w = logits.shape
        prob = torch.softmax(
            logits.flatten(1) / self.temperature, dim=1
        ).view(b, h, w)
        ys = torch.linspace(0, 1, h, device=logits.device)
        xs = torch.linspace(0, 1, w, device=logits.device)
        x = (prob.sum(dim=1) * xs).sum(dim=1)
        y = (prob.sum(dim=2) * ys).sum(dim=1)
        return torch.stack((x, y), dim=1)
```

### 13.2. Gaussian target

```python
def make_gaussian_target(xy, h=64, w=64, sigma=2.0):
    # xy: (B, 2) normalized coordinates
    device = xy.device
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device), indexing="ij"
    )
    cx = xy[:, 0].view(-1, 1, 1) * (w - 1)
    cy = xy[:, 1].view(-1, 1, 1) * (h - 1)
    target = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return target.unsqueeze(1)
```

### 13.3. FPN-lite neck khung

```python
class HeatmapNeck(nn.Module):
    def __init__(self, c4=48, c8=80, hidden=96):
        super().__init__()
        self.lat4 = nn.Conv2d(c4, hidden, 1)
        self.lat8 = nn.Conv2d(c8, hidden, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, f4, f8):
        f8 = F.interpolate(
            self.lat8(f8), size=f4.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return self.refine(self.lat4(f4) + f8)  # (B,1,64,64)
```

### 13.4. Các kiểm tra bắt buộc

```python
assert heatmap.shape[-2:] == (64, 64)
assert torch.isfinite(loss).all()
assert (pred_xy >= 0).all() and (pred_xy <= 1).all()
```

Không được tuyên bố V3 thành công nếu chưa có so sánh V2/V3 cùng split, cùng metric, cùng benchmark latency và kiểm tra trên test set giữ kín.

---

## 14. INSTRUCTIONS CHO AI TIẾP NHẬN

1. Đọc toàn bộ file này trước khi sửa code.
2. Không xóa hoặc ghi đè checkpoint V2.
3. Không tự khẳng định MAE <5px trước khi train và đánh giá.
4. Không gọi heatmap là module đo nhiệt độ.
5. Không đổi backbone trước khi thử V3 FPN-lite trên MobileNetV4 hiện tại.
6. Bắt đầu bằng smoke test model, sau đó chạy một epoch thử, rồi mới chạy full Kaggle.
7. Mọi metric phải ghi rõ resolution, split, threshold, thiết bị và số lượng mẫu.
8. Khi kết thúc, cập nhật HANDOFF với kết quả thực nghiệm thật, các lỗi còn lại và quyết định bước tiếp theo.

---

## 15. CẬP NHẬT SAU KHI TẠO PIPELINE V3

### Trạng thái file hiện tại

- Các file `notebook4_v2_mnv4.py` và `notebook4_cell1.py` đến `notebook4_cell7.py` đã được loại bỏ.
- Notebook/script V3 hiện tại là `notebook4_v3_fpn.py`.
- Model V3 và checkpoint loader nằm trong `firegrounder_v3.py`.
- Inference V3 nằm trong `inference_v3.py`; `inference.py` vẫn dành cho V1/V2.
- `README.md` là hướng dẫn triển khai chính cho GitHub, máy local và Kaggle.

### Dataset HomeFire và V3

`fire_ground_dataset/dataset_labels.json` là file nhãn cần thiết cho V3:

- 6.500 bản ghi;
- 4.509 ảnh có lửa và 1.991 ảnh không lửa;
- tọa độ `p_fire` là điểm gốc lửa chuẩn hóa;
- các đường dẫn trong JSON là đường dẫn mount của Kaggle, không phải đường dẫn local.

Thư mục này chỉ chứa nhãn và cache, không chứa ảnh gốc. Khi chạy trên Kaggle
hoặc máy khác, phải attach/download đúng bộ ảnh HomeFire và đặt biến
`FIRE_IMAGE_ROOT` nếu đường dẫn trong JSON không còn tồn tại.

`fire_ground_dataset/stream_cache.jsonl` chỉ phục vụ resume quá trình SAM3
labeling. V3 không đọc file này.

`dataset_labels (1).json` là bộ legacy khác, 1.034 bản ghi; không dùng làm
dataset chính cho V3.

### Kết quả kiểm tra đã thực hiện

- `firegrounder_v3.py`, `inference_v3.py`, `notebook4_v3_fpn.py` đã qua kiểm tra cú pháp.
- Smoke test V3 đã chạy thành công: output `(2, 3)`, heatmap `(2, 1, 64, 64)`, loss hữu hạn.
- Chưa chạy huấn luyện V3 trong workspace này theo yêu cầu; chưa được phép kết luận metric V3.
- Không sửa hoặc ghi đè các checkpoint V2 hiện có.
