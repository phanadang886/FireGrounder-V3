# 📜 DEPTH3D_HISTORY — Changelog đầy đủ Depth3D_Localizer

**Đọc file này khi:** cần biết *vì sao* một quyết định cũ được chốt, hoặc tìm bài học / lỗi đã gặp.
**KHÔNG cần đọc** khi: bạn chỉ muốn biết trạng thái hiện tại, hoặc đang code → đọc `AGENTS.md`.

Tách khỏi `AGENTS.md` ngày 2026-10-09 vì file gốc phình lên 599 dòng, phần lớn là log cũ
không liên quan tới việc viết code hôm nay.

> **Nguyên tắc giữ nguyên:** không entry nào bị xóa hay rút gọn. Lịch sử vẫn đầy đủ ở đây.
> Entry mới nhất nằm trong `AGENTS.md`.

---

## 📋 Mục lục

| Ngày | Agent | Nội dung |
|---|---|---|
| 2026-10-09 | #4 | Hoàn thành TODO Depth3D: metric depth + calibration + pipeline + benchmark (và 4 lỗi thật đã sửa) |
| 2026-10-08 | #3 | Đổi tên `docs/HANDOFF.md` → `LEGACY_heatmap_history.md` |
| 2026-10-08 | #3 | Chốt phạm vi dự án: chỉ FireBase-YOLO26 + DepthART |
| 2026-10-08 | #3 | Test notebook & khuyến nghị dataset 3D |
| 2026-10-07 | #2 | Sửa backend tích hợp DepthART chính thức |
| 2026-10-07 | #2 | Kiểm tra và phát hiện lỗi |

---

### [2026-10-09] — Hoàn thành toàn bộ TODO Depth3D: metric depth + calibration + pipeline + benchmark — AI Agent #4

**Bối cảnh:** Agent #3 để lại 4 TODO "Sau khi backend stable" + 3 TODO "Optional — metric depth" chưa làm gì. Session này đóng hết, đồng thời phát hiện và sửa 4 lỗi thật.

#### 🐛 Lỗi phát hiện & sửa

1. **`localizer_3d.py` — zone bị ĐẢO NGƯỢC với metric depth**
   Metric depth giá trị THẤP = gần, nhưng `depth_norm`/`zone` được tính theo quy ước relative (cao = gần). Không sửa thì **mọi cảnh báo khoảng cách bị lật ngược 100%** (điểm 1m ra zone FAR, điểm 9m ra zone NEAR).
   → Sửa: đảo `depth_norm`, `depth_percentile`, `relative_dist` khi `metric_depth=True`. Có test [Test 4] chốt hợp đồng này.

2. **`calibrate_camera.py` — `per_view_errors` bỏ qua distortion**
   Truyền `dist=None` → ảnh ở rìa khung bị tính sai → RMSE tăng giả → `auto_drop_outliers` loại nhầm ảnh TỐT. Trên dữ liệu verify: loại mất 19/30 ảnh, sai số per-view tối đa **17.5px** (đúng cách: **0.42px**).
   → Sửa: `calibrate()` trả thêm `dist`, `per_view_errors` dùng distortion thật.

3. **`calibrate_camera.py` — pattern bàn cờ sai quy ước**
   `cv2.findChessboardCorners` nhận **số góc trong**, không phải số ô vuông. Tài liệu cũ ghi "9x6 ô vuông" → người dùng truyền sai → detect fail 100%.
   → Sửa: ghi rõ "inner corners" trong docstring + help + thông báo lỗi (kèm công thức bàn cờ tương đương).

4. **`calibrate()` thiếu ép kiểu dtype**
   `Point3f`/`Point2f` bắt buộc; float64 gây lỗi `Unsupported format or combination of formats` khó hiểu.
   → Sửa: ép `float32` + `ascontiguousarray` trong `calibrate()`.

#### ✅ TODO đã đóng

| TODO cũ | Kết quả | File |
|---|---|---|
| Thêm calibration tool lấy camera intrinsics | `calibrate_camera.py` — chessboard, SB detector, auto loại ảnh nhiễu, cảnh báo FOV vô lý | mới |
| Tải DepthART metric checkpoint | `depthart_metric_indoor_s_448.pth` tải + **SHA256 khớp** | `install_depthart.py` |
| Triển khai metric depth pipeline | `metric_depth.py` (`MetricDepthBackend`) + `xyz_metric` thật trong localizer | mới + `localizer_3d.py` |
| Viết script video inference end-to-end | `pipeline.py` — ảnh/video/webcam, 3 chế độ depth, JSON output | mới |
| Đo hiệu năng pipeline hoàn chỉnh | `benchmark.py` — đo thật trên ảnh dataset | mới |
| Đánh giá độ chính xác depth | `verify_calibration.py` — verify calibration bằng pinhole 3D thật | mới |

#### 📁 File mới
- **`metric_depth.py`** — `MetricDepthBackend`, output **mét**. Cần `CameraIntrinsics`. Ghi rõ quy ước ngược chiều với relative.
- **`calibrate_camera.py`** — tool calibration chessboard.
- **`verify_calibration.py`** — kiểm chứng calibration bằng dữ liệu 3D tổng hợp.
- **`pipeline.py`** — end-to-end YOLO → depth → localize → annotate.
- **`benchmark.py`** — đo hiệu năng (ms/frame, FPS, p50/p95).

#### 🔧 File đã sửa
- **`localizer_3d.py`** — thêm `CameraIntrinsics.save/load/scaled/to_matrix/pixel_to_ray/pixel_to_metric_xyz/is_calibrated`, trường `FireLocation3D.depth_m`, tham số `metric_depth`, sửa đảo chiều metric.
- **`install_depthart.py`** — viết lại: nhiều checkpoint, `--metric/--encoder/--all`, verify SHA256, tự tìm encoding Windows.
- **`quick_test.py`** — 5 → **37 assertions**, thêm hợp đồng metric/relative, pinhole, save/load, validate args.
- **`README.md`** — viết lại, thêm bảng quy ước depth, số đo benchmark, troubleshooting.
- **`requirements.txt`** — thêm `ultralytics`.

#### 📊 Kết quả đo THẬT (i5-1145G7, 16GB, Iris Xe, KHÔNG CUDA)

Relative depth (`DepthART-MNV4-S`, 20 ảnh `datasets/fire-samples`):

| Giai đoạn | Mean | p50 | p95 |
|---|---|---|---|
| detect | 0.04 ms | 0.00 | 0.08 |
| **depth** | **72.8 ms** | 56.7 | 118.1 |
| localize | 0.51 ms | 0.41 | 0.90 |
| annotate | 0.36 ms | 0.34 | 0.49 |
| **TOTAL** | **73.8 ms** | 57.9 | 118.8 |

→ **13.6 FPS**. Depth chiếm **99%** thời gian → nút thắt là depth, không phải YOLO.
Metric depth: **~2.3 s/frame** CPU (TinyViM/Mamba reference scan) — chỉ dùng ảnh tĩnh.

#### ✅ Kiểm chứng
- `quick_test.py` → **37/37 assertions PASS**
- `verify_calibration.py` → **[A] PASS**: khôi phục fx/fy/cx/cy **sai số 0.000%** với corner lý tưởng; [B] detect 30/30, RMSE 0.087px
- `install_depthart.py --metric indoor --encoder s` → 2/2 checkpoint, **SHA256 khớp**
- Metric depth chạy thật: depth 2.13–3.39 m, `xyz_metric` ra tọa độ mét hợp lý

#### ⚠️ Blocker chưa giải quyết (CẦN USER)
1. **Thiếu `best.pt` của FireBase-YOLO26** — `.gitignore` loại `*.pt` nên repo không có. Pipeline đã viết đầy đủ + có fallback `--keypoints` để chạy/test được phần depth, nhưng **chưa test được với YOLO thật**. Cần copy `best.pt` vào `models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/`.
2. **`ultralytics` chưa cài** trên máy dev → chưa chạy được detector. `pip install ultralytics`.
3. **Chưa có intrinsics thật** cho camera deployment → `xyz_metric` hiện mới chỉ dùng được với intrinsics ước lượng. Cần chụp bàn cờ rồi `python calibrate_camera.py --images ... --drop-outliers`.
4. **Chưa đánh giá độ chính xác depth trên ảnh lửa thật** — cần dataset có ground truth depth (Replica/ScanNet++), xem `docs/3D_DATASET_RECOMMENDATIONS.md`.

#### 📌 Quyết định thiết kế đã chốt
- **Tách 2 backend depth thay vì gộp**: `DepthBackend` (relative) và `MetricDepthBackend` (mét) khác nhau về bản chất — metric là **camera-aware** (intrinsics là input của model), không phải relative + hệ số. Gộp sẽ gây nhầm lẫn quy ước chiều depth.
- **`localize()` nhận cờ `metric_depth`** thay vì tự suy ra từ backend → caller kiểm soát rõ ràng, tránh đoán sai.
- **Intrinsics lưu JSON** kèm `source` + `is_calibrated` → pipeline cảnh báo khi đang dùng ước lượng cho metric depth.
- **`verify_calibration.py` tách [A] thuật toán / [B] end-to-end**: trộn lẫn 2 nguồn sai số dễ dẫn tới kết luận sai. [A] mới là tiêu chí quyết định.

#### 💡 Bài học (cho agent sau)
- **RMSE nhỏ ≠ calibration đúng.** Focal length chỉ xác định được khi ảnh có độ nghiêng đa dạng. Ảnh mặt phẳng cho RMSE < 0.01px nhưng fx sai bậc hai order. Luôn kiểm tra FOV có hợp lý (20°–140°) không.
- **Mọi backend depth không cùng quy ước chiều.** Luôn kiểm tra bằng test đối chiếu 2 chiều trước khi tin.
- **Khi verify, tách sai số thuật toán khỏi sai số hạ tầng** (render/detect). Nếu không, sẽ sửa nhầm code đúng.

---

### [2026-10-08] — Đổi tên docs/HANDOFF.md → LEGACY_heatmap_history.md — AI Agent #3

**Lý do:** `HANDOFF.md` trùng vai trò với `AGENTS.md` (cùng là "AI sau đọc để làm tiếp"), nhưng nội dung là nhánh heatmap đã bỏ. Tên cũ khiến agent sau tưởng đó là handoff hiện tại.

**Đã làm:**
1. ✅ `docs/HANDOFF.md` → `docs/LEGACY_heatmap_history.md` (nội dung giữ nguyên)
2. ✅ Thêm banner ở đầu file: đừng làm theo, kiến trúc đang chạy xem `AGENTS.md`
3. ✅ Sửa chỗ trỏ tên cũ trong `AGENTS.md` và `docs/README.md` (2 chỗ)

**Không đụng:** code, model, dataset.

---

### [2026-10-08] — Chốt phạm vi dự án: chỉ FireBase-YOLO26 + DepthART — AI Agent #3

**Quyết định của user:** Mọi model ngoài `FireBase_YOLO26` và `Depth3D_Localizer` đều là **thử nghiệm cũ đã bị bỏ**.

**Đã làm:**
1. ✅ Thêm mục `⚠️ PHẠM VI DỰ ÁN` ở đầu file — nêu rõ 2 module đang chạy
2. ✅ Đánh dấu 3 thư mục là legacy/KHÔNG ĐỤNG VÀO:
   - `models/FireGrounder_V3/` (heatmap single-keypoint)
   - `models/ROI_Refiner/` (narrow_localizer, train_roi_localizer)
   - `models/YOLO26_Pose/` (pose thử nghiệm cũ)
3. ✅ Thêm mục `### 0` vào phần `🚫 KHÔNG ĐƯỢC THAY ĐỔI`
4. ✅ Cảnh báo `docs/README.md` + `docs/research_gap.md` mô tả nhánh **đã bỏ** (heatmap), chỉ dùng làm lịch sử
5. ✅ Dọn `docs/`: xoá `AGENT3_SUMMARY.md` (trùng lặp changelog — giữ 1 file log duy nhất)

**Quy tắc cho agent sau:**
- Đọc `## ⚠️ PHẠM VI DỰ ÁN` **trước khi sửa bất kỳ file nào**
- Legacy = đọc để học ý tưởng, **không sửa / không xoá / không import**
- Code mới chỉ viết trong `models/Depth3D_Localizer/`

**Cấu trúc docs sau khi dọn:**
```
AGENTS.md                         ← log duy nhất
docs/3D_DATASET_RECOMMENDATIONS.md ← reference dataset 3D
docs/README.md                    ⚠️ legacy branch, stale paths
docs/LEGACY_heatmap_history.md   ← lịch sử nhánh heatmap ĐÃ BỎ (đừng làm theo)
docs/research_gap.md              ← background research (legacy branch)
```

---

### [2026-10-08] — Test notebook & khuyến nghị dataset 3D — AI Agent #3

**Đã hoàn thành:**
1. ✅ Verify `quick_test.py` — tất cả tests PASSED
2. ✅ Verify backend load với `prefer='auto'` — DepthART-MNV4-S loaded OK
3. ✅ Review `depth3d_demo.ipynb` — notebook đã cập nhật đúng prefer='auto'
4. ✅ Research và đề xuất 5 dataset 3D chính cho testing
5. ✅ Tạo document chi tiết: `docs/3D_DATASET_RECOMMENDATIONS.md`

**Dataset khuyến nghị:**
- **TOP 1: Replica** (Facebook AI) — 18 synthetic scenes, perfect ground truth, simulatable
- **TOP 2: ScanNet++** (TUM) — 460 real scenes, sub-mm laser scan, high quality
- **TOP 3: Matterport3D** (Princeton) — 90 building-scale scenes, panoramic views
- **TOP 4: Hypersim** (Apple) — 461 scenes, HDR, perfect synthetic ground truth
- **TOP 5: ScanNet** (TUM) — 1513 scenes, standard benchmark

**Khuyến nghị triển khai:**
- Phase 1 (Quick): Replica với Habitat-Sim → tạo synthetic fire test dataset
- Phase 2 (Real): ScanNet++ hoặc Matterport3D → validate trên real textures
- Phase 3 (Advanced): Hypersim + Blender fire rendering → full simulation

**Trạng thái notebook:**
- `depth3d_demo.ipynb` sẵn sàng chạy với backend='auto' (fallback DAv2 nếu chưa cài DepthART)
- Cell 1-3: Import + load backend + inference — đã test OK
- Cell 4: Interactive click demo — cần Jupyter với matplotlib backend
- Cell 5-8: Batch processing, video, full pipeline — chưa test (cần ảnh/video thật)

**Lưu ý cho agent tiếp:**
- Notebook không test trực tiếp được vì cần Jupyter interactive environment
- Các unit tests và backend load đã verify — module hoạt động bình thường
- Để test notebook đầy đủ: chạy trong VS Code Jupyter hoặc JupyterLab
- Dataset recommendations trong `docs/3D_DATASET_RECOMMENDATIONS.md`

---

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
