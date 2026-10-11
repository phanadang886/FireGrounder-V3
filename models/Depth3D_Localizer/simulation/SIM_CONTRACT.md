# 📋 SIM_CONTRACT — Đánh giá depth + toán 2D→3D bằng ảnh và depth có sẵn

**Phiên bản:** v3 (2026-10-09) — hướng ảnh+depth có sẵn, đã gộp task list vào đây.
**Đối tượng đọc:** developer / AI agent được giao code simulator.
**Cấu trúc:** mục **1–10** = spec kỹ thuật (format, CLI, công thức). Mục **11** = task list có tick.
Muốn biết "vì sao đổi hướng" thì đọc `AGENTS.md` (log quyết định). Muốn chạy thì đọc `README.md`.

---

---

## 0. Quy tắc về số liệu — KHÔNG BỊA SỐ (user yêu cầu 2026-10-09)

**Ba loại trạng thái, tuyệt đối không gộp:**

| Ký hiệu | Nghĩa là | Ví dụ |
|---|---|---|
| `[x]` + "logic đã chứng minh" | **Đã verify, không cần data thật** | `utils.py --selftest` 22/22; round-trip 5.6e-17 m; `scale=6553.5` ra 5.9535 m chứ không phải 39.016 m |
| `[ ] ⬜` | **CHƯA CÓ SỐ** — chưa đo, hoặc đo trên data giả | Mọi metric DepthART trên ảnh thật |
| `[ ]` | Chưa làm | Phase 5, 6 |

**Cấm tuyệt đối:**
1. ❌ Bịa số, bịa output, bịa kết quả — kể cả số "đẹp" hay "hợp lý".
2. ❌ Tick `[x]` cho điều kiện xong dựa trên **dataset giả / dữ liệu tự sinh**.
   Test trên dữ liệu tự sinh chỉ chứng minh *máy chạy đúng*, **không** phải kết quả đánh giá.
3. ❌ Đoán "chắc là khoảng X m" rồi ghi như số thật. Nói rõ là **dự đoán**, giữ trạng thái `[ ] ⬜`.
4. ❌ Trộn số của dataset giả vào bảng số liệu dùng để báo cáo.

**Bắt buộc:**
- Số nào in ra cho người dùng phải **kèm nguồn**: script đã chạy, và dataset là thật hay giả.
- Nếu không làm được hoặc thiếu data → **báo lại user kèm hướng giải quyết**,
  không tự bịa kết quả tạm để lấp chỗ trống.

> **Hiện trạng (2026-10-09):** Phase 1, 3, 4 xong phần *logic*.
> **CHƯA có bất kỳ con số nào về độ chính xác DepthART trên ảnh thật** — đang chờ Replica
> tải xong (Phase 2), rồi Phase 6 mới sinh ra số đó.

---

## 1. Hướng đi (đọc 1 phút là hiểu)

Để chấm điểm DepthART và phép biến đổi 2D→3D, mỗi ảnh chỉ cần đúng 2 thứ:

1. **Ảnh RGB**
2. **Bản đồ depth thật** của ảnh đó (tính bằng mét)

Quy trình cho mỗi điểm test:

```
chọn 1 pixel (u,v) có depth thật hợp lệ
        │
        ├─→ depth thật tại (u,v) + ma trận nội quan → tọa độ 3D CHUẨN
        │
        └─→ đưa ảnh cho DepthART → depth đoán tại (u,v) + ma trận nội quan → tọa độ 3D ĐOÁN
                                                        │
                                              sai số = khoảng cách giữa 2 tọa độ 3D
```

**Không cần:** tải mesh 31 GB, tự render, đặt vật 3D, chạy detector, dán texture lửa.
Vì không chạy detector nên đáp án 3D chuẩn được tính thẳng từ depth thật — sạch 100%.

### Vì sao sai số vị trí khác sai số depth

Cùng một lỗi depth, pixel ở **góc ảnh** cho sai số vị trí lớn hơn pixel ở **giữa ảnh**:

```
sai số vị trí ≈ sai số depth × √(1 + ((u−cx)/fx)² + ((v−cy)/fy)²)
```

Hệ số nhân ở giữa ảnh ≈ 1.0, ở góc ảnh có thể tới 2–2.5. Vì vậy metric phải tách theo
**vùng ảnh (giữa/rìa)** — chấm chung một số sẽ che mất hiện tượng này.

---

## 2. Dataset

### 2.1 Replica preprocessed (ưu tiên 1) — xác minh link thật ngày 2026-10-09

- **Là gì:** ảnh RGB + depth + pose do cộng đồng render sẵn từ Replica (đóng gói cho NICE-SLAM).
- **Link chính thức (đã HEAD-check, HTTP 200):**
  `https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip` — **11.59 GB**, cập nhật
  2022-04-05. Máy chủ ETH Zurich (CVG), **không phải link cộng đồng** nên ổn định hơn hẳn.
- **Link dự phòng** (nếu chính chết): Baidu caiyun, mật khẩu `v3fY`, trong
  `cvg/nice-slam` → `scripts/download_replica.sh`.
- **Vì sao ưu tiên:** ảnh trong nhà, khớp mục tiêu depth indoor, 11.59 GB vừa ổ D.
- **Giới hạn:** góc camera cố định theo video quay sẵn — chấp nhận được vì ta không cần đổi góc.

> ⚠️ **Format thật (đã đọc từ source, không đoán).** Nguồn: `cvg/nice-slam` →
> `src/utils/datasets.py` class `Replica` và `configs/Replica/replica.yaml`.
> Bốn điểm dễ sai nhất:
>
> | Mắt xích | Thực tế |
> |---|---|
> | **Hệ số depth** | `png_depth_scale = 6553.5` — **KHÔNG phải 1000** |
> | **Tên file ảnh** | `results/frame000000.jpg` (**jpg**, zero-pad 6 số) |
> | **Tên file depth** | `results/depth000000.png` (**cùng thư mục `results/`**) |
> | **Quy ước pose** | `traj.txt` **đã ở hệ OpenCV** — KHÔNG đảo cột. Đã kiểm bằng mesh (mục 5.1) |
>
> Dùng sai hệ số depth ⇒ **mọi số đo sai 6.55 lần**, không báo lỗi. Đây là lý do
> `load_depth_map(path, scale=...)` **bắt buộc** truyền `scale` thay vì tự đoán.

### 2.2 NYUv2 (dự phòng / so sánh với paper)

- **Là gì:** ảnh thật trong nhà kèm depth thật, chuẩn vàng để chấm model depth.
- **Vì sao dự phòng:** ảnh thật nên chấm DepthART công bằng hơn; có số của paper để đối chiếu
  (AbsRel kỳ vọng ~0.06–0.08).
- **Khác Replica:** depth PNG16 theo **milimét** (`scale=1000.0`), **không có** `traj.txt`
  (pose là ma trận đơn vị). `sample_points.py` phải chịu được cả hai.
- **Dùng khi nào:** khi link Replica chết, hoặc muốn số so được với literature.

### 2.3 Không dùng

- Mesh Replica gốc (`replica_v1_0.tar.gz`, 31.53 GB): vượt ổ D, SDK không chạy Windows.
- `win_download.bat` gốc: ghi cứng `c:\replica` — **cấm chạy** (ổ C chỉ còn 0.8 GB).

---

## 3. Quy tắc ổ đĩa (bắt buộc)

1. **Cấm ghi file dữ liệu vào ổ C** (ngoại trừ pip install vào site-packages).
2. Mọi dữ liệu + kết quả nằm dưới **`D:\replica_sim\`**.
3. Mọi script phải gọi `check_disk_space()` trước mọi I/O:
   - **Ngưỡng nghiêm (I/O nặng: tải, giải nén 11 GB):** `min_c_gb=2`, `min_d_gb=10`
   - **Ngưỡng nhẹ (I/O nhẹ: ghi vài chục file JSON, vẽ preview):** `min_c_gb=0.5`, `min_d_gb=1`
   - Lý do phân tầng: nguy cơ Windows treo đến từ **tải/giải nén**, không phải từ
     việc ghi 200 file JSON vài MB. Ổ C dao động 0.8–2.6 GB tuỳ Chrome/Docker có mở hay
     không — dùng ngưỡng 2 GB cho mọi thứ sẽ chặn oan các thao tác hoàn toàn vô hại.
4. Không hard-code `C:\` trong code. Dùng `pathlib.Path`.

---

## 4. Cấu trúc thư mục

```
D:\replica_sim\
├── data\
│   ├── replica_pp\
│   │   ├── office0\                      ← ⚠️ KHÔNG có gạch dưới
│   │   │   ├── results\
│   │   │   │   ├── frame000000.jpg       ← RGB, zero-pad 6 số
│   │   │   │   ├── frame000001.jpg
│   │   │   │   ├── depth000000.png       ← depth, CÙNG thư mục results/
│   │   │   │   └── depth000001.png
│   │   │   └── traj.txt                  ← pose, ở GỐC scene (không phải trong results/)
│   │   ├── office1\ … room0\ room1\ room2\
│   │   └── _dl\Replica.zip               ← file tải về; XOÁ sau khi giải nén
│   └── nyuv2\
│       └── {split}\
│           ├── rgb\
│           ├── depth\
│           └── intrinsics.txt             ← NYUv2 dùng chung 1 intrinsics
├── samples\                              ← điểm test đã chọn (xem 5.2)
│   ├── replica_pp_office0_0001.json
│   └── ...
└── results\
    ├── case_A\                           ← depth GT (kiểm tra toán)
    │   └── per_sample.csv
    ├── case_B_clean\                     ← DepthART + pose chuẩn
    │   └── per_sample.csv
    ├── case_B_noisy\                     ← DepthART + pose nhiễu (số thực tế)
    │   └── per_sample.csv
    └── metrics_summary.json
```

⚠️ **Replica không có file `intrinsics.txt`.** Dùng `utils.replica_intrinsics()`:
1200×680, fx=fy=600, cx=599.5, cy=339.5. Không suy `width = 2·cx` (ra 1199, lệch 1 px).

Code mới chỉ nằm trong `models/Depth3D_Localizer/simulation/`:

```
simulation\
├── SIM_CONTRACT.md      ← file này
├── README.md            ← hướng dẫn chạy từng bước
├── utils.py             ← hàm dùng chung (disk check, parse, đọc depth)
├── download_dataset.py  ← tải dataset
├── sample_points.py     ← chọn điểm test
├── run_eval.py          ← chạy case A / B
└── eval_metrics.py      ← tính metric + vẽ biểu đồ
```

---

## 5. Format dữ liệu

### 5.1 Dataset tải về

**Pose — Replica `traj.txt`** (16 số mỗi dòng, ma trận 4×4 row-major):
```
r00 r01 r02 tx   r10 r11 r12 ty   r20 r21 r22 tz   0 0 0 1
```
- Dòng N ứng với frame thứ N (`frame{N:06d}.jpg` / `depth{N:06d}.png`).
- ✅ **Đã ở hệ OpenCV** (X phải, Y xuống, Z xa) — khớp `localizer_3d.py`.
  → `parse_pose_line(line)` (mặc định `opengl_to_opencv=False`). **KHÔNG đảo.**
  
  **Cách chứng minh (2026-10-09, đo trên mesh Replica thật):** depth GT được render từ chính
  mesh đó, nên điểm back-project đúng quy ước phải **nằm sát bề mặt mesh**. Đo khoảng cách
  median tới đỉnh mesh gần nhất (n=96, cả 8 scene):
  
  | Cách xử lý | Khoảng cách median |
  |---|---|
  | **Không đảo (đúng)** | **0.005 m** — đúng độ phân giải mesh |
  | Đảo cột 1,2 | 0.666 m |
  | Đảo cả 3 cột | 0.894 m |
  
  **Lý do dễ nhầm:** `cvg/nice-slam` có `c2w[:3,1] *= -1; c2w[:3,2] *= -1` trong
  `src/utils/datasets.py`, nhưng đó là để chuyển hệ cho **renderer OpenGL của họ**, KHÔNG
  phải để đọc file `traj.txt` đã lưu. Đã từng chuyển nhầm từ code loader sang file data
  → toạ độ world lệch trung vị **6.02 m**. Đã sửa.
- Thứ tự dòng: file được sắp xếp **chuỗi**, nên phải zero-pad đủ 6 số
  (`sorted()` cho `frame9` > `frame10` nếu không pad).

**Pose — dạng 7 số** (một số bản khác):
```
tx ty tz qx qy qz qw
```
- Quy ước quay **mặc định OpenCV**, không đảo. Quy ước quy ước phải kiểm riêng theo nguồn.

**Intrinsics:**
- **Replica: KHÔNG có file** → `utils.replica_intrinsics()` = 1200×680, fx=fy=600, cx=599.5, cy=339.5.
- Dataset khác: `intrinsics.txt` một dòng `fx fy cx cy` / 6 số / ma trận 3×3.

**Depth map:**
- Replica: `results/depth{N:06d}.png`, uint16, **`scale = 6553.5`** → mét. Giá trị 0 = không đo được.
- PNG milimét (NYUv2): `scale = 1000.0`. NPY float32: đã là mét, `scale` bị bỏ qua.
- Đọc bằng `utils.load_depth_map(path, scale=...)` (không tự viết lại ở mỗi script).
- **Kiểm tra nhanh khi đọc:** Replica cắt ở 10 m nên `max(depth) <= 10.01`. Nếu thấy
  depth 30–60 m ⇒ đã chia nhầm hệ số.

### 5.2 Sample JSON (điểm test)

**File:** `D:\replica_sim\samples\{source}_{scene}_{id:04d}.json`

```json
{
  "sample_id": "replica_pp_office0_0001",
  "source": "replica_pp",
  "scene": "office0",
  "frame": 42,
  "frame_path": "D:\\replica_sim\\data\\replica_pp\\office0\\results\\frame000042.jpg",
  "depth_path": "D:\\replica_sim\\data\\replica_pp\\office0\\results\\depth000042.png",
  "pixel": [717, 401],
  "region": "border",
  "depth_gt_m": 2.80,
  "xyz_gt_cam": [0.98, -0.30, 2.80],
  "xyz_gt_world": [1.20, 0.50, 2.80],
  "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5, "width": 1200, "height": 680},
  "depth_scale": 6553.5,
  "pose_convention": "opencv",
  "pose": {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.0, 1.5, 0.0]}
}
```

- `region`: `center` nếu pixel nằm trong nửa giữa ảnh (u trong [W/4, 3W/4] và v trong [H/4, 3H/4]), còn lại là `border`.
- `xyz_gt_cam`: tính từ pixel + depth GT bằng công thức pinhole (chính là đáp án chuẩn ở hệ camera).
- `xyz_gt_world`: `xyz_gt_cam` đổi qua pose (đáp án chuẩn ở hệ thế giới).
- `pose`: lưu **dạng ma trận `R` + vector `t`, đã ở hệ OpenCV** — không lưu quaternion,
  để `run_eval.py` không phải parse lại. `pose_convention` ghi lại để dễ kiểm khi đọc.
- `depth_scale`: lưu kèm hệ số đã dùng để đọc depth, để tự kiểm chéo với `metrics_summary.json`.
- Quy tắc chọn điểm (trong `sample_points.py`): depth GT trong [1.5, 6.0] m, pixel cách rìa ảnh ≥ 10 px,
  mỗi frame tối đa 5 điểm, trộn đều `center`/`border`, seed cố định để chạy lại ra kết quả giống nhau.

### 5.3 Per-sample CSV (output của run_eval)

**File:** `D:\replica_sim\results\{case}\per_sample.csv`

```
sample_id,source,scene,frame,region,depth_gt,depth_pred,depth_err,gt_x,gt_y,gt_z,pred_x,pred_y,pred_z,err,err_x,err_y,err_z
```

- Case A: `depth_pred = depth_gt` (lấy thẳng từ depth map), `err` kỳ vọng ≈ 0.
- Case B: `depth_pred` từ DepthART. Với model **relative** phải căn scale+shift
  (bình phương tối thiểu theo depth GT của cả ảnh) trước khi so — vì relative không có đơn vị mét.
  Với model **metric** so trực tiếp, không căn.
- Tọa độ so ở **hệ thế giới** (`xyz_world`) để bao gồm cả sai số đổi hệ.

### 5.4 metrics_summary.json (output của eval_metrics)

```json
{
  "metadata": {"date": "2026-10-09", "num_samples": 200, "sources": ["replica_pp"]},
  "depth": {
    "mae": 0.087, "rmse": 0.124, "absrel": 0.061,
    "delta_125": 0.94, "delta_125_2": 0.99, "delta_125_3": 1.0,
    "note": "relative đã căn scale+shift; metric so trực tiếp"
  },
  "position_3d": {
    "mae": 0.11, "median": 0.08, "rmse": 0.16, "p95": 0.32,
    "le_0_25m": 0.82, "le_0_50m": 0.96,
    "by_axis": {"mae_x": 0.04, "mae_y": 0.05, "mae_z": 0.09},
    "by_distance": {"0-2m": {"mae": 0.05, "n": 40}, "2-4m": {"mae": 0.10, "n": 90}, "4-6m": {"mae": 0.17, "n": 70}},
    "by_region": {"center": {"mae": 0.08, "n": 100}, "border": {"mae": 0.14, "n": 100}}
  }
}
```

Có 1 block như trên cho mỗi case (`case_A`, `case_B_clean`, `case_B_noisy`).

---

## 6. Các case đánh giá

| Case | Depth | Pose | Ý nghĩa |
|---|---|---|---|
| **A** | GT | chuẩn | Kiểm tra toán pinhole + đổi hệ. MAE phải < 1 cm, nếu không là code sai, chưa liên quan model |
| **B_clean** | DepthART | chuẩn | Chấm riêng model depth |
| **B_noisy** | DepthART | nhiễu (dịch chuyển σ=2 cm, xoay σ=0.5°) | Số gần thực tế triển khai — đây mới là số để báo cáo |

(Học từ báo cáo ngoài: pose chuẩn cho 0.13 m nhưng pose nhiễu lên 0.27 m. Chỉ báo cáo số pose chuẩn là đánh lừa.)

---

## 7. CLI của từng script

```bash
# 1. Tải dataset (có --dry-run để xem dung lượng trước khi tải thật)
python download_dataset.py --source replica_pp --scenes office_0,room_0 --output D:\replica_sim\data [--dry-run]
python download_dataset.py --source nyuv2 --split test --output D:\replica_sim\data [--dry-run]

# 2. Chọn 200 điểm test
python sample_points.py --data-dir D:\replica_sim\data\replica_pp --source replica_pp \
    --num-samples 200 --output D:\replica_sim\samples --seed 42 [--visualize]

# 3. Chạy đánh giá (3 lệnh độc lập)
python run_eval.py --case A --samples D:\replica_sim\samples --data-dir D:\replica_sim\data\replica_pp \
    --output D:\replica_sim\results\case_A
python run_eval.py --case B --pose-mode clean --depth-model depthart-mnv4s \
    --samples D:\replica_sim\samples --data-dir D:\replica_sim\data\replica_pp \
    --output D:\replica_sim\results\case_B_clean
python run_eval.py --case B --pose-mode noisy \
    --samples D:\replica_sim\samples --data-dir D:\replica_sim\data\replica_pp \
    --output D:\replica_sim\results\case_B_noisy

# 4. Tổng hợp metric + vẽ biểu đồ
python eval_metrics.py --results-dir D:\replica_sim\results --output D:\replica_sim\results\metrics_summary.json [--plot]
```

Yêu cầu chung: `--help` đầy đủ, progress bar cho vòng lặp dài, log mỗi sample
`[B_clean] replica_pp_office_0_0001 (3/200): depth_err=0.19m err=0.21m`,
validate input trước khi xử lý, thông báo lỗi kèm cách sửa.

---

## 8. `utils.py` — hàm dùng chung (không viết lại ở script khác)

- `check_disk_space(min_c_gb=2, min_d_gb=10)` — kiểm tra ổ đĩa, raise lỗi rõ ràng.
- `load_depth_map(path, scale=None)` — PNG 16-bit / NPY → mét (float32), pixel lỗi = NaN.
  **`scale` BẮT BUỘC khi depth là số nguyên** — Replica `6553.5`, milimét `1000.0`.
  Hàm cố tình *không* đoán hệ số: đoán sai cho số sai 6.55 lần mà không báo lỗi.
- `parse_intrinsics_txt(path)` → dict `{fx, fy, cx, cy, width, height, source}` (width/height suy từ `2×cx`, `2×cy` nếu file không ghi).
- `replica_intrinsics()` → dict intrinsics Replica (dataset **không có file intrinsics**).
- `parse_pose_line(line, opengl_to_opencv=False)` → `(R, t)`.
  Replica `traj.txt` **đã ở hệ OpenCV** — dùng mặc định, **không đảo**. Xem mục 5.1.
- `pixel_to_cam(pixel, depth_m, intr)` → `(X, Y, Z)` hệ camera (X phải, Y xuống, Z xa camera — khớp `localizer_3d.py`).
- `cam_to_world(xyz_cam, R, t)` → `R @ xyz + t`.
- `add_pose_noise(R, t, seed)` → pose nhiễu (σ dịch chuyển 2 cm, σ xoay 0.5°).
- `align_scale_shift(pred, gt)` → `ScaleShift(scale, shift)` cho depth **relative**.
  Dùng `scale, shift = align_scale_shift(pred, gt)` rồi tính `aligned = scale * pred + shift`.

---

## 9. Dependencies

```
numpy>=1.24.0
opencv-python>=4.8.0
pillow>=10.0.0
torch>=2.1.0            # DepthART (đã có từ module cũ)
tqdm>=4.65.0
matplotlib>=3.7.0       # chỉ cần cho --plot
pandas>=2.0.0
scipy>=1.11.0           # quaternion, align
```

Import từ module hiện có (chỉ đọc, không sửa):
```python
sys.path.insert(0, <thư mục cha của simulation>)
from localizer_3d import CameraIntrinsics
from depth_model import DepthBackend        # relative
from metric_depth import MetricDepthBackend # metric
```

**Không bắt buộc Open3D** (không còn render mesh). Gỡ khỏi danh sách cài cho simulator.

---

## 10. Số liệu kỳ vọng

- **Case A:** MAE < 0.01 m. Lớn hơn = code sai, dừng lại sửa trước khi đọc bất kỳ số nào khác.
- **Case B (relative, sau căn):** MAE vị trí khoảng 0.05–0.20 m tùy khoảng cách; AbsRel depth tham khảo ~0.06–0.10.
- **Case B_noisy** luôn tệ hơn B_clean — chênh lệch này chính là giá của pose sai, phải báo cáo cả hai.
- **Dung lượng:** dataset vài GB + samples vài MB + results < 100 MB → ổ D 34 GB đủ thoải mái.
- **Thời gian:** tải 10–30 phút; sample vài phút; Case A < 1 phút; Case B ≈ số sample × ~0.1 s (+ overhead model).

Điều kiện xong cụ thể theo từng phase: xem **mục 11**.

---

## 11. Task list — chia phase & tick

Tick sau mỗi task. Đây là phần bạn đọc khi được giao code simulator.

### ⚠️ Trước khi code — 3 điều bắt buộc

1. **Đọc mục 1–8 của file này trước.** Không tự ý đổi format file, tên trường, hay CLI.
2. **Cấm ghi vào ổ C.** Mọi thứ dưới `D:\replica_sim\`. Gọi `utils.check_disk_space()` trước mọi I/O nặng.
3. **Không sửa code module chính.** `depth_model.py`, `metric_depth.py`, `localizer_3d.py` chỉ **import**, không sửa. Muốn đổi hành vi → hỏi user, đừng tự ý patch.

### Sơ đồ phụ thuộc

```
        utils.py  ← LÀM ĐẦU TIÊN, mọi script còn lại import nó
           │
    ┌──────┴──────┬─────────────┬─────────────┐
    ↓             ↓             ↓             ↓
download_    sample_        run_eval     eval_metrics
dataset.py   points.py      .py          .py
    │             │             │             │
    └──────┬──────┘             └──────┬──────┘
           ↓                           ↓
      cần dataset                 cần samples
      (data tải về)              (JSON từ sample_points)
                                        ↓
                              ⛔ GATE: Case A < 1 cm
                                        ↓
                              đọc được Case B / B_noisy
```

**Phân biệt 2 loại "song song":**

| | Việc | Trạng thái |
|---|---|---|
| **Code** | Sau khi `utils.py` có stub với đúng chữ ký hàm, 4 script kia **code song song được** | Đều chỉ phụ thuộc contract |
| **Chạy** | Thứ tự **cứng**: download → sample → run_eval → eval_metrics | Không đảo được |

---

### Phase 0 — Contract ✅ DONE

- [x] Mục 1–10 của `SIM_CONTRACT.md` (spec)
- [x] `README.md` — 6 lệnh chạy
- [x] Mục 11 (task list này)
- [x] ~~Phê duyệt user trước khi code~~ — user đã duyệt và giao Phase 1 (2026-10-09)

---

### Phase 1 — `utils.py` (chặn tất cả) 🔴 BLOCKING

**Phụ thuộc:** không có · **Ưu tiên cao nhất**

- [x] `check_disk_space(min_c_gb=2, min_d_gb=10)` — raise lỗi tiếng Việt kèm cách sửa
- [x] `load_depth_map(path)` — PNG 16-bit (mm→m, 0→NaN) + NPY (m). Tự nhận diện, không cần tham số
- [x] `parse_intrinsics_txt(path)` → `{fx, fy, cx, cy, width, height}`; suy `width=2·cx` nếu file thiếu
- [x] `parse_pose_line(line)` → `(R, t)` từ `tx ty tz qx qy qz qw`
- [x] `pixel_to_cam(pixel, depth_m, intr)` → `(X,Y,Z)` — **đúng quy ước trục của `localizer_3d.py`** (X phải, Y xuống, Z xa camera)
- [x] `cam_to_world(xyz_cam, R, t)` → `R @ xyz + t`
- [x] `add_pose_noise(R, t, seed)` — σ dịch 2 cm, σ xoay 0.5°
- [x] `align_scale_shift(pred, gt)` — bình phương tối thiểu trên pixel có GT hợp lệ
- [x] `load_module_backend(name)` — import `depth_model` / `metric_depth` từ thư mục cha, `sys.path` đúng

**Điều kiện xong:**
- [x] Test tự kiểm **không cần model**: 1 tọa độ 3D biết trước → `pixel_to_cam` → `cam_to_world` → về lại gần như y hệt.
      → Chạy `python utils.py --selftest`: **22/22 PASS, exit 0** (sai số round-trip ≤ 5.6e-17 m)
- [x] `--help` in ra mô tả từng hàm
- [x] **Không** import torch/model nặng ở cấp module (chỉ khi gọi hàm cần)
      → Đo bằng process sạch: `import utils` = **151 ms**, `sys.modules` không có `torch`/`cv2`/`depth_model`/`metric_depth`

**Ghi chú khi làm (Agent #5):**
- `align_scale_shift` trả về `ScaleShift(scale, shift)` (NamedTuple) chứ không phải mảng depth đã căn — vì `run_eval.py` cần áp hệ số lên **cả bản đồ**, không chỉ pixel đã dùng để fit. Dùng `scale, shift = align_scale_shift(pred, gt)` rồi tính `aligned = scale * pred + shift`.
- `parse_pose_line` nhận thêm **dạng 16 số** (ma trận 4×4 của NICE-SLAM) vì Replica preprocessed có cả 7 số lẫn 16 số tuỳ bản phát hành.
- `load_module_backend` mặc định `load=False` để test chữ ký hàm không mất vài giây tải checkpoint. `run_eval.py` gọi `.load()` sau đó.
- ⚠️ **Phát hiện môi trường (quan trọng cho Phase 4):** máy này có 15.7 GB RAM nhưng khi
  Chrome + Docker + IDE cùng mở thì **chỉ còn ~1 GB trống**. OpenBLAS cấp buffer theo số luồng
  (8 luồng) và khi thiếu RAM nó **abort cứng tiến trình** — `try/except` của Python không bắt được.
  → `utils.py` đã đặt `OPENBLAS_NUM_THREADS=1` **trước** khi import numpy (chỉ OPENBLAS, **không**
    đụng OMP/MKL để khỏi làm chậm torch). Import `utils` giờ không bao giờ crash vì lý do này.
  → Nhưng **khi chạy DepthART thật (Phase 4) vẫn phải đóng Chrome/Docker** để có ≥ 3 GB trống.

> **Tip:** commit stub sớm để 4 script kia bắt đầu code song song, không bị chặn.

---

### Phase 2 — `download_dataset.py` ⛔ RISK CAO

**Phụ thuộc:** Phase 1

> **Rủi ro lớn nhất của cả simulator:** link tải Replica preprocessed đến từ cộng đồng (NICE-SLAM, iMAP, vMAP), **có thể chết hoặc đổi tên bất cứ lúc nào**. Nếu không xử lý, mọi phase sau đứng lại.

- [ ] `--dry-run` in dung lượng từng scene **trước khi tải thật**
- [ ] Tải `replica_pp` — mặc định ưu tiên, ưu tiên scene nhẹ (`office_0`, `room_0`)
- [ ] Tải `nyuv2` (dự phòng) — có `--split test`
- [ ] Giải nén an toàn (kiểm tra path chống zip-slip, kiểm tra dung lượng cần)
- [ ] `--verify` sau tải: đủ `rgb/` + `depth/` + `poses.txt` + `intrinsics.txt` không
- [ ] Tự dò mirror dự phòng khi link chính fail, báo rõ đã dùng link nào
- [ ] Báo cáo dung lượng thực tế sau tải, cảnh báo nếu ổ D còn < 10 GB

**Điều kiện xong:** `office_0` + `room_0` nằm đúng cấu trúc mục 4, `--verify` pass.

**Nếu mọi link Replica đều chết:** dừng, báo user, chuyển sang `--source nyuv2` (nhiều mirror hơn vì là dataset phổ biến). Không tự bịa link.

---

### Phase 3 — `sample_points.py`

**Phụ thuộc:** Phase 1 + data Phase 2

- [x] Chọn pixel theo luật mục 5.2: depth GT ∈ [1.5, 6.0] m, cách rìa ≥ 10 px, **tối đa 5 điểm/frame**
- [x] Trộn đều `center` / `border` (bắt buộc — không thì metric `by_region` vô nghĩa)
- [x] Seed cố định → chạy lại ra **cùng kết quả** (verify: cùng seed → SHA256 file giống hệt; seed khác → khác file)
- [x] Xuất JSON đúng format 5.2, tên file `{source}_{scene}_{id:04d}.json`
- [x] Tính sẵn `xyz_gt_cam` và `xyz_gt_world` từ depth GT
- [x] `--visualize`: vẽ chấm lên RGB, lưu ảnh preview để **kiểm bằng mắt**
- [x] Báo cáo phân bố: bao nhiêu center/border, depth trung bình, số frame dùng
- [x] `--dry-run`: chỉ báo cáo phân bố, không ghi file
- [x] Cảnh báo khi kích thước `depth` ≠ `intrinsics` (trộn lẫn là sai số hệ thống → bỏ qua frame, không âm thầm dùng)
- [x] Chia quota đều cho mỗi scene, không dồn hết vào phòng đầu

**Điều kiện xong:**
- [x] 60 JSON đúng schema (dataset giả tạo **đúng format Replica thật**), không pixel nào nằm ngoài ảnh, không depth NaN
- [x] **Tự nhất quán toán:** chiếu ngược `xyz_gt_cam` → pixel, sai số **5.68e-14 px**;
      `xyz_gt_world` khớp `R @ xyz_cam + t` **sai số 0.00 m**; `RᵀR = I`, `det R = 1`;
      `Z == depth_gt_m`; depth trong [1.5, 6.0]; pixel trong khung 1200×680
- [x] Cân bằng vùng ảnh: **52% center** (31/29, tỉ lệ 0.94)
- [ ] ~~Mở ảnh preview **tự nhìn** xác nhận chấm nằm đúng trên vật thể~~ ← chờ làm trên data thật

**Ghi chú khi làm (Agent #5):**
- Logic cân bằng vùng ảnh dùng **xen kẽ**, không cắt cứng 2 center + 3 border. Bản cắt cứng
  cho 37% center (lệch 40/60); xen kẽ cho 52% (31/29).
- `pose` lưu dạng ma trận `R` + vector `t` **đã ở hệ OpenCV**, kèm `pose_convention` và
  `depth_scale` để tự kiểm chéo khi đọc lại. `run_eval.py` không phải parse quaternion.
- Ngưỡng ổ đĩa của script này là **ngưỡng nhẹ** (`min_c_gb=0.5`), khác `download_dataset.py`
  (nghiêm, 2 GB) — ghi 200 JSON không nguy hiểm bằng giải nén 11 GB. Xem mục 3.

> **Tip (đã dùng thật):** test được **trước khi tải dataset xong** bằng cách tự sinh dataset
> giả đúng format Replica: `results/frame%06d.jpg` + `results/depth%06d.png` với `scale=6553.5`
> + `traj.txt` 16 số. Toàn bộ DoD ở trên đã verify bằng dataset giả này.

---

### Phase 4 — `run_eval.py`

**Phụ thuộc:** Phase 1 + samples Phase 3

- [x] `--case A`: `depth_pred = depth_gt`, **đọc lại từ depth map trên đĩa** (không lấy từ JSON để tránh tự tròn)
- [x] `--case A` đọc lại **`traj.txt` từ scene** — lấy pose từ JSON thì quy ước trục sai sẽ không bao giờ bị phát hiện
- [x] `--case B --pose-mode clean`
- [x] `--case B --pose-mode noisy` — seed cố định, nhiễu tính theo `seed + frame` nên tái lập được
- [x] Ghi CSV **18 cột** đúng mục 5.3 (mục 5.3 liệt kê 18 cột; dòng "17 cột" ở đây là đếm sai)
- [x] So ở **hệ thế giới** (`xyz_world`) để bao gồm cả sai số đổi hệ
- [x] Model relative → gọi `align_scale_shift` **trước** khi so; model metric → so thẳng
- [x] Progress + log từng mẫu dạng `[B_clean] xxx (3/200): depth_err=0.19m err=0.21m`
- [x] Validate input: samples rỗng / JSON hỏng / thiếu trường / thiếu ảnh → báo rõ, không crash im lặng
- [x] **Cache depth map + depth dự đoán theo frame** (`FrameCache`) — 5 điểm/frame chỉ infer 1 lần
- [x] `--limit N` để thử nhanh; `--align affine|disparity`

**Điều kiện xong — chia làm 2 loại, đừng gộp:**

**A. Logic máy chạy đúng (verify được KHÔNG cần data thật):**

- [x] **Case A MAE = 0.0000 m** (exit 0) trên dataset giả đúng format Replica
- [x] **Xác minh độc lập, không qua `utils`:** đọc PNG bằng `cv2` thuần + intrinsics lấy tay
      từ yaml → `xyz_gt_world` khớp CSV tới **2.3e-07 m**. Xác nhận `scale=6553.5` chạy đúng:
      đọc sai hệ số sẽ ra 39.016 m thay vì 5.9535 m
- [x] ~~Phép đảo quy ước OpenGL→OpenCV cho `traj.txt` chính xác (test với `traj.txt` có xoay)~~ — **SAI, đã bịc
      ghi, khái điện.** Xem mục 5.1. Test ban đầu dùng `traj.txt` có xoay nhưng **vẫn ra đúng** — nên
      không chứng minh được quy ước đúng. Cách chứng minh thật: đo điểm back-project với bề mặt mesh
      Replica thật (0.005 m khi không đảo, 0.666 m khi đảo cột 1,2).
- [x] CSV đủ 18 cột, ghi đúng số dòng
- [x] `B_noisy MAE > B_clean MAE` — nhiễu pose tác dụng đúng chiều
- [x] Chẩn đoán affine vs disparity phân biệt đúng (unit-test với dữ liệu biết trước:
      `affine_err=0.000000` khi dữ liệu affine thuần, ngược lại khi disparity thuần)

**B. Đánh giá thật — CHƯA CÓ SỐ, đang chờ Replica tải xong:**

- [ ] ⬜ Case A trên **Replica thật** → GATE < 1 cm lặp lại (Phase 6)
- [ ] ⬜ Case B_clean / B_noisy trên **Replica thật**
- [ ] ⬜ Chọn chế độ `--align` dựa trên `affine_err` vs `disparity_err` **trên Replica**
- [ ] ⬜ Mở ảnh preview xem bằng mắt (Phase 3 DoD tương ứng)

> ⚠️ **KHÔNG ĐƯỢC ĐỌC NHẦM.** Mọi số Case B đã ghi ra cho tới giờ đều đo trên
> **dataset giả tự sinh** (depth ngẫu nhiên, ảnh đen có hình chữ nhật). Chúng chỉ chứng minh
> *máy chạy đúng*, **không nói gì về độ chính xác của DepthART**. **Hiện chưa có con số nào
> về DepthART trên ảnh thật.** Phase 6 mới sinh ra số đó.

**Ghi chú khi làm (Agent #5):**
- **Chẩn đoán chế độ căn thay vì tin docstring.** `depth_model.py` ghi depth relative là
  *affine-invariant* (`d = s·d_rel + b`), nhưng nếu model thực sự trả **disparity** thì phải căn
  `s/d + b`. Script vì vậy **đo cả hai** và in `affine_err` vs `disparity_err` từng frame để quyết
  định bằng số liệu.
- **Bắt 2 lỗi của chính mình khi test:** (1) `by_region` suy vùng từ `samples[:total]` trong khi
  `errs` chỉ chứa mẫu chạy được → có mẫu bị bỏ qua thì cặp chỉ số lệch, số thống kê sai; sửa bằng
  cách lưu `err_regions` đi cặp với `errs`. (2) Báo cáo căn lặp 5 lần cho cùng 1 frame; sửa bằng
  `set` frame đã báo.
- `utils.load_module_backend` phải **chuẩn hoá tên encoder**: DepthART chỉ nhận
  `MNV4-S`/`MNV4-M`/`S`/`B`/`L`, **không nhận** `mnv4s`. Rút gọn được ánh xạ sang tên chuẩn.
- **🔴 LỗI NGHIÊM TRỆNG khi chạy trên data thật (2026-10-09):** đã để
  `traj.txt` ra chiến đến 100% cho case A. Đã dùng mesh thật mới phát hiện: cách
  tôi đáo cột 1,2 chỉ **sai** (lệch 0.666 m so với 0.005 m). Đã sửa.
  Để đặnh hước cho file data **không suy ra được cho file data**.
  - Số MAE **của Case B_clean không đổi** (kiểm chứng `np.allclose` = True): vì cã gt và pred
    dùng cùng một `R`, mà `|R·(x_pred − x_gt)| = |x_pred − x_gt|` vì R trực giao.
  - Số **Case B_noisy đổi nhẹ** (MAE 0.3648 → 0.3678): nhiễu dịch `t` được cộng thẳng
    ở hệ thố giọc mà **không đi qua xoay**, nên tương tác của nó với phần dư
    đã xoay của `R`.
  - Toạ độ world trong JSON cũ lệch **trung vị 6.02 m** — đã sửa, cần sinh lại JSON.

- **Sửa lại chính mình:** ban đầu tick `[x]` cho toàn bộ DoD chỉ dựa trên dataset giả. Đã tách lại
  thành A (logic, đã chứng minh) và B (đánh giá thật, chưa có số) để không ai đọc nhầm.

**Ghi nhận (không phải kết quả):** trên dataset giả, `B_noisy` chỉ cao hơn `B_clean` 0.005 m vì
sai số depth 1.63 m áp đảo lấn át nhiễu pose 2 cm. Trên Replica thật sai số depth nhỏ hơn nhiều
nên tỉ lệ này dự kiến đảo lại — nhưng **đó là dự đoán, chưa đo**.

---

### Phase 5 — `eval_metrics.py`

**Phụ thuộc:** Phase 1 + kết quả Phase 4

- [x] Depth metrics: MAE, RMSE, **AbsRel**, δ<1.25 / 1.25² / 1.25³
- [x] Vị trí 3D: MAE, **Median, P95**, RMSE
- [x] Tỉ lệ dưới ngưỡng: `le_0_25m`, `le_0_50m` (quy ước lấy từ báo cáo tuần 8)
- [x] Tách trục: `by_axis` (mae_x/y/z)
- [x] Tách khoảng cách: `by_distance` 0-2m / 2-4m / 4-6m — **kèm `n` mỗi nhóm**
- [x] Tách vùng ảnh: `by_region` center / border — **kèm `n`**; nhóm nào `n` quá nhỏ thì ghi cảnh báo
- [x] `metrics_summary.json` đúng schema 5.4, một block cho mỗi case
- [x] `--plot`: biểu đồ MAE theo khoảng cách, center vs border (histogram sai số vẽ gộp trong panel 3)
- [x] Case A thiếu hoặc ≥ 1 cm → **cảnh báo to in ra + exit code 1**, không im lặng xuất số Case B

**Điều kiện xong:**
- [x] `metrics_summary.json` đủ trường cho cả 3 case (400 mẫu/case), chạy trên Replica thật
- [x] Bắt 2 lỗi hiển thị của chính mình: header cột khoảng cách in `0.0/2.0/4.0` thay vì
      `0-2m/2-4m/4-6m`; cảnh báo nhóm nhỏ in nguyên dict thay vì tên nhóm

**Phát hiện khi phân tích (2026-10-09, đo trên 400 mẫu B_noisy — KHÔNG phải dự đoán):**
- `center` MAE (0.353 m) cao hơn `border` (0.261 m), trái với lý thuyết khuếch đại hình học
  ở mục 1. Nguyên nhân đo được: **hiệu ứng chọn mẫu, không phải hình học**.
  Điểm giữa ảnh ở xa hơn trung bình (3.15 m vs 2.55 m) vì nhìn xuyên phòng/hành lang, còn điểm
  rìa ảnh hay rơi vào tường/đồ gần (tương quan err–depth trong nhóm center = 0.44).
  Hệ số khuếch đại hình học chỉ 1.05 (center) vs 1.22 (border) — quá nhỏ so với chênh lệch
  sai số depth (0.33 vs 0.20 m, chênh 60%).
- Ngay cả khi so **trong cùng khoảng cách**, center vẫn ở xa hơn (VD nhóm 2–4 m:
  center 2.88 m vs border 2.74 m), nên chênh lệch còn lại vẫn giải thích được bằng khoảng cách.
- ⚠️ Hệ quả cho metric: `by_region` hiện tại **bị lẫn với khoảng cách**. Muốn đo riêng hiệu ứng
  khuếch đại hình học phải so trong dải depth hẹp hoặc ghép cặp cùng depth — để dành cho
  `eval_metrics.py` bản sau, không sửa vội.

---

### Phase 6 — Chạy thật & đóng gói

**Phụ thuộc:** hết Phase 1–5

- [ ] Chạy trọn 6 lệnh trong `README.md` từ đầu tới cuối, **không bỏ bước nào**
- [ ] Ghi số thật vào changelog `AGENTS.md` (entry mới ở đầu mục CHANGELOG, giữ nguyên log cũ)
- [ ] Dán vào changelog: số Case A, số Case B_noisy, **giải thích ngắn vì sao tách center/border**
- [ ] Xoá cache/ảnh preview tạm nếu nặng (giữ `samples/` + `results/`)
- [ ] Xác nhận **không có file nào** được ghi ra ổ C

---

### 📌 Definition of Done (chung)

Simulator coi là xong khi **đủ cả 6 điều**:

1. [ ] 6 lệnh trong `README.md` chạy được từ đầu tới cuối, không lỗi
2. [ ] **Case A MAE < 0.01 m**
3. [ ] `metrics_summary.json` đủ trường mục 5.4 cho cả 3 case
4. [ ] Case B_noisy có số MAE/Median/P95 + tỉ lệ < 0.25 m để báo cáo
5. [ ] Changelog `AGENTS.md` đã ghi số thật
6. [ ] Không file nào nằm trên ổ C

---

### 🚨 Đừng làm

| Sai lầm | Vì sao |
|---|---|
| Báo cáo số Case B khi Case A chưa pass | Số sai do toán sai, không phải do model |
| Chỉ báo số `B_clean` | Pose chuẩn không phải thực tế; **số để báo cáo là `B_noisy`** |
| Căn scale+shift cho model metric | Metric đã có đơn vị mét, căn là hỏng |
| Dùng 1 nhóm `region` duy nhất | Che mất hiện tượng khuếch đại sai số ở rìa ảnh |
| Tự sửa `localizer_3d.py` / `depth_model.py` | Ngoài phạm vi simulator; phải hỏi user |
| Hard-code `C:\` hoặc đường dẫn tuyệt đối | Dùng `pathlib.Path` + tham số `--output` |
| Nhảy thẳng sang `eval_metrics.py` khi Case A lỗi | Sẽ ra số đẹp giả |

---

**Lịch sử file:** v1 (2026-10-09) viết theo hướng mesh/Open3D — đã bỏ. v2 (2026-10-09) hướng ảnh+depth có sẵn. v3 (2026-10-09) gộp task list (`TASKS.md`) vào đây để giảm số file phải đọc.
