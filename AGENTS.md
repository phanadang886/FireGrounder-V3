# 🤖 FireGrounder-V3 — Agent Collaboration Log

**Dự án:** Hệ thống phát hiện và định vị lửa 3D  
**Ngày tạo:** 2026-10-07  
**Mục đích file:** Ghi lại kiến trúc, quyết định thiết kế, và những gì **không được thay đổi** để các AI agent sau đọc và hiểu context mà không phụ thuộc vào conversation history.

---

## 🧭 ĐỌC FILE NÀY TRƯỚC — nhưng đừng đọc hết repo

**Luôn bắt buộc đọc:** mục `⚠️ PHẠM VI DỰ ÁN` + `🚫 KHÔNG ĐƯỢC THAY ĐỔI` (~30 dòng).
Không đọc 2 mục đó thì rất dễ sửa nhầm code legacy hoặc dataset đã đóng băng.

Sau đó **chỉ đọc phần bạn cần** theo bảng dưới — đừng đọc cả file này từ dòng 1 tới cuối.

| Việc bạn được giao | Chỉ cần đọc |
|---|---|
| **Code simulator** (`simulation/*.py`) | `simulation/SIM_CONTRACT.md` — mục **11 (Task list)** + mục **5, 7** đúng script bạn làm |
| Chạy simulator | `simulation/README.md` (76 dòng, 6 lệnh) |
| Sửa `localizer_3d.py` / `depth_model.py` / `metric_depth.py` | `📐 Thiết kế quan trọng` (quy ước depth, trục 3D) |
| Đụng tới model / dataset / config YOLO | `🚫 KHÔNG ĐƯỢC THAY ĐỔI` — **cấm** |
| Chạy được nhưng ra kết quả sai | `🔍 Debug hints` |
| Muốn biết *vì sao* một quyết định cũ chốt thế nào | `docs/DEPTH3D_HISTORY.md` |
| **Click kiểm tra depth trong phòng 3D (Replica preprocessed + mesh, Windows)** | **`models/Depth3D_Localizer/simulation/PLAN_2026-10-16.md`** (đã viết lại, ngắn). Code: `mesh_io.py`, `anchor.py`, `room_viewer.py` |
| Lửa giả / Phase 2 | `PLAN_2026-10-16.md` mục **6** + **8** (chờ lệnh user) |
| Open3D đọc PLY thiếu mặt / viewer lỗi API / Δ mesh lớn bất thường | `🔍 Debug hints` — 3 mục mới ở cuối phần EGL |
| Biết trạng thái hiện tại / việc gì đang dở | `📝 SUMMARY` ở cuối file này |
| Trước khi commit thay đổi | `🧪 Testing checklist` |

**File khác trong repo:**

| File | Vai trò |
|---|---|
| `models/Depth3D_Localizer/simulation/SIM_CONTRACT.md` | Spec kỹ thuật **+ task list** của simulator — nguồn chân lý khi code simulator |
| `models/Depth3D_Localizer/simulation/README.md` | 6 lệnh chạy simulator từ đầu tới cuối |
| `docs/DEPTH3D_HISTORY.md` | Changelog đầy đủ Agent #2–#4, có mục lục + bài học cũ |
| `docs/LEGACY_heatmap_history.md` | ⚠️ Nhánh heatmap **đã bị bỏ** — đừng làm theo |
| `docs/3D_DATASET_RECOMMENDATIONS.md` | Reference dataset 3D |

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

### ⚠️ PHẠM VI DỰ ÁN — ĐỌC TRƯỚC KHI LÀM VIỆC

**Chỉ 2 module này là phần đang chạy:**

| Module | File chính | Chức năng | Trạng thái |
|--------|-----------|-----------|-----------|
| **FireBase-YOLO26** | `firebase_yolo26n_pose.yaml` | Phát hiện lửa + keypoint BASE (gốc lửa) | ✅ Đã train, **KHÔNG SỬA** |
| **Depth3D_Localizer** | `depth_model.py`, `metric_depth.py`, `localizer_3d.py` | Ước lượng depth + chuyển 2D→3D | ✅ Hoàn tất, **đang chạy** |
| **Datasets** | `datasets/dataset_fire_pose/` | Dataset đã gán nhãn | ✅ Hoàn chỉnh, **KHÔNG SỬA** |

### Các entrypoint của Depth3D_Localizer

| Script | Vai trò |
|--------|---------|
| `pipeline.py` | **Điểm vào chính** — YOLO → depth → localize → annotate (ảnh/video/webcam) |
| `depth_model.py` | Backend relative depth (DepthART → DAv2 → DAv1) |
| `metric_depth.py` | Backend metric depth (đơn vị mét, cần intrinsics) |
| `localizer_3d.py` | Core 3D logic + `CameraIntrinsics` + `FireLocation3D` |
| `calibrate_camera.py` | Lấy intrinsics thật (chessboard) |
| `verify_calibration.py` | Kiểm chứng calibration bằng dữ liệu 3D tổng hợp |
| `benchmark.py` | Đo hiệu năng |
| `quick_test.py` | Unit test (37 assertions, không cần model) |
| `install_depthart.py` | Cài repo + tải checkpoint (verify SHA256) |

**Tất cả model/module khác là THỬ NGHIỆM CŨ — KHÔNG ĐỤNG VÀO:**

| Thư mục | Nội dung | Trạng thái |
|---------|----------|-----------|
| `models/FireGrounder_V3/` | Heatmap single-keypoint model (`firegrounder_v3.py`, `inference_v3.py`, `notebook4_v3_fpn.py`) | 🗄️ **Legacy — thử nghiệm cũ, KHÔNG SỬA** |
| `models/ROI_Refiner/` | `narrow_localizer.py`, `train_roi_localizer.py` | 🗄️ **Legacy — thử nghiệm cũ, KHÔNG SỬA** |
| `models/YOLO26_Pose/` | Thử nghiệm pose cũ | 🗄️ **Legacy — thử nghiệm cũ, KHÔNG SỬA** |

**Quy tắc bắt buộc cho agent:**
1. ❌ **KHÔNG** sửa, xoá, di chuyển, hoặc refactor code trong 3 thư mục legacy trên
2. ❌ **KHÔNG** import chúng vào pipeline mới
3. ❌ **KHÔNG** xoá chỉ vì "thấy không dùng" — chúng là tài liệu lịch sử thử nghiệm
4. ✅ Nếu cần tham khảo ý tưởng cũ → **đọc** để học, nhưng **viết code mới** trong `models/Depth3D_Localizer/`
5. ✅ Dữ liệu đầu ra của chúng **không liên quan** tới metrics hiện tại

**Lưu ý về `docs/`:** `docs/README.md` và `docs/research_gap.md` mô tả nhánh **FireGrounder_V3 (heatmap)** — đó là kiến trúc **đã bị bỏ**. Dùng chúng làm **lịch sử/background** thôi, đừng làm theo. Kiến trúc đang chạy là **YOLO26-pose + Depth3D** như trên.

---

## 🚫 KHÔNG ĐƯỢC THAY ĐỔI

### 0. Các module thử nghiệm cũ (legacy)
- **Path:** `models/FireGrounder_V3/`, `models/ROI_Refiner/`, `models/YOLO26_Pose/`
- **Lý do:** Đã thử và bị bỏ; chỉ giữ lại làm tham khảo lịch sử
- **Hành động:** KHÔNG sửa / KHÔNG xoá / KHÔNG import vào pipeline mới

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

*Quy ước: file này chỉ giữ các entry **mới nhất**. Lịch sử đầy đủ (Agent #2–#4, 6 entry) nằm ở
`docs/DEPTH3D_HISTORY.md` — đọc khi cần biết vì sao một quyết định cũ được chốt, hoặc tìm lỗi đã gặp.*

### [2026-10-16] — ĐỔI HƯỚNG: bỏ WSL2/Habitat/máy B, làm viewer click-to-check trên mesh có sẵn — AI Agent #6

**Đã làm (Windows, 1 máy, không tải thêm):** `mesh_io.py` (parser PLY quad tay — Open3D nuốt 99,1 % mặt),
`anchor.py` (pixel→3D GT + raycast mesh + Δ + phân loại cạnh che bằng `plane_residual` 11×11, ngưỡng 5 cm),
`room_viewer.py` (2 cửa sổ). Cả 3 `--selftest` PASS; `room_viewer.py --smoke 8` chạy cửa sổ thật.
`--smoke` bắt 3 lỗi API Open3D 0.20 mà selftest không thấy (`set_up_vector`, `remove_geometry(tên)`,
xoá nhầm mesh phòng). Mesh check = bắt lỗi code, không phải phép đo (room0 sạch: median 2,04 / p95 6,76 mm).
`PLAN_2026-10-16.md` đã viết lại ngắn. **Chưa có:** `fire_sprite.py`, Phase 2.
Mục dưới đây (EGL/máy B) là **lịch sử, đã bỏ**.

---

### [2026-10-16] — (LỊCH SỬ, ĐÃ BỎ) Kiểm chứng nguồn cho plan Replica gốc: rủi ro EGL — AI Agent #6

**Bối cảnh:** User yêu cầu "kiểm tra lại thông tin chi tiết trên mọi nguồn để confirm". Đây là bước
kiểm chứng cho `simulation/PLAN_2026-10-16.md` — plan dựng trình mô phỏng mới trên **Replica gốc**
ở **máy B (Windows 10 + RTX 3060)**, phục vụ Phase 1 (click pixel → so depth) và Phase 2 (lửa giả).

#### 🔴 Lỗi đã sửa

| # | Đã ghi sai | Thực tế | Hậu quả nếu để nguyên |
|---|---|---|---|
| 1 | `conda create -n habitat python=3.12 cmake=3.27` | **`python=3.9 cmake=3.14.0`** — gói conda **ổn định 0.3.3** ràng buộc `>=3.9,<3.10` | Conda giải không ra / cài hỏng env |

Nguyên nhân: đọc README **nhánh `main`** (chưa phát hành) rồi tưởng là bản ổn định.

#### 🔴 Rủi ro mới phát hiện — **EGL trên WSL2** (chưa ai vượt qua)

Habitat-Sim headless **cần EGL của NVIDIA** khớp thiết bị CUDA. Bằng chứng đã tra:
- Tài liệu CUDA-on-WSL của NVIDIA: *"OpenGL-CUDA Interop is not yet supported. Applications relying on OpenGL will not work."*
- Nền tảng habitat-sim hỗ trợ **chỉ `linux-64` và `macOS-arm64`** — **không Windows, không WSL**.
- GitHub issue #127: người dùng báo lỗi GPU trên WSL2 — **cả headless lẫn không headless**.
- Lỗi điển hình: `EGL_BAD_PARAMETER` · `unable to find CUDA device 0 among N EGL devices` · `EGL device 0, CUDA device 0 is not readable`.

→ **Không có tài liệu nào xác nhận habitat-sim chạy được trên WSL2.** Đã thêm **⛔ CỔNG EGL**
(`PLAN` mục 4.3b) chặn **trước khi tải 34 GB Replica**: tải `habitat_test_scenes` (~trăm MB) →
render thử → **chỉ khi PASS mới được tải dataset**.

#### ✅ Xác minh đúng

- `wsl --install` cần Win10 build ≥ **19041** (máy B: **19045**) — Microsoft Learn.
- WSL2 cần bật **cả** `VirtualMachinePlatform` + `Microsoft-Windows-Subsystem-Linux`, **buộc restart**.
- **WSLg (GUI) cần Windows 11 build 22000+** → Win10 **không có GUI**, buộc render headless.
- `--location` cần **WSL ≥ 2.4.4**. `wsl.exe` trên máy B **chưa có** → phải cập nhật trước.
- **Máy B CÓ quyền admin** (đã mở được PowerShell Administrator) — khác giả định trước đó.
- **Replica**: 18 scene; `mesh.ply` có **màu đỉnh**, `textures/` là **Ptex HDR**.
- **Blender**: `Blender Internal` (engine duy nhất từng hỗ trợ Ptex) **đã xoá từ 2.80**; Cycles **chưa
  bao giờ** hỗ trợ Ptex → Blender chỉ render được màu đỉnh. Replica tự nói "not recommended".
- **UltraViewer**: mới nhất **6.6.127**; blog chính thức nói restart từ xa cần **cả 2 máy ≥ 6.2.79**.
  ⚠️ **Chưa xác minh được:** có cần bản quyền không, và có điều khiển được **màn hình login** không.

#### 📄 Đã cập nhật

- `simulation/PLAN_2026-10-16.md`: mục **4.2** (bảng các đường đi + bằng chứng EGL + bảng sự thật đã
  xác minh), mục **4.2b** (quy trình an toàn restart: UltraViewer mật khẩu riêng + auto-login Windows
  + cảnh báo `netplwiz` bị ẩn), mục **4.3** (DISM bật feature + `python=3.9`), mục **4.3b** (⛔ cổng
  EGL), mục **4.7** (Blender + sprite 2D + Open3D), checklist mục 7 và bẫy mục 8.
- Thêm đường **ghép sprite 2D lên Replica preprocessed** (mục 4.7.2) — không cần mesh/WSL/admin;
  BASE 3D GT = back-project **pixel neo** bằng depth GT.

**Chưa có:** bất kỳ số liệu nào ở máy B. Chưa cài WSL2, chưa tải gì, chưa bật feature.

---

### [2026-10-09] — Notebook so 4 model depth, chưa có số — AI Agent #5

**Bối cảnh:** User muốn tự bấm so DepthART MNV4-S, DepthART MNV4-M-slim-SPF, YOLO26n-depth
và Depth Anything V2 Small trên cùng bộ điểm Replica. Chưa chạy chấm.

**Đã làm:**
- `simulation/compare_depth_models.ipynb` (10 ô). Ô 4 mặc định tắt (`RUN = False`,
  `LIMIT_SAMPLES = 40`). Kết quả sẽ ghi `D:\replica_sim\results\cmp_<tên>\per_sample.csv`.
  Model nào không nạp được thì bỏ qua và in lỗi — không điền số.
- Ô 3 tải phần còn thiếu về ổ D: `depthart_relative_mnv4m_slim_224.pth` (~25 MB, Hugging Face
  `Fengxue93/DepthART`, **chưa có trong máy**) và `yolo26n-depth.pt` (cần `ultralytics`,
  cũng chưa cài). Depth Anything V2 Small tải lúc chạy.
- Căn **affine trên output gốc**, giống `run_eval.py`: `mét = scale × dự_đoán + shift`.
  Không đảo `1/z` trước. Scale âm nghĩa là model để giá trị cao = gần; ô 4 in cảnh báo.

**Quyết định:** YOLO26n-depth theo tài liệu Ultralytics (`result.depth.data`) là mét, nhưng
head thật là log-depth và thang mét tách riêng (`exp(a·log d + b)`). Notebook vẫn căn affine
như 3 model kia để so công bằng. **Không** được báo số đó là mét thật không cần GT.

**Chưa có:** mọi số so sánh. MNV4-S 0.3119 m vẫn là số relative duy nhất đã chấm (400 điểm,
pose nhiễu). Chưa chấm TinyViM-S dù weight `depthart_relative_s_224.pth` đã có trong máy.

---

### [2026-10-09] — Notebook tự kiểm số + sửa nạp bản metric — AI Agent #5

**Bối cảnh:** User muốn tự bấm xem metric, không chỉ đọc số trong chat.

**Đã làm:**
- `simulation/check_metrics.ipynb`: đọc CSV có sẵn, tính lại bằng đúng hàm `eval_metrics.py`,
  đối chiếu MAE tay với MAE script (đã chạy thử: `case_B_noisy` khớp **0.3119 m**).
  Ô chấm bản metric mặc định tắt (`RUN_METRIC = False`) — user tự bật.
- Sửa `run_eval.py`: bản metric bắt buộc truyền intrinsics từ sample (trước đó crash
  `No module named localizer_3d` vì import trước khi chèn `sys.path`). Log không còn in
  `affine_err=nan` cho model đã có đơn vị mét.
- `eval_metrics.py`: case có chữ `metric` trong tên được ghi chú "so trực tiếp, không căn",
  không bị gán nhầm là bản relative.

**Chưa có:** số 400 mẫu của bản metric. Thư mục thử 2 mẫu đã xoá để khỏi lẫn với kết quả.

---

### [2026-10-09] — Giải thích được điểm lạ center/border + khóa scope tuần này — AI Agent #5

**Bối cảnh:** Số liệu thật cho thấy `center` MAE 0.353 m cao hơn `border` 0.261 m, trái với lý thuyết
khuếch đại hình học ở `SIM_CONTRACT.md` mục 1. User yêu cầu giải thích.

**Kết quả (đo trên 400 mẫu B_noisy — KHÔNG phải dự đoán):**
- Điểm giữa ảnh ở **xa hơn** trung bình: 3.15 m vs 2.55 m. Vì chọn điểm ngẫu nhiên: giữa ảnh
  thường nhìn xuyên phòng/hành lang (xa), rìa ảnh hay rơi vào tường/đồ gần.
- Sai số depth của nhóm center lớn hơn 60% (0.33 vs 0.20 m), trong khi hệ số khuếch đại hình học
  chỉ chênh 1.05 vs 1.22 (khoảng 16%). **Sai số depth lấn át hoàn toàn yếu tố hình học.**
- Ngay cả trong cùng nhóm khoảng cách, center vẫn ở xa hơn (VD nhóm 2–4 m: 2.88 vs 2.74 m).
- Hiện tượng xuất hiện ở 6/8 scene (room0/room1/room2 hai nhóm gần bằng nhau) — không phải do
  1–2 phòng lạ.
- ⚠️ Hệ quả: metric `by_region` hiện tại **bị lẫn với khoảng cách**. Muốn đo riêng hiệu ứng hình học
  phải so trong dải depth hẹp — đã ghi vào contract, để dành bản sau.

**Scope tuần này (user chốt 2026-10-09):** chỉ chấm các model đo độ sâu bằng simulator.
- ✅ Làm: tick Phase 5, điều tra center/border, viết `download_dataset.py`, chấm bản metric,
  chấm chéo NYUv2, so DepthART với Depth Anything V1/V2 (backend đã có sẵn trong repo).
- ⏸️ Tạm dừng: gắn YOLO vào pipeline (cần `best.pt` + `ultralytics` + ảnh bàn cờ — việc của user).
- ⏸️ Tạm dừng: tối ưu tốc độ, ảnh lửa thật (không dataset công khai nào có).

---

### [2026-10-09] — Xác minh link Replica thật, sửa 4 sai sót format, code `utils.py` + `sample_points.py` — AI Agent #5

**Bối cảnh:** User hỏi "nếu Phase 2 không làm được thì đỡ mất công làm cái khác không?" — câu hỏi
chạm đúng chỗ tôi từng đề xuất sai. Trả lời bằng cách **kiểm tra link + đọc source trước khi code tiếp**.

#### 🔴 4 sai sót trong contract đã phát hiện và sửa

Tôi viết spec Replica theo trí nhớ. Đọc source thật (`cvg/nice-slam`) thì sai 4 chỗ —
**cái nào cũng đủ làm hỏng kết quả âm thầm, không báo lỗi**:

| # | Tôi đã viết | Thực tế | Hậu quả nếu để nguyên |
|---|---|---|---|
| 1 | depth PNG16 chia **1000** | **`png_depth_scale = 6553.5`** | **mọi số sai 6.55 lần** |
| 2 | `rgb/0000.png` + `depth/0000.png` | **`results/frame%06d.jpg`** + **`results/depth%06d.png`** (cùng thư mục) | không tìm được file nào |
| 3 | scene `office_0`, có `intrinsics.txt` | tên **`office0`**, **không có file intrinsics** | sai tên scene, thiếu dữ liệu |
| 4 | pose lấy thẳng từ file | `traj.txt` lưu theo **quy ước OpenGL** | trục Y và Z bị đảo → toạ độ 3D sai |

Nguồn xác minh: `cvg/nice-slam` → `scripts/download_replica.sh`, `configs/Replica/replica.yaml`,
`src/utils/datasets.py` (class `Replica`). Người dùng nghi ngờ đúng chỗ: kiểm tra 2 phút tránh được
việc làm lại từ đầu.

#### ✅ Link tải đã xác minh (HTTP 200)

```
https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip   11.59 GB   cập nhật 2022-04-05
```

Máy chủ **ETH Zurich (CVG)** — link chính thức của nhóm tác giả, không phải link cộng đồng nên ổn định.
Link dự phòng: Baidu caiyun (mật khẩu `v3fY`), ghi trong `scripts/download_replica.sh`.

#### 🐛 Lỗi môi trường: RAM, không phải lỗi code

`import torch` fail với `OpenBLAS error: Memory allocation still failed`. Nguyên nhân: máy có
15.7 GB nhưng **chỉ còn 0.93 GB trống** (Chrome + Docker + IDE); OpenBLAS cấp buffer cho 8 luồng rồi
**abort cứng tiến trình** — `try/except` của Python không bắt được (đã thử, không ăn).
→ `utils.py` đặt `OPENBLAS_NUM_THREADS=1` **trước** khi import numpy (chỉ OPENBLAS, **không** đụng
OMP/MKL để khỏi làm chậm torch ở Phase 4). Import `utils` giờ 159 ms.

#### 📦 Đã code + verify (không tick mục nào chưa kiểm chứng)

| File | Kết quả |
|---|---|
| `simulation/utils.py` (9 hàm) | **22/22 assertions PASS**, exit 0, round-trip ≤ 5.6e-17 m |
| `simulation/sample_points.py` | 60 JSON đúng schema; chiếu ngược `xyz_gt_cam` → pixel sai số **5.68e-14 px**; `xyz_gt_world` khớp `R@cam+t` **sai số 0.00 m**; cân bằng vùng ảnh **52% center**; cùng seed → SHA256 file giống hệt |

**Sửa thêm trong `utils.py`:** `load_depth_map` giờ **bắt buộc** truyền `scale` cho depth nguyên
(cố tình *không* đoán — đoán sai cho số sai 6.55 lần mà không báo lỗi); `parse_pose_line` có tham số
`opengl_to_opencv`; thêm `replica_intrinsics()` + hằng số `REPLICA_PNG_DEPTH_SCALE`, `REPLICA_CAM`.

**Quyết định thiết kế:** `align_scale_shift` trả `ScaleShift(scale, shift)` chứ không phải mảng depth
đã căn — `run_eval.py` cần áp hệ số lên **cả bản đồ**, không chỉ pixel dùng để fit.

**Quyết định thiết kế:** phân tầng ngưỡng ổ đĩa. `download_dataset.py` dùng ngưỡng nghiêm
(`min_c_gb=2`) vì tải + giải nén 11 GB; `sample_points.py` dùng ngưỡng nhẹ (`min_c_gb=0.5`) vì ghi
200 JSON vài MB. Ổ C dao động 0.8–2.6 GB tuỳ Chrome/Docker — ngưỡng 2 GB cho mọi thứ sẽ chặn oan.

**Bài học:** *đừng viết spec dataset từ trí nhớ.* Chi tiết như hệ số depth và quy ước trục nằm ở
`configs/*.yaml` và code loader, **không** nằm ở README. 2 phút đọc source tiết kiệm hàng giờ code lại.

---

### [2026-10-09] — Chạy thật trên Replica: tìm & sửa lỗi quy ước pose, có số liệu đầu tiên — AI Agent #5

**Bối cảnh:** Tải xong `Replica.zip` 11.59 GB (HTTP 200, CRC sạch, 32,017 file), giải nén 8 scene
× 2000 frame. Chạy pipeline lần đầu trên **data thật**.

#### 🔴🔴 LỖI NGHIÊM TRỌNG — quy ước pose tôi đọc sai

Tôi đọc `cvg/nice-slam/src/utils/datasets.py`, thấy `c2w[:3,1] *= -1; c2w[:3,2] *= -1`, và kết luận
`traj.txt` lưu theo hệ OpenGL. **Sai.** Dòng đó là để chuyển hệ cho **renderer OpenGL của họ**, không
phải để đọc file `traj.txt` đã lưu.

**Cách phát hiện — và đây là điểm mấu chốt:** Case A chạy PASS (MAE 0.0000 m) nhưng **Case A không
thể bắt được lỗi này** — vì gt và pred đều đi qua cùng một `R` nên tự triệt tiêu, đúng/sai đều ra 0.
Phải dùng **mesh Replica thật** làm chuẩn độc lập: depth GT được render từ chính mesh đó, nên điểm
back-project đúng quy ước phải **nằm sát bề mặt mesh**. Đo khoảng cách median tới đỉnh gần nhất
(n=96, cả 8 scene):

| Cách xử lý | Khoảng cách median |
|---|---|
| **Không đảo (đúng)** | **0.005 m** — đúng độ phân giải mesh |
| Đảo cột 1,2 (cách tôi làm) | 0.666 m |
| Đảo cả 3 cột | 0.894 m |

**Ảnh hưởng (đo được, không đoán):**
- Toạ độ world trong JSON lệch trung vị **6.02 m** → đã sửa, đã sinh lại toàn bộ sample.
- Số Case **B_clean không đổi** (kiểm chứng `np.allclose` = True): gt và pred dùng cùng `R` nên
  `|R·(x_pred − x_gt)| = |x_pred − x_gt|`, `R` trực giao.
- Số Case **B_noisy CÓ đổi** (MAE 0.3648 → 0.3678 trên 40 mẫu): nhiễu dịch chuyển `δ` được cộng
  thẳng ở hệ thế giới **không đi qua xoay**, nên tương tác của nó với phần dư đã xoay phụ thuộc `R`.

#### 📊 SỐ LIỆU THẬT — Replica preprocessed, 400 điểm, 8 scene, DepthART-MNV4-S (relative, 224px)

| Case | n | MAE | Median | P95 | <0.25 m | AbsRel |
|---|---|---|---|---|---|---|
| A (kiểm tra toán) | 400 | 0.0000 m | 0.000 | 0.000 | 100% | 0.000 |
| B_clean (pose chuẩn) | 400 | 0.3012 m | 0.2374 | 0.7787 | 52.2% | 0.096 |
| **B_noisy (số để báo cáo)** | 400 | **0.3119 m** | 0.2459 | 0.7865 | **50.7%** | 0.096 |

Tách theo khoảng cách (B_noisy): 0–2 m **0.255 m** (n=71) · 2–4 m **0.277 m** (n=273) ·
4–6 m **0.553 m** (n=56). Sai số tăng rõ theo khoảng cách.

⚠️ **Một quan sát trái với dự đoán của tôi:** `center` MAE 0.353 m **cao hơn** `border` 0.261 m
(tỉ lệ 1.35), trong khi contract mục 1 dự đoán ngược lại (rìa ảnh xa tâm nên bị khuếch đại).
n=222/178 đủ lớn để không phải nhiễu thống kê, nhưng **tôi chưa giải thích được** và không bịa lý do.
Giả thuyết đáng kiểm sau: sai số depth lấn át yếu tố khuếch đại hình học.

⚠️ **Giới hạn:** chỉ mới có **ảnh trong nhà, không có ảnh lửa**. Chưa có con số nào về việc DepthART
nhận ra lửa thật hay không — đó vẫn là khoảng trống không dataset công khai nào lấp được.

#### ✅ Thêm được
- `eval_metrics.py` (Phase 5): AbsRel, δ<1.25ᵏ, tách trục/khoảng cách/vùng ảnh kèm `n`, cảnh báo
  nhóm quá nhỏ, `--plot`, **exit code 1 nếu GATE fail**.
- `replica_intrinsics(data_dir)` giờ **đọc `cam_params.json` của dataset** thay vì hardcode.
- Script giải nén tự viết (chống zip-slip, chặn ghi ổ C, bỏ `.ipynb_checkpoints`).

#### 💡 Bài học (cho agent sau)

1. **Case A trùng lặp không phải bằng chứng.** Nó chỉ chứng minh hai đoạn code thống nhất với nhau.
   Muốn kiểm chứng *đúng* phải có **nguồn độc lập** — ở đây là mesh thật.
2. **Đừng chuyển đặc tính từ code loader sang file data.** Cùng một repo nhưng `traj.txt` trên đĩa và
   biến trong code xử lý là hai thứ khác nhau.
3. **Test bằng dữ liệu giả dễ tự thỏa mãn.** Test "traj.txt có xoay" của tôi vẫn ra kết quả đúng
   dù quy ước sai — vì test không có tín hiệu nào phân biệt được.
4. **Số liệu chạy trên data giả không được tick cùng trạng thái với data thật.** Phải tách rõ.

---


### [2026-10-09] — Đổi hướng simulator: chấm depth bằng ảnh+depth có sẵn, bỏ mesh 31 GB — AI Agent #5

**Bối cảnh:** Kiểm tra thực tế cho thấy mesh Replica gốc cần tải 31.53 GB + ~70 GB lúc giải nén
(vượt ổ D 34 GB), Replica SDK/Habitat-Sim không chạy trên Windows, và quan trọng nhất:
để chấm DepthART + toán 2D→3D **không cần mesh, không cần đặt vật 3D, không cần detector** —
chỉ cần cặp ảnh RGB + depth thật (mét) kèm intrinsics + pose.

**Quyết định mới (thay toàn bộ plan mesh/Open3D trước đó):**
1. **Dataset:** Replica preprocessed (ưu tiên, nhẹ vài GB) + NYUv2 (dự phòng, chuẩn so với paper).
2. **Phương pháp:** chọn pixel có depth GT hợp lệ → tính tọa độ 3D chuẩn từ depth GT →
   DepthART đoán depth tại pixel đó → tọa độ 3D dự đoán → sai số Euclid. Metric tách theo
   khoảng cách và vùng ảnh giữa/rìa (sai số vị trí bị khuếch đại ở rìa ảnh).
3. **3 case:** A (depth GT — kiểm tra toán, kỳ vọng < 1 cm), B_clean (DepthART + pose chuẩn),
   B_noisy (DepthART + pose nhiễu σ=2 cm/0.5° — số gần thực tế, mới là số để báo cáo).
4. **Đánh giá báo cáo của bạn Tuấn Dũng (tuần 8):** không lấy code (nhánh ROI thuộc legacy cấm đụng;
   ray casting cần mesh; Scene-MLP sai 1.18 m không dùng được). Chỉ lấy quy ước metric
   (MAE/Median/P95/tỉ lệ < 0.25 m) và protocol pose-nhiễu.

**Dọn trùng lặp (user yêu cầu):** plan chi tiết Phase 0–6 cũ (~370 dòng, viết theo hướng mesh) đã xóa
khỏi file này vì trùng vai trò với spec. Sau đó user phát hiện phải đọc 4 file mới xong 1 việc
(1166 dòng) → gọn tiếp lần 2. Cấu trúc tài liệu cuối cùng:

| File | Vai trò | ~Dòng |
|---|---|---|
| `AGENTS.md` (file này) | Quy tắc cấm + trạng thái hiện tại + changelog **entry mới nhất** | 380 |
| `docs/DEPTH3D_HISTORY.md` | Changelog đầy đủ Agent #2–#4 + mục lục + bài học cũ | 240 |
| `simulation/SIM_CONTRACT.md` | Spec kỹ thuật (mục 1–10) **+ task list tick** (mục 11) | 490 |
| `simulation/README.md` | 6 lệnh chạy simulator | 76 |

- Thêm bảng **🧭 định tuyến** ở đầu file này: "việc X → đọc file Y mục Z" để agent mới không phải đọc hết repo.
- `TASKS.md` đã **xoá** (gộp vào `SIM_CONTRACT.md` mục 11) — spec và task list cùng đối tượng đọc.
- Nguyên tắc "không xóa log cũ" vẫn giữ: 6 entry cũ nằm đầy đủ trong `docs/DEPTH3D_HISTORY.md`.

**Quy tắc ổ đĩa (ghi cứng):** ổ C chỉ còn 0.8 GB — **CẤM ghi mọi file dữ liệu vào ổ C**.
Mọi thứ của simulator nằm trên `D:\replica_sim\`. Script nào cũng phải check disk trước khi chạy.

---

## 📐 Thiết kế quan trọng

### ⚠️ Depth output convention — HAI QUY ƯỚC NGƯỢC NHAU

Đây là nguồn lỗi nguy hiểm nhất của module:

| | Relative (`depth_model.py`) | Metric (`metric_depth.py`) |
|---|---|---|
| Bản chất | affine-invariant | **khoảng cách thật (mét)** |
| Giá trị CAO | = **gần** | = **xa** |
| Camera-aware | không | **có** (intrinsics là input model) |
| Cho `xyz_metric` | ❌ | ✅ |
| Checkpoint | `relative/*.pth` | `metric/*.pth` |

- **DepthART:** relative, giá trị **cao** = **gần**
- **Depth Anything V1/V2:** tương tự, disparity-like, **cao** = **gần**
- **DepthART metric:** **thấp** = **gần** (khoảng cách tăng dần khi ra xa)
- **FireLocalizer3D:** `depth_normalized = 1.0` = gần nhất, `0.0` = xa nhất (đã chuẩn hoá cho cả 2)
  → `localize()` tự đảo chiều khi `metric_depth=True`

**Zone logic:**
- `depth_normalized > 0.70` → NEAR
- `0.30 < depth_normalized < 0.70` → MID
- `depth_normalized < 0.30` → FAR

**Visualize:** mỗi backend có `colorize()` riêng đã xử lý đúng chiều. Đừng dùng chung.

### Camera intrinsics
- **Ước lượng** (`CameraIntrinsics.from_fov`): `fx = W / (2 * tan(45°))` — chỉ đủ cho zone/xyz_normalized
- **Thật** (`calibrate_camera.py`): chessboard → RMSE < 1px → dùng được cho metric depth
- File JSON có `source` + `is_calibrated` để phân biệt
- **Không dùng `from_fov` cho metric depth nếu cần chính xác** — metric model dùng intrinsics làm input, sai intrinsics → sai số mét ngay

### Metric depth cần gì
1. Checkpoint `checkpoints/metric/depthart_metric_{indoor,outdoor}_{s,b,l}_448.pth`
2. Intrinsics thật tương ứng ảnh input
3. Công thức: `Z = depth`, `X = (u - cx)·Z/fx`, `Y = (v - cy)·Z/fy`

### Quy ước trục 3D
- **X**: dương = sang phải
- **Y**: dương = **xuống dưới** (khớp hệ toạ độ ảnh)
- **Z**: dương = ra xa camera

### Hiệu năng (đo thật, i5-1145G7 CPU không CUDA)
- Relative depth: **~74 ms/frame** → **13.6 FPS**
- **Depth chiếm 99% thời gian** — nút thắt, không phải YOLO
- Metric depth: **~2.3 s/frame** (TinyViM/Mamba reference scan) — chỉ dùng ảnh tĩnh

---

## 🎯 TODO — Ưu tiên cao

### Ngay lập tức — ĐÃ XONG ✅
- [x] Tạo file AGENTS.md này
- [x] Sửa `depth_model.py` tích hợp API DepthART chính thức
- [x] Sửa `install_depthart.py` tải đúng checkpoint MobileNetV4-S
- [x] Sửa `localizer_3d.py` loại bỏ "mét giả"
- [x] Test `depth3d_demo.ipynb` với backend mới
- [x] Research và đề xuất dataset 3D phù hợp (xem `docs/3D_DATASET_RECOMMENDATIONS.md`)
- [x] Viết script video inference end-to-end → `pipeline.py`
- [x] Đo hiệu năng pipeline hoàn chỉnh → `benchmark.py` (**73.8ms/frame, 13.6 FPS CPU**)
- [x] Thêm calibration tool lấy camera intrinsics → `calibrate_camera.py`
- [x] Tải DepthART metric indoor checkpoint (SHA256 đã verify)
- [x] Triển khai metric depth pipeline → `metric_depth.py` + `xyz_metric` thật
- [x] Kiểm chứng calibration → `verify_calibration.py` (khôi phục chính xác 0.000%)

### Cần USER cung cấp — BLOCKED ⛔
- [ ] **Copy `best.pt` FireBase-YOLO26** vào `models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/` (gitignore `*.pt`)
- [ ] **`pip install ultralytics`** (chưa cài trên máy dev)
- [ ] **Chụp 20-30 ảnh bàn cờ** → `python calibrate_camera.py --images ... --drop-outliers` để có intrinsics thật
- [ ] **Test với camera thật** — cần phần cứng, không làm được trong môi trường code
- [ ] **Ra lệnh Phase 2** (lửa giả A/B/C) — xem `PLAN_2026-10-16.md` mục 6

### Tương lai
- [x] Code simulator theo `models/Depth3D_Localizer/simulation/SIM_CONTRACT.md` — **5/6 phase xong**, Phase 6 còn dở (xem 📝 SUMMARY)
- [ ] Đánh giá độ chính xác depth trên ảnh lửa thật (cần dataset GT: Replica/ScanNet++)
- [ ] Tối ưu tốc độ depth (nút thắt 99% thời gian) — thử ONNX / TensorRT / giảm resolution
- [ ] Tách thread YOLO + depth để tăng FPS
- [ ] Batch nhiều keypoint thay vì gọi `infer_pil` nhiều lần

### Đã loại bỏ
- ~~Tải DepthART metric outdoor~~ (chưa cần — dùng indoor cho môi trường trong nhà)
- ~~Replica dataset render pipeline~~ (cần 10GB + habitat-sim; ưu tiên sau)
  → Đã thử mở lại 2026-10-16 (máy B/WSL2/EGL) rồi **bỏ hẳn**: dùng mesh có sẵn trong Replica preprocessed.

---

## 🧪 Testing checklist

Chạy trước khi commit thay đổi quan trọng:

```bash
cd models/Depth3D_Localizer

# 1. Unit test (không cần model, ~1s)
python quick_test.py

# 2. Kiểm chứng calibration bằng dữ liệu 3D tổng hợp (~30s)
python verify_calibration.py

# 3. Đo hiệu năng (tải model ~4s + đo)
python benchmark.py --num 20 --no-detector

# 4. Load cả 2 backend
python -c "
import sys; sys.path.insert(0,'.')
from depth_model import DepthBackend
from metric_depth import MetricDepthBackend
from localizer_3d import CameraIntrinsics
b=DepthBackend(prefer='auto'); b.load(); print('relative:', b.model_name)
i=CameraIntrinsics.from_fov(640,480)
m=MetricDepthBackend(intrinsics=i,domain='indoor',encoder='s'); m.load()
print('metric:', m.model_name)"

# 5. Pipeline không cần YOLO (test được khi chưa có best.pt)
python pipeline.py --image <một ảnh trong datasets/fire-samples> \
    --keypoints 112,150 --out /tmp/out

# 6. Nếu đã cài ultralytics + có best.pt → test detector thật
python pipeline.py --image <ảnh> --out /tmp/out
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
ultralytics>=8.3.0    # FireBase-YOLO26 (chỉ cần cho detector)
```

> **Trạng thái máy dev (2026-10-09):** đã cài tất cả **trừ `ultralytics`**.
> Phần depth/localize chạy đầy đủ; detector cần `pip install ultralytics` + `best.pt`.

---

## 🔍 Debug hints

### Lỗi "No module named 'relative.model'"
- **Nguyên nhân:** API cũ không tồn tại
- **Fix:** Dùng `load_relative_model()` từ `relative/models.py` (đã xử lý trong `depth_model.py`)

### Zone luôn NEAR hoặc FAR
- Điều chỉnh `near_thresh` / `far_thresh` trong `FireLocalizer3D()`
- Kiểm tra depth map có phải hằng số: `print(depth.min(), depth.max())`
  ảnh đồng nhất (bầu trời trắng, tường trắng) → depth hằng → zone vô nghĩa

### 🔴 Zone bị ngược với metric depth
- **Triệu chứng:** điểm 1m ra zone FAR, điểm 9m ra zone NEAR
- **Nguyên nhân:** dùng `metric_depth=False` cho depth_map metric
- **Fix:** `localize(..., metric_depth=True, intrinsics=intr)`
- Metric (thấp = gần) và relative (cao = gần) là **hai quy ước ngược nhau**

### Calibration sai nhưng RMSE nhỏ
- **Triệu nhân:** ảnh chụp quá mặt phẳng → focal length không xác định được
- **Dấu hiệu:** FOV ước tính ngoài 20°–140° (tool sẽ cảnh báo)
- **Fix:** chụp lại ở **nhiều góc nghiêng** (20°–45°), nhiều khoảng cách

### Không detect được bàn cờ
- **Nguyên nhân số 1:** sai số pattern. `--pattern` nhận **số góc trong** (OpenCV convention)
  pattern `9x6` = bàn cờ **10x7 ô vuông**
- Kiểm tra: `--preview <dir>` xem ảnh nào detect được
- Ảnh mờ, thiếu sáng, bàn cờ quá nhỏ → không detect

### Metric depth cho kết quả sai số mét lớn
- Intrinsics sai → sai số tỉ lệ (metric model dùng intrinsics làm **input**)
- Kiểm tra: `intr.is_calibrated` phải `True`
- Chạy `python verify_calibration.py` trước khi tin tưởng tool

### DepthART không tải được trên CPU
- Kiểm tra `sys.path` có trỏ đến thư mục chứa `relative/`
- Kiểm tra `einops` và `timm` đã cài
- Dùng `prefer='auto'` để fallback DAv2 nếu cần
- **Encoder MNV4-S/M** chạy tốt trên CPU. TinyViM S/B/L có reference fallback nhưng **chậm**

### `UnicodeEncodeError` trên Windows
- Đã xử lý trong tất cả script (bọc stdout UTF-8)
- Nếu gặp ở script cũ: `python -X utf8 <script>`

### `OpenBLAS error: Memory allocation still failed after 10 retries`
- **Triệu chứng:** `import torch` (qua `DepthBackend` / `MetricDepthBackend`) fail, không load được model
- **Nguyên nhân:** **RAM trống quá ít**, không phải lỗi code. Máy có 15.7 GB nhưng nếu Chrome +
  Docker + IDE đang mở thì chỉ còn **~1.2 GB trống** — torch + OpenBLAS không cấp được buffer
- **Kiểm tra:** `Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory` (chia 1MB ra GB)
- **Fix:** đặt `OPENBLAS_NUM_THREADS=1` và `OMP_NUM_THREADS=1` trước khi chạy, và **đóng Chrome/Docker**
  trước khi chạy DepthART. Cần nâng RAM trống lên > 3 GB.
- **Liên quan:** Phase 4 simulator (`run_eval.py`) chạy DepthART thật → chắc chắn gặp lỗi này nếu RAM chật

### 🗄️ (Lịch sử) `EGL_BAD_PARAMETER` — hướng Habitat-Sim/WSL2 đã BỎ, không còn áp dụng
- Chỉ gặp nếu ai đó mở lại hướng Habitat-Sim headless trong WSL2. Hướng đó bị bỏ vì mesh GT đã có sẵn.

### `open3d` read PLY ra thiếu 99 % tam giác / `A polygon in the mesh could not be decomposed`
- Mesh Replica toàn **tứ giác**; Open3D nuốt gần hết. **Luôn dùng `mesh_io.load_ply`**, không `o3d.io.read_triangle_mesh`.

### Viewer Open3D 0.20: `set_up_vector` / `remove_geometry(name)` lỗi
- Đúng API: `ViewControl.set_up(...)`; `Visualizer.remove_geometry(<đối tượng>)` (không nhận tên);
  `set_front` = hướng từ điểm nhìn **tới** camera (ngược hướng nhìn). Chỉ `--smoke` mới bắt được, `--selftest` thì không.

### Cạnh che làm Δ mesh↔GT lớn (0,26–0,84 m) mà không phải bug
- Dùng `anchor.plane_residual` (cửa sổ 11×11, ngưỡng 5 cm); `nan` = "không phân loại được", **không** phải sạch.
  Đừng thay bằng `max−min` hay cửa sổ 7×7 — đã đo, đều sai.

### Thiếu weights FireBase-YOLO26
- `*.pt` bị `.gitignore` → repo không có
- Copy vào `models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/best.pt`
- Hoặc test phần depth không cần YOLO: `python pipeline.py --image <ảnh> --keypoints x,y`

---

## 📞 Contact / Handoff

Nếu agent tiếp theo cần context nhanh:

1. Đọc phần **KHÔNG ĐƯỢC THAY ĐỔI** trước
2. Đọc **CHANGELOG** mới nhất
3. Xem **TODO** để biết công việc đang dở (chú ý mục "Cần USER cung cấp")
4. Chạy **Testing checklist** để verify trạng thái hiện tại
5. Cập nhật file này với thay đổi của bạn trước khi commit

**Nguyên tắc:** Không xóa changelog cũ; thêm mục mới ở đầu mỗi phần.

---

**Last updated:** 2026-10-16 (lần 2) by AI Agent #6 (kiểm chứng nguồn cho plan Replica gốc ở máy B:
sửa `python=3.12` → **`3.9`**; thêm **⛔ cổng EGL** + **quy trình an toàn restart**; sửa đánh giá
Blender; thêm đường ghép sprite 2D. **Chưa chạy số nào.**)

---

## 📝 SUMMARY — Trạng thái hiện tại

### Module đang chạy
- ✅ **Depth3D_Localizer** — relative/metric depth + 2D→3D + calibration + pipeline
- ⚠️ **FireBase-YOLO26** — chưa test với detector thật (thiếu `best.pt` + `ultralytics`)

### Trình mô phỏng đánh giá depth (hướng mới — bỏ mesh)
- ✅ **Spec v3 + task list** — DONE: `models/Depth3D_Localizer/simulation/SIM_CONTRACT.md` (mục 1–10 spec, **mục 11 = phase & tick**)
- ✅ **README chạy từng bước** — DONE: `models/Depth3D_Localizer/simulation/README.md`
- ✅ **Phase 1 — `utils.py`** — logic đã chứng minh, **22/22 assertions PASS** (`python utils.py --selftest`, không cần data). Test bao gồm cả quy ước OpenGL→OpenCV, `scale=6553.5`, encoder chuẩn hoá
- ✅ **Phase 3 — `sample_points.py`** — logic đã chứng minh trên dataset giả đúng format Replica: 60 JSON, chiếu ngược `xyz_gt_cam`→pixel sai số 5.68e-14 px, cân bằng vùng ảnh 52% center, seed tái lập được (SHA256 giống hệt)
- ✅ **Phase 4 — `run_eval.py`** — logic đã chứng minh: Case A MAE 0.0000 m + verify độc lập bằng `cv2` thuần (khớp 2.3e-07 m), CSV 18 cột, nhiễu pose đúng chiều, chẩn đoán affine/disparity phân biệt đúng
- ✅ **Phase 2 — dataset** — `Replica.zip` đã giải nén tại `D:\replica_sim\data\replica_pp\Replica\` (thêm một cấp `Replica\`), zip đã xoá
- ✅ **Phase 5 — `eval_metrics.py`** — đã có; `check_metrics.ipynb` tính lại `case_B_noisy` khớp **0.3119 m**
- 🔄 **Phase 6 — đánh giá thật** — mới có số **DepthART relative MNV4-S** (B_noisy MAE 0.3119 m, 400 điểm). Bản metric, slim, YOLO26n-depth, Depth Anything V2 **chưa có số**
- ⬜ **`compare_depth_models.ipynb`** — notebook so 4 model đã viết, ô chấm tắt, **chưa chạy**
- ✅ **Hướng WSL2 / Habitat-Sim / EGL / máy B / tải 34 GB đã BỎ (2026-10-16)** — mesh GT đã có sẵn trong
  Replica preprocessed (`*_mesh.ply`, 8 file, 267 MB). Không cần Linux, reboot hay tải thêm.
- ⛔ **Không đụng nhánh ROI/mesh của báo cáo ngoài** (legacy cấm + Scene-MLP sai 1.18 m)

### Trình mô phỏng click-to-check trên Replica preprocessed + mesh (Windows, 1 máy) ✅
- ✅ **Plan** — `simulation/PLAN_2026-10-16.md` (đã viết lại hoàn toàn)
- ✅ **`mesh_io.py`** — parser PLY quad bằng tay + raycast mesh gốc; `--selftest` PASS
- ✅ **`anchor.py`** — pixel → 3D (GT) + raycast mesh + Δ + bộ phân loại cạnh che 3 lớp; `--selftest` PASS
- ✅ **`room_viewer.py`** — 2 cửa sổ (ảnh bấm + phòng 3D); `--selftest` PASS **và** `--smoke 8` chạy cửa sổ thật
- ⬜ **`fire_sprite.py` + Phase 2 (A không lửa / B sprite / C miếng xám)** — **chưa viết, chờ lệnh user**
- ⚠️ Mesh check = bắt lỗi code, **không phải phép đo**. Báo median/p95 trên điểm sạch (room0: 2,04 / 6,76 mm), không báo "2 mm cố định".

### Blocker cần USER
1. **Copy `best.pt`** → `models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/`
2. **`pip install ultralytics`** (chưa cài)
3. **Chụp 20-30 ảnh bàn cờ** → calibrate intrinsics thật (nếu muốn metric depth chính xác)
4. Muốn số so sánh: mở `compare_depth_models.ipynb`, chạy ô 1–3, bật `RUN = True` ở ô 4. Đóng Chrome trước (cần RAM trống).
5. Muốn dùng viewer: `cd models/Depth3D_Localizer/simulation && python room_viewer.py --scene room0` (đóng Chrome/Docker nếu RAM chật).
6. Ra lệnh **Phase 2** (lửa giả) khi muốn viết `fire_sprite.py` và chạy A/B/C.

### Next action (cho agent tiếp)
- **Việc A (notebook so 4 model):** Đừng chạy chấm hộ nếu user chưa bật. Khi có CSV trong `D:\replica_sim\results\cmp_*`, đọc ô 5 — chỉ báo số khi cột `n` của các model **bằng nhau**. Đừng gọi số YOLO26 là mét thật: notebook căn affine như các model relative.
- **Việc B (Phase 2 — lửa giả):** đọc `PLAN_2026-10-16.md` mục 6 + 8. **Chỉ làm khi user ra lệnh.** Viết `fire_sprite.py`
  rồi chạy 3 điều kiện A/B/C trên cùng frame/pixel. Neo tránh pixel `edge`/`unknown`. Ảnh vào DepthART là JPEG gốc + sprite.
  **Cấm ghi vào ổ C.**
