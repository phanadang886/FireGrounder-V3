# 🧪 Simulator — Đánh giá depth + toán 2D→3D

**Đọc spec đầy đủ + task list:** `SIM_CONTRACT.md` (mục 11 = phase & tick) — file này chỉ là hướng dẫn chạy.

## Simulator này làm gì (1 đoạn)

Chấm điểm DepthART và phép biến đổi 2D→3D bằng cách so với depth thật có sẵn.
Không cần tải mesh 31 GB, không cần đặt vật 3D, không cần detector.
Mỗi điểm test = 1 pixel có depth thật → tính tọa độ 3D chuẩn → so với tọa độ DepthART đoán ra.

## Yêu cầu

- Python 3.14, PyTorch + DepthART đã cài (từ module `Depth3D_Localizer`).
- Ổ **D còn ≥ 10 GB**. Ổ C không dùng tới — script nào cũng tự kiểm tra trước khi chạy.
- Không cần Open3D, không cần card rời.

## Chạy từng bước

```bash
cd models/Depth3D_Localizer/simulation

# 0. Kiểm tra môi trường
python -c "import torch, cv2, numpy, scipy; print('Dependencies OK')"

# 1. Xem dataset tốn bao nhiêu trước khi tải thật
python download_dataset.py --source replica_pp --scenes office_0,room_0 --output D:\replica_sim\data --dry-run

# 2. Tải thật (vài GB)
python download_dataset.py --source replica_pp --scenes office_0,room_0 --output D:\replica_sim\data

# 3. Chọn 200 điểm test
python sample_points.py --data-dir D:\replica_sim\data\replica_pp --source replica_pp \
    --num-samples 200 --output D:\replica_sim\samples --seed 42 --visualize

# 4a. Case A: kiểm tra toán (phải < 1 cm, nếu không thì code sai — dừng lại)
python run_eval.py --case A --samples D:\replica_sim\samples \
    --data-dir D:\replica_sim\data\replica_pp --output D:\replica_sim\results\case_A

# 4b. Case B: chấm DepthART (pose chuẩn và pose nhiễu)
python run_eval.py --case B --pose-mode clean --depth-model depthart-mnv4s \
    --samples D:\replica_sim\samples --data-dir D:\replica_sim\data\replica_pp \
    --output D:\replica_sim\results\case_B_clean
python run_eval.py --case B --pose-mode noisy \
    --samples D:\replica_sim\samples --data-dir D:\replica_sim\data\replica_pp \
    --output D:\replica_sim\results\case_B_noisy

# 5. Tổng hợp + vẽ biểu đồ
python eval_metrics.py --results-dir D:\replica_sim\results \
    --output D:\replica_sim\results\metrics_summary.json --plot
```

## Xem số bằng notebook (bấm từng ô)

Mở `check_metrics.ipynb` trong thư mục này. Chạy từ trên xuống: ô đầu chỉ **đọc CSV đã có**,
không chạy AI. Ô cuối (để `RUN_METRIC = True`) mới chấm bản DepthART metric — vài phút, cần RAM trống.

## Đọc kết quả

- `results\case_*\per_sample.csv` — sai số từng điểm.
- `results\metrics_summary.json` — số tổng hợp: `depth` (AbsRel/RMSE), `position_3d`
  (MAE/Median/P95, tách theo trục / khoảng cách / vùng ảnh giữa-rìa).
- Số để báo cáo là **`case_B_noisy`** (pose nhiễu = gần thực tế), không phải pose chuẩn.

## Sự cố thường gặp

| Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|
| `RuntimeError` ổ C/D | Hết chỗ | Dọn ổ D; không bao giờ ghi sang ổ C |
| Link tải chết | Mirror đổi | Đổi `--source nyuv2`, hoặc tìm mirror NICE-SLAM/iMAP mới rồi cập nhật `download_dataset.py` |
| Case A MAE > 1 cm | Code sai (pinhole/pose/parse) | Dừng, sửa `utils.py`, chạy lại Case A trước khi đụng tới DepthART |
| Depth toàn NaN | Sai đơn vị PNG/NPY | Kiểm tra `load_depth_map`: PNG 16-bit là mm (chia 1000), NPY là m |

## Giao việc cho agent khác

**Đọc `SIM_CONTRACT.md` mục 11 trước** — có chia phase, tick từng việc, điều kiện xong, và cảnh báo
⛔ GATE (Case A phải < 1 cm trước khi tin bất kỳ số Case B nào).

Tóm tắt thứ tự: `utils.py` (làm trước, chặn tất cả) → 4 script kia code song song được
→ nhưng khi **chạy** thứ tự cứng là download → sample → run_eval → eval_metrics.

Xong code thì chạy đúng lệnh ở trên, rồi tick mục 11 của `SIM_CONTRACT.md` và dán số thật vào changelog `AGENTS.md`.
