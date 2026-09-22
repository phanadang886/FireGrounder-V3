# 🔍 RESEARCH GAP ANALYSIS — Fire Grounding Project
> **File này ghi chép**:
> - Những gì đã có người làm rồi (có link paper).
> - Những gì là "điểm mới / chưa có" trong lĩnh vực này.
> - Dùng để AI tiếp nhận biết được: **chúng ta đang đi sau bao nhiêu bước**, và **điểm cần chú trọng không được lặp lại**.

---

## 1. Lĩnh vực chúng ta đang làm là gì?

Chúng ta không làm:
- ❌ Fire Detection (có lửa/không?)
- ❌ Fire Segmentation (mask toàn bộ lửa)

Chúng ta làm:
- ✅ **Real-time Fire Base Grounding** — tìm **một điểm duy nhất** (`has_fire`, `x_norm`, `y_norm`) cho biết **chân nền ngọn lửa** tiếp xúc với mặt sàn, để robot tự hành hướng tới.

Đây là một dạng **single-keypoint regression** — **chưa có benchmark chuẩn**. Các paper lớn đều làm multi-keypoint hoặc segmentation.

---

## 2. Những gì đã có người làm rồi?

### 2.1. Kỹ thuật heatmap + soft-argmax
- **Soft-argmax gốc**: [Luvizon et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0131.pdf)
- **Soft-argmax 2D trong Pytorch (tham khảo code):** [PyTorch-Soft-Argmax (GitHub)](https://github.com/Fdevmsy/PyTorch-Soft-Argmax)
- **Bài phân tích calibration:** [On the Calibration of Human Pose Estimation, 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Gu_On_the_Calibration_of_Human_Pose_Estimation_CVPR_2024_paper.pdf)

### 2.2. FPN Fusion
- **FPN gốc:** [Lin et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html)

### 2.3. Focal Loss
- **Focal Loss gốc:** [Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

### 2.4. Gaussian heatmap target
- Dùng sigma = 2 heuristic (được khẳng định trong calibration paper 2024); cần ablation sigma = 1, 2, 3.

### 2.5. Hard negative mining
- **OHEM gốc:** [Shrivastava et al., CVPR 2016](https://arxiv.org/abs/1604.03540)
- **False positive suppression trong fire detection:** [Ultralytics Glossary: Hard Negative Mining](https://www.ultralytics.com/glossary/hard-negative-mining)
- **2025/2026:** [Risk-Aware False-Alarm Suppression for Resource-Constrained...](https://ieeexplore.ieee.org/abstract/document/11658591) — áp dụng giảm báo giả trong fire detection thực tế.

### 2.6. MobileNetV4
- **Paper gốc:** [MobileNetV4, arXiv 2024](https://arxiv.org/abs/2404.10518)
- **timm implementation** đã hỗ trợ `features_only=True` với output indices đã verified: `[2, 4, 8, 16, 32]`.

### 2.7. HRNet / SimpleBaseline
- **SimpleBaseline (deconv head):** [Xiao et al., CVPR 2018 (arXiv:1804.06208)](https://arxiv.org/abs/1804.06208)
- **HRNet:** [Sun et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html)

### 2.8. YOLOv8 Pose
- **YOLOv8 Pose output format:** [Ultralytics Docs - Pose](https://docs.ultralytics.com/modes/pose/)
- **YOLOv8 Pose inference example:** [Roboflow YOLOv8 Pose](https://universe.roboflow.com/yolo-xvnzo/yolov8-pose-utovc)

### 2.9. Fire Detection papers (2025–2026)
| Paper | Năm | Dataset | Phương pháp | Dataset có keypoint? |
|---|---|---|---|---|
| Real-Time Wildfire Localization (NASA) | 2026 | AMS Sensor | Classification + Segmentation | ❌ |
| PyroFocus (multispectral) | 2025 | NASA MASTER | Regression + Segmentation | ❌ |
| FASDD_RS (remote sensing) | 2025 | Self-collected | Swin Transformer | ❌ (bbox) |
| CFMap (fire risk map) | 2026 | Self-collected | CNN | ❌ |
| FlameFinder (flame obscured by smoke) | 2024 | UAV RGB-T | Deep Metric Learning | ❌ |
| Industrial & Mining Fire Detection | 2026 | Coalmine-Fire | CMF-Net | ❌ (bbox) |
| Waste Fire Surveillance | 2026 | Satellite | FIRMS + PlanetScope | ❌ |
| Nighttime Fire Detection (NIR) | 2025 | Self-collected | YOLOv11n + EfficientNet | ❌ (bbox) |

### 2.10. Datasets liên quan
| Dataset | Năm | Frames/samples | Annotations | Notes |
|---|---|---|---|---|
| FLAME | 2017 | 42k | Classification | UAV RGB video |
| FLAME-2 | 2022 | 53k | 4-class classification | RGB/LWIR paired |
| FASDD_RS | 2025 | ~2k | Bbox | Remote sensing |
| Visifire | — | 7 videos | Không có annotation | Raw footage only |
| Coalmine-Fire | 2026 | ~1k (synthetic) | Bbox | GAN-based flame synthesis |
| Fire Detection Dataset (Vietnam NIR) | 2025 | Self-collected | Bbox | NIR-only |

### 2.11. Hard Negatives
- **False positive sources** trong các paper 2025–2026 bao gồm:
  - Đèn neon, đèn pha, phản xạ ánh sáng
  - Thùng rác, kim loại sáng
  - Áo bảo hộ màu cam/nâu
  - Hoàng hôn, ánh nắng rực rỡ
=> Đây **chính là vấn đề chúng ta đang gặp** ở `fire7.png`.

---

## 3. Những gì chưa có người làm (Research Gap)?

| Research Gap | Mô tả | Tại sao chúng ta cần làm? |
|---|---|---|
| 🔸 Single-point fire base annotation | Không có dataset nào cung cấp `p_fire = [x, y]` cho chân nền ngọn lửa | Chúng ta đang xây dựng pipeline đầu tiên cho task này |
| 🔸 Real-time keypoint fire grounding trên edge | Các paper đều dùng seg/classification; không có paper nào làm keypoint real-time trên Jetson Orin | Chúng ta đang làm realtime + edge-first |
| 🔸 Soft-argmax + MSE trên heatmap tại 64x64 | Chưa có paper nào áp dụng soft-argmax trực tiếp trên heatmap 64x64 cho 1 keypoint chân nền | Nhỏ gọn, đủ tốc độ, nhưng chưa có ai công bố |
| 🔸 Fusion FPN stride-4 + stride-8 | HRNet dùng multi-resolution; SimpleBaseline dùng deconv; nhưng không có paper nào fusion FPN từ backbone MobileNetV4 cho 1 keypoint | Chúng ta đang thử cách này |
| 🔸 Hard negative mining dựa trên confidence score + visual overlay | Các paper dùng OHEM/EMA/Dynamic Head, nhưng chưa có paper nào visual hard-negative collection dựa trên false positives thực tế | Chúng ta đang làm theo style `fire7.png` |
| 🔸 Label noise estimation từ pseudo-GT | Chưa ai đo label noise từ SAM3 pseudo-GT cho keypoint | Chúng ta cần biết ngưỡng dưới của MAE (SAM3 ≈ 3.2px) |

---

## 4. Các lỗi thường gặp (Common Pitfalls) — tránh lặp lại

| Sai | Mô tả | Liên quan |
|---|---|---|
| ❗ Pseudo-bbox từ single point | Tạo bbox xung quanh 1 point rồi dùng làm GT cho YOLO-Pose = vòng lặp tự tham chiếu sai lệch | YOLO-Pose V3 |
| ❗ Upsample từ stride-8 không tạo detail | Chỉ upsample 32x32 lên 64x64 không bù thông tin đã mất | V3 Heatmap |
| ❗ So sánh không công bằng | Dùng history file V1 để so sánh với V2/V3 | File log |
| ❗ Không chuẩn hóa metric | Tính MAE ở 256x256 nhưng báo ở ảnh gốc 1080p | File report |
| ❗ Không đo label noise | Hứa MAE < 5px mà chưa biết GT chính xác tới đâu | V2 Kaggle |
| ❗ Hard negative mà không ảnh hưởng recall | Thêm negative quá nhiều có thể làm mất sensitivity với fire thật | Hard Negs |

---

## 5. Quyết định thiết kế (Decision Log)

| Tính năng | Quyết định | Nguồn |
|---|---|---|
| Heatmap + soft-argmax | ✅ Áp dụng | Luvizon 2018, Xiao 2018 |
| FPN-lite fusion | ✅ Áp dụng | Lin 2017 |
| Focal Loss | ✅ Giữ nguyên | Lin 2017 |
| MobileNetV4 | ✅ Giữ nguyên | MobileNetV4 paper, verified timm |
| Sigma = 2 | ✅ Thử nghiệm ban đầu | Calibration 2024 |
| YOLOv8-Pose | ⏸ Hoãn | Chưa có bbox GT |
| Hard negatives | ✅ Thu thập 300–500 ảnh | 2025–2026 FP papers |
| Edge deployment | ✅ Jetson Orin | EdgeNeXt paper |

---

## 6. Kết luận cho AI thực thi

- Chúng ta **không phải** đang làm một phần nhỏ của một paper nào — mà đang **kết hợp các khối** từ nhiều paper (FPN + Soft-argmax + Focal Loss + MobileNetV4) để giải quyết một bài toán mới.
- Giá trị mới thực sự nằm ở: **pipeline hoàn chỉnh real-time cho single-point fire base grounding trên edge**, cùng bộ **benchmark + label-noise measurement** đáng tin cậy.
- Mọi quyết định kỹ thuật trong bảng Decision Log phải được giữ nguyên trừ khi có dữ liệu thực nghiệm bác bỏ.