# 📊 Dataset 3D Khuyến Nghị cho FireGrounder-V3

**Mục đích:** Tạo môi trường mô phỏng để test module định vị lửa 3D trong không gian phòng kín với ground truth depth chính xác.

**Ngày tạo:** 2026-10-08  
**Agent:** #3

---

## 🎯 Yêu cầu Dataset

Để test và đánh giá độ chính xác của **Depth3D_Localizer**, cần dataset có:

1. ✅ **RGB-D images** với ground truth depth chính xác
2. ✅ **Indoor scenes** (phòng kín, văn phòng, nhà ở)
3. ✅ **Camera intrinsics** (fx, fy, cx, cy) để chuyển 2D→3D chính xác
4. ✅ **Diverse viewpoints** để test với góc nhìn khác nhau
5. ✅ **Metric depth** (đơn vị mét) thay vì relative depth
6. 🔄 **Simulatable** — có thể render ảnh mới từ bất kỳ góc nào (optional nhưng rất hữu ích)

---

## 🏆 TOP 5 DATASET KHUYẾN NGHỊ

### 1. **Replica Dataset** ⭐⭐⭐⭐⭐ (KHUYẾN NGHỊ NHẤT)

**Nguồn:** Facebook AI Research  
**GitHub:** https://github.com/facebookresearch/Replica-Dataset  
**Paper:** [The Replica Dataset: A Digital Replica of Indoor Spaces (2019)](https://arxiv.org/abs/1906.05797)

**Ưu điểm:**
- ✅ **18 synthetic indoor scenes** với độ chi tiết cực cao
- ✅ **Perfect ground truth depth** — không có noise sensor
- ✅ **High-resolution HDR textures** (photorealistic)
- ✅ **Semantic segmentation** (88 classes)
- ✅ **Glass/mirror surface annotations** — quan trọng với depth estimation
- ✅ **Fully simulatable** — render từ bất kỳ camera pose nào
- ✅ **Compatible với AI Habitat** — có thể tạo môi trường interactive
- ✅ **6 scenes có multiple furniture configurations** — test với layout khác nhau

**Tại sao phù hợp:**
- Không có noise từ sensor → ground truth depth hoàn hảo để đánh giá model
- Có thể **thêm lửa synthetic** vào scene và render chính xác depth tại vị trí lửa
- Scene realistic đủ để model DepthART/DAv2 hoạt động tốt
- Dễ tích hợp với pipeline hiện tại (render RGB + depth → YOLO → Localizer)

**Download:**
- Dataset size: ~10-15GB (mỗi scene ~500MB-1GB)
- License: Creative Commons (free for research)
- Format: `.ply` mesh + HDR textures + semantic annotations

**Cách sử dụng:**
1. Clone repo và tải scenes: `git clone https://github.com/facebookresearch/Replica-Dataset`
2. Setup Habitat-Sim để render: `pip install habitat-sim`
3. Render RGB + depth từ camera poses mong muốn
4. (Optional) Thêm fire texture vào một số vị trí và gán nhãn keypoint BASE
5. Chạy pipeline: YOLO26 + DepthART + Localizer3D
6. So sánh xyz_normalized với ground truth position

**Code mẫu:**
```python
import habitat_sim

# Load scene
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = "replica_dataset/apartment_0/mesh_semantic.ply"
sim = habitat_sim.Simulator(cfg)

# Set camera pose
agent = sim.get_agent(0)
agent.set_state(habitat_sim.AgentState(
    position=[x, y, z],
    rotation=habitat_sim.utils.quat_from_angle_axis(angle, axis)
))

# Render RGB + Depth
obs = sim.get_sensor_observations()
rgb = obs["color_sensor"]  # (H, W, 4)
depth = obs["depth_sensor"]  # (H, W) — metric depth in meters
```

---

### 2. **ScanNet++** ⭐⭐⭐⭐⭐

**Nguồn:** Technical University of Munich  
**Website:** https://kaldir.vc.in.tum.de/scannetpp/  
**Paper:** [ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes (2023)](https://arxiv.org/abs/2308.11417)

**Ưu điểm:**
- ✅ **460 real indoor scenes** — data thực từ laser scanner
- ✅ **Sub-millimeter accuracy** — laser scan chất lượng cao
- ✅ **33MP DSLR images** + **iPhone RGB-D** (consumer-grade)
- ✅ **Ground truth depth** từ laser scan
- ✅ **Comprehensive semantic annotations** (ambiguous labels included)
- ✅ **Camera intrinsics** đầy đủ cho tất cả captures

**Tại sao phù hợp:**
- Real-world scenes → test generalization của model
- Có cả high-end (DSLR) và consumer-grade (iPhone) data
- Ground truth depth từ laser scan rất chính xác
- Scene đa dạng: văn phòng, phòng khách, nhà bếp, v.v.

**Hạn chế:**
- Không có fire annotations → cần tự tạo synthetic fire overlay
- Dataset lớn (~hundreds of GB)
- Cần đăng ký để download

**Download:**
- Đăng ký tại: https://kaldir.vc.in.tum.de/scannetpp/
- License: Research-only (Terms of Use)

---

### 3. **Matterport3D** ⭐⭐⭐⭐

**Nguồn:** Princeton University + Stanford  
**Website:** https://niessner.github.io/Matterport/  
**Paper:** [Matterport3D: Learning from RGB-D Data in Indoor Environments (2017)](https://arxiv.org/abs/1709.06158)

**Ưu điểm:**
- ✅ **90 building-scale scenes** (194,400 RGB-D images)
- ✅ **10,800 panoramic views** (360° coverage)
- ✅ **Ground truth depth** + camera poses
- ✅ **2D & 3D semantic segmentation**
- ✅ **Building-scale** — nhiều phòng liên kết, hữu ích cho multi-room fire scenarios

**Tại sao phù hợp:**
- Panoramic views → test với FOV rộng (quan trọng với fire detection)
- Depth quality tốt, ít noise
- Scene rất đa dạng: nhà ở, văn phòng, khách sạn

**Hạn chế:**
- Cần ký Terms of Use và request access
- Dataset rất lớn (~100GB+)
- Depth từ Matterport camera (proprietary) — không phải consumer camera

**Download:**
- Email Terms of Use đến: matterport3d@googlegroups.com
- License: Academic research only

---

### 4. **Hypersim** ⭐⭐⭐⭐

**Nguồn:** Apple AI Research  
**GitHub:** https://github.com/apple/ml-hypersim  
**Paper:** [Hypersim: A Photorealistic Synthetic Dataset (2020)](https://arxiv.org/abs/2011.02523)

**Ưu điểm:**
- ✅ **77,400 images from 461 indoor scenes** — synthetic nhưng photorealistic
- ✅ **Perfect ground truth** cho depth, normals, semantic segmentation
- ✅ **Diffuse reflectance + illumination** — phân tích lighting (hữu ích với fire glow)
- ✅ **High dynamic range** — captures extreme brightness (lửa)
- ✅ **Complete camera intrinsics**
- ✅ **Open-source 3D assets** — có thể modify scenes

**Tại sao phù hợp:**
- Synthetic → dễ thêm fire objects với ground truth chính xác
- HDR + diffuse lighting → test với fire illumination effects
- Photorealistic đủ để model depth hoạt động tốt
- Free and open dataset

**Hạn chế:**
- Synthetic scenes có thể thiếu "realism artifacts" mà model học từ real data
- Average depth ~5.4m → có thể khác camera deployment thực tế

**Download:**
- GitHub: https://github.com/apple/ml-hypersim
- License: Creative Commons (free)
- Size: ~461GB full dataset

---

### 5. **ScanNet (Original)** ⭐⭐⭐⭐

**Nguồn:** Technical University of Munich  
**Website:** http://www.scan-net.org/  
**Paper:** [ScanNet: Richly-annotated 3D Reconstructions (2017)](https://arxiv.org/abs/1702.04405)

**Ưu điểm:**
- ✅ **1,513 scans** (2.5M RGB-D frames) — largest real-world indoor dataset
- ✅ **Ground truth depth** from Structure Sensor
- ✅ **Camera poses + surface reconstruction**
- ✅ **Semantic segmentation** (36k objects, 18 categories)
- ✅ **Widely used benchmark** — dễ so sánh với research khác

**Tại sao phù hợp:**
- Standard benchmark cho indoor scene understanding
- Nhiều scene đa dạng
- Real-world data → test generalization

**Hạn chế:**
- Depth quality thấp hơn ScanNet++ (Structure Sensor noise)
- Không có fire annotations
- Cần đăng ký Terms of Use

**Download:**
- Website: http://www.scan-net.org/
- License: Research-only

---

## 🛠️ KHUYẾN NGHỊ TRIỂN KHAI

### Phase 1: Quick Prototyping (1-2 tuần)
**Dataset:** **Replica** (18 scenes, ~10GB)

**Tại sao:**
- Nhẹ, download nhanh
- Perfect ground truth
- Dễ setup với Habitat-Sim
- Có thể thêm synthetic fire ngay

**Tasks:**
1. Download 2-3 Replica scenes (apartment, office, hotel)
2. Setup Habitat-Sim render pipeline
3. Tạo synthetic fire overlay với keypoint BASE
4. Render 100-200 test images với camera poses random
5. Chạy full pipeline: YOLO26 → DepthART → Localizer3D
6. Đo error: `||xyz_predicted - xyz_ground_truth||`

---

### Phase 2: Real-World Validation (3-4 tuần)
**Dataset:** **ScanNet++** hoặc **Matterport3D**

**Tại sao:**
- Real-world data → test generalization
- Diverse scenes
- Ground truth depth chính xác từ laser scan / Matterport camera

**Tasks:**
1. Download subset scenes (~50 scenes)
2. Overlay synthetic fire vào ảnh real (sử dụng Poisson blending)
3. Test pipeline trên real textures + lighting
4. Đánh giá performance drop so với Replica
5. Fine-tune threshold `near_thresh` / `far_thresh` nếu cần

---

### Phase 3: Synthetic Fire Simulation (nếu có thời gian)
**Dataset:** **Hypersim** + fire rendering plugin

**Tasks:**
1. Tải Hypersim scenes với HDR lighting
2. Sử dụng Blender hoặc Unity để render fire particles realistic
3. Export RGB + depth + fire mask
4. Train hoặc fine-tune YOLO26 với synthetic fire (optional)
5. Benchmark end-to-end accuracy

---

## 📊 So Sánh Tổng Quan

| Dataset | Scenes | Ground Truth Depth | Photorealistic | Simulatable | Fire-Ready | Size | License |
|---------|--------|-------------------|----------------|-------------|------------|------|---------|
| **Replica** | 18 | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐⭐ Easy | 10GB | Free |
| **ScanNet++** | 460 | ⭐⭐⭐⭐⭐ Laser | ⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐ Manual | 100GB+ | Research |
| **Matterport3D** | 90 | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐ Manual | 100GB+ | Research |
| **Hypersim** | 461 | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐ Easy | 461GB | Free |
| **ScanNet** | 1513 | ⭐⭐⭐ Fair | ⭐⭐⭐ | ❌ No | ⭐⭐⭐ Manual | 50GB+ | Research |

---

## 🔥 Tạo Fire Test Dataset từ Replica

### Script Mẫu: `create_fire_test_dataset.py`

```python
"""
Tạo dataset test với synthetic fire overlay trên Replica scenes.
"""
import numpy as np
import cv2
from PIL import Image
import habitat_sim
from pathlib import Path

def add_fire_to_scene(rgb, depth, fire_position_3d, intrinsics):
    """
    Thêm fire texture vào RGB image và tính keypoint BASE.
    
    Args:
        rgb: (H, W, 3) RGB image
        depth: (H, W) depth map (meters)
        fire_position_3d: (x, y, z) vị trí lửa trong không gian 3D
        intrinsics: (fx, fy, cx, cy)
    
    Returns:
        rgb_with_fire: RGB image có fire overlay
        keypoint_base: (kx, ky) pixel coordinates of fire base
        ground_truth: dict with 3D position
    """
    fx, fy, cx, cy = intrinsics
    x, y, z = fire_position_3d
    
    # Project 3D → 2D
    kx = int(fx * x / z + cx)
    ky = int(fy * y / z + cy)
    
    # Load fire texture (transparent PNG)
    fire_tex = cv2.imread('assets/fire_texture.png', cv2.IMREAD_UNCHANGED)
    fire_h, fire_w = fire_tex.shape[:2]
    
    # Resize based on depth (farther = smaller)
    scale = 1.0 / z  # simple perspective scaling
    new_h = int(fire_h * scale)
    new_w = int(fire_w * scale)
    fire_resized = cv2.resize(fire_tex, (new_w, new_h))
    
    # Alpha blend onto RGB
    rgb_with_fire = rgb.copy()
    # ... (alpha blending code)
    
    ground_truth = {
        'pixel': (kx, ky),
        'xyz_metric': fire_position_3d,
        'depth_metric': z,
        'scene_name': 'replica_apartment_0',
    }
    
    return rgb_with_fire, (kx, ky), ground_truth


def create_test_dataset(replica_path, output_path, num_samples=200):
    """
    Render samples từ Replica và thêm fire synthetic.
    """
    # Setup Habitat
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(replica_path / "apartment_0/mesh.ply")
    sim = habitat_sim.Simulator(cfg)
    
    dataset = []
    for i in range(num_samples):
        # Random camera pose
        pos = sample_random_position()
        rot = sample_random_rotation()
        
        # Render
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][..., :3]
        depth = obs["depth_sensor"]
        
        # Add fire at random location
        fire_pos = sample_fire_position_in_view(depth)
        rgb_fire, kp, gt = add_fire_to_scene(rgb, depth, fire_pos, intrinsics)
        
        # Save
        sample = {
            'image': rgb_fire,
            'depth_gt': depth,
            'keypoint': kp,
            'ground_truth': gt,
        }
        dataset.append(sample)
    
    return dataset
```

---

## 📦 Resources

### Download Links
- **Replica:** https://github.com/facebookresearch/Replica-Dataset
- **ScanNet++:** https://kaldir.vc.in.tum.de/scannetpp/
- **Matterport3D:** https://niessner.github.io/Matterport/
- **Hypersim:** https://github.com/apple/ml-hypersim
- **ScanNet:** http://www.scan-net.org/

### Tools
- **Habitat-Sim:** https://github.com/facebookresearch/habitat-sim (render Replica)
- **AI Habitat:** https://aihabitat.org/ (embodied AI platform)
- **Blender:** https://www.blender.org/ (3D rendering cho fire simulation)
- **PyRender:** https://github.com/mmatl/pyrender (lightweight 3D rendering)

### Papers
1. Replica Dataset (2019): https://arxiv.org/abs/1906.05797
2. ScanNet (2017): https://arxiv.org/abs/1702.04405
3. ScanNet++ (2023): https://arxiv.org/abs/2308.11417
4. Matterport3D (2017): https://arxiv.org/abs/1709.06158
5. Hypersim (2020): https://arxiv.org/abs/2011.02523

---

**Cập nhật cuối:** 2026-10-08 by AI Agent #3  
**Next steps:** Download Replica dataset và tạo prototype fire test pipeline
