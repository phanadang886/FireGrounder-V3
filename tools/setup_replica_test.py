#!/usr/bin/env python3
"""
Setup và test với Replica Dataset cho FireGrounder-V3

Script này hướng dẫn:
1. Download Replica dataset
2. Setup Habitat-Sim
3. Render test images với synthetic fire
4. Chạy full pipeline: YOLO26 → DepthART → Localizer3D

Author: AI Agent #3
Date: 2026-10-08
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models" / "Depth3D_Localizer"))

def check_dependencies():
    """Kiểm tra dependencies cần thiết."""
    print("🔍 Checking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'PIL': 'PIL',
        'torch': 'torch',
    }
    
    optional = {
        'habitat-sim': 'habitat_sim',
    }
    
    missing = []
    for pkg, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} (REQUIRED)")
            missing.append(pkg)
    
    for pkg, import_name in optional.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ⚠️  {pkg} (optional, needed for Replica rendering)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def download_replica_guide():
    """Hướng dẫn download Replica dataset."""
    print("\n" + "="*60)
    print("📦 REPLICA DATASET DOWNLOAD GUIDE")
    print("="*60)
    
    print("""
Replica Dataset là synthetic indoor scenes với perfect ground truth depth.

🔗 GitHub: https://github.com/facebookresearch/Replica-Dataset

📥 CÁCH 1: Download từ GitHub Releases
  1. Truy cập: https://github.com/facebookresearch/Replica-Dataset/releases
  2. Download các scenes (mỗi scene ~500MB-1GB):
     - apartment_0.zip  (apartment)
     - office_0.zip     (office) 
     - hotel_0.zip      (hotel)
     - room_0.zip       (single room)
  3. Extract vào thư mục: data/replica/
  
📥 CÁCH 2: Clone repo (chỉ download scenes cần)
  ```bash
  cd data/
  git clone https://github.com/facebookresearch/Replica-Dataset.git replica
  cd replica
  # Download individual scenes với script của họ
  ./download.sh apartment_0
  ./download.sh office_0
  ```

📂 Cấu trúc thư mục sau khi download:
  data/replica/
  ├── apartment_0/
  │   ├── mesh.ply
  │   ├── habitat/
  │   │   ├── mesh_semantic.ply
  │   │   └── ...
  │   └── textures/
  ├── office_0/
  └── ...

🔧 Setup Habitat-Sim (để render từ Replica):
  ```bash
  # With CUDA (recommended nếu có GPU)
  conda install habitat-sim -c conda-forge -c aihabitat
  
  # CPU only (chậm hơn nhưng OK cho testing)
  pip install habitat-sim
  ```

⚡ Quick Test (sau khi download):
  python tools/replica_render_test.py --scene apartment_0
""")


def setup_fire_overlay_guide():
    """Hướng dẫn tạo fire overlay cho testing."""
    print("\n" + "="*60)
    print("🔥 FIRE OVERLAY SETUP GUIDE")
    print("="*60)
    
    print("""
Để test định vị lửa 3D, cần overlay fire texture lên Replica images.

📥 Option 1: Sử dụng fire texture có sẵn
  1. Download fire PNG với alpha channel:
     - https://opengameart.org/content/animated-fire
     - https://itch.io/game-assets/free/tag-fire
  2. Lưu vào: assets/fire_textures/
  3. Chọn frames có transparent background (alpha channel)

📥 Option 2: Tạo fire texture từ video
  ```python
  import cv2
  import numpy as np
  
  # Extract frame từ video lửa
  cap = cv2.VideoCapture('fire_video.mp4')
  ret, frame = cap.read()
  
  # Chroma key hoặc background removal để tạo alpha
  # ... (code xử lý)
  
  # Save as PNG với alpha
  cv2.imwrite('assets/fire_textures/fire_001.png', frame_with_alpha)
  ```

🎨 Yêu cầu fire texture:
  - Format: PNG với alpha channel (RGBA)
  - Size: 256x512 px trở lên (sẽ resize theo depth)
  - Background: transparent (alpha = 0)
  - Fire: bright orange/yellow với alpha gradient

🧪 Test fire overlay:
  python tools/test_fire_overlay.py --texture assets/fire_textures/fire_001.png
""")


def pipeline_test_guide():
    """Hướng dẫn test full pipeline."""
    print("\n" + "="*60)
    print("🚀 FULL PIPELINE TEST GUIDE")
    print("="*60)
    
    print("""
Pipeline đầy đủ: Replica Render → YOLO26 → DepthART → Localizer3D

📝 Workflow:
  
  1️⃣ Render scene từ Replica
     → RGB image + ground truth depth
  
  2️⃣ Overlay synthetic fire
     → Fire position (x,y,z) trong world coords
     → Project 3D→2D để tính keypoint BASE
  
  3️⃣ Run YOLO26 detection
     → Detect fire bbox + keypoint BASE
     (hoặc dùng ground truth keypoint để isolate depth module)
  
  4️⃣ Run DepthART depth estimation
     → Depth map từ RGB
  
  5️⃣ Run Localizer3D
     → Input: depth map + keypoint (kx, ky)
     → Output: FireLocation3D với xyz_normalized
  
  6️⃣ Compare với ground truth
     → Error = ||predicted_xyz - ground_truth_xyz||
     → Zone accuracy: predicted_zone == ground_truth_zone

📊 Metrics để đo:
  - Position error (L2 distance): target < 0.1 (normalized coords)
  - Zone accuracy: target > 90%
  - Depth error: |predicted_depth - gt_depth| / gt_depth < 10%
  - Inference time: target < 200ms per frame (YOLO + Depth + Localize)

🔧 Chạy test:
  # Test với 1 ảnh
  python tools/test_single_image.py \\
    --scene apartment_0 \\
    --fire-pos 2.5,0.5,3.0 \\
    --yolo-weights models/FireBase_YOLO26/.../best.pt

  # Test batch với nhiều góc camera
  python tools/test_batch_replica.py \\
    --scene apartment_0 \\
    --num-samples 100 \\
    --output results/replica_test/

  # Visualize results
  python tools/visualize_results.py --input results/replica_test/
""")


def create_sample_script():
    """Tạo sample script để bắt đầu nhanh."""
    script_path = REPO_ROOT / "tools" / "replica_quick_start.py"
    
    sample_code = '''#!/usr/bin/env python3
"""
Quick start script - Test Replica rendering và fire overlay

Usage:
  python tools/replica_quick_start.py
"""

import numpy as np
import cv2
from pathlib import Path

def main():
    print("🚀 Replica Quick Start")
    print("="*60)
    
    # Check Habitat-Sim
    try:
        import habitat_sim
        print("✅ habitat-sim installed")
        has_habitat = True
    except ImportError:
        print("⚠️  habitat-sim not installed")
        print("   Install: pip install habitat-sim")
        has_habitat = False
    
    # Check Replica data
    replica_path = Path("data/replica")
    if not replica_path.exists():
        print("⚠️  Replica dataset not found at:", replica_path)
        print("   Download from: https://github.com/facebookresearch/Replica-Dataset")
        return
    
    scenes = list(replica_path.glob("*/mesh.ply"))
    if not scenes:
        print("⚠️  No scenes found in", replica_path)
        return
    
    print(f"✅ Found {len(scenes)} Replica scenes:")
    for scene in scenes:
        print(f"   - {scene.parent.name}")
    
    if not has_habitat:
        print("\\n⚠️  Cannot render without habitat-sim")
        return
    
    # Test rendering (commented để không crash nếu setup chưa đủ)
    print("\\n📸 TODO: Add rendering code here")
    print("   See docs/3D_DATASET_RECOMMENDATIONS.md for examples")

if __name__ == "__main__":
    main()
'''
    
    script_path.write_text(sample_code, encoding='utf-8')
    print(f"\n✅ Created sample script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup và test Replica dataset cho FireGrounder-V3"
    )
    parser.add_argument(
        '--check-only', 
        action='store_true',
        help="Chỉ kiểm tra dependencies"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔥 FireGrounder-V3 — Replica Test Setup")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_only:
        print("\n✅ Dependency check complete")
        return
    
    # Show guides
    download_replica_guide()
    setup_fire_overlay_guide()
    pipeline_test_guide()
    
    # Create sample scripts
    create_sample_script()
    
    print("\n" + "="*60)
    print("📚 NEXT STEPS")
    print("="*60)
    print("""
1. Download Replica dataset (xem hướng dẫn trên)
2. Install habitat-sim: pip install habitat-sim
3. Chuẩn bị fire textures vào assets/fire_textures/
4. Chạy sample script: python tools/replica_quick_start.py
5. Đọc thêm: docs/3D_DATASET_RECOMMENDATIONS.md

📖 Full documentation:
   - AGENTS.md — Project log và decisions
   - docs/3D_DATASET_RECOMMENDATIONS.md — Dataset details
   - docs/AGENT3_SUMMARY.md — Summary của session này

💡 Nếu gặp issue, check AGENTS.md → Debug hints section
""")


if __name__ == "__main__":
    main()
