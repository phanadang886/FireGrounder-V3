# ══════════════════════════════════════════════════════════════════════════════
# KAGGLE NOTEBOOK — 5 CELLS
# SAM3 Label + Train MobileNetV4 + Preview + So sanh Truoc/Sau khi Train
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — CAI DAT (chay 1 lan, Restart Session sau do)
# ─────────────────────────────────────────────────────────────────────────────
"""
# Chi cai timm — KHONG DUONG den torch/torchvision/transformers/Pillow
# de tranh pha CUDA environment cua Kaggle
!pip install -q timm
print("Done. Restart Session truoc khi chay Cell 2!")
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — SAM3 LABEL FIRE + LABEL DEFAULT -> dataset_labels.json
#
# JSON output format:
#   Fire   : {"image_path":..., "has_fire":1, "p_fire":[x,y]}
#   Default: {"image_path":..., "has_fire":0, "p_fire":[0,0]}
# ─────────────────────────────────────────────────────────────────────────────
"""
import sys, os, json, pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# ── Fix loi torchaudio voi transformers tren Kaggle ───────────────────────────
# Van de: is_torchaudio_available() = True (da cai), nhung import bi circular
# Fix: monkeypatch ham do thanh False TRUOC khi transformers load SAM3
# -> audio_utils.py se bo qua "import torchaudio" va khong crash

# Xoa state cu neu co
for _k in list(sys.modules.keys()):
    if "torchaudio" in _k:
        del sys.modules[_k]

# Patch transformers de bao "torchaudio khong co"
import transformers.utils.import_utils as _tu
_tu.is_torchaudio_available = lambda: False
import transformers.utils as _tutils
_tutils.is_torchaudio_available = lambda: False

# Bay gio an toan de import SAM3
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = "/kaggle/working/fire_ground_dataset"
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_labels.json")
STREAM_CACHE     = os.path.join(OUTPUT_DIR, "stream_cache.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ten thu muc tuong ung voi moi loai anh
FIRE_KEYWORDS    = ["fire"]
DEFAULT_KEYWORDS = ["default", "normal", "no_fire", "negative", "none"]
SKIP_KEYWORDS    = ["smoke"]   # bo qua khoi (se train sau)
valid_exts       = ('.jpg', '.jpeg', '.png', '.webp')

# ── Ham phan loai thu muc ─────────────────────────────────────────────────────
def classify_path(path: str):
    """
    Tra ve 'fire', 'default', hoac 'skip'.

    Dung EXACT match cho FIRE_KEYWORDS (kw == ten_thu_muc)
    de tranh truong hop dataset co ten chua 'fire'
    (vd: 'fire-detection-from-cctv') lam phan loai nham.

    Thu tu uu tien:
      1. Thu muc la (leaf dir): ro rang nhat
      2. Bat ky thu muc nao trong duong dan
    """
    parts    = path.lower().replace("\\", "/").split("/")
    leaf_dir = parts[-2] if len(parts) >= 2 else ""   # thu muc chua truc tiep file anh

    # -- Uu tien 1: kiem tra leaf dir (chinh xac nhat) --
    for kw in SKIP_KEYWORDS:
        if kw == leaf_dir: return "skip"
    for kw in FIRE_KEYWORDS:
        if kw == leaf_dir: return "fire"
    for kw in DEFAULT_KEYWORDS:
        if kw == leaf_dir: return "default"

    # -- Uu tien 2: kiem tra toan duong dan (exact match, tranh "fire-detection-from-cctv") --
    for p in parts:
        for kw in SKIP_KEYWORDS:
            if kw == p: return "skip"
    for p in parts:
        for kw in FIRE_KEYWORDS:
            if kw == p: return "fire"       # chi khop khi ten thu muc la "fire" chinh xac
    for p in parts:
        for kw in DEFAULT_KEYWORDS:
            if kw == p: return "default"

    return "skip"   # khong ro loai -> bo qua

# ── Quet tat ca anh ───────────────────────────────────────────────────────────
all_raw = sorted({
    os.path.join(root, f)
    for root, _, files in os.walk("/kaggle/input")
    for f in files if f.lower().endswith(valid_exts)
})

# In cau truc thu muc
dirs = sorted({str(pathlib.Path(p).parent) for p in all_raw})
print("=== Cau truc thu muc ===")
for d in dirs[:20]:
    n    = sum(1 for p in all_raw if str(pathlib.Path(p).parent) == d)
    kind = classify_path(d + "/x.jpg")
    print(f"  [{kind:7s}] {d}  ({n} anh)")
print()

fire_imgs    = [p for p in all_raw if classify_path(p) == "fire"]
default_imgs = [p for p in all_raw if classify_path(p) == "default"]
print(f"Fire   : {len(fire_imgs)} anh  (se dung SAM3 gan nhan toa do)")
print(f"Default: {len(default_imgs)} anh  (se gan nhan has_fire=0 tu dong)")
print(f"Skip   : {len(all_raw)-len(fire_imgs)-len(default_imgs)} anh  (bo qua)")

if not fire_imgs and not default_imgs:
    print("\nKHONG TIM THAY ANH! Chinh FIRE_KEYWORDS / DEFAULT_KEYWORDS!")
    raise SystemExit

# Resume
done = set()
if os.path.exists(STREAM_CACHE):
    with open(STREAM_CACHE, encoding="utf-8") as f:
        for line in f:
            try: done.add(json.loads(line)["image_path"])
            except: pass
    print(f"\nResume: da xu ly {len(done)} anh truoc do")

# ── Khoi tao SAM3 (chi can neu co anh fire chua xu ly) ────────────────────────
fire_todo = [p for p in fire_imgs if p not in done]
if fire_todo:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
    sam_model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    sam_processor = Sam3Processor.from_pretrained("facebook/sam3")
    sam_model.eval()
    print("SAM3 ready")
else:
    print("\nTat ca anh fire da duoc xu ly truoc, bo qua SAM3.")

# ── Adaptive Baseline ─────────────────────────────────────────────────────────
def adaptive_baseline(mask_np, alpha=0.35, jump_thresh=1.8):
    ys, xs = np.where(mask_np > 0)
    if len(ys) < 50: return None
    g_ymin, g_ymax = int(ys.min()), int(ys.max())
    if g_ymin == g_ymax: return None
    y_cut = g_ymax - alpha * (g_ymax - g_ymin)
    profile = []
    for ux in np.unique(xs):
        my = int(np.max(ys[xs == ux]))
        if my >= y_cut: profile.append((int(ux), my))
    if len(profile) < 2: return None
    base_line = [profile[0]]
    for cx, cy in profile[1:]:
        px, py = base_line[-1]
        dx, dy = abs(cx - px), abs(cy - py)
        if dx > 0 and (dy / dx) > jump_thresh and cy < py: continue
        base_line.append((cx, cy))
    if not base_line: return None
    H, W = mask_np.shape
    best = base_line[int(np.argmax([p[1] for p in base_line]))]
    return round(float(best[0]) / W, 6), round(float(best[1]) / H, 6)

# ── Vong lap gan nhan ─────────────────────────────────────────────────────────
n_fire_ok = n_fire_skip = n_default = 0

with open(STREAM_CACHE, "a", encoding="utf-8") as cache_f:

    # ── 1. Anh LUA: dung SAM3 ─────────────────────────────────────────────────
    if fire_todo:
        for img_path in tqdm(fire_todo, desc="SAM3 fire"):
            try:
                image = Image.open(img_path).convert("RGB")
                W, H  = image.size
                inputs = sam_processor(images=image, text="fire",
                                       return_tensors="pt").to(device)
                if device == "cuda":
                    inputs = {k: v.to(torch.float16)
                              if v.dtype == torch.float32 else v
                              for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = sam_model(**inputs)
                results = sam_processor.post_process_instance_segmentation(
                    outputs, threshold=0.30, mask_threshold=0.5,
                    target_sizes=[(H, W)]
                )[0]

                # Lay masks an toan: kiem tra None truoc, KHONG dung "or" voi Tensor
                raw = results.get("masks")
                if raw is None:
                    raw = results.get("segmentation")
                if raw is None:
                    masks_list = []
                elif isinstance(raw, torch.Tensor):
                    # raw co dang (N, H, W) -> chuyen thanh list N masks
                    masks_list = [raw[i] for i in range(raw.shape[0])]
                else:
                    masks_list = list(raw)

                pts = [p for m in masks_list
                       if (p := adaptive_baseline(
                               (m.cpu().numpy() > 0).astype(np.uint8)))]
                if pts:
                    best = max(pts, key=lambda p: p[1])
                    cache_f.write(json.dumps({
                        "image_path": img_path,
                        "has_fire"  : 1,
                        "p_fire"    : list(best)
                    }) + "\n")
                    cache_f.flush()
                    n_fire_ok += 1
                else:
                    n_fire_skip += 1
            except Exception as e:
                print(f"[SKIP fire] {os.path.basename(img_path)}: {e}")
                n_fire_skip += 1

    # ── 2. Anh DEFAULT: gan nhan has_fire=0 tu dong ───────────────────────────
    default_todo = [p for p in default_imgs if p not in done]
    for img_path in tqdm(default_todo, desc="Default (no-fire)"):
        cache_f.write(json.dumps({
            "image_path": img_path,
            "has_fire"  : 0,
            "p_fire"    : [0.0, 0.0]
        }) + "\n")
        cache_f.flush()
        n_default += 1

# ── Gop thanh JSON cuoi ───────────────────────────────────────────────────────
final = []
with open(STREAM_CACHE, encoding="utf-8") as f:
    for line in f:
        try: final.append(json.loads(line.strip()))
        except: pass
with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)

n_fire_total   = sum(1 for d in final if d.get("has_fire") == 1)
n_nf_total     = sum(1 for d in final if d.get("has_fire") == 0)
print(f"\nKet qua:")
print(f"  Fire label   : {n_fire_total} anh  (SAM3 ok={n_fire_ok}, skip={n_fire_skip})")
print(f"  No-fire label: {n_nf_total} anh  (default tu dong)")
print(f"  Tong JSON    : {len(final)} ban ghi -> {JSON_OUTPUT_PATH}")

# Giai phong VRAM neu co
if fire_todo:
    del sam_model, sam_processor
    torch.cuda.empty_cache()
    print("VRAM cleared, san sang train!")
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — TRAIN MOBILENETV4 MULTI-TASK TREN KAGGLE T4
# Output [p_fire, x_norm, y_norm]: phat hien + dinh vi day lua
# ─────────────────────────────────────────────────────────────────────────────
"""
import os, json, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# ── Transforms ────────────────────────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Dataset ────────────────────────────────────────────────────────────────────
class FireDataset(Dataset):
    def __init__(self, json_path, split="train", val_ratio=0.15, seed=42):
        with open(json_path, encoding="utf-8") as f:
            all_data = json.load(f)
        random.seed(seed)
        idx   = list(range(len(all_data)))
        random.shuffle(idx)
        n_val = max(1, int(len(all_data) * val_ratio))
        self.data = ([all_data[i] for i in idx[:n_val]]  if split == "val"
                     else [all_data[i] for i in idx[n_val:]])
        self.tfm  = train_tfm if split == "train" else val_tfm
        n_fire = sum(1 for d in self.data if d.get("has_fire", 1))
        print(f"  [{split}] {len(self.data)} samples  "
              f"(fire={n_fire}, no-fire={len(self.data)-n_fire})")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d        = self.data[i]
        img      = Image.open(d["image_path"]).convert("RGB")
        has_fire = float(d.get("has_fire", 1))
        x, y     = d["p_fire"] if has_fire else (0.0, 0.0)
        return self.tfm(img), torch.tensor([has_fire, x, y], dtype=torch.float32)

# ── Model ──────────────────────────────────────────────────────────────────────
BACKBONE    = "mobilenetv4_conv_medium"
IN_FEATURES = 1280  # mobilenetv4_conv_medium output feature dimension

class FireGrounder(nn.Module):
    def __init__(self, backbone=BACKBONE, pretrained=True, in_features=IN_FEATURES):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        self.head = nn.Linear(in_features, 3)   # [p_fire, x_norm, y_norm]
        print(f"  Backbone={backbone}  features={in_features}  "
              f"params={sum(p.numel() for p in self.parameters()):,}")
    def forward(self, x):
        return torch.sigmoid(self.head(self.backbone(x)))

# ── Loss ───────────────────────────────────────────────────────────────────────
class FireLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_reg=2.0):
        super().__init__()
        self.w_cls, self.w_reg = w_cls, w_reg
        self.bce  = nn.BCELoss()
        self.sl1  = nn.SmoothL1Loss()
    def forward(self, preds, labels):
        cls_loss = self.bce(preds[:, 0], labels[:, 0])
        mask     = labels[:, 0] > 0.5
        reg_loss = self.sl1(preds[mask, 1:], labels[mask, 1:]) if mask.any() \
                   else torch.tensor(0.0, device=preds.device)
        return self.w_cls * cls_loss + self.w_reg * reg_loss

# ── Config ─────────────────────────────────────────────────────────────────────
JSON_PATH  = "/kaggle/working/fire_ground_dataset/dataset_labels.json"
OUT_DIR    = "/kaggle/working"
EPOCHS     = 30
BATCH      = 64
LR         = 1e-4
VAL_RATIO  = 0.15

# ── Init ───────────────────────────────────────────────────────────────────────
print("Dataset:")
train_ds = FireDataset(JSON_PATH, "train", VAL_RATIO)
val_ds   = FireDataset(JSON_PATH, "val",   VAL_RATIO)
train_dl = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2)

print("Model:")
model     = FireGrounder(BACKBONE, pretrained=True).to(device)
criterion = FireLoss(w_cls=1.0, w_reg=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, LR/100)

# ── Train ──────────────────────────────────────────────────────────────────────
best_loss = float("inf")
best_path = os.path.join(OUT_DIR, "best.pth")
history   = []

print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'Acc%':>7} {'MAE':>8} {'LR':>10}")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    trn_loss = 0.0
    for imgs, lbl in train_dl:
        imgs, lbl = imgs.to(device), lbl.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbl)
        loss.backward(); optimizer.step()
        trn_loss += loss.item() * len(imgs)
    trn_loss /= len(train_dl.dataset)

    # Eval
    model.eval()
    val_loss = 0.0
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbl in val_dl:
            imgs, lbl = imgs.to(device), lbl.to(device)
            p = model(imgs)
            val_loss += criterion(p, lbl).item() * len(imgs)
            all_p.append(p.cpu()); all_l.append(lbl.cpu())
    val_loss /= len(val_dl.dataset)

    pall, lall = torch.cat(all_p), torch.cat(all_l)
    acc = ((pall[:,0]>0.5) == (lall[:,0]>0.5)).float().mean().item() * 100
    mask = lall[:,0] > 0.5
    mae  = ((pall[mask,1:]-lall[mask,1:]).abs()*224).pow(2).sum(1).sqrt().mean().item() \
           if mask.any() else float("nan")

    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]
    mae_s  = f"{mae:.1f}" if mae == mae else "  -"
    history.append({"epoch":epoch,"train":trn_loss,"val":val_loss,"acc":acc,"mae":mae})

    mark = " ★" if val_loss < best_loss else ""
    print(f"{epoch:>6d} {trn_loss:>10.5f} {val_loss:>10.5f} "
          f"{acc:>6.1f}% {mae_s:>7}px {lr_now:>10.2e}{mark}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({"model": model.state_dict(), "backbone": BACKBONE,
                    "epoch": epoch, "val_loss": val_loss}, best_path)

    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss, "backbone": BACKBONE},
               os.path.join(OUT_DIR, "last.pth"))

with open(os.path.join(OUT_DIR, "train_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nBest val loss: {best_loss:.5f}")
print(f"Saved: {best_path}")

# Hien link download truc tiep tren man hinh
from IPython.display import FileLink, display
print("\n>>> CLICK VAO LINK DUOI DE TAI VE MAY:")
display(FileLink("best.pth"))
display(FileLink("last.pth"))
display(FileLink("train_history.json"))
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — PREVIEW KET QUA
# ─────────────────────────────────────────────────────────────────────────────
"""
import json, random
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ckpt = torch.load("/kaggle/working/best.pth", map_location="cpu")
print(f"Backbone : {ckpt['backbone']}")
print(f"Val loss : {ckpt['val_loss']:.5f}")

class FireGrounder(nn.Module):
    def __init__(self, backbone, pretrained=False, in_features=1280):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        self.head = nn.Linear(in_features, 3)
    def forward(self, x):
        return torch.sigmoid(self.head(self.backbone(x)))

model = FireGrounder(ckpt["backbone"])
model.load_state_dict(ckpt["model"])
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

with open("/kaggle/working/fire_ground_dataset/dataset_labels.json") as f:
    data = json.load(f)

fire_data = [d for d in data if d.get("has_fire", 1)]
samples   = random.sample(fire_data, min(6, len(fire_data)))
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, item in zip(axes.flatten(), samples):
    img  = Image.open(item["image_path"]).convert("RGB")
    W, H = img.size
    with torch.no_grad():
        pred = model(tfm(img).unsqueeze(0)).squeeze()
    p_fire, px, py = pred[0].item(), int(pred[1]*W), int(pred[2]*H)
    lx, ly = int(item["p_fire"][0]*W), int(item["p_fire"][1]*H)
    r = max(8, min(W,H)//40)
    draw = ImageDraw.Draw(img)
    draw.ellipse([lx-r,ly-r,lx+r,ly+r], outline="lime",  width=3)  # SAM3 label
    draw.ellipse([px-r,py-r,px+r,py+r], outline="red",   width=3)  # prediction
    draw.line([px-r,py,px+r,py], fill="red", width=2)
    draw.line([px,py-r,px,py+r], fill="red", width=2)
    ax.imshow(img)
    ax.set_title(f"p_fire={p_fire:.2f}  err={(((lx-px)**2+(ly-py)**2)**0.5):.1f}px",
                 fontsize=9)
    ax.axis("off")
fig.suptitle("SAM3 label (xanh) vs MobileNetV4 predict (do)", fontsize=12)
plt.tight_layout()
plt.savefig("/kaggle/working/prediction_preview.png", dpi=120)
plt.show()

# Learning curve
with open("/kaggle/working/train_history.json") as f: hist = json.load(f)
ep = [h["epoch"] for h in hist]
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
axes2[0].plot(ep, [h["train"] for h in hist], label="Train")
axes2[0].plot(ep, [h["val"]   for h in hist], label="Val")
axes2[0].set_title("Loss"); axes2[0].legend()
axes2[1].plot(ep, [h["acc"] for h in hist], color="green")
axes2[1].set_title("Accuracy (%)")
axes2[2].plot(ep, [h.get("mae", 0) for h in hist], color="orange")
axes2[2].set_title("MAE (pixels)")
plt.tight_layout()
plt.savefig("/kaggle/working/learning_curve.png", dpi=120)
plt.show()

from IPython.display import FileLink, display
print("\n>>> TAI ANH KET QUA VE MAY:")
display(FileLink("prediction_preview.png"))
display(FileLink("learning_curve.png"))
print("Sau do: python inference.py --mode image --input fire.jpg --checkpoint best.pth")
"""


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — SO SANH MOBILENET TRUOC VA SAU KHI TRAIN
#
# 1. Bang so sanh dinh luong (Accuracy, Precision, Recall, F1, MAE error pixel)
# 2. Hinh anh truc quan 3 cot: Ground Truth vs Truoc train vs Sau train
# 3. Bieu do so sanh (Metrics Bar Chart + Error Distribution)
# ─────────────────────────────────────────────────────────────────────────────
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# ── 1. Load Checkpoint & Thiet lap 2 Model ────────────────────────────────────
CKPT_PATH = "/kaggle/working/best.pth"
JSON_PATH = "/kaggle/working/fire_ground_dataset/dataset_labels.json"

# Tu dong fallback sang /kaggle/input/ neu trong /kaggle/working/ khong co
if not os.path.exists(CKPT_PATH):
    found_ckpt = [os.path.join(root, f) for root, _, files in os.walk("/kaggle/input") for f in files if f == "best.pth"]
    if found_ckpt:
        CKPT_PATH = found_ckpt[0]

if not os.path.exists(JSON_PATH):
    found_json = [os.path.join(root, f) for root, _, files in os.walk("/kaggle/input") for f in files if "dataset_labels" in f and f.endswith(".json")]
    if found_json:
        JSON_PATH = found_json[0]

print(f"-> Su dung Checkpoint: {CKPT_PATH}")
print(f"-> Su dung Nhan JSON : {JSON_PATH}")

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Chua tim thay best.pth o ca /kaggle/working lan /kaggle/input!")
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"Chua tim thay dataset_labels.json o ca /kaggle/working lan /kaggle/input!")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
backbone_name = ckpt.get("backbone", "mobilenetv4_conv_medium")
print(f"Backbone: {backbone_name}")
print(f"Checkpoint Best Val Loss: {ckpt.get('val_loss', 0.0):.5f} (Epoch {ckpt.get('epoch', '-')})")

class FireGrounder(nn.Module):
    def __init__(self, backbone, pretrained=True, in_features=1280):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, global_pool="avg")
        self.head = nn.Linear(in_features, 3)  # [p_fire, x_norm, y_norm]
    def forward(self, x):
        return torch.sigmoid(self.head(self.backbone(x)))

# Model TRUOC KHI TRAIN: ImageNet pretrained backbone + Random initialized head
torch.manual_seed(42)
print("\n[1/4] Khoi tao Model TRUOC KHI TRAIN (Untrained Head)...")
model_before = FireGrounder(backbone_name, pretrained=True).to(device)
model_before.eval()

# Model SAU KHI TRAIN: Load trong so tu best.pth
print("[2/4] Khoi tao Model SAU KHI TRAIN (Checkpoint best.pth)...")
model_after = FireGrounder(backbone_name, pretrained=False).to(device)
model_after.load_state_dict(ckpt["model"])
model_after.eval()

# ── 2. Load Validation Set ────────────────────────────────────────────────────
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

with open(JSON_PATH, encoding="utf-8") as f:
    all_data = json.load(f)

# Dung seed=42 giong het luc train o Cell 3 de tach tap Val
random.seed(42)
indices = list(range(len(all_data)))
random.shuffle(indices)
n_val = max(1, int(len(all_data) * 0.15))
val_data = [all_data[i] for i in indices[:n_val]]

val_fire    = [d for d in val_data if d.get("has_fire", 1) == 1]
val_default = [d for d in val_data if d.get("has_fire", 1) == 0]
print(f"\nTap Validation: {len(val_data)} mau (Fire={len(val_fire)}, Default/No-Fire={len(val_default)})")

# Tao cache duong dan anh de tranh truong hop dataset mount ten hoi khac
img_cache = {f: os.path.join(root, f)
             for root, _, files in os.walk("/kaggle/input")
             for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))}

def resolve_path(p):
    if os.path.exists(p): return p
    return img_cache.get(os.path.basename(p), p)

# ── 3. Danh gia Dinh luong tren Validation Set ────────────────────────────────
def evaluate(model, data_list):
    y_true, y_pred, p_list = [], [], []
    fire_errors = []
    
    with torch.no_grad():
        for item in data_list:
            real_path = resolve_path(item["image_path"])
            if not os.path.exists(real_path):
                continue
            img = Image.open(real_path).convert("RGB")
            t_img = tfm(img).unsqueeze(0).to(device)
            out = model(t_img).squeeze(0).cpu().numpy()
            
            p_fire = float(out[0])
            pred_cls = 1 if p_fire >= 0.5 else 0
            true_cls = int(item.get("has_fire", 1))
            
            y_true.append(true_cls)
            y_pred.append(pred_cls)
            p_list.append(p_fire)
            
            if true_cls == 1:
                lx, ly = item["p_fire"][0] * 224.0, item["p_fire"][1] * 224.0
                px, py = out[1] * 224.0, out[2] * 224.0
                dist = np.sqrt((lx - px)**2 + (ly - py)**2)
                fire_errors.append(dist)
                
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    acc = (tp + tn) / len(y_true) * 100.0 if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    errs = np.array(fire_errors) if len(fire_errors) > 0 else np.array([0.0])
    mean_err = float(np.mean(errs))
    median_err = float(np.median(errs))
    acc_15px = float(np.mean(errs <= 15.0) * 100.0)
    acc_30px = float(np.mean(errs <= 30.0) * 100.0)
    
    return {
        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1-Score (%)": f1,
        "False Positive (anh bao sai)": int(fp),
        "False Negative (anh bo sot)": int(fn),
        "Mean Error (px)": mean_err,
        "Median Error (px)": median_err,
        "Do chinh xac < 15px (%)": acc_15px,
        "Do chinh xac < 30px (%)": acc_30px,
        "errors": errs,
        "p_fire": p_list
    }

print("\n[3/4] Dang tinh toan metrics so sanh...")
res_before = evaluate(model_before, val_data)
res_after  = evaluate(model_after,  val_data)

# In bang so sanh
print("\n" + "═"*78)
print(f"{'CHI SO (METRIC)':<32} | {'TRUOC KHI TRAIN':<16} | {'SAU KHI TRAIN':<16} | {'THAY DOI':<10}")
print("─"*78)

metrics_to_show = [
    ("Accuracy (Phan loai)", "Accuracy (%)", "%", True),
    ("Precision (Do xac thuc)", "Precision (%)", "%", True),
    ("Recall (Do nhay)", "Recall (%)", "%", True),
    ("F1-Score", "F1-Score (%)", "%", True),
    ("Bao dong gia (False Pos)", "False Positive (anh bao sai)", "", False),
    ("Bo sot lua (False Neg)", "False Negative (anh bo sot)", "", False),
    ("Sai so trung binh day lua", "Mean Error (px)", " px", False),
    ("Sai so trung vi (Median)", "Median Error (px)", " px", False),
    ("Ty le sai so rat nho (<15px)", "Do chinh xac < 15px (%)", "%", True),
    ("Ty le sai so chap nhan (<30px)", "Do chinh xac < 30px (%)", "%", True),
]

for label, key, unit, higher_is_better in metrics_to_show:
    vb = res_before[key]
    va = res_after[key]
    diff = va - vb
    sign = "+" if diff > 0 else ""
    # Mui ten danh gia tot/xau
    if higher_is_better:
        status = "▲ Tot" if diff > 0 else ("▼ Giam" if diff < 0 else "=")
    else:
        status = "▼ Tot" if diff < 0 else ("▲ Tang" if diff > 0 else "=")
    
    if isinstance(vb, float):
        print(f"{label:<32} | {vb:>13.2f}{unit:<3} | {va:>13.2f}{unit:<3} | {sign}{diff:>6.2f}{unit} ({status})")
    else:
        print(f"{label:<32} | {vb:>16} | {va:>16} | {sign}{diff:>6} ({status})")
print("═"*78)

# ── 4. Truc quan hoa Hinh anh: 3 Cot (Ground Truth vs Truoc vs Sau) ────────────
print("\n[4/4] Dang tao hinh anh so sanh truc quan...")

# Chon mau da dang: 4 anh lua (ngau nhien co dinh) + 2 anh default
valid_val_fire = [d for d in val_fire if os.path.exists(resolve_path(d["image_path"]))]
valid_val_def  = [d for d in val_default if os.path.exists(resolve_path(d["image_path"]))]

rng = random.Random(123)
sample_fire = rng.sample(valid_val_fire, min(4, len(valid_val_fire)))
sample_def  = rng.sample(valid_val_def, min(2, len(valid_val_def))) if len(valid_val_def) >= 2 else []
samples = sample_fire + sample_def

n_rows = len(samples)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
plt.subplots_adjust(hspace=0.25, wspace=0.1)

for i, item in enumerate(samples):
    real_path = resolve_path(item["image_path"])
    img_orig = Image.open(real_path).convert("RGB")
    W, H = img_orig.size
    t_img = tfm(img_orig).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_b = model_before(t_img).squeeze(0).cpu().numpy()
        out_a = model_after(t_img).squeeze(0).cpu().numpy()
        
    has_fire = int(item.get("has_fire", 1))
    radius = max(6, min(W, H) // 40)
    
    # --- Cot 1: Ground Truth ---
    ax1 = axes[i, 0] if n_rows > 1 else axes[0]
    img1 = img_orig.copy()
    draw1 = ImageDraw.Draw(img1)
    if has_fire == 1:
        lx, ly = int(item["p_fire"][0] * W), int(item["p_fire"][1] * H)
        draw1.ellipse([lx-radius, ly-radius, lx+radius, ly+radius], outline="#00FF00", width=4)
        draw1.ellipse([lx-2, ly-2, lx+2, ly+2], fill="#00FF00")
        title_gt = f"Ground Truth: CO LUA\nDay lua: ({lx}, {ly})"
        box_color = "#00FF00"
    else:
        title_gt = "Ground Truth: KHONG LUA (Default)"
        box_color = "#333333"
    ax1.imshow(img1)
    ax1.set_title(title_gt, fontsize=11, fontweight="bold", color="darkgreen" if has_fire else "black")
    ax1.axis("off")
    
    # --- Cot 2: Truoc khi Train ---
    ax2 = axes[i, 1] if n_rows > 1 else axes[1]
    img2 = img_orig.copy()
    draw2 = ImageDraw.Draw(img2)
    pb = float(out_b[0])
    pxb, pyb = int(out_b[1] * W), int(out_b[2] * H)
    
    if pb >= 0.5:
        # Ve diem du doan cua model truoc train
        draw2.ellipse([pxb-radius, pyb-radius, pxb+radius, pyb+radius], outline="#FF3333", width=3)
        draw2.line([pxb-radius*1.5, pyb, pxb+radius*1.5, pyb], fill="#FF3333", width=2)
        draw2.line([pxb, pyb-radius*1.5, pxb, pyb+radius*1.5], fill="#FF3333", width=2)
        if has_fire == 1:
            err_b = np.sqrt((item["p_fire"][0]*W - pxb)**2 + (item["p_fire"][1]*H - pyb)**2)
            title_b = f"Truoc Train: DOAN CO LUA (p={pb:.2f})\nSai so: {err_b:.1f}px (Lech nhieu)"
        else:
            title_b = f"Truoc Train: DOAN CO LUA (p={pb:.2f})\n[BAO SAI / FALSE ALARM]"
    else:
        title_b = f"Truoc Train: DOAN KHONG LUA (p={pb:.2f})"
        if has_fire == 1:
            title_b += "\n[BO SOT / FALSE NEGATIVE]"
            
    ax2.imshow(img2)
    ax2.set_title(title_b, fontsize=11, color="red" if (has_fire and pb < 0.5) or (not has_fire and pb >= 0.5) else "black")
    ax2.axis("off")
    
    # --- Cot 3: Sau khi Train ---
    ax3 = axes[i, 2] if n_rows > 1 else axes[2]
    img3 = img_orig.copy()
    draw3 = ImageDraw.Draw(img3)
    pa = float(out_a[0])
    pxa, pya = int(out_a[1] * W), int(out_a[2] * H)
    
    if pa >= 0.5:
        # Ve diem Ground Truth mau xanh nhe de so sanh
        if has_fire == 1:
            draw3.ellipse([lx-radius, ly-radius, lx+radius, ly+radius], outline="#00FF00", width=2)
        # Ve diem du doan Sau Train mau xanh duong/cam
        draw3.ellipse([pxa-radius, pya-radius, pxa+radius, pya+radius], outline="#FF8800", width=3)
        draw3.line([pxa-radius*1.5, pya, pxa+radius*1.5, pya], fill="#FF8800", width=2)
        draw3.line([pxa, pya-radius*1.5, pxa, pya+radius*1.5], fill="#FF8800", width=2)
        
        if has_fire == 1:
            err_a = np.sqrt((item["p_fire"][0]*W - pxa)**2 + (item["p_fire"][1]*H - pya)**2)
            title_a = f"Sau Train: CO LUA (p={pa:.2f})\nSai so: {err_a:.1f}px (Chinh xac cao)"
        else:
            title_a = f"Sau Train: DOAN CO LUA (p={pa:.2f})\n[BAO SAI]"
    else:
        title_a = f"Sau Train: KHONG LUA (p={pa:.2f})\n[Dung thuc te]"
        
    ax3.imshow(img3)
    ax3.set_title(title_a, fontsize=11, color="blue" if pa >= 0.5 and has_fire else ("darkgreen" if pa < 0.5 and not has_fire else "red"))
    ax3.axis("off")

plt.suptitle("SO SANH CHI TIET: GROUND TRUTH (Xanh la) vs TRUOC TRAIN (Do) vs SAU TRAIN (Cam/Xanh)", fontsize=14, y=0.99)
plt.tight_layout()
VISUAL_PATH = "/kaggle/working/compare_visual_before_after.png"
plt.savefig(VISUAL_PATH, dpi=120, bbox_inches="tight")
plt.show()

# ── 5. Bieu do So sanh Dinh luong (Bar Chart & Error Distribution) ─────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# 1. So sanh Kha nang Phan loai (%)
cls_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
vals_b = [res_before['Accuracy (%)'], res_before['Precision (%)'], res_before['Recall (%)'], res_before['F1-Score (%)']]
vals_a = [res_after['Accuracy (%)'], res_after['Precision (%)'], res_after['Recall (%)'], res_after['F1-Score (%)']]
x = np.arange(len(cls_labels))
width = 0.35

axes2[0].bar(x - width/2, vals_b, width, label='Truoc Train', color='#e74c3c')
axes2[0].bar(x + width/2, vals_a, width, label='Sau Train', color='#2ecc71')
axes2[0].set_ylabel('Phan tram (%)')
axes2[0].set_title('Chi so Phan loai Lua (Classification)')
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(cls_labels)
axes2[0].set_ylim(0, 105)
axes2[0].legend()
axes2[0].grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(cls_labels)):
    axes2[0].text(x[i] - width/2, vals_b[i] + 1.5, f"{vals_b[i]:.1f}%", ha='center', fontsize=9)
    axes2[0].text(x[i] + width/2, vals_a[i] + 1.5, f"{vals_a[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

# 2. So sanh Sai so Toa do Day lua (pixels tren anh 224x224)
loc_labels = ['Sai so TB (Mean)', 'Sai so Trung vi (Median)']
loc_b = [res_before['Mean Error (px)'], res_before['Median Error (px)']]
loc_a = [res_after['Mean Error (px)'], res_after['Median Error (px)']]
x2 = np.arange(len(loc_labels))

axes2[1].bar(x2 - width/2, loc_b, width, label='Truoc Train', color='#e74c3c')
axes2[1].bar(x2 + width/2, loc_a, width, label='Sau Train', color='#3498db')
axes2[1].set_ylabel('Khoang cach sai so (pixels) - Thap hon la tot')
axes2[1].set_title('Sai so Dinh vi Day lua (Pixel Error)')
axes2[1].set_xticks(x2)
axes2[1].set_xticklabels(loc_labels)
axes2[1].legend()
axes2[1].grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(loc_labels)):
    axes2[1].text(x2[i] - width/2, loc_b[i] + 1.0, f"{loc_b[i]:.1f}px", ha='center', fontsize=9)
    axes2[1].text(x2[i] + width/2, loc_a[i] + 1.0, f"{loc_a[i]:.1f}px", ha='center', fontsize=9, fontweight='bold')

# 3. Phan bo Sai so Toa do (Error Distribution Histogram)
axes2[2].hist(res_before['errors'], bins=15, alpha=0.6, label='Truoc Train', color='#e74c3c', edgecolor='black')
axes2[2].hist(res_after['errors'],  bins=15, alpha=0.7, label='Sau Train',  color='#2ecc71', edgecolor='black')
axes2[2].axvline(res_after['Mean Error (px)'], color='darkgreen', linestyle='dashed', linewidth=2, label=f"Mean Sau: {res_after['Mean Error (px)']:.1f}px")
axes2[2].set_xlabel('Sai so khoang cach (pixels)')
axes2[2].set_ylabel('So luong mau')
axes2[2].set_title('Phan phoi Sai so Toa do tren Tap Val')
axes2[2].legend()
axes2[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
CHARTS_PATH = "/kaggle/working/compare_metrics_charts.png"
plt.savefig(CHARTS_PATH, dpi=120)
plt.show()

# ── 6. Luu bao cao JSON & Link Tai ───────────────────────────────────────────
report_data = {
    "backbone": backbone_name,
    "val_samples_total": len(val_data),
    "val_fire_count": len(val_fire),
    "val_default_count": len(val_default),
    "before_training": {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in res_before.items() if k not in ["errors", "p_fire"]},
    "after_training": {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in res_after.items() if k not in ["errors", "p_fire"]},
}
REPORT_PATH = "/kaggle/working/comparison_report.json"
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)

print("\n" + "═"*78)
print(">>> HOAN THANH SO SANH! CLICK VAO LINK DUOI DE TAI CAC FILE VE:")
display(FileLink("compare_visual_before_after.png"))
display(FileLink("compare_metrics_charts.png"))
display(FileLink("comparison_report.json"))
print("═"*78)
"""