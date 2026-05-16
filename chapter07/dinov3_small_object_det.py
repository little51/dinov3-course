import os, sys, re
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

from skimage.measure import label as connected_label, regionprops
from sklearn.decomposition import PCA

# ─── Paths (relative) ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(SCRIPT_DIR, 'test_samples')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ────────────────────────────────────────────────
IMG_SIZE = 2048
PATCH_SIZE = 16
THRESHOLD_PCT = 99
MIN_AREA_PCT = 0.00001
LAYER_BEST = 19       # layer with highest Dice for box detection
LAYERS_VIS = [3, 7, 11, 15, 19, 23]  # layers for feature vis (extract all)
LAYERS_DISPLAY = [3, 11, 15, 19]     # layers to show in vis grid (4 cols)
PCA_LAYERS = [7, 15, 23]             # layers for PCA vis
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Input: {IMG_SIZE}x{IMG_SIZE}  |  Feature grid: {IMG_SIZE//PATCH_SIZE}x{IMG_SIZE//PATCH_SIZE}")

# ─── 1. Load test image ────────────────────────────────────
img_path = os.path.join(SAMPLE_DIR, '2253.tif')
mask_path = os.path.join(SAMPLE_DIR, '2253_mask.tif')

if not os.path.exists(img_path):
    print(f"[ERROR] Test image not found: {img_path}")
    sys.exit(1)

img_pil = Image.open(img_path).convert('RGB')
mask_pil = Image.open(mask_path).convert('L')
W, H = img_pil.size
print(f"Original size: {W}x{H}")

# ─── 2. Load DINOv3 model ──────────────────────────────────
print("Loading DINOv3.sat493m...")
model = timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0)
model = model.to(DEVICE)
model.eval()
print(f"DINOv3 ViT-L: {len(model.blocks)} layers, {model.embed_dim}-dim")

# ─── 3. Preprocess + multi-layer feature extraction ────────
print("Extracting features from multiple layers...")

def preprocess(pil_img, size):
    img = np.array(pil_img.resize((size, size), Image.LANCZOS))
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    m = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    return ((t.unsqueeze(0) - m) / s).to(DEVICE), img

# ---- 3a. 2048x2048 for box detection (layer 19 only) ----
img_big = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
img_np_2048 = np.array(img_big)
img_t_2048, _ = preprocess(img_pil, IMG_SIZE)

feat_store = {}
def hook_fn(m, i, o):
    feat_store['out'] = o[0].detach()
handle = model.blocks[LAYER_BEST].register_forward_hook(hook_fn)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model.forward_features(img_t_2048)
handle.remove()
mem_peak = torch.cuda.max_memory_allocated() / 1e9

grid_size = IMG_SIZE // PATCH_SIZE
patches_best = feat_store['out'][5:]
patch_norm = patches_best.norm(dim=1).cpu().numpy()
feat_2d_best = patch_norm.reshape(grid_size, grid_size)

# upsample for heatmap
feat_t = torch.from_numpy(feat_2d_best).float().unsqueeze(0).unsqueeze(0)
feat_up_best = F.interpolate(feat_t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')[0, 0].numpy()
vmin, vmax = np.percentile(feat_up_best, [5, 98])
feat_disp = np.clip((feat_up_best - vmin) / (vmax - vmin + 1e-8), 0, 1)

# threshold + connected components
threshold_val = np.percentile(patch_norm, THRESHOLD_PCT)
binary = (patch_norm > threshold_val).astype(np.uint8).reshape(grid_size, grid_size)
labeled = connected_label(binary)
regions = regionprops(labeled)
MIN_AREA = max(1, int(grid_size * grid_size * MIN_AREA_PCT))
boxes = []
for reg in regions:
    if reg.area < MIN_AREA: continue
    minr, minc, maxr, maxc = reg.bbox
    scale = IMG_SIZE / grid_size
    x1, y1 = int(minc * scale), int(minr * scale)
    x2, y2 = int(maxc * scale), int(maxr * scale)
    if max(x2-x1, y2-y1) / (min(x2-x1, y2-y1) + 1e-8) < 5:
        boxes.append((x1, y1, x2, y2, reg.area))
boxes.sort(key=lambda b: b[4], reverse=True)
print(f"Box detection: {len(boxes)} candidates, VRAM={mem_peak:.2f}GB")

# ---- 3b. 896x896 for multi-layer feature extraction ----
VIS_SIZE = 896
grid_vis = VIS_SIZE // PATCH_SIZE
img_t_vis, img_np_vis = preprocess(img_pil, VIS_SIZE)

feature_maps = {}
def make_hook(name):
    def hook(m, i, o): feature_maps[name] = o[0].detach()
    return hook

handles = []
for idx in LAYERS_VIS:
    handles.append(model.blocks[idx].register_forward_hook(make_hook(f'block_{idx}')))
with torch.no_grad():
    _ = model.forward_features(img_t_vis)
for h in handles: h.remove()

# ─── 4. Comprehensive visualization ────────────────────────
print("Generating visualization...")

fig, axes = plt.subplots(3, 4, figsize=(24, 18))

# ==================== ROW 0: Box detection result ====================
ax = axes[0, 0]
ax.imshow(np.array(img_pil))
ax.set_title('(a) 原图 (512×512)', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[0, 1]
ax.imshow(img_np_2048)
for x1, y1, x2, y2, area in boxes:
    box_ratio = (x2-x1)*(y2-y1) / (IMG_SIZE*IMG_SIZE)
    if box_ratio < 0.005:      color, lw = 'red', 0.8
    elif box_ratio < 0.03:     color, lw = 'orange', 1.2
    else:                      color, lw = 'blue', 1.8
    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=lw, edgecolor=color, facecolor='none', alpha=0.6)
    ax.add_patch(rect)
ax.set_title(f'(b) 放大 2048×2048 + 框选 ({len(boxes)} 个)', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[0, 2]
ax.imshow(np.array(mask_pil), cmap='gray')
ax.set_title('(c) 建筑标注 (GT)', fontsize=12, fontweight='bold')
ax.axis('off')

ax = axes[0, 3]
ax.imshow(img_np_2048, alpha=0.5)
ax.imshow(feat_disp, cmap='jet', alpha=0.5)
ax.set_title(f'(d) DINOv3 第 {LAYER_BEST} 层特征', fontsize=12, fontweight='bold')
ax.axis('off')

# ==================== ROW 1: Multi-layer feature maps ====================
layer_names = {3: 'Layer 3 (浅层/纹理)',
               11: 'Layer 11 (中层/部件)',
               15: 'Layer 15 (中深层/语义)',
               19: 'Layer 19 (深层/目标)'}

for i, idx in enumerate(LAYERS_DISPLAY):
    feat = feature_maps[f'block_{idx}']
    patches = feat[5:]
    acts = patches.norm(dim=1).cpu().numpy().reshape(grid_vis, grid_vis)
    acts_t = torch.from_numpy(acts).float().unsqueeze(0).unsqueeze(0)
    acts_up = F.interpolate(acts_t, size=(VIS_SIZE, VIS_SIZE), mode='bilinear')[0, 0].numpy()
    acts_norm = (acts_up - acts_up.min()) / (acts_up.max() - acts_up.min() + 1e-8)
    
    ax = axes[1, i]
    ax.imshow(img_np_vis, alpha=0.4)
    ax.imshow(acts_norm, cmap='jet', alpha=0.6)
    ax.set_title(f'({chr(ord("e")+i)}) {layer_names[idx]}', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Dice score against GT mask
    mask_down = F.interpolate(
        torch.from_numpy((np.array(mask_pil) > 0).astype(float)).float().unsqueeze(0).unsqueeze(0),
        size=(grid_vis, grid_vis), mode='nearest')[0, 0].numpy()
    pred = (acts > acts.mean()).astype(float)
    inter = (pred * mask_down).sum()
    dice = 2 * inter / (pred.sum() + mask_down.sum() + 1e-8)
    print(f"  Layer {idx:2d}: Dice vs GT = {dice:.4f}")

# Row 1 col 4 (index 3): Binary mask from 2048
ax = axes[1, 3]
ax.imshow(binary, cmap='gray', interpolation='nearest')
ax.set_title(f'(j) 二值掩码 (前 {THRESHOLD_PCT}%)', fontsize=11, fontweight='bold')
ax.axis('off')

# ==================== ROW 2: PCA + Stats ====================
for j, idx in enumerate(PCA_LAYERS):
    feat = feature_maps[f'block_{idx}']
    patches = feat[5:].cpu().numpy()
    pca = PCA(n_components=3)
    pca_r = pca.fit_transform(patches)
    pca_n = (pca_r - pca_r.min(axis=0)) / (pca_r.max(axis=0) - pca_r.min(axis=0) + 1e-8)
    pca_img = pca_n.reshape(grid_vis, grid_vis, 3)
    pca_t = torch.from_numpy(pca_img).float().permute(2, 0, 1).unsqueeze(0)
    pca_up = F.interpolate(pca_t, size=(VIS_SIZE, VIS_SIZE), mode='bilinear')[0].permute(1, 2, 0).numpy()
    
    ax = axes[2, j]
    ax.imshow(img_np_vis, alpha=0.3)
    ax.imshow(pca_up, alpha=0.7)
    labels = ['Layer 7 PCA', 'Layer 15 PCA', 'Layer 23 PCA']
    ax.set_title(f'({chr(ord("k")+j)}) {labels[j]}', fontsize=11, fontweight='bold')
    ax.axis('off')

# Stats panel
ax = axes[2, 3]
ax.axis('off')
small = sum(1 for *_, a in boxes if a/(grid_size*grid_size) < 0.005)
medium = sum(1 for *_, a in boxes if 0.005 <= a/(grid_size*grid_size) < 0.03)
large = sum(1 for *_, a in boxes if a/(grid_size*grid_size) >= 0.03)
stats = (
    f"  统计信息\n"
    f"  {'='*16}\n\n"
    f"  输入尺寸:      {IMG_SIZE}×{IMG_SIZE}\n"
    f"  特征网格:    {grid_size}×{grid_size}\n"
    f"  DINOv3 层:    {LAYER_BEST}\n"
    f"  阈值:       前 {THRESHOLD_PCT}%\n"
    f"  显存:        {mem_peak:.2f} GB\n\n"
    f"  候选区:      {len(boxes)}\n"
    f"    小(红):   {small}\n"
    f"    中(橙):   {medium}\n"
    f"    大(蓝):   {large}\n\n"
    f"  结论\n"
    f"  {'─'*8}\n"
    f"  DINOv3 零训练、零微\n"
    f"  调，仅靠特征范数阈\n"
    f"  值即可自动定位小目\n"
    f"  标候选区域。"
)
ax.text(0.05, 0.97, stats, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', linespacing=1.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, 'dinov3_small_object_result.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print(f"Done! Result saved to:")
print(f"  {save_path}")
print(f"{'='*50}")
print(f"  Candidates: {len(boxes)} (small={small}, medium={medium}, large={large})")
print(f"  VRAM:       {mem_peak:.2f} GB")
print(f"\n  Key claim: DINOv3 with zero training")
print(f"  automatically finds {len(boxes)} small-object")
print(f"  candidates from a single WHU image.")
