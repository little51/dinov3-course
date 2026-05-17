#!/usr/bin/env python3
"""
DINOv3 ViT-L 24层特征可视化
================================
可视化 DINOv3 ViT-Large（24层）每层"看到"什么，
展示从底层纹理到高层语义的逐层变化。

输出：一张 4×6 网格图，每格对应一层的特征热力图。
"""

import os, sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# HF mirror for China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ─── Paths ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(SCRIPT_DIR, 'dinov3.jpg')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Config ─────────────────────────────────────────────
INPUT_SIZE = 560          # 24层同时提取，560×560内存控制合理
PATCH_SIZE = 16
NUM_LAYERS = 24
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── 1. Load image ──────────────────────────────────────
if not os.path.exists(IMG_PATH):
    print(f"[ERROR] Image not found: {IMG_PATH}")
    sys.exit(1)

img_pil = Image.open(IMG_PATH).convert('RGB')
W, H = img_pil.size
print(f"Image: {W}x{H}")

# ─── 2. Load DINOv3 ViT-L ──────────────────────────────
print("Loading DINOv3 ViT-L (24 layers)...")
model = timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0)
model = model.to(DEVICE)
model.eval()
print(f"Model: DINOv3 ViT-L | {len(model.blocks)} layers | {model.embed_dim}-dim")

# ─── 3. Preprocess ─────────────────────────────────────
def preprocess(pil_img, size):
    img = np.array(pil_img.resize((size, size), Image.LANCZOS))
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    m = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    return ((t.unsqueeze(0) - m) / s).to(DEVICE), img

img_t, img_np = preprocess(img_pil, INPUT_SIZE)
grid = INPUT_SIZE // PATCH_SIZE  # 35×35

# ─── 4. Register hooks on ALL 24 layers ─────────────────
feature_maps = {}

def make_hook(name):
    def hook(m, i, o):
        feature_maps[name] = o[0].detach()
    return hook

handles = []
for idx in range(NUM_LAYERS):
    handles.append(model.blocks[idx].register_forward_hook(make_hook(f'block_{idx}')))

print("Extracting features from all 24 layers...")
with torch.no_grad():
    _ = model.forward_features(img_t)

for h in handles:
    h.remove()

print(f"Captured {len(feature_maps)} layers")

# ─── 5. Layer descriptions ──────────────────────────────
layer_labels = {
    0:  "Layer 0\n边缘/颜色检测",
    1:  "Layer 1\n局部纹理",
    2:  "Layer 2\n简单模式",
    3:  "Layer 3\n重复纹理",
    4:  "Layer 4\n方向性纹理",
    5:  "Layer 5\n局部形状",
    6:  "Layer 6\n部件线索",
    7:  "Layer 7\n简单部件",
    8:  "Layer 8\n部件轮廓",
    9:  "Layer 9\n中等部件",
    10: "Layer 10\n部件组合",
    11: "Layer 11\n语义部件",
    12: "Layer 12\n目标局部",
    13: "Layer 13\n目标片段",
    14: "Layer 14\n目标聚焦",
    15: "Layer 15\n目标检测",
    16: "Layer 16\n语义区域",
    17: "Layer 17\n类别区分",
    18: "Layer 18\n目标边界",
    19: "Layer 19\n语义分割",
    20: "Layer 20\n上下文理解",
    21: "Layer 21\n全局关系",
    22: "Layer 22\n抽象语义",
    23: "Layer 23\n高层概念",
}

# ─── 6. Visualize all 24 layers (4×6 grid) ────────────
print("Generating 4×6 grid visualization...")

fig, axes = plt.subplots(4, 6, figsize=(24, 16))

for idx in range(NUM_LAYERS):
    row, col = idx // 6, idx % 6
    ax = axes[row, col]

    feat = feature_maps[f'block_{idx}']
    patches = feat[5:]  # drop cls token
    acts = patches.norm(dim=1).cpu().numpy().reshape(grid, grid)

    # Upsample to input size for overlay
    acts_t = torch.from_numpy(acts).float().unsqueeze(0).unsqueeze(0)
    acts_up = F.interpolate(acts_t, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear')[0, 0].numpy()
    acts_norm = (acts_up - acts_up.min()) / (acts_up.max() - acts_up.min() + 1e-8)

    ax.imshow(img_np, alpha=0.4)
    im = ax.imshow(acts_norm, cmap='jet', alpha=0.6)
    ax.set_title(layer_labels[idx], fontsize=9, fontweight='bold')
    ax.axis('off')

# Add a shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cbar_ax, label='特征响应强度')

plt.suptitle(f'DINOv3 ViT-L (24层) 逐层特征可视化 — 从边缘到语义', fontsize=16, fontweight='bold', y=0.98)

save_path = os.path.join(OUTPUT_DIR, 'dinov3_24layers.png')
plt.savefig(save_path, dpi=180, bbox_inches='tight')
plt.close()

print(f"\nDone! Saved to: {save_path}")

# ─── 7. Print layer summary ────────────────────────────
print("\n" + "="*60)
print("DINOv3 ViT-L 24层语义层级概要")
print("="*60)

stages = [
    (0, 4,  "浅层：边缘、纹理、颜色", "检测局部梯度方向、重复纹理模式、色彩边界"),
    (5, 10, "中浅层：部件、模式", "形成简单形状（角、弧线）、组合为局部部件"),
    (11, 15, "中深层：目标、语义", "识别目标局部→完整目标、开始区分类别"),
    (16, 20, "深层：类别、上下文", "目标级语义、背景与前景分离、场景理解"),
    (21, 23, "最深层：抽象、关系", "目标间关系、全局场景抽象、高层概念"),
]

for start, end, title, desc in stages:
    print(f"\n  Layer {start}–{end} | {title}")
    print(f"    {desc}")

print("\n  💡 越深层的特征越适合分类和语义理解")
print("  💡 中层特征（Layer 11-15）最适合目标检测")
print("  💡 浅层特征（Layer 3-7）适合纹理分析和精细分割")
print("="*60)
