import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== 参数 ==========
MODEL_PATH = "./output/best_model.pth"
DATA_DIR = "./data/LoveDA"
OUTPUT_DIR = "./output"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 7
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_COLORS = {
    0: (0.6, 0.6, 0.6), 1: (1.0, 0.0, 0.0), 2: (1.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0), 4: (0.6, 0.4, 0.0), 5: (0.0, 0.6, 0.0), 6: (0.8, 0.6, 0.2),
}
cls_names = ['bg','bld','road','water','barren','forest','agri']

# ─── 模型 ───────────────────────────────────────────────────────────────
import timm

class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat5  = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.lat11 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.lat17 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.lat23 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.fpn_out = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.head = nn.Conv2d(256, 7, 1)

    def forward(self, features):
        p5, p11, p17, p23 = [lat(f) for lat, f in
                             zip([self.lat5, self.lat11, self.lat17, self.lat23], features)]
        x = p5 + p11 + p17 + p23
        x = self.fpn_out(x)
        x = F.interpolate(x, size=(1024,1024), mode='bilinear', align_corners=False)
        return self.head(x)

class LoveDA1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_large_patch16_dinov3.sat493m',
            pretrained=True, num_classes=0, dynamic_img_size=True,
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.decoder = SimpleDecoder()

    def forward(self, x):
        with torch.no_grad():
            _, feats = self.backbone.forward_intermediates(x, indices=[5, 11, 17, 23])
        return self.decoder(feats)

# ─── 数据 ───────────────────────────────────────────────────────────────
from torchgeo.datasets import LoveDA

class LoveDA7Class:
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        s = self.base[idx]
        img = s['image'].float() / 255.0
        mask = s['mask'].clone()
        remapped = torch.full_like(mask, 255, dtype=torch.long)
        for orig, new in zip(range(1, 8), range(7)):
            remapped[mask == orig] = new
        return img, remapped

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c, color in CLASS_COLORS.items():
        rgb[mask == c] = color
    return rgb

def compute_ious(mask, pred):
    inter = np.array([((pred==c)&(mask==c)).sum() for c in range(N_CLASSES)])
    union = np.array([((pred==c)|(mask==c)).sum() for c in range(N_CLASSES)])
    return inter / np.maximum(union, 1)

# ─── 主流程 ────────────────────────────────────────────────────────────
print(f"加载模型: {MODEL_PATH}")
model = LoveDA1024().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

ds = LoveDA7Class(LoveDA(root=DATA_DIR, split='val', download=False))
print(f"验证集: {len(ds)} 张")

random.seed(42)
indices = random.sample(range(len(ds)), 3)
print(f"选中: {indices}")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for row, idx in enumerate(indices):
    img, mask = ds[idx]
    img_np = img.permute(1,2,0).numpy()
    mask_np = mask.squeeze(0).numpy()

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    ious = compute_ious(mask_np, pred)
    iou_str = ' | '.join(f"{n}={v:.3f}" for n,v in zip(cls_names, ious))

    axes[row][0].imshow(img_np); axes[row][0].set_title(f"Image #{idx}"); axes[row][0].axis('off')
    axes[row][1].imshow(mask_to_rgb(mask_np)); axes[row][1].set_title("Ground Truth"); axes[row][1].axis('off')
    axes[row][2].imshow(mask_to_rgb(pred)); axes[row][2].set_title(f"Prediction (mIoU={ious.mean():.3f})"); axes[row][2].axis('off')
    axes[row][3].imshow(mask_to_rgb(mask_np)*0.5 + mask_to_rgb(pred)*0.5)
    axes[row][3].set_title(f"Overlay\n{iou_str}", fontsize=8); axes[row][3].axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "vis_result.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"保存: {out_path}")

print("\n--- IoU 详情 ---")
for i, idx in enumerate(indices):
    img, mask = ds[idx]
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    ious = compute_ious(mask.squeeze(0).numpy(), pred)
    print(f"#{idx}: mIoU={ious.mean():.4f}  " + " | ".join(f"{n}={v:.3f}" for n,v in zip(cls_names, ious)))
