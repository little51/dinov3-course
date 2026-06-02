import os, sys, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ========== 参数 ==========
DATA_DIR = "./data/LoveDA"
OUTPUT_DIR = "./output"
BATCH_SIZE = 1          # 显存小就设 1
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
N_CLASSES = 7
IGNORE_INDEX = 255
DROPOUT = 0.2
NUM_WORKERS = 0         # Windows 必须 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"LoveDA 1024 DINOv3  | 设备={DEVICE} 批次={BATCH_SIZE} 轮数={EPOCHS}")
print(f"数据={DATA_DIR}  输出={OUTPUT_DIR}")

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

train_ds = LoveDA7Class(LoveDA(root=DATA_DIR, split='train', download=False))
val_ds   = LoveDA7Class(LoveDA(root=DATA_DIR, split='val', download=False))
print(f"Train={len(train_ds)} | Val={len(val_ds)}")
N_BATCHES_TRAIN = len(train_ds) // BATCH_SIZE

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
            nn.Dropout2d(DROPOUT),
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

model = LoveDA1024().to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训参数: {trainable:,}")

opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
sched = CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

# ─── CSV ────────────────────────────────────────────────────────────────
batch_csv = os.path.join(OUTPUT_DIR, "batch_log.csv")
with open(batch_csv, "w") as f:
    f.write("epoch,batch,n_batches,loss,miou\n")

# ─── 训练 ───────────────────────────────────────────────────────────────
cls_names = ['bg','bld','road','water','barren','forest','agri']
best_miou, best_ep = 0.0, 0

print(f"\n训练开始\n")

for epoch in range(EPOCHS):
    t0 = time.time()
    ep_num = epoch + 1

    model.train()
    tr_losses, tr_mious = [], []
    for b, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        loss = crit(logits, masks)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        batch_loss = loss.item()
        tr_losses.append(batch_loss)

        preds = logits.argmax(dim=1)
        b_inter, b_union = np.zeros(N_CLASSES), np.zeros(N_CLASSES, dtype=np.int64)
        for c in range(N_CLASSES):
            b_inter[c] = ((preds == c) & (masks == c)).sum().item()
            b_union[c] = ((preds == c) | (masks == c)).sum().item()
        batch_miou = (b_inter / np.maximum(b_union, 1)).mean()
        tr_mious.append(batch_miou)

        with open(batch_csv, "a") as f:
            f.write(f"{ep_num},{b+1},{N_BATCHES_TRAIN},{batch_loss:.6f},{batch_miou:.6f}\n")

    tr_loss = float(np.mean(tr_losses))
    tr_miou = float(np.mean(tr_mious))

    model.eval()
    vl_loss = 0.0
    intersect, union = np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            vl_loss += crit(preds, masks).item()
            pl = preds.argmax(dim=1)
            for c in range(N_CLASSES):
                intersect[c] += ((pl == c) & (masks == c)).sum().item()
                union[c] += ((pl == c) | (masks == c)).sum().item()
    vl_loss /= len(val_loader)
    ious = intersect / np.maximum(union, 1)
    vl_miou = ious.mean()
    sched.step()

    iou_str = ' | '.join(f"{n}={v:.3f}" for n,v in zip(cls_names, ious))
    print(f"[{ep_num:3d}] tr_loss={tr_loss:.4f} tr_miou={tr_miou:.4f} | "
          f"vl_loss={vl_loss:.4f} vl_miou={vl_miou:.4f} (best={best_miou:.4f}@{best_ep}) | "
          f"{time.time()-t0:.0f}s")
    print(f"  IoU: {iou_str}")

    with open(os.path.join(OUTPUT_DIR, "training.log"), "a") as f:
        f.write(f"[{ep_num:3d}] tr_loss={tr_loss:.4f} tr_miou={tr_miou:.4f} | "
                f"vl_loss={vl_loss:.4f} vl_miou={vl_miou:.4f} (best={best_miou:.4f}@{best_ep}) | "
                f"{time.time()-t0:.0f}s\n  IoU: {iou_str}\n")

    if vl_miou > best_miou + 1e-4:
        best_miou, best_ep = vl_miou, ep_num
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  ** New best! mIoU={best_miou:.4f} @ ep{best_ep} **")

print(f"\nDone! Best: {best_miou:.4f}@ep{best_ep}")
json.dump({'best_miou':best_miou,'best_epoch':best_ep},
          open(os.path.join(OUTPUT_DIR,"results.json"),'w'), indent=2)
