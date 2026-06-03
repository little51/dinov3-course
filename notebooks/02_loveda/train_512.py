#!/usr/bin/env python3
"""LoveDA 7-class — DINOv3 frozen + ASPP decoder, 512×512 input (fair comparison baseline)."""
import os, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

print = lambda *a, **kw: __import__('builtins').print(*a, **kw, flush=True)

# ─── Config ─────────────────────────────────────────────────────────────
DATA_DIR = "./data/LoveDA"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS = 10
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_CLASSES = 7
IGNORE_INDEX = 255
BLOCKS = [4, 8, 16, 23]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

IMG_SIZE = 512   # 512×512 → patch=16 → 32×32 grid
SUBSET = 1.0     # 100% data — full run

# ─── Data ───────────────────────────────────────────────────────────────
from torchgeo.datasets import LoveDA

class LoveDA7Class:
    def __init__(self, base, img_size=IMG_SIZE):
        self.base = base
        self.img_size = img_size
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        s = self.base[idx]
        img = s['image'].float() / 255.0
        mask = s['mask'].clone()
        remapped = torch.full_like(mask, 255, dtype=torch.long)
        for orig, new in zip(range(1, 8), range(7)):
            remapped[mask == orig] = new
        # Resize to 512×512
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size),
                            mode='bilinear', align_corners=False).squeeze(0)
        mask_float = remapped.float().unsqueeze(0).unsqueeze(0)
        mask_rs = F.interpolate(mask_float, size=(self.img_size, self.img_size),
                                mode='nearest').squeeze(0).squeeze(0).long()
        return img, mask_rs

has_train = os.path.isdir(os.path.join(DATA_DIR, "Train"))
if has_train:
    print("=== Using FULL dataset (Train + Val) ===")
    ds_train = LoveDA(root=DATA_DIR, split='train', download=False)
    ds_val   = LoveDA(root=DATA_DIR, split='val', download=False)
    train_ds = LoveDA7Class(ds_train)
    val_ds   = LoveDA7Class(ds_val)
else:
    ds_all = LoveDA(root=DATA_DIR, split='val', download=False)
    ds = LoveDA7Class(ds_all)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    n_train = int(len(ds) * 0.8)
    train_ds = Subset(ds, indices[:n_train])
    val_ds   = Subset(ds, indices[n_train:])

n_sub = max(1, int(len(train_ds) * SUBSET))
train_sub = Subset(train_ds, range(n_sub))
print(f"  Train: {len(train_sub)}/{len(train_ds)} | Val: {len(val_ds)}")

train_loader = DataLoader(train_sub, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ─── ASPP Decoder (for 32×32 feature map → 512×512 output) ──────────────
import timm

class ASPPDecoder(nn.Module):
    """Concat-fusion ASPP for 32×32 grid → 512 output."""
    def __init__(self, in_dim=1024, dec_dim=256, n_classes=N_CLASSES):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, dec_dim, 1),
                nn.BatchNorm2d(dec_dim),
                nn.ReLU(inplace=True),
            ) for _ in BLOCKS
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(dec_dim * len(BLOCKS), dec_dim, 1),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )
        aspp_dim = dec_dim
        self.aspp_b1 = nn.Sequential(
            nn.Conv2d(dec_dim, aspp_dim, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(aspp_dim), nn.ReLU(inplace=True),
        )
        self.aspp_b2 = nn.Sequential(
            nn.Conv2d(dec_dim, aspp_dim, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(aspp_dim), nn.ReLU(inplace=True),
        )
        self.aspp_b3 = nn.Sequential(
            nn.Conv2d(dec_dim, aspp_dim, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(aspp_dim), nn.ReLU(inplace=True),
        )
        self.aspp_b4 = nn.Sequential(
            nn.Conv2d(dec_dim, aspp_dim, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(aspp_dim), nn.ReLU(inplace=True),
        )
        self.aspp_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dec_dim, aspp_dim, 1, bias=False),
            nn.BatchNorm2d(aspp_dim), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_dim * 5, dec_dim, 1),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )
        # 2-stage upsampling: 32→128 (4×) → 128→512 (4×)
        self.refine = nn.Sequential(
            nn.Conv2d(dec_dim, dec_dim, 3, padding=1),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_dim, n_classes, 1)

        total_p = sum(p.numel() for p in self.parameters())
        print(f"  ASPPDecoder (512): {len(BLOCKS)}-block concat + 5-branch ASPP, {total_p/1e6:.2f}M params")

    def forward(self, features):
        proj = [lat(f) for lat, f in zip(self.laterals, features)]
        x = self.fusion(torch.cat(proj, dim=1))      # [B, 256, 32, 32]

        b1 = self.aspp_b1(x); b2 = self.aspp_b2(x)
        b3 = self.aspp_b3(x); b4 = self.aspp_b4(x)
        bg = F.interpolate(self.aspp_global(x), size=b1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([b1, b2, b3, b4, bg], dim=1))  # [B, 256, 32, 32]

        # 2-stage upsampling: 32→128 (4×) → refine → 128→512 (4×)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)  # →128
        x = self.refine(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)  # →512
        return self.head(x)


class LoveDA512(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0,
            dynamic_img_size=True,
        )
        self.dim = self.backbone.embed_dim
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"  Backbone: vit_large_patch16_dinov3.sat493m, dim={self.dim}, {total/1e6:.0f}M (frozen)")
        self.decoder = ASPPDecoder(in_dim=self.dim)

    def forward(self, x):
        with torch.no_grad():
            _, intermediates = self.backbone.forward_intermediates(x, indices=BLOCKS)
        return self.decoder(intermediates)


# ─── Training ───────────────────────────────────────────────────────────
device = 'cuda'
print(f"\nDevice: {torch.cuda.get_device_name(0)}")
model = LoveDA512().to(device)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)\n")

opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY)
sched = CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

best_miou = 0.0
best_ep = 0

print(f"=== Training {EPOCHS} epochs ({SUBSET*100:.0f}% data, 512×512 input) ===\n")

for epoch in range(EPOCHS):
    t0 = time.time()

    model.train()
    tr_loss = 0.0
    tr_intersect = np.zeros(N_CLASSES, dtype=np.float64)
    tr_union = np.zeros(N_CLASSES, dtype=np.float64)

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = crit(preds, masks)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        tr_loss += loss.item()

        pred_labels = preds.argmax(dim=1)
        for c in range(N_CLASSES):
            tr_intersect[c] += ((pred_labels == c) & (masks == c)).sum().item()
            tr_union[c] += ((pred_labels == c) | (masks == c)).sum().item()

    tr_loss /= len(train_loader)
    tr_iou = tr_intersect / np.maximum(tr_union, 1)
    tr_miou = tr_iou.mean()

    model.eval()
    vl_loss = 0.0
    intersect = np.zeros(N_CLASSES, dtype=np.float64)
    union = np.zeros(N_CLASSES, dtype=np.float64)

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            vl_loss += crit(preds, masks).item()
            pred_labels = preds.argmax(dim=1)
            for c in range(N_CLASSES):
                p = (pred_labels == c)
                g = (masks == c)
                intersect[c] += (p & g).sum().item()
                union[c] += (p | g).sum().item()

    vl_loss /= len(val_loader)
    ious = intersect / np.maximum(union, 1)
    vl_miou = ious.mean()

    sched.step()
    elapsed = time.time() - t0

    class_names = ['bg', 'bld', 'road', 'water', 'barren', 'forest', 'agri']
    iou_str = ' | '.join(f"{n}={v:.3f}" for n, v in zip(class_names, ious))

    log_line = (f"[{epoch+1:3d}/{EPOCHS}] loss={tr_loss:.4f}/{vl_loss:.4f} | "
                f"tr_miou={tr_miou:.4f} vl_miou={vl_miou:.4f} "
                f"(best={best_miou:.4f}@{best_ep}) | "
                f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s\n"
                f"  IoU: {iou_str}")
    print(f"  {log_line}")

    with open(os.path.join(OUTPUT_DIR, "training.log"), "a") as f:
        f.write(log_line + "\n")

    if vl_miou > best_miou + 1e-4:
        best_miou = vl_miou
        best_ep = epoch + 1
        decoder_state = {k: v for k, v in model.state_dict().items()
                         if k.startswith('decoder.')}
        torch.save(decoder_state, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  -> New best! mIoU={best_miou:.4f}")

print(f"\nDone! Best: {best_miou:.4f}@ep{best_ep}")
json.dump({
    'best_miou': best_miou, 'best_epoch': best_ep,
    'arch': 'aspp-512',
    'config': {
        'backbone': 'vit_large_patch16_dinov3.sat493m',
        'frozen': True,
        'input_size': IMG_SIZE,
        'lr': LR, 'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP,
        'n_train_sub': n_sub, 'n_val': len(val_ds),
    }
}, open(os.path.join(OUTPUT_DIR, "results.json"), 'w'), indent=2)
