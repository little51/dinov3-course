"""PCADecoder (渐进通道注意力解码器) — DINOv3 sat493m — LoveDA."""
import os, json, time, random, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import builtins
_LOG_FH = None
def tee_print(*a, **kw):
    builtins.print(*a, **kw, flush=True)
    if _LOG_FH and not _LOG_FH.closed:
        builtins.print(*a, file=_LOG_FH, flush=True)
print = tee_print

DATA_DIR = "data/LoveDA"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_LOG_FH = open(os.path.join(OUTPUT_DIR, "training.log"), "a", buffering=1)

BATCH_SIZE = 4
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_CLASSES = 7
IGNORE_INDEX = 255
LAYERS = [1, 17, 21, 23]   # 极跨度：1(最浅)+17/21/23(最深)
DICE_WEIGHT = 1.0
PATIENCE = 10

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
IMG_SIZE = 512
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD  = (0.213, 0.156, 0.143)

# ─── Augmentation ───
def rand_scale(img, mask, scale_range=(0.5, 2.0)):
    s = random.uniform(*scale_range)
    new_h = int(round(img.shape[1] * s)); new_w = int(round(img.shape[2] * s))
    new_h = max(new_h, IMG_SIZE); new_w = max(new_w, IMG_SIZE)
    img_rs = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)
    mask_rs = F.interpolate(mask_f, size=(new_h, new_w), mode='nearest').squeeze(0).squeeze(0).long()
    top = (new_h - IMG_SIZE)//2; left = (new_w - IMG_SIZE)//2
    return img_rs[:, top:top+IMG_SIZE, left:left+IMG_SIZE], mask_rs[top:top+IMG_SIZE, left:left+IMG_SIZE]

def rand_flip(img, mask):
    if random.random() < 0.5: img = img.flip(-1); mask = mask.flip(-1)
    return img, mask

def photometric_distort(img):
    if random.random() < 0.5: img = img + random.uniform(-32/255, 32/255)
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        mean = img.mean(dim=(1,2), keepdim=True)
        img = (img - mean) * factor + mean
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        gray = img.mean(dim=0, keepdim=True)
        img = img * factor + gray * (1 - factor)
    if random.random() < 0.5: img = img + random.uniform(-18/255, 18/255)
    return img.clamp(0, 1)

# ─── Data ───
from torchgeo.datasets import LoveDA

class LoveDA7Class:
    def __init__(self, base, img_size=IMG_SIZE, augment=False):
        self.base = base; self.img_size = img_size; self.augment = augment
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        s = self.base[idx]
        img = s['image'].float() / 255.0
        mask = s['mask'].clone()
        remapped = torch.full_like(mask, 255, dtype=torch.long)
        for orig, new in zip(range(1,8), range(7)): remapped[mask == orig] = new
        img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        mask_f = remapped.float().unsqueeze(0).unsqueeze(0)
        mask_rs = F.interpolate(mask_f, size=(self.img_size, self.img_size), mode='nearest').squeeze(0).squeeze(0).long()
        if self.augment:
            img, mask_rs = rand_scale(img, mask_rs)
            img, mask_rs = rand_flip(img, mask_rs)
            img = photometric_distort(img)
        mean_t = torch.tensor(SAT_MEAN).view(3,1,1); std_t = torch.tensor(SAT_STD).view(3,1,1)
        img = (img - mean_t) / std_t
        return img, mask_rs

ds_train = LoveDA(root=DATA_DIR, split='train', download=False)
ds_val   = LoveDA(root=DATA_DIR, split='val', download=False)
train_ds = LoveDA7Class(ds_train, augment=True)
val_ds   = LoveDA7Class(ds_val, augment=False)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

# ─── Models ───
import timm

class SeparableConvBlock(nn.Module):
    """SegFormer-style separable conv: depthwise→pointwise→BN→GELU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)

class PCADecoder(nn.Module):
    """渐进通道注意力解码器 (Progressive Channel Attention Decoder).
    渐进投影 + 384ch + SE通道注意力."""
    def __init__(self, in_channels=1024, num_classes=7, decoder_channels=384, num_layers=4):
        super().__init__()
        # Deep+Wide projection: 1024→512→384
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.GELU(),
                nn.Conv2d(512, decoder_channels, 1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])
        # SE on concat(4×256=1024)
        self.concat_se = SEModule(decoder_channels * num_layers, reduction=16)
        # Fuse: 1024→256
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * num_layers, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        # Refine blocks with SE
        self.up_refine1 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.up_refine2 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.up_refine3 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.up_refine4 = nn.Sequential(
            SeparableConvBlock(decoder_channels, decoder_channels), SEModule(decoder_channels))
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, feats):
        mlp_feats = [linear(feat) for feat, linear in zip(feats, self.linear_layers)]
        x = torch.cat(mlp_feats, dim=1)
        x = self.concat_se(x)          # SE after concat
        x = self.linear_fuse(x)
        for refine in [self.up_refine1, self.up_refine2, self.up_refine3, self.up_refine4]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = refine(x)
        return self.classifier(x)

class LoveDA_DINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_large_patch16_dinov3', pretrained=False, img_size=512, num_classes=0)
        ckpt = torch.load('/home/user01/models/dinov3_sat/vit_large_dinov3_sat493m.pth', map_location='cpu')
        self.backbone.load_state_dict(ckpt, strict=True)
        print(f"  Loaded DINOv3 sat493m weights")
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.decoder = PCADecoder(num_layers=len(LAYERS))

    def forward(self, x):
        with torch.no_grad():
            inter = self.backbone.forward_intermediates(
                x, indices=LAYERS, norm=True, output_fmt='NCHW', intermediates_only=True,
            )
        return self.decoder(list(inter))


# ─── Dice Loss ───
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        target_clean = target.clone(); target_clean[target==IGNORE_INDEX] = 0
        ps = F.softmax(pred, dim=1)
        oh = F.one_hot(target_clean, num_classes=pred.shape[1]).permute(0,3,1,2).float()
        mask = (target!=IGNORE_INDEX).unsqueeze(1).float()
        ps = ps * mask; oh = oh * mask
        inter = (ps * oh).sum(dim=(2,3)); union = ps.sum(dim=(2,3)) + oh.sum(dim=(2,3))
        dice = (2*inter+self.smooth)/(union+self.smooth)
        return 1 - dice.mean()

# ─── Training ───
device = 'cuda'
model = LoveDA_DINOv3().to(device)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%)")

opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
sched = CosineAnnealingLR(opt, T_max=EPOCHS)
crit_ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
crit_dice = DiceLoss()

best_miou = 0.0
best_ep = 0
no_improve = 0

print(f"\n=== DINOv3 sat493m | PCADecoder(384ch,渐进+SE) | {LAYERS} | {EPOCHS}ep | LR={LR} | patience={PATIENCE} ===\n")

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    tr_loss = 0.0
    tr_intersect = np.zeros(N_CLASSES, dtype=np.float64)
    tr_union = np.zeros(N_CLASSES, dtype=np.float64)

    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = crit_ce(preds, masks) + DICE_WEIGHT * crit_dice(preds, masks)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        tr_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"  [{epoch+1}/{EPOCHS}] batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}")
        pred_labels = preds.argmax(dim=1)
        for c in range(N_CLASSES):
            tr_intersect[c] += ((pred_labels==c)&(masks==c)).sum().item()
            tr_union[c] += ((pred_labels==c)|(masks==c)).sum().item()

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
            vl_loss += (crit_ce(preds, masks) + DICE_WEIGHT * crit_dice(preds, masks)).item()
            pred_labels = preds.argmax(dim=1)
            for c in range(N_CLASSES):
                p = (pred_labels==c); g = (masks==c)
                intersect[c] += (p & g).sum().item()
                union[c] += (p | g).sum().item()

    vl_loss /= len(val_loader)
    ious = intersect / np.maximum(union, 1)
    vl_miou = ious.mean()
    sched.step()
    elapsed = time.time() - t0

    class_names = ['bg','bld','road','water','barren','forest','agri']
    iou_str = ' | '.join(f"{n}={v:.3f}" for n,v in zip(class_names, ious))
    log_line = (f"[{epoch+1:2d}/{EPOCHS}] loss={tr_loss:.4f}/{vl_loss:.4f} | "
                f"tr_miou={tr_miou:.4f} vl_miou={vl_miou:.4f} "
                f"(best={best_miou:.4f}@{best_ep}) | lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s\n"
                f"  IoU: {iou_str}")
    print(f"  {log_line}")

    with open(os.path.join(OUTPUT_DIR, "training.log"), "a") as f:
        f.write(log_line + "\n"); f.flush(); os.fsync(f.fileno())

    if vl_miou > best_miou + 1e-4:
        best_miou = vl_miou; best_ep = epoch + 1; no_improve = 0
        decoder_state = {k:v for k,v in model.state_dict().items() if k.startswith('decoder.')}
        torch.save(decoder_state, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  -> New best! mIoU={best_miou:.4f}")
    else:
        no_improve += 1
        print(f"  -> No improvement ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            print(f"  Early stopping after {epoch+1} epochs.")
            break

print(f"\nDone! Best: {best_miou:.4f}@ep{best_ep}")
json.dump({'best_miou':best_miou,'best_epoch':best_ep,'arch':'dinov3_sat_segformer_pca','config':{
    'backbone':'dinov3_sat493m_vit_large_patch16_dinov3(timm)','frozen':True,'input_size':IMG_SIZE,'layers':LAYERS,
    'lr':LR,'weight_decay':WEIGHT_DECAY,'from_scratch':True,
    'dice_weight':DICE_WEIGHT,'patience':PATIENCE,
    'decoder':'PCADecoder(384ch,4×indep_proj[1×1(1024→512)+BN+GELU→1×1(512→384)+BN+GELU]→SE→concat+3×3fuse+BN+GELU→4×[SepConv+SE]+upsample)',
}}, open(os.path.join(OUTPUT_DIR,"results.json"),'w'), indent=2)
