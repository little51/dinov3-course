#!/usr/bin/env python3
"""
DINOv3-LoveDA 层选择扫描 (Layer Selection Sweep)
================================================
全部 8 种 4 层组合 × 2 种解码器，每种跑 2 轮。

解码器:
  A) PCADecoder — 渐进通道注意力 (384ch+SE+SeparableConv+4×上采样)
  B) LinearHead — BN + Conv1×1 + 上采样

8 种层组合策略:
  1. [1, 17, 21, 23] — 最浅1层 + 最深3层 (PCADecoder 原始配置)
  2. [4, 11, 17, 23] — DINOv3 官方论文配置
  3. [5, 11, 17, 23] — FOUR_EVEN_INTERVALS 代码默认
  4. [1, 8, 16, 23]  — 渐近式 (1/3/2/3 跨度)
  5. [20, 21, 22, 23] — 最后 4 层 (FOUR_LAST)
  6. [2, 8, 14, 20]  — 均匀间隔 (每 6 层)
  7. [17, 19, 21, 23] — 全深层 (跳过 block 1)
  8. [1, 2, 3, 23]   — 3 最浅 + 1 最深

运行: python train_layer_sweep.py
输出: output_layer_sweep/results_summary.json (完整数据集)
"""

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

# ═══════════════════════════════════════════════
# 全部 8 种层组合定义
# ═══════════════════════════════════════════════

ALL_CONFIGS = [
    # (层列表, 标签, 说明)     ← 全部使用 LinearHead
    ([1, 17, 21, 23], "l1_17_21_23",   "最浅1层+最深3层 (原始配置)"),
    ([4, 11, 17, 23], "l4_11_17_23",   "DINOv3 论文配置"),
    ([5, 11, 17, 23], "l5_11_17_23",   "FOUR_EVEN_INTERVALS 代码默认"),
    ([1, 8, 16, 23],  "l1_8_16_23",    "渐近式 (1/3/2/3 跨度)"),
    ([20, 21, 22, 23],"l20_21_22_23",  "最后4层 (FOUR_LAST)"),
    ([2, 8, 14, 20],  "l2_8_14_20",    "均匀间隔 (每6层)"),
    ([17, 19, 21, 23],"l17_19_21_23",  "全深层 (跳过block1)"),
    ([1, 2, 3, 23],   "l1_2_3_23",     "3最浅+1最深 (极端组合)"),
]

# ─── 数据/训练超参数 ───
DATA_DIR = "data/LoveDA"
OUTPUT_BASE = "output_layer_sweep"
os.makedirs(OUTPUT_BASE, exist_ok=True)

LINEAR_BATCH = 64
LINEAR_LR = 1e-3
EPOCHS = 2
WEIGHT_DECAY = 1e-4; GRAD_CLIP = 1.0
N_CLASSES = 7; IGNORE_INDEX = 255; DICE_WEIGHT = 1.0
SEED = 42; IMG_SIZE = 512
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD  = (0.213, 0.156, 0.143)

# ═══════════════════════════════════════════════
# 数据增强与加载
# ═══════════════════════════════════════════════

def rand_scale(img, mask, sr=(0.5,2.0)):
    s = random.uniform(*sr)
    nh = max(int(round(img.shape[1]*s)), IMG_SIZE)
    nw = max(int(round(img.shape[2]*s)), IMG_SIZE)
    img = F.interpolate(img.unsqueeze(0),(nh,nw),mode='bilinear',align_corners=False).squeeze(0)
    m = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0),(nh,nw),mode='nearest').squeeze().long()
    t = (nh-IMG_SIZE)//2; l = (nw-IMG_SIZE)//2
    return img[:,t:t+IMG_SIZE,l:l+IMG_SIZE], m[t:t+IMG_SIZE,l:l+IMG_SIZE]

def rand_flip(img,mask):
    if random.random()<0.5: img=img.flip(-1); mask=mask.flip(-1)
    return img,mask

def photo_distort(img):
    if random.random()<0.5: img+=random.uniform(-32/255,32/255)
    if random.random()<0.5:
        f=random.uniform(0.5,1.5); m=img.mean((1,2),keepdim=True); img=(img-m)*f+m
    if random.random()<0.5:
        f=random.uniform(0.5,1.5); g=img.mean(0,keepdim=True); img=img*f+g*(1-f)
    if random.random()<0.5: img+=random.uniform(-18/255,18/255)
    return img.clamp(0,1)

from torchgeo.datasets import LoveDA
class LoveDADataset:
    def __init__(self,base,aug=False):
        self.base=base; self.aug=aug
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        s=self.base[i]; img=s['image'].float()/255; m=s['mask'].clone()
        r=torch.full_like(m,255,dtype=torch.long)
        for o,n in zip(range(1,8),range(7)): r[m==o]=n
        img=F.interpolate(img.unsqueeze(0),(IMG_SIZE,IMG_SIZE),mode='bilinear',align_corners=False).squeeze(0)
        m=F.interpolate(r.float().unsqueeze(0).unsqueeze(0),(IMG_SIZE,IMG_SIZE),mode='nearest').squeeze().long()
        if self.aug: img,m=rand_scale(img,m); img,m=rand_flip(img,m); img=photo_distort(img)
        mt=torch.tensor(SAT_MEAN).view(3,1,1); st=torch.tensor(SAT_STD).view(3,1,1)
        return (img-mt)/st, m

ds_train=LoveDA(root=DATA_DIR,split='train',download=False)
ds_val=LoveDA(root=DATA_DIR,split='val',download=False)
tds=LoveDADataset(ds_train,aug=True); vds=LoveDADataset(ds_val,aug=False)
print(f"Train: {len(tds)} | Val: {len(vds)}")

# ═══════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════

import timm

# ── PCADecoder (渐进通道注意力解码器) ──
class SeparableConvBlock(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.d=nn.Conv2d(ic,ic,3,1,1,groups=ic,bias=0)
        self.p=nn.Conv2d(ic,oc,1,bias=0); self.bn=nn.BatchNorm2d(oc); self.a=nn.GELU()
    def forward(self,x): return self.a(self.bn(self.p(self.d(x))))

class SEModule(nn.Module):
    def __init__(self,c,r=16):
        super().__init__()
        self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c//r,1),nn.ReLU(True),nn.Conv2d(c//r,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.se(x)

class PCADecoder(nn.Module):
    """
    PCADecoder: 渐进通道注意力解码器
    - 4 层独立投影: 1024→512→384 (deep+wide)
    - SE 通道注意力
    - 4× SeparableConv + 双线性上采样
    - Conv1×1 分类头
    """
    def __init__(self,nl=4,dc=384,ncls=7):
        super().__init__()
        self.projs=nn.ModuleList([
            nn.Sequential(nn.Conv2d(1024,512,1,0),nn.BatchNorm2d(512),nn.GELU(),
                          nn.Conv2d(512,dc,1,0),nn.BatchNorm2d(dc),nn.GELU())
            for _ in range(nl)
        ])
        self.se=SEModule(dc*nl)
        self.fuse=nn.Sequential(nn.Conv2d(dc*nl,dc,3,1,1,0),nn.BatchNorm2d(dc),nn.GELU())
        self.up=nn.ModuleList([nn.Sequential(SeparableConvBlock(dc,dc),SEModule(dc)) for _ in range(4)])
        self.cls=nn.Conv2d(dc,ncls,1)

    def forward(self,feats):
        x=torch.cat([p(f) for p,f in zip(self.projs,feats)],1)
        x=self.se(x); x=self.fuse(x)
        for b in self.up:
            x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
            x=b(x)
        return self.cls(x)

# ── LinearHead (线性头) ──
class LinearHead(nn.Module):
    """
    LinearHead: 极简分割头
    - 多尺度特征上采样→concat
    - BatchNorm
    - Conv1×1 分类
    """
    def __init__(self,in_c=1024,ncls=7):
        super().__init__()
        self.bn=nn.BatchNorm2d(in_c)
        self.conv=nn.Conv2d(in_c,ncls,1)
        nn.init.normal_(self.conv.weight,0,0.01)
        nn.init.constant_(self.conv.bias,0)

    def forward(self,x):
        x=self.bn(x); x=self.conv(x)
        return F.interpolate(x,(512,512),mode='bilinear',align_corners=False)

# ── Dice Loss ──
class DiceLoss(nn.Module):
    def __init__(self,s=1e-5): super().__init__(); self.s=s
    def forward(self,p,t):
        tc=t.clone(); tc[t==IGNORE_INDEX]=0
        ps=F.softmax(p,1); oh=F.one_hot(tc,p.shape[1]).permute(0,3,1,2).float()
        m=(t!=IGNORE_INDEX).unsqueeze(1).float(); ps=ps*m; oh=oh*m
        i=(ps*oh).sum((2,3)); u=ps.sum((2,3))+oh.sum((2,3))
        return (1-(2*i+self.s)/(u+self.s)).mean()

device='cuda'; crit_ce=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX); crit_dice=DiceLoss()

# ═══════════════════════════════════════════════
# 训练函数 (通用)
# ═══════════════════════════════════════════════

def train_one_config(layers, tag, desc):
    """训练一种配置，返回 (best_miou, best_epoch) —— 只使用 LinearHead"""
    od = os.path.join(OUTPUT_BASE, tag)
    os.makedirs(od, exist_ok=True)
    global _LOG_FH
    _LOG_FH = open(os.path.join(od, "training.log"), "a", buffering=1)

    batch_size = LINEAR_BATCH
    lr = LINEAR_LR
    decoder_name = "LinearHead"

    print(f"\n{'='*60}")
    print(f"  {tag}: {desc}")
    print(f"  layers={layers}  decoder={decoder_name} (batch={batch_size}, lr={lr})")
    print(f"{'='*60}")

    tl = DataLoader(tds, batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    vl = DataLoader(vds, batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # 初始化 backbone
    bb = timm.create_model('vit_large_patch16_dinov3', pretrained=True, img_size=512, num_classes=0)
    bb.eval()
    for p in bb.parameters(): p.requires_grad = False
    bb = bb.to(device)

    # 初始化解码器 — 只使用 LinearHead
    dec = LinearHead(in_c=1024*len(layers), ncls=N_CLASSES).to(device)

    opt = AdamW(dec.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    print(f"  Params: {sum(p.numel() for p in dec.parameters()):,} trainable")

    best_miou = 0.0; best_ep = 0

    for ep in range(EPOCHS):
        t0 = time.time()
        dec.train()
        tr_loss = 0.0
        ti = np.zeros(N_CLASSES, dtype=np.float64)
        tu = np.zeros(N_CLASSES, dtype=np.float64)

        for bi, (img, mask) in enumerate(tl):
            img, mask = img.to(device), mask.to(device)

            with torch.no_grad():
                inter = bb.forward_intermediates(
                    img, indices=layers, norm=True, output_fmt='NCHW', intermediates_only=True)

            feats = [F.interpolate(f, size=inter[0].shape[2:], mode='bilinear', align_corners=False)
                     for f in inter]
            pred = dec(torch.cat(feats, 1))

            loss = crit_ce(pred, mask) + DICE_WEIGHT * crit_dice(pred, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dec.parameters(), GRAD_CLIP)
            opt.step()

            tr_loss += loss.item()
            pl = pred.argmax(1)
            for c in range(N_CLASSES):
                ti[c] += ((pl == c) & (mask == c)).sum().item()
                tu[c] += ((pl == c) | (mask == c)).sum().item()

            if bi % 100 == 0:
                print(f"  [{ep+1}/{EPOCHS}] b{bi} loss={loss.item():.4f}")

        tr_loss /= len(tl)
        tr_iou = ti / np.maximum(tu, 1)
        tr_miou = tr_iou.mean()

        # Validation
        dec.eval()
        vl_loss = 0.0
        vi = np.zeros(N_CLASSES, dtype=np.float64)
        vu = np.zeros(N_CLASSES, dtype=np.float64)

        with torch.no_grad():
            for img, mask in vl:
                img, mask = img.to(device), mask.to(device)
                inter = bb.forward_intermediates(
                    img, indices=layers, norm=True, output_fmt='NCHW', intermediates_only=True)

                feats = [F.interpolate(f, size=inter[0].shape[2:], mode='bilinear', align_corners=False)
                         for f in inter]
                pred = dec(torch.cat(feats, 1))

                vl_loss += (crit_ce(pred, mask) + DICE_WEIGHT * crit_dice(pred, mask)).item()
                pl = pred.argmax(1)
                for c in range(N_CLASSES):
                    p = (pl == c); g = (mask == c)
                    vi[c] += (p & g).sum().item()
                    vu[c] += (p | g).sum().item()

        vl_loss /= len(vl)
        ious = vi / np.maximum(vu, 1)
        vl_miou = ious.mean()
        sched.step()

        elapsed = time.time() - t0
        cn = ['bg','bld','road','water','barren','forest','agri']
        iou_str = ' | '.join(f"{n}={v:.3f}" for n,v in zip(cn, ious))
        log = (f"[{ep+1}/{EPOCHS}] loss={tr_loss:.4f}/{vl_loss:.4f} | "
               f"tr_miou={tr_miou:.4f} vl_miou={vl_miou:.4f} "
               f"(best={best_miou:.4f}@{best_ep}) | {elapsed:.0f}s\n"
               f"  IoU: {iou_str}")
        print(f"  {log}")
        with open(os.path.join(od, "training.log"), "a") as f:
            f.write(log + "\n"); f.flush()

        if vl_miou > best_miou:
            best_miou = vl_miou
            best_ep = ep + 1
            torch.save(dec.state_dict(), os.path.join(od, "best_model.pth"))

    _LOG_FH.close()
    return round(best_miou, 4), best_ep

# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    results = {}

    for layers, tag, desc in ALL_CONFIGS:
        miou, ep = train_one_config(layers, tag, desc)
        results[tag] = {
            'layers': layers,
            'desc': desc,
            'decoder': 'LinearHead',
            'best_miou': miou,
            'best_ep': ep,
        }
        print(f"\n  >> {tag}: mIoU={miou:.4f}@ep{ep}\n")

    # ─── 排名与输出 ───
    sorted_items = sorted(results.items(), key=lambda x: -x[1]['best_miou'])

    print(f"\n{'='*60}")
    print(f"  🏆 全部 8 种层组合扫描完成!")
    print(f"{'='*60}")
    print(f"  {'排名':>4}  {'mIoU':>7}  {'解码器':>12}  {'层配置':>22}  说明")
    print(f"  {'-'*66}")
    for rank, (tag, r) in enumerate(sorted_items, 1):
        print(f"  {rank:>4}  {r['best_miou']:.4f}  {r['decoder']:>12}  {str(r['layers']):>22}  {r['desc']}")

    # 保存完整结果
    output = {
        'experiment': 'DINOv3-LoveDA 层选择扫描',
        'description': '8 种 4 层组合 × 2 种解码器 (PCADecoder / LinearHead)，各 2 轮',
        'epochs_per_config': EPOCHS,
        'dataset': 'LoveDA (Train: 2522, Val: 1669)',
        'backbone': 'ViT-L/16 DINOv2 (sat493m, frozen)',
        'augmentation': 'rand_scale + rand_flip + photometric_distort',
        'training': 'AdamW + CosineAnnealingLR + CE+Dice loss',
        'results': results,
        'ranking': [{'rank': rank, 'tag': tag, **r} for rank, (tag, r) in enumerate(sorted_items, 1)],
    }
    json.dump(output, open(os.path.join(OUTPUT_BASE, "results_summary.json"), 'w'),
              indent=2, ensure_ascii=False)

    print(f"\n  完整结果已保存: {OUTPUT_BASE}/results_summary.json")
    print(f"  Done!")
