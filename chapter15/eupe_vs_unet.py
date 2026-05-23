#!/usr/bin/env python3
"""
WHU Building 语义分割 — U-Net vs EUPE-ViT-B+DPT 对比
====================================================
请将以下文件放在当前目录:
  ./whu_building/       - WHU Building 数据集 (train/val/test, 含 image/ 和 label/ 子目录)
  ./eupe/               - EUPE 仓库 (git clone https://github.com/facebookresearch/EUPE)
  ./models/eupe/EUPE-ViT-B.pt  - EUPE 权重文件 (381MB)
====================================================
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─── 路径配置 ───────────────────────────────────────────────────────────────
DATA_ROOT   = './whu_building'
EUPE_DIR    = './eupe'
EUPE_WEIGHT = './models/eupe/EUPE-ViT-B.pt'
OUTPUT      = './outputs'
N_EPOCHS    = 3
SEED        = 42

os.makedirs(OUTPUT, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('设备:', device)
print('数据集:', DATA_ROOT)
print('输出:', OUTPUT)
print()

# ─── 数据集 ─────────────────────────────────────────────────────────────────
class WHUBuilding(Dataset):
    def __init__(self, root, split, img_size, augment=True):
        root = Path(root)
        self.img_dir   = root / split / 'image'
        self.label_dir = root / split / 'label'
        self.files = sorted([f for f in os.listdir(str(self.img_dir)) if f.endswith('.tif')])
        self.img_size = img_size
        self.augment  = augment and (split == 'train')
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img   = Image.open(str(self.img_dir / self.files[idx])).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        label = Image.open(str(self.label_dir / self.files[idx])).resize((self.img_size, self.img_size), Image.NEAREST)
        img_t   = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        label_t = torch.from_numpy(np.array(label, dtype=np.int64)).long()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_t = img_t.flip(dims=(2,)); label_t = label_t.flip(dims=(1,))
            if torch.rand(1).item() > 0.5:
                img_t = img_t.flip(dims=(1,)); label_t = label_t.flip(dims=(0,))
        return (img_t - self.mean) / self.std, label_t

# ─── 评估 & 损失 ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_p, all_t = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
        all_p.append(logits.argmax(dim=1).cpu()); all_t.append(masks.cpu())
    p = torch.cat(all_p).view(-1); t = torch.cat(all_t).view(-1)
    v = t != 255; p, t = p[v], t[v]
    ious = []
    for c in range(2):
        inter = ((p==c)&(t==c)).sum().item()
        union = ((p==c)|(t==c)).sum().item()
        ious.append(inter/union if union else float('nan'))
    return {'mIoU': float(np.nanmean(ious)), 'Building_IoU': ious[1], 'Acc': (p==t).float().mean().item()}

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        ps = F.softmax(pred, dim=1)
        oh = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
        inter = (ps*oh).sum(dim=(2,3)); union = ps.sum(dim=(2,3))+oh.sum(dim=(2,3))
        return 1 - (2*inter+self.smooth).mean()/(union+self.smooth).mean()

# ═══ 1. U-Net ═══════════════════════════════════════════════════════════════
def train_unet():
    print('='*50+'\n  U-Net (ResNet-34) - %d 轮\n'%N_EPOCHS+'='*50)
    import segmentation_models_pytorch as smp
    train_loader = DataLoader(WHUBuilding(DATA_ROOT,'train',256,True),  batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(WHUBuilding(DATA_ROOT,'val',  256,False), batch_size=64, shuffle=False, num_workers=0)
    model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=2).to(device)
    print('  参数量: %d'%sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    best, log = 0, []
    for ep in range(1, N_EPOCHS+1):
        model.train(); ls = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                loss = F.cross_entropy(model(imgs), masks)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            ls += loss.item()
        sched.step()
        m = evaluate(model, val_loader)
        if m['mIoU']>best: best=m['mIoU']; torch.save({'epoch':ep,'state_dict':model.state_dict(),'metrics':m}, OUTPUT+'/unet_best.pth')
        print('  Ep%d/%d L=%.4f mIoU=%.4f Bldg=%.4f'%(ep,N_EPOCHS,ls/len(train_loader),m['mIoU'],m['Building_IoU']))
        log.append(m)
    print('  最佳 mIoU: %.4f'%best)
    return log

# ═══ 2. EUPE + DPT ═════════════════════════════════════════════════════════
def train_eupe():
    print('='*50+'\n  EUPE-ViT-B + DPT - %d 轮\n'%N_EPOCHS+'='*50)
    sys.path.insert(0, EUPE_DIR)
    from eupe.eval.setup import load_model_and_context
    from eupe.eval.utils import ModelWithIntermediateLayers

    @dataclass
    class Cfg: eupe_hub='eupe_vitb16'; pretrained_weights=EUPE_WEIGHT; config_file=None
    backbone, _ = load_model_and_context(Cfg(), OUTPUT)
    ac = partial(torch.amp.autocast, device_type='cuda', enabled=torch.cuda.is_available(), dtype=torch.float32)
    bb = ModelWithIntermediateLayers(backbone, n=[2,5,8,11], autocast_ctx=ac, reshape=True, return_class_token=False).to(device)
    bb.eval()
    for p in bb.parameters(): p.requires_grad_(False)

    class DPTHead(nn.Module):
        def __init__(self, ed=768, nc=2, fd=256):
            super().__init__()
            self.r = nn.ModuleList([nn.Sequential(nn.Conv2d(ed,fd,1),nn.GELU()) for _ in range(4)])
            self.f = nn.ModuleList([nn.Sequential(nn.Conv2d(fd*2,fd,3,padding=1),nn.GELU(),nn.Conv2d(fd,fd,3,padding=1),nn.GELU()) for _ in range(3)])
            self.o = nn.Sequential(nn.Conv2d(fd,fd,3,padding=1),nn.GELU(),nn.Conv2d(fd,nc,1))
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        def forward(self, feats):
            t=[r(f) for r,f in zip(self.r,feats)]
            x=self.up(t[3]); x=torch.cat([x,self.up(t[2])],1); x=self.f[2](x)
            x=self.up(x);   x=torch.cat([x,self.up(self.up(t[1]))],1); x=self.f[1](x)
            x=self.up(x);   x=torch.cat([x,self.up(self.up(self.up(t[0])))],1); x=self.f[0](x)
            return self.up(self.o(x))

    head = DPTHead().to(device)
    print('  可训练参数量: %d'%sum(p.numel() for p in head.parameters()))
    train_loader = DataLoader(WHUBuilding(DATA_ROOT,'train',512,True),  batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(WHUBuilding(DATA_ROOT,'val',  512,False), batch_size=8, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(head.parameters(), lr=6e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    best, log = 0, []
    for ep in range(1, N_EPOCHS+1):
        head.train(); ls = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = head(bb(imgs))
                if logits.shape[2:]!=masks.shape[1:]: logits=F.interpolate(logits,size=masks.shape[1:],mode='bilinear',align_corners=False)
                loss = F.cross_entropy(logits,masks) + DiceLoss()(logits,masks)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            ls += loss.item()
        sched.step()
        class W(nn.Module):
            def __init__(s): super().__init__(); s.bb=bb; s.head=head
            def forward(s,x): return s.head(s.bb(x))
        m = evaluate(W(), val_loader)
        if m['mIoU']>best: best=m['mIoU']; torch.save({'epoch':ep,'head_state_dict':head.state_dict(),'metrics':m}, OUTPUT+'/eupe_best.pth')
        print('  Ep%d/%d L=%.4f mIoU=%.4f Bldg=%.4f'%(ep,N_EPOCHS,ls/len(train_loader),m['mIoU'],m['Building_IoU']))
        log.append(m)
    print('  最佳 mIoU: %.4f'%best)
    return log, bb, head

# ═══ 3. 可视化 ══════════════════════════════════════════════════════════════
def visualize(unet_log, eupe_log, eupe_bb, eupe_head):
    print('='*50+'\n  生成对比图\n'+'='*50)
    ckpt_u = torch.load(OUTPUT+'/unet_best.pth', map_location='cpu', weights_only=False)
    ckpt_e = torch.load(OUTPUT+'/eupe_best.pth', map_location='cpu', weights_only=False)
    mu, me = ckpt_u['metrics']['mIoU'], ckpt_e['metrics']['mIoU']

    # 曲线
    fig,axes = plt.subplots(1,2,figsize=(16,6),facecolor='white')
    for ax,k,t in zip(axes,['mIoU','Building_IoU'],['mIoU','Building IoU']):
        ul=[m[k] for m in unet_log]; el=[m[k] for m in eupe_log]; e=list(range(1,N_EPOCHS+1))
        ax.plot(e,ul,'o-',color='#2196F3',lw=2.5,ms=8,label='U-Net')
        ax.plot(e,el,'s-',color='#FF5722',lw=2.5,ms=8,label='EUPE+DPT')
        ax.fill_between(e,ul,el,alpha=0.1,color='#4CAF50',label='Δ=%.4f'%(el[-1]-ul[-1]))
        ax.set_xlabel('Epoch',fontsize=13); ax.set_ylabel(k,fontsize=13)
        ax.set_title(t,fontsize=14,fontweight='bold'); ax.set_xticks(e)
        ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
        for i,(u,v) in enumerate(zip(ul,el)):
            ax.annotate('%.4f'%u,(e[i],u),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=9,color='#2196F3')
            ax.annotate('%.4f'%v,(e[i],v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color='#FF5722')
    plt.tight_layout(); plt.savefig(OUTPUT+'/curves.png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()

    # 柱状图
    fig,ax = plt.subplots(figsize=(10,6),facecolor='white')
    uv=[mu,ckpt_u['metrics']['Building_IoU']]; ev=[me,ckpt_e['metrics']['Building_IoU']]; x=range(2); w=0.35
    b1=ax.bar([i-w/2 for i in x],uv,w,label='U-Net',color='#2196F3',ec='white',lw=1.5)
    b2=ax.bar([i+w/2 for i in x],ev,w,label='EUPE+DPT',color='#FF5722',ec='white',lw=1.5)
    ax.set_ylabel('指标'); ax.set_title('3 轮最终结果',fontsize=15,fontweight='bold')
    ax.set_xticks(list(x)); ax.set_xticklabels(['mIoU','Building IoU'],fontsize=13)
    ax.legend(fontsize=12); ax.grid(True,axis='y',alpha=0.3)
    for b,v in zip(b1+b2,uv+ev):
        ax.text(b.get_x()+b.get_width()/2.,b.get_height()+0.003,'%.4f'%v,ha='center',va='bottom',fontsize=11,fontweight='bold',color='#2196F3')
    for i,d in enumerate([ev[i]-uv[i] for i in range(2)]):
        ax.annotate('+%.4f 胜出'%d if d>0 else'%.4f'%d,xy=(i+w/2,ev[i]),fontsize=12,fontweight='bold',color='#4CAF50'if d>0 else'#F44336',xytext=(i+w/2,max(uv[i],ev[i])+0.025),ha='center')
    ax.set_ylim(0,max(max(uv),max(ev))+0.08)
    plt.tight_layout(); plt.savefig(OUTPUT+'/bar.png',dpi=150,bbox_inches='tight',facecolor='white'); plt.close()

    # 样本
    import segmentation_models_pytorch as smp
    unet = smp.Unet('resnet34',encoder_weights=None,in_channels=3,classes=2).to(device)
    unet.load_state_dict(ckpt_u['state_dict']); unet.eval()

    root=Path(DATA_ROOT)
    files=sorted([f for f in os.listdir(str(root/'test'/'image')) if f.endswith('.tif')])
    mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    def load(fn):
        img=Image.open(str(root/'test'/'image'/fn)).convert('RGB').resize((512,512),Image.BILINEAR)
        lbl=Image.open(str(root/'test'/'label'/fn)).resize((512,512),Image.NEAREST)
        it=(torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0-mean)/std
        return it,torch.from_numpy(np.array(lbl,dtype=np.int64)).long(),np.array(img),fn

    vis_dir=Path(OUTPUT)/'samples'; vis_dir.mkdir(parents=True,exist_ok=True)
    samples=[]
    for f in files:
        if len(samples)>=6: break
        _,m,r,fname=load(f)
        if m.sum()>500: samples.append((*load(f),))

    def ov(img,mask):
        o=img.copy(); g=np.zeros_like(img); g[:,:,1]=0.8
        o[mask]=o[mask]*0.5+g[mask]*0.5; return o

    for idx,(img_t,mask_t,raw,fname) in enumerate(samples):
        i256=F.interpolate(img_t.unsqueeze(0).to(device),size=(256,256),mode='bilinear',align_corners=False)
        with torch.no_grad(), torch.amp.autocast('cuda',enabled=torch.cuda.is_available()):
            pu=F.interpolate(unet(i256),size=(512,512),mode='bilinear',align_corners=False).argmax(1).squeeze(0).cpu().numpy().astype(bool)
            pe=eupe_head(eupe_bb(img_t.unsqueeze(0).to(device))).argmax(1).squeeze(0).cpu().numpy().astype(bool)
        gt=mask_t.numpy().astype(bool)
        if isinstance(raw,torch.Tensor): d=raw.permute(1,2,0).cpu().numpy() if raw.shape[0]==3 else raw.cpu().numpy()
        else: d=raw
        d=d.astype(np.float32)/255.0

        fig,axes=plt.subplots(2,4,figsize=(20,10))
        for c in range(4):
            axes[0,c].imshow(d if c==0 else[None,gt,pu,pe][c],cmap=None if c==0 else'gray')
            axes[0,c].set_title(['原图','真值','U-Net\nmIoU=%.4f'%mu,'EUPE+DPT\nmIoU=%.4f'%me][c],fontsize=13,fontweight='bold',color=['black','black','#2196F3','#FF5722'][c])
            axes[0,c].axis('off')
            axes[1,c].imshow(d if c==0 else ov(d,[gt,gt,pu,pe][c]))
            axes[1,c].set_title(['原图','真值叠加','U-Net 叠加','EUPE+DPT 叠加'][c],fontsize=13,fontweight='bold',color=['black','black','#2196F3','#FF5722'][c])
            axes[1,c].axis('off')
        if me>mu:
            for a in[axes[0,3],axes[1,3]]:
                for s in a.spines.values(): s.set_edgecolor('#FF5722'); s.set_linewidth(3)
        plt.suptitle('WHU Building 分割对比 - '+fname,fontsize=15,fontweight='bold',y=1.01)
        plt.tight_layout(); plt.savefig(vis_dir/('sample_%02d.png'%idx),dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print('  curves.png bar.png samples/ 已生成')

# ═══ 主程序 ═════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    ul = train_unet()
    el, bb, hd = train_eupe()
    visualize(ul, el, bb, hd)
    print('\n完成! %.0f 分钟, 输出: %s'%((time.time()-t0)/60, OUTPUT))
