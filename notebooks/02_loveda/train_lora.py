#!/usr/bin/env python3
"""DINOv3 + LoRA (rank=8, q/k/v) + PCADecoder — LoveDA. 最佳配置"""
import os, json, time, random, sys, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import builtins
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
_LOG_FH = None
def tee_print(*a, **kw):
    builtins.print(*a, **kw, flush=True)
    if _LOG_FH and not _LOG_FH.closed:
        builtins.print(*a, file=_LOG_FH, flush=True)
print = tee_print

DATA_DIR = "data/LoveDA"
OUTPUT_DIR = "output_lora_best"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_LOG_FH = open(os.path.join(OUTPUT_DIR, "training.log"), "a", buffering=1)

BATCH_SIZE = 4
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_CLASSES = 7
IGNORE_INDEX = 255
LAYERS = [1, 17, 21, 23]
DICE_WEIGHT = 1.0
PATIENCE = 10
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LR = 1e-4

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
IMG_SIZE = 512
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD  = (0.213, 0.156, 0.143)

# ─── Augmentation ───
def rand_scale(img, mask, scale_range=(0.5, 2.0)):
    s = random.uniform(*scale_range)
    new_h = int(round(img.shape[1]*s)); new_w = int(round(img.shape[2]*s))
    new_h = max(new_h, IMG_SIZE); new_w = max(new_w, IMG_SIZE)
    img_rs = F.interpolate(img.unsqueeze(0), size=(new_h,new_w), mode='bilinear', align_corners=False).squeeze(0)
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)
    mask_rs = F.interpolate(mask_f, size=(new_h,new_w), mode='nearest').squeeze(0).squeeze(0).long()
    top = (new_h-IMG_SIZE)//2; left = (new_w-IMG_SIZE)//2
    return img_rs[:,top:top+IMG_SIZE,left:left+IMG_SIZE], mask_rs[top:top+IMG_SIZE,left:left+IMG_SIZE]

def rand_flip(img, mask):
    if random.random()<0.5: img=img.flip(-1); mask=mask.flip(-1)
    return img, mask

def photometric_distort(img):
    if random.random()<0.5: img+=random.uniform(-32/255,32/255)
    if random.random()<0.5:
        f=random.uniform(0.5,1.5); m=img.mean(dim=(1,2),keepdim=True)
        img=(img-m)*f+m
    if random.random()<0.5:
        f=random.uniform(0.5,1.5); g=img.mean(dim=0,keepdim=True)
        img=img*f+g*(1-f)
    if random.random()<0.5: img+=random.uniform(-18/255,18/255)
    return img.clamp(0,1)

# ─── Data ───
from torchgeo.datasets import LoveDA
def _load_image_robust(self, path):
    with Image.open(path) as img:
        try: array=np.array(img.convert('RGB'))
        except(OSError,TypeError,AttributeError): img.load(); array=np.array(img.convert('RGB'))
        return torch.from_numpy(array).float().permute(2,0,1)
LoveDA._load_image=_load_image_robust
def _load_target_robust(self, path):
    with Image.open(path) as img:
        try: array=np.array(img.convert('L'))
        except(OSError,TypeError,AttributeError): img.load(); array=np.array(img.convert('L'))
        return torch.from_numpy(array).long()
LoveDA._load_target=_load_target_robust

class LoveDA7Class:
    def __init__(self,base,img_size=IMG_SIZE,augment=False):
        self.base=base;self.img_size=img_size;self.augment=augment
    def __len__(self):return len(self.base)
    def __getitem__(self,idx):
        for _ in range(10):
            try:s=self.base[idx];break
            except(OSError,TypeError,KeyError):idx=random.randint(0,len(self.base)-1)
        else:s=self.base[0]
        img=s['image'].float()/255.0;mask=s['mask'].clone()
        remapped=torch.full_like(mask,255,dtype=torch.long)
        for o,n in zip(range(1,8),range(7)):remapped[mask==o]=n
        img=F.interpolate(img.unsqueeze(0),size=(self.img_size,self.img_size),mode='bilinear',align_corners=False).squeeze(0)
        mf=remapped.float().unsqueeze(0).unsqueeze(0)
        mr=F.interpolate(mf,size=(self.img_size,self.img_size),mode='nearest').squeeze(0).squeeze(0).long()
        if self.augment:
            img,mr=rand_scale(img,mr);img,mr=rand_flip(img,mr);img=photometric_distort(img)
        mt=torch.tensor(SAT_MEAN).view(3,1,1);st=torch.tensor(SAT_STD).view(3,1,1)
        return (img-mt)/st,mr

ds=LoveDA(root=DATA_DIR,split='train',download=False)
dv=LoveDA(root=DATA_DIR,split='val',download=False)
td=LoveDA7Class(ds,augment=True);vd=LoveDA7Class(dv,augment=False)
print(f"Train:{len(td)}|Val:{len(vd)}")
tl=DataLoader(td,BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
vl=DataLoader(vd,BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)

# ─── LoRA (q+k+v) ───
class LoRALayer(nn.Module):
    def __init__(self,dim,r=LORA_RANK,alpha=LORA_ALPHA):
        super().__init__()
        self.scale=alpha/r
        self.A=nn.Parameter(torch.zeros(r,dim));self.B=nn.Parameter(torch.zeros(dim,r))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,x):return(x@self.A.T)@self.B.T*self.scale

def inject_lora(backbone,rank=LORA_RANK,alpha=LORA_ALPHA):
    from timm.models.eva import EvaAttention
    params=[]
    for name,mod in backbone.named_modules():
        if isinstance(mod,EvaAttention):
            d=mod.qkv.in_features
            lq=LoRALayer(d,rank,alpha).to(mod.qkv.weight.device)
            lk=LoRALayer(d,rank,alpha).to(mod.qkv.weight.device)
            lv=LoRALayer(d,rank,alpha).to(mod.qkv.weight.device)
            orig=mod.qkv.forward
            def mkf(lq,lk,lv,orig):
                def fwd(x):
                    o=orig(x).clone()
                    o[...,:d]+=lq(x);o[...,d:2*d]+=lk(x);o[...,2*d:]+=lv(x)
                    return o
                return fwd
            mod.qkv.forward=mkf(lq,lk,lv,orig)
            mod.qkv._lq=lq;mod.qkv._lk=lk;mod.qkv._lv=lv
            params.extend([lq.A,lq.B,lk.A,lk.B,lv.A,lv.B])
            print(f"LoRA q+k+v: {name} (r={rank})")
    return params

# ─── Models ───
import timm
class SepConv(nn.Module):
    def __init__(self,ic,oc):
        super().__init__()
        self.dw=nn.Conv2d(ic,ic,3,padding=1,groups=ic,bias=False);self.pw=nn.Conv2d(ic,oc,1,bias=False)
        self.bn=nn.BatchNorm2d(oc);self.act=nn.GELU()
    def forward(self,x):return self.act(self.bn(self.pw(self.dw(x))))
class SE(nn.Module):
    def __init__(self,ch,r=16):
        super().__init__()
        self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(ch,ch//r,1),nn.ReLU(True),nn.Conv2d(ch//r,ch,1),nn.Sigmoid())
    def forward(self,x):return x*self.se(x)
class Dec(nn.Module):
    def __init__(self,dc=384):
        super().__init__()
        self.ll=nn.ModuleList([nn.Sequential(nn.Conv2d(1024,512,1,bias=False),nn.BatchNorm2d(512),nn.GELU(),nn.Conv2d(512,dc,1,bias=False),nn.BatchNorm2d(dc),nn.GELU())for _ in range(4)])
        self.cse=SE(dc*4);self.lf=nn.Sequential(nn.Conv2d(dc*4,dc,3,padding=1,bias=False),nn.BatchNorm2d(dc),nn.GELU())
        self.r=[nn.Sequential(SepConv(dc,dc),SE(dc))for _ in range(4)]
        for i,r in enumerate(self.r):setattr(self,f'r{i+1}',r)
        self.out=nn.Conv2d(dc,7,1)
    def forward(self,fs):
        x=torch.cat([l(f)for f,l in zip(fs,self.ll)],1);x=self.lf(self.cse(x))
        for r in [self.r1,self.r2,self.r3,self.r4]:x=r(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False))
        return self.out(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b=timm.create_model('vit_large_patch16_dinov3.sat493m',pretrained=True,img_size=512,num_classes=0)
        self.b.eval();[p.requires_grad_(False)for p in self.b.parameters()]
        self.lora_p=inject_lora(self.b);print(f"LoRA:{sum(p.numel()for p in self.lora_p):,}")
        self.d=Dec()
    def forward(self,x):
        i=self.b.forward_intermediates(x,indices=LAYERS,norm=True,output_fmt='NCHW',intermediates_only=True)
        return self.d(list(i))

# ─── Training ───
device='cuda'
model=Net().to(device)
total=sum(p.numel()for p in model.parameters())
trainable=sum(p.numel()for p in model.parameters()if p.requires_grad)
print(f"Params:{total:,}total,{trainable:,}trainable({trainable/total*100:.1f}%)")
print(f"Decoder:{sum(p.numel()for p in model.d.parameters()):,}|LoRA:{sum(p.numel()for p in model.lora_p):,}")

opt=AdamW([
    {'params':list(model.d.parameters()),'lr':LR,'weight_decay':WEIGHT_DECAY},
    {'params':model.lora_p,'lr':LORA_LR,'weight_decay':WEIGHT_DECAY},
])
sched=CosineAnnealingLR(opt,T_max=EPOCHS)
crit=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
class Dice(nn.Module):
    def __init__(self,s=1e-5):super().__init__();self.s=s
    def forward(self,p,t):
        tc=t.clone();tc[t==IGNORE_INDEX]=0
        ps=F.softmax(p,1);oh=F.one_hot(tc,num_classes=p.shape[1]).permute(0,3,1,2).float()
        m=(t!=IGNORE_INDEX).unsqueeze(1).float()
        ps=ps*m;oh=oh*m
        i=(ps*oh).sum(dim=(2,3));u=ps.sum(dim=(2,3))+oh.sum(dim=(2,3))
        return 1-((2*i+self.s)/(u+self.s)).mean()
dice=Dice()

best=0.0;bep=0;ni=0
print(f"\n=== LoRA(r={LORA_RANK},q+k+v)|Dec(384ch)|{EPOCHS}ep|lr_dec={LR}|lr_lora={LORA_LR}===\n")

for ep in range(EPOCHS):
    t0=time.time()
    model.train();tr_l=0.0
    tr_is=np.zeros(N_CLASSES,dtype=np.float64);tr_us=np.zeros(N_CLASSES,dtype=np.float64)
    for bi,(im,ms)in enumerate(tl):
        im,ms=im.to(device),ms.to(device)
        pr=model(im)
        lo=crit(pr,ms)+DICE_WEIGHT*dice(pr,ms)
        opt.zero_grad();lo.backward();nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP);opt.step()
        tr_l+=lo.item()
        if bi%100==0:print(f"[{ep+1}/{EPOCHS}]batch{bi}/{len(tl)}loss={lo.item():.4f}")
        pl=pr.argmax(1)
        for c in range(N_CLASSES):
            tr_is[c]+=((pl==c)&(ms==c)).sum().item();tr_us[c]+=((pl==c)|(ms==c)).sum().item()
    tr_l/=len(tl);tr_m=(tr_is/np.maximum(tr_us,1)).mean()

    model.eval();vl_l=0.0
    is_=np.zeros(N_CLASSES,dtype=np.float64);us_=np.zeros(N_CLASSES,dtype=np.float64)
    with torch.no_grad():
        for im,ms in vl:
            im,ms=im.to(device),ms.to(device)
            pr=model(im);vl_l+=(crit(pr,ms)+DICE_WEIGHT*dice(pr,ms)).item()
            pl=pr.argmax(1)
            for c in range(N_CLASSES):
                p=pl==c;g=ms==c;is_[c]+=(p&g).sum().item();us_[c]+=(p|g).sum().item()
    vl_l/=len(vl);io=is_/np.maximum(us_,1);vl_m=io.mean();sched.step()
    el=time.time()-t0
    cn=['bg','bld','road','water','barren','forest','agri']
    iu='|'.join(f"{n}={v:.3f}"for n,v in zip(cn,io))
    ll=f"[{ep+1:2d}/{EPOCHS}]loss={tr_l:.4f}/{vl_l:.4f}|tr_miou={tr_m:.4f}vl_miou={vl_m:.4f}(best={best:.4f}@{bep})|lr={opt.param_groups[0]['lr']:.2e}|{el:.0f}s\n  IoU:{iu}"
    print(f"  {ll}")
    with open(os.path.join(OUTPUT_DIR,"training.log"),"a")as f:f.write(ll+"\n");f.flush();os.fsync(f.fileno())
    if vl_m>best+1e-4:
        best=vl_m;bep=ep+1;ni=0
        torch.save(model.state_dict(),os.path.join(OUTPUT_DIR,"best_model.pth"))
        print(f"  -> New best! mIoU={best:.4f}")
    else:
        ni+=1;print(f"  -> No improvement({ni}/{PATIENCE})")
        if ni>=PATIENCE:print(f"Early stopping@{ep+1}");break

print(f"\nDone!Best:{best:.4f}@ep{bep}")
json.dump({'best_miou':best,'best_epoch':bep,'config':{'backbone':'dinov3_sat493m+LoRA(r=8,q+k+v)','decoder':'PCADecoder(384ch)','lora_rank':LORA_RANK,'lora_alpha':LORA_ALPHA,'lora_lr':LORA_LR,'lr':LR,'patience':PATIENCE}},open(os.path.join(OUTPUT_DIR,"results.json"),'w'),indent=2)
