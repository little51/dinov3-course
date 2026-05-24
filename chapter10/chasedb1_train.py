import os, sys, time, platform, warnings
warnings.filterwarnings('ignore')
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

# ── 中文字体设定 (自动适配 Windows/Linux) ─────────────────────
_os = platform.system()
if _os == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif _os == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════════════
#  路径配置 — 使用相对路径（相对于本文件所在的 chasedb1/ 目录）
#  如果你的数据/权重在其他位置，可以改为绝对路径或符号链接
# ════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CHASEDB1 数据集根目录（含 train/、val/ 子文件夹）
DATA_ROOT = os.path.join(_SCRIPT_DIR, 'data', 'CHASEDB1')

# 训练输出目录
OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, 'outputs')

# EUPE 代码库路径（从 GitHub 克隆的 EUPE 根目录）
EUPE_DIR = os.path.join(_SCRIPT_DIR, 'EUPE')

# EUPE-ConvNeXt-B 权重文件路径
# 下载地址：https://huggingface.co/nvidia/EUPE-ConvNeXt-B
EUPE_CKPT = os.path.join(_SCRIPT_DIR, 'models', 'EUPE-ConvNeXt-B.pt')

# 是否使用 HuggingFace 镜像（国内用户建议 True）
USE_HF_MIRROR = True

# ════════════════════════════════════════════════════════════════

OUTPUT = str(Path(OUTPUT_ROOT))
MAX_EPOCHS = 200
PATIENCE = 20           # 早停轮数（连续 N 轮无提升即停止）
SEED = 42
IMG_SIZE = 1024

os.makedirs(OUTPUT, exist_ok=True)
if USE_HF_MIRROR:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
has_gpu = torch.cuda.is_available()
num_w = 0 if _os == 'Windows' else 4   # Windows 多进程 DataLoader 有时出问题

print('设备:', device)
print('系统:', _os)
print('输出:', OUTPUT)
print('数据集:', DATA_ROOT)

# ── Dataset ────────────────────────────────────────────
class CHASEDB1(Dataset):
    def __init__(self, root, split, img_size, augment=True):
        self.input_dir = Path(root) / split / 'input'
        self.label_dir = Path(root) / split / 'label'
        self.files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.png')])
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        base = os.path.splitext(fname)[0]
        img = Image.open(self.input_dir / fname).convert('RGB')
        label = Image.open(self.label_dir / f'{base}_1stHO.png')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        label = label.resize((self.img_size, self.img_size), Image.NEAREST)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        label_t = torch.from_numpy(np.array(label, dtype=np.int64)).long()
        label_t = (label_t > 0).long()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_t = img_t.flip(dims=(2,)); label_t = label_t.flip(dims=(1,))
            if torch.rand(1).item() > 0.5:
                img_t = img_t.flip(dims=(1,)); label_t = label_t.flip(dims=(0,))
        img_t = (img_t - self.mean) / self.std
        return img_t, label_t

# ── Metrics ────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_p, all_t = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
        all_p.append(logits.argmax(dim=1).cpu()); all_t.append(masks.cpu())
    p = torch.cat(all_p).view(-1); t = torch.cat(all_t).view(-1)
    ious = []
    for cls in range(2):
        inter = ((p == cls) & (t == cls)).sum().item()
        union = ((p == cls) | (t == cls)).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return {'mIoU': float(np.nanmean(ious)), 'Vessel_IoU': ious[1],
            'Acc': (p == t).float().mean().item()}

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth
    def forward(self, pred, target):
        ps = F.softmax(pred, dim=1)
        oh = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        inter = (ps * oh).sum(dim=(2, 3))
        union = ps.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        return 1.0 - (2.0 * inter + self.smooth).mean() / (union + self.smooth).mean()


# ════════════════════════════════════════════════════════
#  PART 1: U-Net (resnet34, 全参数微调)
# ════════════════════════════════════════════════════════
def train_unet():
    print('\n' + '=' * 60)
    print('  1/2: U-Net (1024×1024) 全参数训练')
    print('=' * 60)
    import segmentation_models_pytorch as smp

    train_ds = CHASEDB1(DATA_ROOT, 'train', IMG_SIZE, augment=True)
    val_ds   = CHASEDB1(DATA_ROOT, 'val',   IMG_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              num_workers=num_w, pin_memory=has_gpu, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                              num_workers=num_w, pin_memory=has_gpu)

    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2).to(device)
    total = sum(p.numel() for p in model.parameters())
    print('  参数量: %d (全部可训练)' % total)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best, best_epoch, no_improve, log = 0.0, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_t0 = time.time()
        model.train(); loss_sum = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = F.cross_entropy(model(imgs), masks) + DiceLoss()(model(imgs), masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item()
        sched.step()
        metrics = evaluate(model, val_loader)

        if metrics['mIoU'] > best:
            best = metrics['mIoU']; best_epoch = epoch; no_improve = 0
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'metrics': metrics},
                       os.path.join(OUTPUT, 'unet_best.pth'))
        else:
            no_improve += 1

        epoch_time = time.time() - epoch_t0
        print('  U-Net E%d/%d | L=%.4f | mIoU=%.4f | Ves=%.4f | best=%.4f@E%d | no_impr=%d | %.0fs' % (
            epoch, MAX_EPOCHS, loss_sum/len(train_loader), metrics['mIoU'], metrics['Vessel_IoU'],
            best, best_epoch, no_improve, epoch_time))
        log.append(metrics)

        if no_improve >= PATIENCE:
            print('  早停触发! (best mIoU=%.4f @ E%d)' % (best, best_epoch))
            break

    ckpt = torch.load(os.path.join(OUTPUT, 'unet_best.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print('  U-Net 完成, 共 %d 轮, 最佳 mIoU=%.4f, Vessel IoU=%.4f' % (len(log), best, ckpt['metrics']['Vessel_IoU']))
    return model, log, os.path.join(OUTPUT, 'unet_best.pth')


# ════════════════════════════════════════════════════════
#  PART 2: EUPE ConvNeXt Base + DPT (冻结 backbone)
# ════════════════════════════════════════════════════════
class ConvNeXtDPTHead(nn.Module):
    """DPT head 适配 ConvNeXt Base 的 4 级多尺度特征 [128, 256, 512, 1024]"""
    def __init__(self, nc=2, fd=256):
        super().__init__()
        feat_dims = [128, 256, 512, 1024]
        self.r = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, fd, 1), nn.GELU()) for d in feat_dims
        ])
        self.f = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fd * 2, fd, 3, padding=1), nn.GELU(),
                          nn.Conv2d(fd, fd, 3, padding=1), nn.GELU()) for _ in range(3)
        ])
        self.o = nn.Sequential(
            nn.Conv2d(fd, fd, 3, padding=1), nn.GELU(), nn.Conv2d(fd, nc, 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, feats):
        """feats: [1/4(256×256), 1/8(128×128), 1/16(64×64), 1/32(32×32)]"""
        t = [r(f) for r, f in zip(self.r, feats)]
        x = self.up(t[3]); x = torch.cat([x, t[2]], dim=1); x = self.f[2](x)
        x = self.up(x);   x = torch.cat([x, t[1]], dim=1); x = self.f[1](x)
        x = self.up(x);   x = torch.cat([x, t[0]], dim=1); x = self.f[0](x)
        return self.up(self.up(x))


def train_convnext():
    print('\n' + '=' * 60)
    print('  2/2: EUPE ConvNeXt Base + DPT (冻结 Backbone)')
    print('=' * 60)

    # 加载 backbone
    sys.path.insert(0, EUPE_DIR)
    bb = torch.hub.load(EUPE_DIR, 'eupe_convnext_base', source='local', pretrained=False)

    sd = torch.load(EUPE_CKPT, map_location='cpu', weights_only=False)
    m, u = bb.load_state_dict(sd, strict=False)
    if m:
        print('  missing keys:', m)
    print('  unexpected keys: %d (projectors, 忽略)' % len(u))

    bb = bb.to(device).eval()
    for p in bb.parameters(): p.requires_grad_(False)
    bb_params = sum(p.numel() for p in bb.parameters()) / 1e6
    print('  backbone: %.0fM 参数 (冻结)' % bb_params)

    # DPT head
    head = ConvNeXtDPTHead(nc=2).to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print('  DPT 头: %d 可训练参数 (%.1fM)' % (head_params, head_params / 1e6))

    # Data
    train_ds = CHASEDB1(DATA_ROOT, 'train', IMG_SIZE, augment=True)
    val_ds   = CHASEDB1(DATA_ROOT, 'val',   IMG_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=num_w, pin_memory=has_gpu, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=num_w, pin_memory=has_gpu)

    # Training
    opt = torch.optim.AdamW(head.parameters(), lr=6e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best, best_epoch, no_improve, log = 0.0, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_t0 = time.time()
        head.train(); loss_sum = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                feats = bb.get_intermediate_layers(imgs, n=[0, 1, 2, 3], reshape=True)
                logits = head(feats)
                if logits.shape[2:] != masks.shape[1:]:
                    logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = F.cross_entropy(logits, masks) + DiceLoss()(logits, masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item()
        sched.step()

        class Wrap(nn.Module):
            def __init__(self): super().__init__(); self.bb = bb; self.head = head
            def forward(self, x):
                feats = self.bb.get_intermediate_layers(x, n=[0,1,2,3], reshape=True)
                return self.head(feats)
        metrics = evaluate(Wrap(), val_loader)

        if metrics['mIoU'] > best:
            best = metrics['mIoU']; best_epoch = epoch; no_improve = 0
            torch.save({'epoch': epoch, 'head_state_dict': head.state_dict(), 'metrics': metrics},
                       os.path.join(OUTPUT, 'convnext_best.pth'))
        else:
            no_improve += 1

        epoch_time = time.time() - epoch_t0
        print('  CNX E%d/%d | L=%.4f | mIoU=%.4f | Ves=%.4f | best=%.4f@E%d | no_impr=%d | %.0fs' % (
            epoch, MAX_EPOCHS, loss_sum/len(train_loader), metrics['mIoU'], metrics['Vessel_IoU'],
            best, best_epoch, no_improve, epoch_time))
        log.append(metrics)

        if no_improve >= PATIENCE:
            print('  早停触发! (best mIoU=%.4f @ E%d)' % (best, best_epoch))
            break

    ckpt = torch.load(os.path.join(OUTPUT, 'convnext_best.pth'), map_location='cpu', weights_only=False)
    print('  ConvNeXt Base 完成, 共 %d 轮, 最佳 mIoU=%.4f, Vessel IoU=%.4f @ E%d' % (
        len(log), best, ckpt['metrics']['Vessel_IoU'], best_epoch))
    return (bb, head), log, os.path.join(OUTPUT, 'convnext_best.pth')


# ════════════════════════════════════════════════════════
#  PART 3: 对比可视化
# ════════════════════════════════════════════════════════
def visualize(unet_log, convnext_log, unet_ckpt_path, convnext_ckpt_path):
    print('\n' + '=' * 60)
    print('  3/3: 对比可视化')
    print('=' * 60)

    unet_metrics = torch.load(unet_ckpt_path, map_location='cpu', weights_only=False)['metrics']
    cnx_metrics  = torch.load(convnext_ckpt_path, map_location='cpu', weights_only=False)['metrics']

    # ── 曲线图：mIoU 与 Vessel_IoU ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='white')
    colors = {'unet': '#2196F3', 'cnx': '#9C27B0'}
    labels_m = {'unet': 'U-Net (resnet34)', 'cnx': 'EUPE ConvNeXt Base + DPT'}

    for ax, key, title in zip(axes, ['mIoU', 'Vessel_IoU'], ['mIoU', 'Vessel IoU']):
        ax.plot(range(1, len(unet_log)+1), [m[key] for m in unet_log], '-', color=colors['unet'],
                lw=1.5, label=labels_m['unet'], alpha=0.85)
        ax.plot(range(1, len(convnext_log)+1), [m[key] for m in convnext_log], '-', color=colors['cnx'],
                lw=1.5, label=labels_m['cnx'], alpha=0.85)
        ax.axhline(y=unet_metrics[key], color=colors['unet'], ls=':', lw=1, alpha=0.5)
        ax.axhline(y=cnx_metrics[key], color=colors['cnx'], ls=':', lw=1, alpha=0.5)
        ax.set_xlabel('Epoch'); ax.set_ylabel(key)
        ax.set_title('CHASEDB1 1024×1024 ' + title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, 'curves.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(); print('  [OK] curves.png')

    # ── 柱状图 ──
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    names = ['mIoU', 'Vessel IoU']
    x = range(2); w = 0.3
    b1 = ax.bar([i-w/2 for i in x],
                [unet_metrics['mIoU'], unet_metrics['Vessel_IoU']], w,
                label='U-Net', color='#2196F3', ec='white', lw=1.5)
    b2 = ax.bar([i+w/2 for i in x],
                [cnx_metrics['mIoU'], cnx_metrics['Vessel_IoU']], w,
                label='ConvNeXt Base+DPT', color='#9C27B0', ec='white', lw=1.5)

    ax.set_ylabel('Score'); ax.set_title('CHASEDB1 1024×1024 双模型对比', fontsize=15, fontweight='bold')
    ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(b1, [unet_metrics['mIoU'], unet_metrics['Vessel_IoU']]):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005, f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2196F3')
    for bar, val in zip(b2, [cnx_metrics['mIoU'], cnx_metrics['Vessel_IoU']]):
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005, f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#9C27B0')
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT, 'bar.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(); print('  [OK] bar.png')

    # ── 样本可视化 ──
    import segmentation_models_pytorch as smp
    unet_model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=2).to(device)
    unet_ckpt = torch.load(unet_ckpt_path, map_location=device, weights_only=False)
    unet_model.load_state_dict(unet_ckpt['state_dict'])
    unet_model.eval()

    convnext_ckpt = torch.load(convnext_ckpt_path, map_location='cpu', weights_only=False)
    sys.path.insert(0, EUPE_DIR)
    bb = torch.hub.load(EUPE_DIR, 'eupe_convnext_base', source='local', pretrained=False)
    bb.load_state_dict(torch.load(EUPE_CKPT, map_location='cpu', weights_only=False), strict=False)
    bb = bb.to(device).eval()
    for p in bb.parameters(): p.requires_grad_(False)

    head = ConvNeXtDPTHead(nc=2).to(device)
    head.load_state_dict(convnext_ckpt['head_state_dict'])
    head.eval()

    vis_dir = Path(OUTPUT) / 'samples'; vis_dir.mkdir(parents=True, exist_ok=True)
    val_ds = CHASEDB1(DATA_ROOT, 'val', IMG_SIZE, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 收集有血管的样本
    samples = []
    for imgs, masks in val_loader:
        if masks.sum() > 100:
            samples.append((imgs[0], masks[0]))
        if len(samples) >= 6: break

    with torch.no_grad():
        for idx, (img_t, mask_t) in enumerate(samples):
            mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_disp = (img_t.cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
            img_cuda = img_t.unsqueeze(0).to(device)
            gt = mask_t.numpy().astype(bool)

            with torch.amp.autocast('cuda'):
                p_unet = unet_model(img_cuda).argmax(dim=1).squeeze(0).cpu().numpy().astype(bool)
                feats = bb.get_intermediate_layers(img_cuda, n=[0,1,2,3], reshape=True)
                p_cnx = head(feats).argmax(dim=1).squeeze(0).cpu().numpy().astype(bool)

            fig, axes = plt.subplots(2, 3, figsize=(16, 11))

            def overlay(img, mask, color=(0,0.4,1)):
                ov = img.copy(); ov[mask] = ov[mask]*0.5 + np.array(color)*0.5; return ov

            # 第一行：原图 | U-Net 叠加 | ConvNeXt 叠加
            axes[0, 0].imshow(img_disp); axes[0, 0].set_title('原图', fontsize=14, fontweight='bold'); axes[0, 0].axis('off')
            axes[0, 1].imshow(overlay(img_disp, p_unet, (0,0.4,1))); axes[0, 1].set_title('U-Net 叠加', fontsize=14, fontweight='bold', color='#2196F3'); axes[0, 1].axis('off')
            axes[0, 2].imshow(overlay(img_disp, p_cnx, (0.5,0,0.7))); axes[0, 2].set_title('ConvNeXt 叠加', fontsize=14, fontweight='bold', color='#9C27B0'); axes[0, 2].axis('off')

            # 第二行：真值 | U-Net | ConvNeXt
            axes[1, 0].imshow(gt, cmap='gray'); axes[1, 0].set_title('真值', fontsize=14, fontweight='bold'); axes[1, 0].axis('off')
            axes[1, 1].imshow(p_unet, cmap='gray'); axes[1, 1].set_title('U-Net', fontsize=14, fontweight='bold', color='#2196F3'); axes[1, 1].axis('off')
            axes[1, 2].imshow(p_cnx, cmap='gray'); axes[1, 2].set_title('ConvNeXt', fontsize=14, fontweight='bold', color='#9C27B0'); axes[1, 2].axis('off')

            plt.suptitle('CHASEDB1 1024×1024 — U-Net vs EUPE ConvNeXt Base+DPT', fontsize=15, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(vis_dir / ('sample_%02d.png' % idx), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        print('  [OK] samples/ (6 张)')


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_total = time.time()

    # 1. U-Net
    t0 = time.time()
    unet_model, unet_log, unet_ckpt = train_unet()
    unet_time = (time.time() - t0) / 60

    # 2. EUPE ConvNeXt Base + DPT
    t0 = time.time()
    (bb, head), convnext_log, convnext_ckpt = train_convnext()
    cnx_time = (time.time() - t0) / 60

    # 3. 可视化
    visualize(unet_log, convnext_log, unet_ckpt, convnext_ckpt)

    # Summary
    unet_best = torch.load(unet_ckpt, map_location='cpu', weights_only=False)['metrics']
    cnx_best  = torch.load(convnext_ckpt, map_location='cpu', weights_only=False)['metrics']

    print('\n' + '=' * 60)
    print('  CHASEDB1 1024×1024 双模型训练完成')
    print('=' * 60)
    print('  模型                   mIoU     Vessel IoU  耗时')
    print('  ────────────────────────────────────────────────')
    print('  U-Net (resnet34)       %.4f   %.4f    %.0f 分钟' % (
        unet_best['mIoU'], unet_best['Vessel_IoU'], unet_time))
    print('  EUPE ConvNeXt Base+DPT %.4f   %.4f    %.0f 分钟' % (
        cnx_best['mIoU'], cnx_best['Vessel_IoU'], cnx_time))
    print('  ────────────────────────────────────────────────')
    total_time = (time.time() - t_total) / 60
    print('  总计: %.0f 分钟' % total_time)
    print('  输出: %s' % OUTPUT)
    print('=' * 60)
