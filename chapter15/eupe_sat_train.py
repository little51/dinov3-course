import os, sys, time, json
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch.optim.lr_scheduler import OneCycleLR


# WHU 数据集解压后的目录（里面应有 train/ 和 val/ 子目录）
DATA_DIR    = 'whu_building'

# EUPE 仓库克隆路径
EUPE_REPO   = 'eupe'

# EUPE-ViT-B 权重文件路径
EUPE_WEIGHT = 'models/eupe/EUPE-ViT-B.pt'

# 输出目录
OUTPUT_DIR  = 'outputs'

# ─── 训练超参数（通常无需修改） ────────────────────────────────────
BATCH_SIZE = 32       # 如果显存不足（<8GB），改为 4 或 2
EPOCHS = 3
IMG_SIZE = 512
LR = 1e-3
SEED = 42

TOTAL_ITERS = (4736 // BATCH_SIZE) * EPOCHS

# ═══════════════════════════════════════════════════════════════════
#  以下代码无需修改
# ═══════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ─── Dataset ──────────────────────────────────────────────────────
class WHUBuildingDataset(Dataset):
    def __init__(self, root, split='train', transform=None, mask_transform=None):
        self.img_dir = Path(root) / split / 'image'
        self.mask_dir = Path(root) / split / 'label'
        self.transform = transform
        self.mask_transform = mask_transform
        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.tif') and os.path.exists(os.path.join(self.mask_dir, f))
        ])
        print(f'  {split}: {len(self.files)} images')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(self.img_dir / fname).convert('RGB')
        mask = Image.open(self.mask_dir / fname)
        mask = np.array(mask, dtype=np.int64)
        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(Image.fromarray(mask.astype(np.uint8)))
            mask = mask.squeeze(0).long()
        return img, mask

train_tf = v2.Compose([
    v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
train_mask_tf = v2.Compose([
    v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True, interpolation=v2.InterpolationMode.NEAREST),
])
eval_tf = v2.Compose([
    v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
eval_mask_tf = v2.Compose([
    v2.ToImage(), v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True, interpolation=v2.InterpolationMode.NEAREST),
])

print("Loading datasets...")
train_ds = WHUBuildingDataset(DATA_DIR, 'train', train_tf, train_mask_tf)
val_ds = WHUBuildingDataset(DATA_DIR, 'val', eval_tf, eval_mask_tf)

# Windows 上 num_workers=0 最稳定，多进程反而可能卡死
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch
train_iter = infinite_loader(train_loader)

# ─── Linear Head ──────────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.dropout(x)
        x = self.bn(x)
        x = self.conv(x)
        return x

# ─── 指标 ──────────────────────────────────────────────────────────
@torch.no_grad()
def compute_iou(pred, target, num_classes=2):
    pred = pred.view(-1)
    target = target.view(-1)
    mask = target != 255
    pred, target = pred[mask], target[mask]
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious, np.nanmean(ious)

@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    all_preds, all_targets = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            feats = backbone(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            logits = head(feats)
        preds = logits.argmax(dim=1)
        if preds.shape[1:] != masks.shape[1:]:
            preds = F.interpolate(preds.float().unsqueeze(1), size=masks.shape[1:], mode='nearest').squeeze(1).long()
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    ious, miou = compute_iou(all_preds, all_targets)
    acc = (all_preds == all_targets).float().mean().item()
    return {'mIoU': miou, 'Building_IoU': ious[1], 'Background_IoU': ious[0], 'Pixel_Acc': acc}


# ─── 可视化函数：训练完成后抽2张验证集图，出四栏图 ────────────────
def visualize_predictions(backbone_name, backbone, embed_dim, dataset,
                          device, output_dir, num_samples=2, seed=42):
    """在验证集上抽 num_samples 张图，显示 原图 | 分割结果 | 标注 | 合并叠加"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'SimHei'

    head_path = os.path.join(output_dir, f'{backbone_name}_best.pth')
    if not os.path.exists(head_path):
        print(f'  [警告] 找不到 head 权重: {head_path}，跳过可视化')
        return

    chk = torch.load(head_path, map_location=device, weights_only=False)
    head = LinearHead(embed_dim, 2).to(device)
    head.load_state_dict(chk['head_state_dict'])
    head.eval()
    backbone.eval()

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        mask_np = mask.cpu().numpy()

        with torch.no_grad():
            inp = img.unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                feats = backbone(inp)
                if isinstance(feats, (list, tuple)):
                    feats = feats[0]
                logits = head(feats)
            pred = logits.argmax(dim=1).cpu()
        if pred.shape[1:] != mask.shape:
            pred = F.interpolate(pred.float().unsqueeze(1),
                                 size=mask.shape, mode='nearest').squeeze(1)
        pred_np = pred[0].numpy()

        img_np = img.cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'原图 (Sample {idx})', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f'{backbone_name} 分割结果', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title('标注', fontsize=10)
        axes[row, 2].axis('off')

        overlay = img_np.copy()
        tp = (pred_np == 1) & (mask_np == 1)
        fp = (pred_np == 1) & (mask_np == 0)
        fn = (pred_np == 0) & (mask_np == 1)
        overlay[tp] = overlay[tp] * 0.5 + np.array([0.0, 0.5, 0.0])
        overlay[fp] = overlay[fp] * 0.5 + np.array([0.5, 0.0, 0.0])
        overlay[fn] = overlay[fn] * 0.5 + np.array([0.0, 0.0, 0.5])
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title('合并 (绿=TP 红=FP 蓝=FN)', fontsize=9)
        axes[row, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{backbone_name}_vis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [{backbone_name}] 可视化已保存: {save_path}')


# ─── 训练函数 ──────────────────────────────────────────────────────
def train_one_backbone(name, backbone, embed_dim, output_path):
    print(f'\n{"="*60}')
    print(f'  Training: {name}')
    print(f'  Embed dim: {embed_dim}')
    print(f'{"="*60}')

    head = LinearHead(embed_dim, 2).to(device)
    print(f'  Head params: {sum(p.numel() for p in head.parameters()):,}')

    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=TOTAL_ITERS,
                           pct_start=0.1, anneal_strategy='cos', final_div_factor=1e4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    start = time.time()

    for step in range(1, TOTAL_ITERS + 1):
        images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        head.train()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            feats = backbone(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            logits = head(feats)
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(logits, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            elapsed = time.time() - start
            print(f'  iter {step:5d}/{TOTAL_ITERS} | loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {step/elapsed:.1f} it/s')

    # Final eval
    metrics = evaluate(backbone, head, val_loader, device)
    print(f'\n  >>> {name} Final: mIoU={metrics["mIoU"]:.4f}, Building={metrics["Building_IoU"]:.4f}, '
          f'Background={metrics["Background_IoU"]:.4f}, Acc={metrics["Pixel_Acc"]:.4f}')

    torch.save({
        'name': name,
        'head_state_dict': head.state_dict(),
        'metrics': metrics,
        'training_time': time.time() - start,
    }, output_path)

    return metrics


# ═══════════════════════════════════════════════════════════════════
#  EUPE-ViT-B 线性 probing 分割（WHU Building 数据集）
# ═══════════════════════════════════════════════════════════════════
print("\n=== Loading EUPE-ViT-B backbone ===")
sys.path.insert(0, EUPE_REPO)
from eupe.eval.setup import load_model_and_context
from eupe.eval.utils import ModelWithIntermediateLayers
from functools import partial

@dataclass
class MCfg:
    eupe_hub: str = 'eupe_vitb16'
    pretrained_weights: str = EUPE_WEIGHT
    config_file: str = None

eupe_backbone, ctx = load_model_and_context(MCfg(), OUTPUT_DIR)
eupe_embed = eupe_backbone.embed_dim

autocast_ctx = partial(torch.amp.autocast, device_type='cuda', enabled=torch.cuda.is_available())
eupe_backbone = ModelWithIntermediateLayers(
    eupe_backbone, n=[eupe_backbone.n_blocks - 1],
    autocast_ctx=autocast_ctx, reshape=True, return_class_token=False,
).to(device)
eupe_backbone.eval()
for p in eupe_backbone.parameters():
    p.requires_grad_(False)

eupe_metrics = train_one_backbone(
    'EUPE', eupe_backbone, eupe_embed,
    os.path.join(OUTPUT_DIR, 'eupe_best.pth')
)

# 可视化分割结果（2张验证集图，4栏展示）
visualize_predictions('EUPE', eupe_backbone, eupe_embed,
                      val_ds, device, OUTPUT_DIR, num_samples=2, seed=42)

# 保存结果
results = {k: float(eupe_metrics[k]) for k in ['mIoU', 'Building_IoU', 'Background_IoU', 'Pixel_Acc']}
with open(os.path.join(OUTPUT_DIR, 'EUPE_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n结果已保存到 {os.path.join(OUTPUT_DIR, "EUPE_results.json")}')
print('Done!')
