"""
DINOv3 (sat493m) + DPT Head 语义分割训练程序
目标数据集: WHU Building Dataset (二值建筑分割)
功能: 冻结 DINOv3 backbone，只训练分割头
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
import timm
from tqdm import tqdm


# =====================================================================
#  1. 模型定义
# =====================================================================

class DPTHead(nn.Module):
    """
    Dense Prediction Transformer 解码器.
    从 ViT 的 4 个中间层特征中重建密集预测.
    """
    def __init__(self, embed_dim=1024, num_classes=2, fusion_dim=256):
        super().__init__()
        self.fusion_dim = fusion_dim

        # 4 个读出头，将不同深度的 ViT 特征投影到 fusion_dim
        self.read_ops = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, fusion_dim, kernel_size=1),
                nn.GELU(),
            )
            for _ in range(4)
        ])

        # 3 个逆转置融合模块 (从深到浅)
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fusion_dim * 2, fusion_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for _ in range(3)
        ])

        # 最终输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_dim, num_classes, kernel_size=1),
        )

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, features):
        """
        Args:
            features: list of 4 tensors, 每个 [B, N_patches+1+4, D]
                      对应 block 5, 11, 17, 23 的输出
        Returns:
            [B, num_classes, H, W] 其中 H=W=224 (原始输入尺寸)
        """
        B = features[0].shape[0]
        # 14x14 grid (224/16=14)
        grid_size = int((features[0].shape[1] - 5) ** 0.5)  # 196 = 14^2

        tokens = []
        for feat in features:
            # 取 patch tokens (跳过 CLS token，取前 N_patches 个)
            patch = feat[:, 1:1+grid_size*grid_size, :]       # [B, 196, D]
            patch = patch.reshape(B, grid_size, grid_size, -1) # [B, 14, 14, D]
            patch = patch.permute(0, 3, 1, 2)                 # [B, D, 14, 14]
            patch = self.read_ops[len(tokens)](patch)          # [B, 256, 14, 14]
            tokens.append(patch)

        # 所有 ViT 中间层都是 14×14 分辨率
        # 从深到浅融合，每步先上采样浅层特征到当前分辨率
        target_sizes = [112, 56, 28, 14]  # 最终要融合到的分辨率

        # 从最深层 (tokens[3]) 开始
        x = tokens[3]                      # [B, 256, 14, 14]
        x = self.upsample2x(x)             # 14 -> 28

        # 融合 tokens[2] (block_17) — 上采样 14→28
        t2 = self.upsample2x(tokens[2])    # 14 -> 28
        x = torch.cat([x, t2], dim=1)     # [B, 512, 28, 28]
        x = self.refine[2](x)             # [B, 256, 28, 28]

        x = self.upsample2x(x)             # 28 -> 56

        # 融合 tokens[1] (block_11) — 上采样 14→56 (2次)
        t1 = self.upsample2x(self.upsample2x(tokens[1]))  # 14 -> 28 -> 56
        x = torch.cat([x, t1], dim=1)
        x = self.refine[1](x)

        x = self.upsample2x(x)             # 56 -> 112

        # 融合 tokens[0] (block_5) — 上采样 14→112 (3次)
        t0 = self.upsample2x(self.upsample2x(self.upsample2x(tokens[0])))
        x = torch.cat([x, t0], dim=1)
        x = self.refine[0](x)

        x = self.output_conv(x)            # [B, 2, 112, 112]
        x = self.upsample2x(x)             # [B, 2, 224, 224]

        return x  # [B, num_classes, 224, 224]


class Dinov3Seg(nn.Module):
    """DINOv3 ViT-L/16 + DPT Head 语义分割模型.

    冻结 backbone，仅训练分割头.
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # ---- backbone: DINOv3 ViT-L/16 (sat493m 权重) ----
        print("Loading DINOv3 backbone: vit_large_patch16_dinov3.sat493m ...")
        self.backbone = timm.create_model(
            'vit_large_patch16_dinov3.sat493m',
            pretrained=True,
            num_classes=0,        # 去掉分类头
        )

        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 固定为 eval 模式 (关闭 dropout)
        self.backbone.eval()

        # ---- 分割头 ----
        self.head = DPTHead(
            embed_dim=1024,
            num_classes=num_classes,
            fusion_dim=256,
        )

        # ---- 注册前向钩子，提取中间层特征 ----
        self.features = {}
        self.hook_handles = []
        # 取 4 个中间层: blocks 5, 11, 17, 23
        self.hook_block_ids = [5, 11, 17, 23]

        for bid in self.hook_block_ids:
            def make_hook(name):
                def hook(_, __, output):
                    self.features[name] = output
                return hook
            handle = self.backbone.blocks[bid].register_forward_hook(
                make_hook(f'block_{bid}')
            )
            self.hook_handles.append(handle)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 归一化后的图像
        Returns:
            [B, num_classes, H, W] 分割 logits
        """
        self.features = {}

        # 前向 backbone（钩子自动填充 self.features）
        _ = self.backbone(x)

        # 收集 4 层特征，按深度顺序
        feat_list = [self.features[f'block_{bid}'] for bid in self.hook_block_ids]

        # 通过 DPT Head 解码
        out = self.head(feat_list)
        return out

    def train(self, mode=True):
        """重写 train()，确保 backbone 始终 eval."""
        super().train(mode)
        if mode:
            # 训练模式时，backbone 保持 eval，head 切换为 train
            self.backbone.eval()
        return self


# =====================================================================
#  2. 损失函数
# =====================================================================

class DiceLoss(nn.Module):
    """多类别 Dice Loss."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:   [B, C, H, W] logits
        target: [B, H, W] 类别索引 (long)
        """
        pred_softmax = F.softmax(pred, dim=1)                             # [B, C, H, W]
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])      # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()         # [B, C, H, W]

        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))     # [B, C]
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        return 1.0 - dice.mean()


def combined_loss(pred, target):
    """BCE + Dice 混合损失."""
    ce = F.cross_entropy(pred, target)
    dice = DiceLoss()(pred, target)
    return ce + dice


# =====================================================================
#  3. 数据集
# =====================================================================

class WHUBuildingDataset(Dataset):
    
    def __init__(self, root, split='train', img_size=224, crop_size=224, augment=True):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')

        img_dir = self.root / split / 'image'
        label_dir = self.root / split / 'label'

        self.images = sorted(img_dir.glob('*'))
        self.labels = sorted(label_dir.glob('*'))

        assert len(self.images) == len(self.labels), \
            f"图片数 ({len(self.images)}) != 标签数 ({len(self.labels)})"
        assert len(self.images) > 0, \
            f"在 {img_dir} 中未找到图片"

        # DINOv2/v3 推荐标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图片和标签
        img = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx]).convert('L')

        # 缩放到目标尺寸
        if img.size[0] != self.img_size:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            label = label.resize((self.img_size, self.img_size), Image.NEAREST)

        # 转为 Tensor
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(np.array(label)).long()

        # WHU 标签: 0=背景, 255=建筑 → 映射为 0, 1
        label = (label > 0).long()

        # 数据增强 (训练集)
        if self.augment:
            # 随机水平翻转
            if torch.rand(1).item() > 0.5:
                img = img.flip(dims=(2,))
                label = label.flip(dims=(1,))
            # 随机垂直翻转
            if torch.rand(1).item() > 0.5:
                img = img.flip(dims=(1,))
                label = label.flip(dims=(0,))

        # 标准化
        img = (img - self.mean) / self.std

        return img, label


# =====================================================================
#  4. 评估指标
# =====================================================================

@torch.no_grad()
def compute_metrics(pred, target, num_classes=2):
    """计算 mIoU 和每个类别的 IoU."""
    pred_cls = pred.argmax(dim=1)  # [B, H, W]
    ious = []
    for cls in range(num_classes):
        inter = ((pred_cls == cls) & (target == cls)).sum().float()
        union = ((pred_cls == cls) | (target == cls)).sum().float()
        iou = (inter + 1e-6) / (union + 1e-6)
        ious.append(iou.item())

    # 总体精度
    acc = (pred_cls == target).float().mean().item()
    return {
        'mIoU': np.mean(ious),
        'IoU_bg': ious[0],
        'IoU_building': ious[1],
        'Acc': acc,
    }


# =====================================================================
#  5. 训练与验证函数
# =====================================================================

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', ncols=80)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = combined_loss(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    all_metrics = []

    pbar = tqdm(loader, desc=f'Val   Epoch {epoch}', ncols=80)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = combined_loss(preds, labels)
        total_loss += loss.item()

        metrics = compute_metrics(preds, labels)
        all_metrics.append(metrics)
        pbar.set_postfix(loss=f'{loss.item():.4f}', mIoU=f'{metrics["mIoU"]:.4f}')

    # 汇总
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    avg_metrics['loss'] = total_loss / len(loader)
    return avg_metrics


# =====================================================================
#  6. 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DINOv3 + DPT 语义分割（WHU Building 数据集）'
    )
    # 数据
    parser.add_argument('--data_root', type=str, required=True,
                        help='WHU Building Dataset 根目录')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图片尺寸 (默认 224)')
    # 训练
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=6e-5,
                        help='学习率 (分割头)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    # 保存
    parser.add_argument('--save_dir', type=str, default='./dinov3_seg_whu',
                        help='模型与日志保存目录')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每 N 轮保存一次 checkpoint')
    args = parser.parse_args()

    # ---- 固定随机种子 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ---- 输出目录 ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    print('=' * 60)
    print('WHU Building Dataset - DINOv3 语义分割训练')
    print(f'设备: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('=' * 60)

    print(f'\n[数据] 加载 WHU Building 从: {args.data_root}')
    train_set = WHUBuildingDataset(args.data_root, 'train',
                                    img_size=args.img_size, augment=True)
    val_set   = WHUBuildingDataset(args.data_root, 'val',
                                    img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    print(f'  训练集: {len(train_set)} 张')
    print(f'  验证集: {len(val_set)} 张')
    print(f'  批次大小: {args.batch_size}')

    # ---- 模型 ----
    print(f'\n[模型] 构建 DINOv3 (sat493m) + DPT Head ...')
    model = Dinov3Seg(num_classes=2).to(device)
    # 验证 backbone 确实冻结了
    frozen_params = sum(p.numel() for p in model.backbone.parameters())
    trainable_params = sum(p.numel() for p in model.head.parameters())
    print(f'  Backbone (冻结): {frozen_params:,} 参数')
    print(f'  Head (可训练):   {trainable_params:,} 参数')
    print(f'  总计:            {frozen_params + trainable_params:,} 参数')

    # ---- 优化器 & 调度器 ----
    optimizer = optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- 训练循环 ----
    best_miou = 0.0
    best_epoch = 0
    history = []

    print(f'\n[训练] 共 {args.epochs} epochs')
    print('-' * 60)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # 验证
        val_metrics = val_epoch(model, val_loader, device, epoch)

        # 更新学习率
        scheduler.step()

        # 日志
        elapsed = time.time() - epoch_start
        log = (
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_metrics["loss"]:.4f} | '
            f'mIoU: {val_metrics["mIoU"]:.4f} | '
            f'Building IoU: {val_metrics["IoU_building"]:.4f} | '
            f'Acc: {val_metrics["Acc"]:.4f} | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
            f'Time: {elapsed:.1f}s'
        )
        print(log)

        # 保存历史
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'mIoU': val_metrics['mIoU'],
            'IoU_building': val_metrics['IoU_building'],
            'IoU_bg': val_metrics['IoU_bg'],
            'acc': val_metrics['Acc'],
            'lr': scheduler.get_last_lr()[0],
        })

        # 保存最佳模型
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            best_epoch = epoch
            ckpt_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mIoU': best_miou,
                'args': vars(args),
            }, ckpt_path)
            print(f'  >>> 保存最佳模型 [{ckpt_path}] (mIoU={best_miou:.4f})')

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mIoU': best_miou,
                'args': vars(args),
            }, ckpt_path)

    # ---- 训练结束 ----
    print('=' * 60)
    print(f'训练完成!')
    print(f'最佳验证 mIoU: {best_miou:.4f} (epoch {best_epoch})')

    # 保存训练日志
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 保存最终模型
    final_path = save_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'best_mIoU': best_miou,
        'args': vars(args),
    }, final_path)
    print(f'最终模型保存至: {final_path}')
    print(f'训练日志保存至: {save_dir / "training_history.json"}')
    print(f'最佳模型保存至: {save_dir / "best_model.pth"}')


if __name__ == '__main__':
    main()
