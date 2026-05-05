import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import timm
import numpy as np
import os
import glob

# ==================================== 配置 ====================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224          # 输入分辨率
BATCH_SIZE = 8            # PH2 只有 200 张，调小 batch size
EPOCHS = 100              # 小数据集需要更多 epoch
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2           # 20% 做验证（40 张）
DATA_DIR = "./data/PH2Dataset/PH2 Dataset images"

# ================================== 数据集 ====================================
class PH2Dataset(Dataset):

    def __init__(self, image_paths, mask_paths, image_size=224, is_train=True):
        self.images = image_paths
        self.masks = mask_paths
        self.image_size = image_size
        self.is_train = is_train

        # 图像：缩放 + 张量化 + ImageNet归一化
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        # 掩码：缩放 + 张量化（最近邻插值，不做归一化）
        self.mask_resize = T.Resize((image_size, image_size),
                                    interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")   # 灰度单通道

        # 训练时数据增强
        if self.is_train:
            if np.random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if np.random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        img = self.img_transform(img)          # (3, H, W)
        mask = self.mask_resize(mask)           # (1, H, W)
        mask = TF.to_tensor(mask)               # (1, H, W), [0, 1]
        mask = (mask > 0.5).float()             # 二值化

        return img, mask


# ================================ 解码器 =======================================
class SimpleDecoder(nn.Module):
    """
    轻量解码器：DINOv3 patch 特征 → 上采样 → 分割掩码

    DINOv3 ViT-S/16 输出 (B, 384, 14, 14)，4次 2× 上采样到 (B, 1, 224, 224)
    """

    def __init__(self, in_channels=384, mid_channels=[256, 128, 64, 32]):
        super().__init__()
        layers = []
        prev = in_channels
        for mc in mid_channels:
            layers.extend([
                nn.Conv2d(prev, mc, kernel_size=3, padding=1),
                nn.BatchNorm2d(mc),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ])
            prev = mc
        layers.append(nn.Conv2d(mid_channels[-1], 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# ================================ 损失函数 =====================================
def dice_loss(pred, target, smooth=1.0):
    """Dice Loss for binary segmentation"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(pred, target):
    """BCE + Dice 联合损失"""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ================================ 评价指标 =====================================
@torch.no_grad()
def compute_metrics(pred, target):
    """计算 IoU、Dice、像素准确率"""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred_bin * target).sum().item()
    union = (pred_bin + target).clamp(0, 1).sum().item()
    tp = intersection
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    tn = ((1 - pred_bin) * (1 - target)).sum().item()

    iou = tp / (union + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return iou, dice, acc


# ================================= 主流程 ======================================
def main():
    print("=" * 60)
    print("DINOv3 + 轻量解码器 — PH2 皮肤病变分割")
    print("=" * 60)

    # ---- 1. 扫描数据集 ----
    print("\n[1/4] 扫描数据...")
    case_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "IMD*")))

    if not case_dirs:
        print(f"  ❌ 未找到数据，请确认目录结构:")
        print(f"     {DATA_DIR}/IMD240/IMD240_Dermoscopic_Image/IMD240.bmp")
        return

    image_paths = []
    mask_paths = []
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        # 原图
        img_match = glob.glob(os.path.join(case_dir, f"{case_name}_Dermoscopic_Image", "*.bmp"))
        # 掩码
        msk_match = glob.glob(os.path.join(case_dir, f"{case_name}_lesion", "*_lesion.bmp"))
        if img_match and msk_match:
            image_paths.append(img_match[0])
            mask_paths.append(msk_match[0])

    print(f"  共 {len(image_paths)} 对 图像+掩码（共 {len(case_dirs)} 个病例）")

    # ---- 2. 划分训练/验证集 ----
    n_total = len(image_paths)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    indices = list(range(n_total))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")

    train_ds = PH2Dataset(train_images, train_masks, IMAGE_SIZE, is_train=True)
    val_ds   = PH2Dataset(val_images,   val_masks,   IMAGE_SIZE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- 3. 模型 ----
    print("\n[2/4] 构建模型...")
    print(f"  加载 DINOv3 (ViT-S/16)...")
    dinov3 = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True).to(DEVICE)
    dinov3.eval()
    for p in dinov3.parameters():
        p.requires_grad = False
    print(f"  DINOv3 已冻结")

    decoder = SimpleDecoder(in_channels=384).to(DEVICE)
    print(f"  解码器参数量: {sum(p.numel() for p in decoder.parameters()):,}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def extract_patch_features(images):
        """从 DINOv3 提取 patch 特征，reshape 为 (B, C, H, W)"""
        with torch.no_grad():
            # forward_features 返回 (B, N_total, C)
            # DINOv3 包含 CLS + register tokens + patch tokens
            # 直接取最后 N_patch 个 token（兼容不同 register 数量）
            img_size = images.shape[-1]         # 224
            n_patches = (img_size // 16) ** 2   # 196
            feats = dinov3.forward_features(images)[:, -n_patches:, :]  # (B, 196, 384)
        B = feats.shape[0]
        H = W = int(feats.shape[1] ** 0.5)    # 14
        return feats.transpose(1, 2).reshape(B, -1, H, W)      # (B, 384, 14, 14)

    # ---- 4. 训练 ----
    print(f"\n[3/4] 训练解码器 ({EPOCHS} epochs)...")
    best_dice = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        decoder.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            feats = extract_patch_features(images)     # (B, 384, 14, 14)
            pred = decoder(feats)                       # (B, 1, 224, 224)

            loss = combined_loss(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        # 验证
        decoder.eval()
        val_iou_sum = val_dice_sum = val_acc_sum = 0.0
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            feats = extract_patch_features(images)
            pred = decoder(feats)
            iou, dice, acc = compute_metrics(pred, masks)
            n = images.size(0)
            val_iou_sum  += iou * n
            val_dice_sum += dice * n
            val_acc_sum  += acc * n

        n_val = len(val_ds)
        val_iou  = val_iou_sum  / n_val
        val_dice = val_dice_sum / n_val
        val_acc  = val_acc_sum  / n_val

        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            marker = " ★"

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}  "
                  f"IoU={val_iou:.4f}  Dice={val_dice:.4f}  Acc={val_acc:.4f}{marker}")

    # 恢复最佳模型
    decoder.load_state_dict(best_state)
    print(f"\n  最佳验证 Dice: {best_dice:.4f}  (IoU: {val_iou:.4f}, Acc: {val_acc:.4f})")

    # ---- 5. 可视化 ----
    print("\n[4/4] 可视化分割结果...")
    decoder.eval()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 解决中文显示问题
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
    matplotlib.rcParams['axes.unicode_minus'] = False

    demo_images, demo_masks = next(iter(val_loader))
    demo_images, demo_masks = demo_images[:4].to(DEVICE), demo_masks[:4]

    with torch.no_grad():
        feats = extract_patch_features(demo_images)
        preds = torch.sigmoid(decoder(feats))
        preds_bin = (preds > 0.5).float()

    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    imgs_show = demo_images * std + mean

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        axes[i, 0].imshow(imgs_show[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[i, 0].set_title("原图" if i == 0 else "")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(demo_masks[i, 0].cpu(), cmap="gray")
        axes[i, 1].set_title("真值" if i == 0 else "")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i, 0].cpu(), cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title("预测热图" if i == 0 else "")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(preds_bin[i, 0].cpu(), cmap="gray")
        axes[i, 3].set_title("预测二值" if i == 0 else "")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig("ph2_seg_results.png", dpi=150, bbox_inches="tight")
    print(f"  结果图保存在 ph2_seg_results.png")

    print(f"\n{'='*60}")
    print(f"✅ 完成！最佳 Dice: {best_dice:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
