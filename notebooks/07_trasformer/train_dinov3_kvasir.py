"""Kvasir-SEG：Transformers加载冻结DINOv3，使用多层融合分割头。

分割头按 notebooks/04_deepseek/train_kvasir.py 的思路实现：
4层ViT特征 -> 投影和融合 -> ASPP -> 逐级上采样；
并在1/4尺度引入原图细节分支和中间监督。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel

MODEL_ID = "kryveil/dinov3-vitl16-pretrain-lvd1689m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class KvasirDataset(Dataset):
    def __init__(self, root, size=448, train=False):
        root = Path(root)
        self.images = sorted((root / "images").glob("*"))
        self.masks = {p.stem: p for p in (root / "masks").glob("*")}
        self.size, self.train = size, train
        self.image_tf = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        mask = Image.open(self.masks[path.stem]).convert("L")
        if self.train:
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < 0.3:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        image = self.image_tf(image)
        mask = (self.mask_tf(mask) > 0.5).float()
        return image, mask


class ASPP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate, bias=False),
                          nn.BatchNorm2d(channels), nn.GELU())
            for rate in (1, 3, 6)
        ])
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         nn.Conv2d(channels, channels, 1, bias=False),
                                         nn.BatchNorm2d(channels), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(channels * 4, channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.GELU())

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        features.append(F.interpolate(self.global_pool(x), size=x.shape[-2:], mode="nearest"))
        return self.fuse(torch.cat(features, dim=1))


class DINOv3MultiLayerSegmenter(nn.Module):
    def __init__(self, mid_channels=256, unfreeze_last=4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_ID)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.unfreeze_last = unfreeze_last
        if unfreeze_last:
            for layer in self.backbone.model.layer[-unfreeze_last:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True

        dim = self.backbone.config.hidden_size
        self.prefix_tokens = 1 + self.backbone.config.num_register_tokens
        self.layer_indices = (6, 12, 18, 24)  # ViT-L第6、12、18、24层
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, mid_channels, 1, bias=False),
                          nn.BatchNorm2d(mid_channels), nn.GELU())
            for _ in self.layer_indices
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 4, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.GELU(),
        )
        self.aspp = ASPP(mid_channels)
        self.up1 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(mid_channels // 2), nn.GELU())
        self.detail = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.fuse_mid = nn.Sequential(
            nn.Conv2d(mid_channels // 2 + 64, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2), nn.GELU(),
        )
        self.mid_head = nn.Conv2d(mid_channels // 2, 1, 1)
        self.up2 = nn.Sequential(nn.Conv2d(mid_channels // 2, mid_channels // 4, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(mid_channels // 4), nn.GELU())
        self.head = nn.Conv2d(mid_channels // 4, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        if not self.unfreeze_last:
            self.backbone.eval()
        return self

    def _to_map(self, tokens):
        tokens = tokens[:, self.prefix_tokens:]
        side = int(tokens.shape[1] ** 0.5)
        return tokens.transpose(1, 2).reshape(-1, tokens.shape[2], side, side)

    def forward(self, images):
        if self.unfreeze_last:
            states = self.backbone(pixel_values=images, output_hidden_states=True).hidden_states
        else:
            with torch.no_grad():
                states = self.backbone(pixel_values=images, output_hidden_states=True).hidden_states
        features = [self._to_map(states[i]) for i in self.layer_indices]
        x = torch.cat([proj(feature) for proj, feature in zip(self.proj, features)], dim=1)
        x = self.aspp(self.fuse(x))
        h, w = images.shape[-2:]
        x = F.interpolate(x, size=(h // 4, w // 4), mode="bilinear", align_corners=False)
        x = self.up1(x)
        detail = self.detail(images)
        x = self.fuse_mid(torch.cat([x, detail], dim=1))
        mid_logits = self.mid_head(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(self.up2(x)), mid_logits


def bce_dice_loss(logits, masks):
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probability = logits.sigmoid()
    numerator = 2 * (probability * masks).sum((1, 2, 3)) + 1e-6
    denominator = probability.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1e-6
    return 0.5 * bce + 0.5 * (1 - (numerator / denominator).mean())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0.0
    for images, masks in loader:
        logits, _ = model(images.to(DEVICE))
        prediction = logits.sigmoid() > 0.5
        masks = masks.to(DEVICE)
        intersection = (prediction * masks).sum((1, 2, 3))
        union = prediction.sum((1, 2, 3)) + masks.sum((1, 2, 3))
        total += ((2 * intersection + 1) / (union + 1)).sum().item()
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Kvasir-SEG")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--unfreeze-last", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(42)
    full_dataset = KvasirDataset(args.data)
    train_set, val_set = random_split(full_dataset, [800, 200],
                                      generator=torch.Generator().manual_seed(42))
    train_set.dataset.train = True
    # 给验证集独立的无增强dataset，但保持相同的随机划分索引
    val_dataset = KvasirDataset(args.data, train=False)
    val_set = torch.utils.data.Subset(val_dataset, val_set.indices)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=4, pin_memory=True)

    model = DINOv3MultiLayerSegmenter(unfreeze_last=args.unfreeze_last).to(DEVICE)
    encoder_params = [p for p in model.backbone.parameters() if p.requires_grad]
    decoder_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    print(f"ViT-L | layers={model.layer_indices} | unfreeze last={args.unfreeze_last} | "
          f"encoder trainable={sum(p.numel() for p in encoder_params) / 1e6:.2f}M | "
          f"decoder trainable={sum(p.numel() for p in decoder_params) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)
    best_dice, stale_epochs = 0.0, 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits, mid_logits = model(images)
            mid_masks = F.interpolate(masks, size=mid_logits.shape[-2:], mode="area")
            loss = bce_dice_loss(logits, masks) + 0.4 * bce_dice_loss(mid_logits, mid_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)

        val_dice = evaluate(model, val_loader)
        print(f"epoch {epoch + 1}: loss = {loss_sum / len(train_set):.4f}, val dice = {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice, stale_epochs = val_dice, 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stop: best val dice = {best_dice:.4f}")
                break


if __name__ == "__main__":
    main()
