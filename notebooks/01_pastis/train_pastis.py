#!/usr/bin/env python3
"""
DINOv3 ViT-L SAT + Multi-scale FPN + Reg-TAE
PASTIS 时序语义分割 (全43时相 × 10波段 → 19类)

A: forward_intermediates 提取中间层 [8,16,23]
B: FPN 三尺度金字塔融合 (28×28 / 14×14 / 7×7)
C: Register tokens 调制 L-TAE 时序注意力

用法（在 PASTIS 数据目录下运行）：
    python train_pastis.py

目录结构：
    .
    ├── train_pastis.py           ← 本文件
    ├── data/                     ← PASTIS 原始数据
    └── outputs                   ← 训练输出（自动创建）
"""

import os, json, time, random, builtins
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

print = lambda *a, **kw: builtins.print(*a, **kw, flush=True)


# ─── 配置（按需修改） ──────────────────────────────────────────────────────────
class Config:
    # 数据目录（相对路径）
    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs"

    # 训练参数
    BATCH_SIZE = 2
    EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    SEED = 42

    # 数据参数
    MAX_DATES = 43
    USE_10BAND = True

    # 模型参数
    MODEL_NAME = "vit_large_patch16_dinov3.sat493m"
    FEAT_BLOCKS = [8, 16, 23]  # 中间层索引 (A)
    TAE_GROUPS = 16
    DEC_DIM = 256

    # PASTIS 参数
    NUM_CLASSES = 19
    IGNORE_INDEX = 19

    # PASTIS 类别名称
    CLASS_NAMES = [
        "background", "meadow", "soft_winter_wheat", "corn", "winter_barley",
        "winter_rapeseed", "spring_barley", "sunflower", "grapevine", "beet",
        "winter_triticale", "winter_durum_wheat", "fruits_vegetables_flowers",
        "potatoes", "leguminous_fodder", "soybeans", "orchard", "mixed_cereal",
        "sorghum",
    ]


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)


# ─── 数据集 ────────────────────────────────────────────────────────────────────
from torchgeo.datasets import PASTIS


class MultiDatePASTIS:
    """包装 PASTIS 数据集，固定时序数为 MAX_DATES"""
    def __init__(self, base, max_dates=43, use_10band=True):
        self.base = base
        self.max_dates = max_dates
        self.use_10band = use_10band

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        img = s["image"].float()
        mask = s["mask"].long()
        T = img.shape[0]
        n_dates = min(T, self.max_dates)
        indices = torch.linspace(0, T - 1, n_dates).long().tolist()
        while len(indices) < self.max_dates:
            indices.append(indices[-1])
        bands = [(img[t] / 10000.0).clamp(0, 1) for t in indices]
        return torch.stack(bands, dim=0), mask, len(indices)


def create_dataloaders(cfg):
    """创建训练/验证/测试 DataLoader"""
    print("Loading PASTIS dataset...")
    ds_train = PASTIS(root=cfg.DATA_DIR, folds=(1, 2, 3), bands="s2", mode="semantic", download=False)
    ds_val = PASTIS(root=cfg.DATA_DIR, folds=(4,), bands="s2", mode="semantic", download=False)
    ds_test = PASTIS(root=cfg.DATA_DIR, folds=(5,), bands="s2", mode="semantic", download=False)

    wr_train = MultiDatePASTIS(ds_train, cfg.MAX_DATES, cfg.USE_10BAND)
    wr_val = MultiDatePASTIS(ds_val, cfg.MAX_DATES, cfg.USE_10BAND)
    wr_test = MultiDatePASTIS(ds_test, cfg.MAX_DATES, cfg.USE_10BAND)

    print(f"  Train={len(wr_train)}  Val={len(wr_val)}  Test={len(wr_test)}")

    train_loader = DataLoader(wr_train, cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(wr_val, cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(wr_test, cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─── 模型组件 ──────────────────────────────────────────────────────────────────
import timm


class BandProjection(nn.Module):
    """10波段 → 3波段 (可学习 Conv1x1 + BN)"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(10, 3, 1), nn.BatchNorm2d(3))

    def forward(self, x):
        return self.proj(x)


class MultiLevelEncoder(nn.Module):
    """
    (A) 多层级编码器
    使用 timm 的 forward_intermediates 提取 [8,16,23] 三层的 patch tokens
    和最终层的 register tokens
    """
    def __init__(self, model_name, blocks=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.dim = self.model.embed_dim            # 1024
        self.n_prefix = self.model.num_prefix_tokens  # 5 (1 CLS + 4 reg)
        self.target_size = 224
        self.blocks = blocks or [8, 16, 23]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"  [A] Encoder: {model_name}, dim={self.dim}, blocks={self.blocks}")

    def forward(self, x):
        B = x.shape[0]
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size,) * 2,
                              mode="bilinear", align_corners=False)

        with torch.no_grad():
            _, aux = self.model.forward_intermediates(
                x, indices=self.blocks, return_prefix_tokens=True
            )

        multi_feats = []       # [3 × (B, 1024, 14, 14)]
        reg_tokens_list = []   # [3 × (B, 5, 1024)]
        for patches, prefixes in aux:
            multi_feats.append(patches)
            reg_tokens_list.append(prefixes)

        # 取最后一层的 register tokens (跳过 CLS 位置 0 → 取 1:)
        registers = reg_tokens_list[-1][:, 1:, :]  # (B, 4, 1024)
        return multi_feats, registers


class MultiScaleFPN(nn.Module):
    """
    (B) 多尺度 FPN 融合
    将 3 层 14×14 特征构建为 28×28 / 14×14 / 7×7 金字塔
    自上而下融合后 concat 回 14×14
    """
    def __init__(self, in_dim=1024, dec_dim=256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(in_dim, dec_dim, 1) for _ in range(3)
        ])
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_dim, dec_dim, 3, padding=1),
                nn.BatchNorm2d(dec_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(dec_dim * 3, dec_dim, 1),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True),
        )
        print(f"  [B] FPN: {in_dim}→{dec_dim}, pyramid=28×28/14×14/7×7")

    def forward(self, multi_feats):
        # 第1步：建金字塔
        scales = []
        for i, (f, proj) in enumerate(zip(multi_feats, self.projs)):
            f = proj(f)  # (B, dec_dim, 14, 14)
            if i == 0:
                f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)   # 28×28
            elif i == 2:
                f = F.interpolate(f, scale_factor=0.5, mode="bilinear", align_corners=False)  #  7×7
            scales.append(f)

        # 第2步：自上而下融合
        for i in range(2, -1, -1):
            if i < 2:
                up = F.interpolate(scales[i + 1], size=scales[i].shape[-2:],
                                   mode="bilinear", align_corners=False)
                scales[i] = scales[i] + up
            scales[i] = self.laterals[i](scales[i])

        # 第3步：统一到 14×14 后融合
        fused = torch.cat([
            F.interpolate(scales[0], size=14),
            scales[1],
            F.interpolate(scales[2], size=14),
        ], dim=1)

        return self.fusion(fused)  # (B, dec_dim, 14, 14)


class RegLTAE(nn.Module):
    """
    (C) Register 调制 L-TAE
    Register tokens 通过 MLP 产生 query bias，调整各时相注意力
    """
    def __init__(self, dim=256, n_groups=16):
        super().__init__()
        assert dim % n_groups == 0
        self.dim = dim
        self.n_groups = n_groups
        self.group_dim = dim // n_groups

        self.query = nn.Parameter(torch.randn(1, 1, 1, n_groups, self.group_dim))
        nn.init.normal_(self.query, 0, 0.02)

        self.reg_mod = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, n_groups * self.group_dim)
        )

        self.key_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.key_proj.weight)
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        print(f"  [C] Reg-TAE: dim={dim}, groups={n_groups}")

    def forward(self, x, registers=None):
        """
        x: (B, T, N, D) — patch tokens over timesteps
        registers: (B, 4, D) — register tokens (或 None)
        """
        B, T, N, D = x.shape
        G, GD = self.n_groups, self.group_dim
        k = self.key_proj(x).view(B, T, N, G, GD)

        if registers is not None:
            reg_ctx = registers.mean(dim=1, keepdim=True)
            reg_bias = self.reg_mod(reg_ctx).view(B, 1, 1, G, GD)
            q = self.query + reg_bias
        else:
            q = self.query

        scores = (k * q).sum(dim=-1)
        attn = F.softmax(scores / (GD ** 0.5) * torch.exp(self.log_temp), dim=1)
        result = (attn.unsqueeze(-1) * x.view(B, T, N, G, GD)).sum(dim=1)
        return result.view(B, N, D)


class UpDecoder(nn.Module):
    """简单上采样解码器 14×14 → 128×128"""
    def __init__(self, in_dim=256, n_classes=19, dropout=0.1):
        super().__init__()
        self.ups = nn.ModuleList()
        cur = 14
        while cur < 128:
            nxt = min(cur * 2, 128)
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
            ))
            cur = nxt
        self.head = nn.Conv2d(in_dim, n_classes, 1)
        print(f"  Decoder: {in_dim}, upsample 14→128")

    def forward(self, x):
        for up in self.ups:
            x = up(x)
        if x.shape[-1] != 128:
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        return self.head(x)


class TemporalSegmenter(nn.Module):
    """完整时序分割模型"""
    def __init__(self, cfg):
        super().__init__()
        self.band_proj = BandProjection()
        self.encoder = MultiLevelEncoder(cfg.MODEL_NAME, cfg.FEAT_BLOCKS)
        self.fpn = MultiScaleFPN(in_dim=1024, dec_dim=cfg.DEC_DIM)
        self.reg_tae = RegLTAE(dim=cfg.DEC_DIM, n_groups=cfg.TAE_GROUPS)
        self.decoder = UpDecoder(in_dim=cfg.DEC_DIM, n_classes=cfg.NUM_CLASSES)
        self.reg2tae = nn.Linear(1024, cfg.DEC_DIM)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x3 = self.band_proj(x.view(B * T, C, H, W))       # (B*T, 3, 128, 128)
        multi_feats, registers = self.encoder(x3)           # (A)
        fused = self.fpn(multi_feats)                       # (B) → (B*T, DEC_DIM, 14, 14)

        N = 14 * 14
        fused_seq = fused.view(B * T, cfg.DEC_DIM, -1).transpose(1, 2)
        fused_seq = fused_seq.view(B, T, N, cfg.DEC_DIM)

        registers_mean = registers.view(B, T, 4, -1).mean(dim=1)  # (B, 4, 1024)
        registers_dd = self.reg2tae(registers_mean)                # (B, 4, DEC_DIM)
        agg = self.reg_tae(fused_seq, registers=registers_dd)      # (C) → (B, 196, DEC_DIM)

        feat_map = agg.transpose(1, 2).view(B, cfg.DEC_DIM, 14, 14)
        return self.decoder(feat_map)


# ─── 训练函数 ──────────────────────────────────────────────────────────────────
def compute_global_iou(pred_labels, masks, n_classes=19, ignore_idx=19):
    """全局 IoU (非逐类平均)，与原论文一致"""
    intersect, union = 0, 0
    valid = masks != ignore_idx
    for c in range(n_classes):
        p = (pred_labels == c) & valid
        g = (masks == c) & valid
        intersect += (p & g).sum().item()
        union += (p | g).sum().item()
    return intersect / max(union, 1)


def train_one_epoch(model, loader, opt, crit, device, cfg):
    model.train()
    total_loss = 0.0
    for batch_idx, (rgbs, masks, _) in enumerate(loader):
        rgbs, masks = rgbs.to(device), masks.to(device)
        preds = model(rgbs)
        loss = crit(preds, masks)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, crit, device, cfg):
    model.eval()
    total_loss = 0.0
    intersect, union = 0, 0
    for rgbs, masks, _ in loader:
        rgbs, masks = rgbs.to(device), masks.to(device)
        preds = model(rgbs)
        total_loss += crit(preds, masks).item()
        pred_labels = preds.argmax(dim=1)
        valid = masks != cfg.IGNORE_INDEX
        for c in range(cfg.NUM_CLASSES):
            p = (pred_labels == c) & valid
            g = (masks == c) & valid
            intersect += (p & g).sum().item()
            union += (p | g).sum().item()
    val_loss = total_loss / max(len(loader), 1)
    miou = intersect / max(union, 1)
    return val_loss, miou


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model: {cfg.MODEL_NAME} + A(intermediates) + B(FPN) + C(Reg-TAE)")

    # 数据
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # 模型
    model = TemporalSegmenter(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")

    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    sched = CosineAnnealingLR(opt, T_max=cfg.EPOCHS)
    crit = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)

    # 训练
    best_miou, best_ep, no_impr = 0.0, 0, 0

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, opt, crit, device, cfg)
        vl_loss, vl_miou = validate(model, val_loader, crit, device, cfg)

        sched.step()
        elapsed = time.time() - t0

        print(
            f"[{epoch+1:3d}/{cfg.EPOCHS}] "
            f"loss={tr_loss:.4f}/{vl_loss:.4f} | "
            f"mIoU={vl_miou:.4f} (best={best_miou:.4f}@{best_ep}) | "
            f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.0f}s"
        )

        if vl_miou > best_miou + 1e-4:
            best_miou = vl_miou
            best_ep = epoch + 1
            no_impr = 0
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))
            print(f"  → New best! mIoU={best_miou:.4f}")
        else:
            no_impr += 1
            if no_impr >= 30:
                print(f"Early stop at epoch {epoch + 1}")
                break

    # 测试
    print("\n=== Testing ===")
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth")))
    _, test_miou = validate(model, test_loader, crit, device, cfg)
    print(f"Test mIoU: {test_miou:.4f}")

    json.dump(
        {
            "mIoU": test_miou,
            "best_val_miou": best_miou,
            "best_epoch": best_ep,
            "config": {
                "model": f"{cfg.MODEL_NAME} + A+B+C",
                "batch_size": cfg.BATCH_SIZE,
                "max_dates": cfg.MAX_DATES,
                "tae_groups": cfg.TAE_GROUPS,
                "bands": "10-band S2",
                "dataset": "PASTIS (folds 1-3 train, 4 val, 5 test)",
            },
        },
        open(os.path.join(cfg.OUTPUT_DIR, "results.json"), "w"),
        indent=2,
    )
    print(f"\nDone → {cfg.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
