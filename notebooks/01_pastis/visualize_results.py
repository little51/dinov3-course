#!/usr/bin/env python3
"""
PASTIS 推理可视化
装载最优权重 (outputs/best_model.pth)
随机选 3 张测试集图片，可视化：原图RGB | 真值 | 推理
"""
import os, random, builtins
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# 设置中文字体
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

print = lambda *a, **kw: builtins.print(*a, **kw, flush=True)

# ─── 配置 ──────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CKPT = os.path.join(OUTPUT_DIR, "best_model.pth")
SAVE_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

NUM_CLASSES = 19
IGNORE_INDEX = 19

CLASS_NAMES = [
    "背景", "草甸", "软冬小麦", "玉米", "冬大麦",
    "冬油菜", "春大麦", "向日葵", "葡萄藤", "甜菜",
    "冬小黑麦", "硬粒冬小麦", "果蔬花卉", "土豆",
    "豆科饲料", "大豆", "果园", "混合谷物", "高粱",
]

# ─── PASTIS 19 类颜色映射（可区分、视觉友好）─────────────────────────────────
PASTIS_COLORS = [
    (0, 0, 0),           # 0  背景
    (255, 255, 100),     # 1  草甸
    (170, 255, 0),       # 2  软冬小麦
    (255, 200, 0),       # 3  玉米
    (200, 150, 0),       # 4  冬大麦
    (255, 230, 170),     # 5  冬油菜
    (210, 245, 120),     # 6  春大麦
    (255, 255, 0),       # 7  向日葵
    (128, 0, 128),       # 8  葡萄藤
    (255, 100, 0),       # 9  甜菜
    (255, 180, 100),     # 10 冬小黑麦
    (180, 220, 0),       # 11 硬粒冬小麦
    (255, 100, 100),     # 12 果蔬花卉
    (200, 80, 0),        # 13 土豆
    (0, 200, 0),         # 14 豆科饲料
    (0, 150, 0),         # 15 大豆
    (100, 50, 0),        # 16 果园
    (255, 200, 200),     # 17 混合谷物
    (200, 100, 50),      # 18 高粱
]
# 归一化到 [0,1]
cmap_rgb = np.array(PASTIS_COLORS) / 255.0
cmap = ListedColormap(cmap_rgb)

# 用于 overlay 的 colormap（半透明）
OVERLAY_COLORS = cmap_rgb.copy()
OVERLAY_COLORS[0] = [0, 0, 0]  # 背景保持黑色

# ─── 模型定义（与训练完全一致）─────────────────────────────────────────────────
import timm
from torchgeo.datasets import PASTIS

class BandProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(10, 3, 1), nn.BatchNorm2d(3))
    def forward(self, x):
        return self.proj(x)

class MultiLevelEncoder(nn.Module):
    def __init__(self, model_name="vit_large_patch16_dinov3.sat493m", blocks=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.dim = self.model.embed_dim
        self.n_prefix = self.model.num_prefix_tokens
        self.target_size = 224
        self.blocks = blocks or [8, 16, 23]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size,) * 2,
                              mode="bilinear", align_corners=False)
        with torch.no_grad():
            _, aux = self.model.forward_intermediates(
                x, indices=self.blocks, return_prefix_tokens=True
            )
        multi_feats = []
        reg_tokens_list = []
        for patches, prefixes in aux:
            multi_feats.append(patches)
            reg_tokens_list.append(prefixes)
        registers = reg_tokens_list[-1][:, 1:, :]  # (B, 4, 1024)
        return multi_feats, registers

class MultiScaleFPN(nn.Module):
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

    def forward(self, multi_feats):
        scales = []
        for i, (f, proj) in enumerate(zip(multi_feats, self.projs)):
            f = proj(f)
            if i == 0:
                f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
            elif i == 2:
                f = F.interpolate(f, scale_factor=0.5, mode="bilinear", align_corners=False)
            scales.append(f)
        for i in range(2, -1, -1):
            if i < 2:
                up = F.interpolate(scales[i + 1], size=scales[i].shape[-2:],
                                   mode="bilinear", align_corners=False)
                scales[i] = scales[i] + up
            scales[i] = self.laterals[i](scales[i])
        fused = torch.cat([
            F.interpolate(scales[0], size=14),
            scales[1],
            F.interpolate(scales[2], size=14),
        ], dim=1)
        return self.fusion(fused)

class RegLTAE(nn.Module):
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

    def forward(self, x, registers=None):
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
    def __init__(self, in_dim=256, n_classes=19):
        super().__init__()
        self.ups = nn.ModuleList()
        cur = 14
        while cur < 128:
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_dim, in_dim, 3, padding=1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
            ))
            cur = min(cur * 2, 128)
        self.head = nn.Conv2d(in_dim, n_classes, 1)

    def forward(self, x):
        for up in self.ups:
            x = up(x)
        if x.shape[-1] != 128:
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        return self.head(x)

class TemporalSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_proj = BandProjection()
        self.encoder = MultiLevelEncoder()
        self.fpn = MultiScaleFPN()
        self.reg_tae = RegLTAE()
        self.decoder = UpDecoder()
        self.reg2tae = nn.Linear(1024, 256)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x3 = self.band_proj(x.view(B * T, C, H, W))
        multi_feats, registers = self.encoder(x3)
        fused = self.fpn(multi_feats)
        N = 14 * 14
        fused_seq = fused.view(B * T, 256, -1).transpose(1, 2)
        fused_seq = fused_seq.view(B, T, N, 256)
        registers_mean = registers.view(B, T, 4, -1).mean(dim=1)
        registers_dd = self.reg2tae(registers_mean)
        agg = self.reg_tae(fused_seq, registers=registers_dd)
        feat_map = agg.transpose(1, 2).view(B, 256, 14, 14)
        return self.decoder(feat_map)


# ─── 数据集封装 ──────────────────────────────────────────────────────────────
class MultiDatePASTIS:
    def __init__(self, base, max_dates=43):
        self.base = base
        self.max_dates = max_dates

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
        return torch.stack(bands, dim=0), mask, len(indices), img  # 返回原始 img 用于可视化


# ─── 单张可视化 ──────────────────────────────────────────────────────────────
def make_rgb_composite(raw_img):
    """从原始 10 波段图像生成 True Color RGB（B4=Red, B3=Green, B2=Blue）"""
    # raw_img: (10, 128, 128), uint16 scale
    rgb = raw_img[[2, 1, 0]]  # R=band4(idx2), G=band3(idx1), B=band2(idx0)
    # 自动拉伸到 [0,1]（百分比拉伸）
    for c in range(3):
        lo, hi = torch.quantile(rgb[c], 0.02), torch.quantile(rgb[c], 0.98)
        rgb[c] = rgb[c].float().clamp(lo, hi)
        if hi > lo:
            rgb[c] = (rgb[c] - lo) / (hi - lo)
    return rgb.numpy().transpose(1, 2, 0)  # (H, W, 3)


# ─── 主程序 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model = TemporalSegmenter().to(device)
    model.eval()
    state = torch.load(CKPT, map_location=device)
    # 去掉 'module.' 前缀（如果有 DDP 保存的权重）
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {CKPT}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}, Trainable: {trainable:,}")

    # 数据
    ds_test = PASTIS(root=DATA_DIR, folds=(5,), bands="s2", mode="semantic", download=False)
    wr_test = MultiDatePASTIS(ds_test, max_dates=43)
    print(f"Test samples: {len(wr_test)}")

    # 选 3 张有内容的测试图（排除全背景）
    candidates = []
    for i in range(len(wr_test)):
        _, mask, _, _ = wr_test[i]
        valid_pixels = (mask != IGNORE_INDEX).sum().item()
        if valid_pixels > 1000:  # 至少有内容的像素
            candidates.append(i)
        if len(candidates) >= 50:
            break
    indices = random.sample(candidates, 3)
    print(f"Selected test indices: {indices}")

    # 推理
    for plot_idx, sample_idx in enumerate(indices):
        print(f"\nProcessing sample {sample_idx} ({plot_idx+1}/3)...")
        x, gt_mask, n_dates, raw_img = wr_test[sample_idx]
        x = x.unsqueeze(0).to(device)  # (1, T, 10, 128, 128)
        gt_mask = gt_mask.cpu()

        # 推理
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().squeeze(0)  # (128, 128)

        # RGB 复合图（取中间时相）
        t_mid = raw_img.shape[0] // 2
        rgb = make_rgb_composite(raw_img[t_mid])

        # ── 画图 ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        axes[0].imshow(rgb)
        axes[0].set_title("RGB 复合图 (B4/B3/B2)", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        axes[1].set_title("真值 (Ground Truth)", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        axes[2].set_title(f"推理结果 (Prediction)", fontsize=13)
        axes[2].axis("off")

        plt.suptitle(f"PASTIS 时序分割结果 (DINOv3 SAT + ABC 方案) — 测试样本 #{sample_idx}",
                     fontsize=14, y=1.02)
        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"result_sample_{sample_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # 额外：图例图
    fig_legend, ax_legend = plt.subplots(1, 1, figsize=(14, 1.8))
    ax_legend.axis("off")
    # 绘制色块图例
    n_col = 7  # 每行列数
    for i in range(NUM_CLASSES):
        row = i // n_col
        col = i % n_col
        x_pos = col * 2.0
        y_pos = -row * 0.6
        ax_legend.fill_between([x_pos, x_pos + 0.8], y_pos - 0.2, y_pos + 0.2,
                                color=cmap_rgb[i], edgecolor="gray", linewidth=0.5)
        ax_legend.text(x_pos + 1.0, y_pos, f"{i}: {CLASS_NAMES[i]}", fontsize=8,
                       verticalalignment="center")
    ax_legend.set_xlim(0, n_col * 2.0)
    ax_legend.set_ylim(-2.0, 0.5)
    ax_legend.set_title("PASTIS 19 类图例", fontsize=12, pad=10)
    legend_path = os.path.join(SAVE_DIR, "legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nLegend saved: {legend_path}")

    print("\n所有可视化完成！")
    print(f"输出目录: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
