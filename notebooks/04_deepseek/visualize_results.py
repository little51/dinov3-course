#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kvasir-SEG 推理可视化：装载 outputs/best_model.pth，随机挑几张验证集图片，
画出 原图 | 真值 | 预测概率 | 预测叠加，并打印单张 Dice。
结果保存到 outputs/visualizations/。
"""
import argparse
import builtins
import random
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _setup_cjk_font():
    """本机没有 SimHei；按可用中文字体回退，避免中文标题变方块。"""
    from matplotlib import font_manager
    cands = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
             "/usr/share/fonts/truetype/arphic/uming.ttc",
             "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
    for c in cands:
        if Path(c).exists():
            try:
                font_manager.fontManager.addfont(c)
                name = font_manager.FontProperties(fname=c).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["font.family"] = "sans-serif"
                return name
            except Exception:
                continue
    return None


_CJK_FONT = _setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

from train_kvasir import SegModel, KvasirSegDataset, build_splits, IMAGENET_MEAN, IMAGENET_STD

print = lambda *a, **kw: builtins.print(*a, **kw, flush=True)


def denorm(t):
    """(3,H,W) 归一化张量 → 可显示的 numpy 图"""
    x = t.cpu().numpy().transpose(1, 2, 0)
    x = (x * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32))
    return np.clip(x, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/best_model.pth")
    ap.add_argument("--data-root", default="data/kvasir-seg")
    ap.add_argument("--out-dir", default="outputs/visualizations")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    targs = ck.get("args", {})
    img_size = targs.get("img_size", 392)
    out_indices = tuple(ck.get("out_indices", (7, 15, 23, 31)))
    arch = targs.get("arch", "deepseek_vit_412m.deepseek_v4_1_flash")
    print(f"ckpt={args.ckpt}  epoch={ck.get('epoch')}  val={ck.get('val')}")
    print(f"arch={arch}  img_size={img_size}  out_indices={out_indices}")

    model = SegModel(arch, targs.get("weights", "model/model.safetensors"), out_indices, img_size,
                     freeze=True, unfreeze_last=targs.get("unfreeze_last", 0),
                     mid=targs.get("mid_channels", 256),
                     progressive=targs.get("progressive", False)).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"加载权重: missing={len(missing)} unexpected={len(unexpected)} "
          f"(frozen_encoder={ck.get('frozen_encoder')})")
    model.eval()

    _, va_pairs = build_splits(Path(args.data_root), targs.get("val_ratio", 0.12), targs.get("seed", 42),
                               targs.get("split_json"))
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(va_pairs)), min(args.n, len(va_pairs)))
    ds = KvasirSegDataset(va_pairs, img_size, train=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(picks), 4, figsize=(13, 3.2 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    dice_list = []
    for r, k in enumerate(picks):
        img, mask = ds[k]
        with torch.inference_mode():
            out = model(img.unsqueeze(0).to(device))
            out = out[0] if isinstance(out, tuple) else out    # progressive 版返回 (logits, mid)
            prob = torch.sigmoid(out)[0, 0].float().cpu().numpy()
        pred = (prob > args.threshold).astype(np.float32)
        gt = mask[0].numpy()
        dice = (2 * (pred * gt).sum() + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        dice_list.append(float(dice))
        name = va_pairs[k][0].name
        print(f"  {name}: dice={dice:.4f}")

        rgb = denorm(img)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"原图  {name}", fontsize=9)
        axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title("真值", fontsize=9)
        axes[r, 2].imshow(prob, cmap="jet", vmin=0, vmax=1)
        axes[r, 2].set_title("预测概率", fontsize=9)
        overlay = rgb.copy()
        overlay[..., 0] = np.where(pred > 0.5, 1.0, overlay[..., 0])   # 红色叠加预测
        axes[r, 3].imshow(overlay)
        axes[r, 3].set_title(f"预测叠加  Dice={dice:.3f}", fontsize=9)
        for c in range(4):
            axes[r, c].axis("off")
    fig.tight_layout()
    save = out_dir / "kvasir_predictions.png"
    fig.savefig(save, dpi=130, bbox_inches="tight")
    print(f"平均 Dice={np.mean(dice_list):.4f}   图已保存: {save}")


if __name__ == "__main__":
    main()
