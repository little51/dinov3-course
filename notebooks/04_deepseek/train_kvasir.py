# -*- coding: utf-8 -*-
"""
Kvasir-SEG 息肉分割 —— 用 DeepSeek-V4.1-Flash 的纯视觉编码器（ViT-412M）当 backbone。

编码器来自 timm 对 DeepSeek-V4.1-Flash 视觉塔的 remap（无语言模型权重、无预训练分类头），
本脚本冻结/微调该编码器，接一个轻量分割解码器，在 Kvasir-SEG 上做二分类（息肉/背景）分割。

要点（针对这个编码器的特性）：
  * patch=14，所有 block 的特征都是 1/14 分辨率 → 没有尺度金字塔，所以多取几个中间层做"深度融合"
  * 位置编码是 RoPE（无绝对位置嵌入）→ 输入尺寸只要能被 14 整除即可，本脚本默认 392 (=14×28)
  * 6GB 显存下默认冻结编码器只训解码器；--finetune 打开全量微调（配 --grad-checkpoint 省显存）

用法示例：
  # 冒烟测试（16 张训练图，1 个 epoch）
  python train_kvasir.py --epochs 1 --limit-train 16 --limit-val 8

  # 正式训练：冻结编码器，只训解码器
  python train_kvasir.py --epochs 40 --batch-size 8 --img-size 392 --lr 1e-3

  # 全量微调（显存紧张，需要开梯度检查点）
  python train_kvasir.py --finetune --grad-checkpoint --epochs 20 --batch-size 2 --lr 2e-5
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import timm
from safetensors.torch import load_file as load_safetensors

IMAGENET_MEAN = (0.5, 0.5, 0.5)  # 与该编码器原始预处理一致
IMAGENET_STD = (0.5, 0.5, 0.5)
DEFAULT_ARCH = "deepseek_vit_412m.deepseek_v4_1_flash"


# ----------------------------------------------------------------------------- 数据
def find_pairs(root: Path):
    """在若干常见目录结构里找到 (原图, 掩码) 对。返回 [(img_path, mask_path), ...]"""
    root = Path(root)
    layouts = [
        (root / "images", root / "masks"),
        (root / "images", root / "masks_jpg"),
        (root / "Kvasir-SEG" / "images", root / "Kvasir-SEG" / "masks"),
        (root / "train" / "images", root / "train" / "masks"),
    ]
    for img_dir, mask_dir in layouts:
        if img_dir.is_dir() and mask_dir.is_dir():
            pairs = []
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                for cand in (mask_dir / p.name, mask_dir / (p.stem + ".jpg"), mask_dir / (p.stem + ".png")):
                    if cand.exists():
                        pairs.append((p, cand))
                        break
            if pairs:
                return pairs
    # 兜底：递归找 images 同级/相邻的 masks 目录
    pairs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or "mask" in str(p).lower():
            continue
        mask_dir = p.parent.parent / "masks"
        if mask_dir.is_dir():
            for cand in mask_dir.glob(p.stem + ".*"):
                pairs.append((p, cand))
                break
    return pairs


class FeatureCache:
    """
    冻结编码器时的加速通路：先把所有图的中间层特征算一遍存成 fp16 的 .npy（memmap），
    之后训练只在 1/14 分辨率的特征图上进行，编码器不再参与 → 每轮从十几分钟降到几秒。
    """

    def __init__(self, backbone, pairs, img_size, path, batch_size, device, log_every=20):
        self.path = Path(path)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        n = len(pairs)
        levels = len(backbone.out_indices)
        ch = backbone.num_features
        _h, _w = _hw(img_size)
        _p = getattr(backbone, "patch_size", 14)
        s = (_h // _p, _w // _p)
        if self.path.exists():
            mm = np.load(self.path, mmap_mode="r")
            if mm.shape == (n, levels, ch, *s):
                print(f"[cache] 复用 {self.path.name}  {mm.shape}  {self.path.stat().st_size/2**30:.2f} GB")
                return
            print(f"[cache] {self.path.name} 形状不匹配，重建")
        if tmp.exists():
            tmp.unlink()                      # 上次中断留下的半成品，丢掉
        ds = KvasirSegDataset(pairs, img_size, train=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        # 先写临时文件，全部写完再改名 → 中途被打断不会留下"看起来完整"的缓存
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, levels, ch, *s))
        t0, done = time.time(), 0
        backbone.eval()
        for i, (img, _) in enumerate(loader):
            with torch.inference_mode():
                feats = backbone(img.to(device, non_blocking=True))
            arr = torch.stack(feats, dim=1).to(torch.float16).cpu().numpy()   # (B,L,C,S,S)
            mm[done:done + arr.shape[0]] = arr
            done += arr.shape[0]
            if (i + 1) % log_every == 0 or done == n:
                el = time.time() - t0
                print(f"[cache] {done}/{n}  {el:.0f}s  eta {el/max(1,done)*(n-done):.0f}s", flush=True)
        mm.flush()
        del mm
        os.replace(tmp, self.path)
        print(f"[cache] 写好 {self.path}  {self.path.stat().st_size/2**30:.2f} GB")


class FeatDataset(Dataset):
    """从特征缓存里取 (特征, 掩码)。翻转/旋转直接作用在 1/14 特征图上（与掩码同网格）。"""

    def __init__(self, pairs, cache_path, img_size, train=False):
        self.pairs = pairs
        self.cache_path = str(cache_path)
        self.img_size = img_size
        self.train = train
        self._mm = None

    def _mmap(self):
        if self._mm is None:                      # 每个 dataloader worker 各自开只读句柄
            self._mm = np.load(self.cache_path, mmap_mode="r")
        return self._mm

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        f = np.asarray(self._mmap()[i], dtype=np.float32)          # (L,C,S,S)
        _h, _w = _hw(self.img_size)
        mask = Image.open(self.pairs[i][1]).convert("L").resize((_w, _h), Image.NEAREST)
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[None]   # (1,H,W)
        if self.train:
            if random.random() < 0.5:
                f, mask = f[:, :, :, ::-1], mask[:, :, ::-1]
            if random.random() < 0.5:
                f, mask = f[:, :, ::-1], mask[:, ::-1]
            k = random.randint(0, 3)
            if k:
                f, mask = np.rot90(f, k, axes=(2, 3)), np.rot90(mask, k, axes=(1, 2))
            f, mask = np.ascontiguousarray(f), np.ascontiguousarray(mask)
        return torch.from_numpy(f), torch.from_numpy(mask)


class KvasirSegDataset(Dataset):
    def __init__(self, pairs, img_size=392, train=False):
        self.pairs = pairs
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def _load(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        return img, mask

    def __getitem__(self, i):
        img, mask = self._load(i)
        h, w = _hw(self.img_size)
        img = img.resize((w, h), Image.BICUBIC)
        mask = mask.resize((w, h), Image.NEAREST)
        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.train:
            if random.random() < 0.5:                       # 水平翻转
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:                       # 垂直翻转
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3) if h == w else random.choice([0, 2])   # 非正方形：避免 90° 旋转交换 H/W
            if k:
                img, mask = np.rot90(img, k), np.rot90(mask, k)
            if random.random() < 0.3:                       # 亮度/对比度抖动
                a = 1.0 + random.uniform(-0.2, 0.2)
                b = random.uniform(-0.1, 0.1)
                img = np.clip(img * a + b, 0.0, 1.0)
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask


def build_splits(root: Path, val_ratio=0.12, seed=42, split_json=None):
    pairs = find_pairs(root)
    if not pairs:
        raise SystemExit(f"No image/mask pairs found under {root}")
    if split_json and Path(split_json).exists():
        spec = json.loads(Path(split_json).read_text(encoding="utf-8"))
        names_tr, names_va = set(spec["train"]), set(spec["val"])
        tr = [p for p in pairs if p[0].stem in names_tr]
        va = [p for p in pairs if p[0].stem in names_va]
        return tr, va
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(pairs) * val_ratio)))
    va = [pairs[i] for i in idx[:n_val]]
    tr = [pairs[i] for i in idx[n_val:]]
    return tr, va


# ----------------------------------------------------------------------------- 模型
def _hw(sz):
    """统一尺寸成 (h, w)：接受 int / (h,w) / 'HxW' / 'S'。"""
    if isinstance(sz, str):
        parts = sz.lower().replace(" ", "").split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return int(parts[0]), int(parts[0])
    if isinstance(sz, (tuple, list)):
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


def _get_patch_size(arch, default=14):
    """从 timm 模型定义里读 patch size（DeepSeek=14 / DINOv3=16 ...）。"""
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="")
        for path in ("patch_embed.patch_size", "encoder.patch_embed.patch_size",
                     "encoder.encoder.patch_embed.patch_size"):
            obj = m
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return int(obj[0] if isinstance(obj, (tuple, list)) else obj)
    except Exception:
        pass
    return default


def _find_blocks(model):
    """兼容不同 backbone 的层列表位置：DeepSeek 藏在 encoder 内，标准 ViT 在顶层。"""
    for path in ("blocks", "encoder.blocks", "encoder.encoder.blocks"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    raise AttributeError(f"找不到 transformer blocks: {type(model).__name__}")


class DeepSeekViTBackbone(nn.Module):
    """timm 的 DeepSeek ViT 编码器：加载本地 safetensors，取多个中间层 (都是 1/14 分辨率)。"""

    def __init__(self, arch=DEFAULT_ARCH, ckpt=None, out_indices=(7, 15, 23, 31),
                 img_size=392, freeze=True, unfreeze_last=0, grad_checkpoint=False,
                 prune_to=None, verbose=True):
        super().__init__()
        self.unfreeze_last = int(unfreeze_last)
        self.arch = arch
        self.verbose = verbose
        self.out_indices = tuple(out_indices)
        # num_classes=0 / global_pool='' → 只要编码器本体，不带分类头
        self.enc = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="")
        self.enc.set_grad_checkpointing(grad_checkpoint)

        if prune_to is not None:                      # 可选：只保留前 N 个 block（0..N-1），省算力
            self.out_indices = tuple(i for i in self.out_indices if i < prune_to)
            if not self.out_indices:
                self.out_indices = (prune_to - 1,)
            # timm 的约定：传 int 表示"最后 n 层"，传 list 才是具体下标；
            # 这里用 [N-1] 达到"截断到前 N 层、并只输出第 N-1 层特征"的效果
            self.enc.prune_intermediate_layers([prune_to - 1], prune_head=True)

        if ckpt:
            sd = load_safetensors(ckpt) if str(ckpt).endswith(".safetensors") else torch.load(ckpt, map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.enc.load_state_dict(sd, strict=False)
            if verbose:
                print(f"[ckpt] {Path(ckpt).name}: missing={len(missing)} unexpected={len(unexpected)}")
                for k in missing[:5]:
                    print(f"        missing   : {k}")
                for k in unexpected[:5]:
                    print(f"        unexpected: {k}")
            self.ckpt_report = (len(missing), len(unexpected), list(missing)[:8], list(unexpected)[:8])
        else:
            self.ckpt_report = None

        try:
            _ps = self.enc.patch_embed.patch_size
            self.patch_size = int(_ps[0] if isinstance(_ps, (tuple, list)) else _ps)
        except Exception:
            self.patch_size = 14
        self.freeze = freeze
        self._apply_freeze()
        if self.unfreeze_last > 0:                 # ★解冻编码器最后 N 个 block + 尾部两个 norm
            self._unfreeze_tail(self.unfreeze_last)

    def _unfreeze_tail(self, n):
        blks = _find_blocks(self.enc)
        for blk in blks[-n:]:
            for p_ in blk.parameters():
                p_.requires_grad_(True)
        for path in ("encoder.norm", "norm"):
            mod = self.enc
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is not None:
                for p_ in mod.parameters():
                    p_.requires_grad_(True)
        n_un = sum(p_.numel() for p_ in self.enc.parameters() if p_.requires_grad)
        if getattr(self, "verbose", True):
            print(f"[unfreeze] 编码器最后 {n} 层已解冻 (blocks {len(blks)-n}..{len(blks)-1}), "
                  f"可训练 {n_un/1e6:.1f}M / 全部 {sum(p_.numel() for p_ in self.enc.parameters())/1e6:.1f}M")

    def _apply_freeze(self):
        for p in self.enc.parameters():
            p.requires_grad = not self.freeze

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:                    # 冻结时编码器一直 eval（BN/RMSNorm 统计与 dropout 固定）
            self.enc.eval()
        return self

    @property
    def num_features(self):
        return 1024

    def forward(self, x):
        if self.freeze and self.unfreeze_last == 0:
            # 全冻结时用 no_grad 前向：不建图 → 省显存、略快；特征交给解码器继续训
            with torch.no_grad():
                feats = self.enc.forward_intermediates(
                    x, indices=list(self.out_indices), norm=True,
                    output_fmt="NCHW", intermediates_only=True,
                )
            return [f.detach() for f in feats]
        return self.enc.forward_intermediates(
            x, indices=list(self.out_indices), norm=True,
            output_fmt="NCHW", intermediates_only=True,
        )


class ASPP(nn.Module):
    """轻量 ASPP：多空洞率拿上下文（编码器只有 1/14 单尺度，上下文全靠这里补）。"""

    def __init__(self, in_ch, out_ch, rates=(1, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(out_ch), nn.GELU()) for r in rates
        ])
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                 nn.BatchNorm2d(out_ch), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.GELU())
        self.out_ch = out_ch

    def forward(self, x):
        ys = [b(x) for b in self.branches]
        ys.append(F.interpolate(self.gap(x), size=x.shape[-2:], mode="nearest"))
        return self.fuse(torch.cat(ys, dim=1))


class SegDecoder(nn.Module):
    """
    多深度特征融合解码器：
      4 层 1/14 特征 → 各自 1x1 降到 mid → 拼接 → 卷积融合 → ASPP → x4 上采样 → 逐级上采样到原图
    因为编码器特征都是 1/14，这里用"深度"代替通常的"尺度金字塔"。
    """

    def __init__(self, in_chs, mid=256, out_ch=1, img_size=392, progressive=False):
        super().__init__()
        self.progressive = progressive
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU()) for c in in_chs
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * len(in_chs), mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.aspp = ASPP(mid, mid)
        self.up1 = nn.Sequential(nn.Conv2d(mid, mid // 2, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(mid // 2), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(mid // 2, mid // 4, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(mid // 4), nn.GELU())
        self.head = nn.Conv2d(mid // 4, out_ch, 1)
        self.img_size = img_size
        if progressive:
            # ★浅层细节分支: 原图 -> 1/4 分辨率(98x98), 手工补一个"细尺度"
            self.detail = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            )
            self.fuse_mid = nn.Sequential(
                nn.Conv2d(mid // 2 + 64, mid // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid // 2), nn.GELU())
            self.mid_head = nn.Conv2d(mid // 2, out_ch, 1)      # 中间监督输出

    def forward(self, feats, out_size=None, x_img=None):
        if out_size is None:
            out_size = _hw(self.img_size)
        x = torch.cat([p(f) for p, f in zip(self.proj, feats)], dim=1)      # 1/14
        x = self.fuse(x)
        x = self.aspp(x)
        h, w = out_size
        x = F.interpolate(x, size=(max(1, h // 4), max(1, w // 4)), mode="bilinear", align_corners=False)
        x = self.up1(x)
        mid = None
        if self.progressive and x_img is not None:        # ★第一级精化: 注入原图细节 + 产出中间 logits
            d = self.detail(x_img)
            if d.shape[-2:] != x.shape[-2:]:
                d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse_mid(torch.cat([x, d], dim=1))
            mid = self.mid_head(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.up2(x)
        return self.head(x), mid


class SegModel(nn.Module):
    def __init__(self, arch, ckpt, out_indices, img_size, freeze=True, unfreeze_last=0,
                 grad_checkpoint=False, mid=256, prune_to=None, progressive=False):
        super().__init__()
        self.backbone = DeepSeekViTBackbone(arch, ckpt, out_indices, img_size,
                                            freeze=freeze, unfreeze_last=unfreeze_last,
                                            grad_checkpoint=grad_checkpoint, prune_to=prune_to)
        self.decoder = SegDecoder([self.backbone.num_features] * len(self.backbone.out_indices),
                                  mid=mid, img_size=img_size, progressive=progressive)

    def forward(self, x):
        feats = self.backbone(x)
        return self.decoder(feats, out_size=x.shape[-2:], x_img=x)

    def param_groups(self, lr_enc, lr_dec):
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": lr_enc},
            {"params": [p for p in self.decoder.parameters() if p.requires_grad], "lr": lr_dec},
        ]


# ----------------------------------------------------------------------------- 损失/指标
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1e-6
        den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        dice = 1 - (num / den).mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


def save_ckpt(model, path, args, epoch, val, out_indices, frozen_encoder):
    """冻结编码器时只存解码器（编码器权重复用 model/model.safetensors），省 1.6GB。"""
    sd = model.state_dict()
    if frozen_encoder:
        sd = {k: v for k, v in sd.items() if k.startswith("decoder.")}
    torch.save({"model": sd, "frozen_encoder": frozen_encoder, "args": args,
                "epoch": epoch, "val": val, "out_indices": out_indices}, path)


def forward_logits(model, x, mode, x_img=None):
    """返回 (logits, mid_logits)。mode='feat' 时 x 是缓存特征、无原图，细节分支自动跳过。"""
    if mode == "feat":
        return model.decoder(list(x.unbind(1)), x_img=x_img,
                             out_size=_hw(model.decoder.img_size))
    return model(x)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, mode="image"):
    model.eval()
    inter = union = pred_sum = tgt_sum = 0.0
    loss_sum, n = 0.0, 0
    crit = BCEDiceLoss()
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)
        logits, _ = forward_logits(model, x, mode)
        loss_sum += float(crit(logits, mask))
        n += 1
        pred = (torch.sigmoid(logits) > threshold).float()
        inter += float((pred * mask).sum())
        union += float(((pred + mask) > 0).sum())
        pred_sum += float(pred.sum())
        tgt_sum += float(mask.sum())
    dice = (2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return {"dice": dice, "iou": iou, "loss": loss_sum / max(1, n)}


# ----------------------------------------------------------------------------- 训练
class _Tee:
    """把 stdout 同时写入日志文件, 行缓冲 + 每行 flush, 便于外部 tail -f 实时查看。"""

    def __init__(self, path):
        self._f = open(path, 'a', encoding='utf-8', buffering=1)
        self._out = sys.stdout

    def write(self, s):
        self._out.write(s)
        try:
            self._f.write(s)
            self._f.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._out.flush()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/kvasir-seg")
    ap.add_argument("--weights", default="model/model.safetensors")
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--img-size", default="392", help="必须能被 14 整除")
    ap.add_argument("--out-indices", default="7,15,23,31", help="取哪些 block 的中间特征")
    ap.add_argument("--prune-to", type=int, default=None, help="只用前 N 个 block（省算力）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="解码器学习率")
    ap.add_argument("--lr-encoder", type=float, default=2e-5, help="--finetune 时的编码器学习率")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--mid-channels", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--finetune", action="store_true", help="不冻结编码器，全量微调")
    ap.add_argument("--unfreeze-last", type=int, default=0, help="解冻编码器最后 N 个 block（自动关特征缓存）")
    ap.add_argument("--progressive", action="store_true", help="解码器逐步精化：浅层细节分支 + 中间监督")
    ap.add_argument("--mid-loss-weight", type=float, default=0.4, help="中间监督权重")
    ap.add_argument("--patience", type=int, default=0, help="连续 N 轮验证无提升则早停(0=不早停)")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--amp", action="store_true", help="混合精度（Pascal 卡上 fp16 反而慢，默认关）")
    ap.add_argument("--limit-train", type=int, default=None, help="只用 N 张训练图（冒烟测试）")
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--cache-features", dest="cache_features", action="store_true", default=None,
                    help="冻结编码器时先把特征算好缓存（默认自动开启）")
    ap.add_argument("--no-cache-features", dest="cache_features", action="store_false",
                    help="不缓存，每轮都跑编码器（慢，但每轮特征都是最新的）")
    ap.add_argument("--cache-dir", default=None, help="特征缓存目录，默认 outputs/cache")
    args = ap.parse_args()

    _patch = _get_patch_size(args.arch)
    _h, _w = _hw(args.img_size)
    assert _h % _patch == 0 and _w % _patch == 0, (
        f"--img-size 的高({_h})和宽({_w})都必须能被 {_patch} 整除（{args.arch} 的 patch={_patch}）")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _logfile = out_dir / "train.log"
    sys.stdout = _Tee(str(_logfile))
    _argv = " ".join(sys.argv[1:])
    print("=" * 100, flush=True)
    print(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')}  python train_kvasir.py {_argv}", flush=True)
    print(f"===== 日志同时写入: {_logfile}  (tail -f 可实时查看)", flush=True)
    print("=" * 100, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  img_size={args.img_size}  arch={args.arch}")
    print(f"torch={torch.__version__}  timm={timm.__version__}")

    # 数据
    tr_pairs, va_pairs = build_splits(Path(args.data_root), args.val_ratio, args.seed, args.split_json)
    if args.limit_train:
        tr_pairs = tr_pairs[: args.limit_train]
    if args.limit_val:
        va_pairs = va_pairs[: args.limit_val]
    print(f"train={len(tr_pairs)}  val={len(va_pairs)}   (source: {args.data_root})")

    # 模型（要先建好，缓存特征需要它）
    out_indices = tuple(int(x) for x in args.out_indices.split(",") if x != "")
    model = SegModel(args.arch, args.weights, out_indices, args.img_size,
                     freeze=not args.finetune, unfreeze_last=args.unfreeze_last,
                     grad_checkpoint=args.grad_checkpoint,
                     mid=args.mid_channels, prune_to=args.prune_to,
                     progressive=args.progressive).to(device)
    n_enc = sum(p.numel() for p in model.backbone.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"encoder params {n_enc/1e6:.1f}M | decoder params {n_dec/1e6:.2f}M | trainable {n_train/1e6:.2f}M")
    print(f"freeze_encoder={not args.finetune}  out_indices={out_indices}")

    # 数据：冻结编码器时默认走特征缓存（编码器只跑一遍）
    use_cache = args.cache_features if args.cache_features is not None else (
        not args.finetune and args.unfreeze_last == 0 and not args.progressive)
    if use_cache and (args.finetune or args.unfreeze_last or args.progressive):
        print("注意：--finetune/--unfreeze-last/--progressive 时必须关掉特征缓存，已强制关闭")
        use_cache = False
    if use_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"sz{args.img_size}_idx{'-'.join(map(str, out_indices))}_b{len(_find_blocks(model.backbone.enc))}"
        print("[cache] 特征缓存模式：编码器只跑一遍，之后只训解码器")
        FeatureCache(model.backbone, tr_pairs, args.img_size, cache_dir / f"train_{tag}.npy",
                     max(1, args.batch_size), device)
        FeatureCache(model.backbone, va_pairs, args.img_size, cache_dir / f"val_{tag}.npy",
                     max(1, args.batch_size), device)
        ds_tr = FeatDataset(tr_pairs, cache_dir / f"train_{tag}.npy", args.img_size, train=True)
        ds_va = FeatDataset(va_pairs, cache_dir / f"val_{tag}.npy", args.img_size, train=False)
        mode = "feat"
    else:
        ds_tr = KvasirSegDataset(tr_pairs, args.img_size, train=True)
        ds_va = KvasirSegDataset(va_pairs, args.img_size, train=False)
        mode = "image"
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.num_workers,
                       pin_memory=True)

    # 模型（已在上面创建，这里不再重复建）

    crit = BCEDiceLoss().to(device)
    groups = [g for g in model.param_groups(args.lr_encoder, args.lr) if g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    for g in opt.param_groups:                      # 记下基准学习率，供余弦调度用
        g["initial_lr"] = g["lr"]
    iters_per_epoch = max(1, len(dl_tr))
    total_iters = iters_per_epoch * args.epochs
    warmup = min(args.warmup_iters, max(1, total_iters // 10))

    def lr_at(it):
        if it < warmup:
            return (it + 1) / warmup
        prog = (it - warmup) / max(1, total_iters - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * prog))   # 余弦到 1%

    use_amp = bool(args.amp and device == "cuda")
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,iter,train_loss,val_dice,val_iou,val_loss,lr_sec,lr_dec,seconds\n",
                            encoding="utf-8")

    best_dice = -1.0
    no_improve = 0
    history = []
    it_global = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for i, (x, mask) in enumerate(dl_tr):
            lr_scale = lr_at(it_global)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lr_scale
            x, mask = x.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, mid_logits = forward_logits(model, x, mode)
                    loss = crit(logits, mask)
                    if mid_logits is not None:                       # ★中间监督 (98x98 那一级)
                        mid_t = F.interpolate(mask, size=mid_logits.shape[-2:], mode="area")
                        loss = loss + args.mid_loss_weight * crit(mid_logits, mid_t)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                logits, mid_logits = forward_logits(model, x, mode)
                loss = crit(logits, mask)
                if mid_logits is not None:                           # ★中间监督 (98x98 那一级)
                    mid_t = F.interpolate(mask, size=mid_logits.shape[-2:], mode="area")
                    loss = loss + args.mid_loss_weight * crit(mid_logits, mid_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            run_loss += float(loss) * x.size(0)
            seen += x.size(0)
            it_global += 1
            if (i + 1) % 20 == 0 or (i + 1) == iters_per_epoch:
                print(f"  epoch {epoch}/{args.epochs}  iter {i+1}/{iters_per_epoch}  "
                      f"loss {run_loss/max(1,seen):.4f}  {(time.time()-t0):.0f}s", flush=True)

        train_loss = run_loss / max(1, seen)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, dl_va, device, mode=mode)
        else:
            m = {"dice": float("nan"), "iou": float("nan"), "loss": float("nan")}
        secs = time.time() - t0
        lrs = "|".join(f"{g['lr']:.2e}" for g in opt.param_groups)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_dice={m['dice']:.4f}  "
              f"val_iou={m['iou']:.4f}  val_loss={m['loss']:.4f}  lr={lrs}  {secs:.0f}s", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{it_global},{train_loss:.6f},{m['dice']:.6f},{m['iou']:.6f},{m['loss']:.6f},"
                    f"{lrs},{secs:.1f}\n")
        history.append({"epoch": epoch, "train_loss": train_loss, **m})

        if m["dice"] == m["dice"] and m["dice"] > best_dice:      # dice 非 nan 且更优
            best_dice = m["dice"]
            no_improve = 0
            save_ckpt(model, out_dir / "best_model.pth", vars(args), epoch, m, out_indices,
                      frozen_encoder=not args.finetune)
            size_mb = (out_dir / "best_model.pth").stat().st_size / 2**20
            print(f"  -> saved best_model.pth ({size_mb:.1f} MB, val_dice={best_dice:.4f})", flush=True)
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:      # ★早停
                print(f"早停: 连续 {args.patience} 轮无提升, best val_dice={best_dice:.4f}", flush=True)
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_ckpt(model, out_dir / "last_model.pth", vars(args), args.epochs,
              history[-1] if history else {}, out_indices, frozen_encoder=not args.finetune)
    print(f"done. best val_dice={best_dice:.4f}  total {time.time()-t0:.0f}s  out={out_dir}")


if __name__ == "__main__":
    main()
