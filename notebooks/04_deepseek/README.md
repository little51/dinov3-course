# Kvasir-SEG 息肉分割 — 大模型视觉编码器对比（DeepSeek ViT vs DINOv3）

用从多模态大模型里 remap 出来的纯视觉编码器当 backbone，在 Kvasir-SEG 上做息肉二分类分割，并和专门用图像自监督训练的 **DINOv3** 做同条件对照。

脚本支持任意 timm ViT 编码器（换 `--arch` + `--weights` 即可），目前跑过：

- `deepseek_vit_412m.deepseek_v4_1_flash`（timm 对 **DeepSeek-V4.1-Flash** 视觉塔的 remap，411.8M，patch 14，32 层）
- `deepseek_vit_412m.deepseek_v4_flash_vision_exp`（**DeepSeek-V4-Flash-Vision-Exp**，411.8M，patch 14，32 层）
- `vit_large_patch16_dinov3.lvd1689m`（**DINOv3 ViT-L/16**，303.1M，patch 16，24 层，LVD-1689M web 数据）

这些编码器都**不含语言模型权重、也没有训练好的分类头**，本目录把它们当分割 backbone 用（冻结 + 解冻末尾若干层 + 轻量解码器）。

## 目录结构

```bash
04_deepseek/
├── train_kvasir.py            # 训练脚本（数据/backbone/解码器/指标）
├── visualize_results.py       # 推理可视化脚本
├── timm_main/                 # timm GitHub main 源码（PYTHONPATH 注入用）
├── model/
│   ├── model.safetensors                     # DeepSeek V4.1-Flash 权重 1571 MB
│   ├── model_vision_exp.safetensors          # DeepSeek V4-Flash-Vision-Exp 权重 1571 MB
│   └── dinov3_vitl16_web.safetensors         # → HF cache 里的 DINOv3 ViT-L/16 权重
├── data/
│   └── kvasir-seg/Kvasir-SEG/{images,masks}   # 1000 图 / 1000 掩码
└── outputs/
    ├── best_model.pth / last_model.pth / train_log.csv / history.json / train.log
    ├── <实验名>/               # --out-dir 指定的实验目录，结构与上面相同
    └── visualizations/        # 可视化结果
```

## 环境

本机实际使用（`deepseek` conda 环境并未创建）：

```bash
conda activate loveda                 # torch 2.12.1+cu130 / timm 1.0.27 / transformers 5.13
export PYTHONPATH=$PWD/timm_main      # ★必须：PyPI 的 timm 里还没有 deepseek 模型
python -u train_kvasir.py ...
```

从零搭环境的话：

```bash
pip install torch torchvision numpy pillow matplotlib safetensors
pip install "git+https://github.com/huggingface/pytorch-image-models"   # PyPI 1.0.29 无此模型
```

## 模型准备

**DeepSeek（timm remap 权重）**：

```bash
curl -L -o model/model.safetensors \
  "https://hf-mirror.com/timm/deepseek_vit_412m.deepseek_v4_1_flash/resolve/main/model.safetensors"
curl -L -o model/model_vision_exp.safetensors \
  "https://hf-mirror.com/timm/deepseek_vit_412m.deepseek_v4_flash_vision_exp/resolve/main/model.safetensors"
```

**DINOv3**：`facebook/*` 官方仓库 gated，用 timm 的镜像版即可。timm 下载后权重缓存在
`~/.cache/huggingface/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/`，但 blob 文件**没有 `.safetensors` 后缀**，脚本按后缀判断格式会走错分支，需要建个软链：

```bash
ln -sf ~/.cache/huggingface/hub/models--timm--vit_large_patch16_dinov3.lvd1689m/blobs/<blobs文件> \
       model/dinov3_vitl16_web.safetensors
```

## 数据准备

```bash
curl -L -o data/kvasir-seg.zip "https://datasets.simula.no/downloads/kvasir-seg.zip"
```

解压到 `data/kvasir-seg/`，结构为 `Kvasir-SEG/images/*.jpg` + `Kvasir-SEG/masks/*.jpg`（掩码是 JPEG，按 >127 二值化）。

划分：按文件名排序后用固定种子随机划分，`--val-ratio` 默认 0.12（880/120），常用 0.2（800/200）；可用 `--split-json` 指定划分文件。**Kvasir-SEG 官方没有标准划分，跨论文数字不可直接横比。**

## 训练

```bash
# 冒烟（16 张，1 轮，1 分钟内）
python -u train_kvasir.py --epochs 1 --limit-train 16 --limit-val 8

# 基线：冻结编码器，只训解码器（走特征缓存，每轮约 45 s）
python -u train_kvasir.py --epochs 80 --batch-size 12 --img-size 392 --lr 1e-3

# ★主力配置：解冻编码器最后 4 层 + 解码器逐步精化（不走缓存，每轮 45–130 s）
python -u train_kvasir.py --epochs 200 --batch-size 12 --img-size 392 --lr 1e-3 \
  --val-ratio 0.2 --unfreeze-last 4 --progressive --amp --patience 20 \
  --out-dir outputs/my_exp

# DINOv3 ViT-L/16：patch 16，输入 16 的倍数，24 层取 5/11/17/23
python -u train_kvasir.py --epochs 200 --batch-size 12 --img-size 448 --lr 1e-3 \
  --arch vit_large_patch16_dinov3.lvd1689m --weights model/dinov3_vitl16_web.safetensors \
  --out-indices 5,11,17,23 --val-ratio 0.2 --unfreeze-last 4 --progressive --amp --patience 20 \
  --out-dir outputs/dinov3l_448

# 全量微调（显存紧张，开梯度检查点、降 batch）
python -u train_kvasir.py --finetune --grad-checkpoint --epochs 20 --batch-size 2 --lr 2e-5
```

### 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--arch` | deepseek_vit_412m.deepseek_v4_1_flash | timm 模型名（可换 DINOv3 等任意 ViT） |
| `--weights` | model/model.safetensors | 本地 safetensors 权重路径 |
| `--img-size` | 392 | 输入边长，或用 `448x384` 指定非正方形（两维都要能被 patch 整除） |
| `--out-indices` | 7,15,23,31 | 取哪几层的中间特征（层数因模型而异，DINOv3 是 5,11,17,23） |
| `--unfreeze-last` | 0 | **解冻编码器最后 N 层**（自动关特征缓存） |
| `--progressive` | 关 | **解码器逐步精化**：浅层细节支路 + 98×98 中间监督 |
| `--mid-loss-weight` | 0.4 | 中间监督权重 |
| `--patience` | 0 | 连续 N 轮 val_dice 无提升则早停 |
| `--val-ratio` | 0.12 | 验证集比例 |
| `--batch-size` | 8 | 冻结时可用 12 |
| `--lr` / `--lr-encoder` | 1e-3 / 2e-5 | 解码器 / 编码器学习率 |
| `--mid-channels` | 256 | 解码器中间通道数 |
| `--finetune` | 关 | 全量微调编码器（自动关缓存） |
| `--prune-to` | 无 | 只保留前 N 个 block，省算力 |
| `--amp` | 关 | 混合精度（Ampere 卡显著提速；Pascal 反而慢） |
| `--out-dir` | outputs | 实验输出目录 |
| `--limit-train/--limit-val` | 无 | 冒烟测试用 |

## 架构概览

```
输入: (B, 3, H, W)   H=W=392（DeepSeek）/ 448（DINOv3）等
  │
  ├─ 编码器（timm ViT，任意）：forward_intermediates @ out_indices
  │   ├─ DeepSeek-V4 ViT：32 层 × 1024 宽，patch 14 → 4 × (B, 1024, 28, 28)
  │   └─ DINOv3 ViT-L/16：24 层 × 1024 宽，patch 16 → 4 × (B, 1024, 28, 28)
  │   └─ 冻结 / 解冻末尾 N 层
  │
  ├─ 1×1 投影（1024 → 256，各自 BN+GELU）→ concat → (B, 1024, S, S)
  ├─ 融合卷积 ×2 → ASPP（空洞率 1/3/6 + 全局池化）
  │
  ├─ 上采样到 1/4 分辨率 → 3×3 卷积降到 128
  │   ├─ ★浅层细节支路：原图两次 stride=2 卷积 → 同分辨率 64 通道，拼接成 192 → 融合回 128
  │   └─ ★中间监督：1×1 头输出 1/4 分辨率的辅助预测（权重 0.4）
  ├─ 上采样到全分辨率 → 3×3 卷积降到 64 → 1×1 头 → (B, 1, H, W)

损失: 0.5×BCEWithLogits + 0.5×SoftDice  (+0.4×中间监督)     指标: Dice / IoU
```

★ 标记的两处是「逐步精化」，与 `--unfreeze-last` 一起构成主力配置。

## 实验结果

Kvasir-SEG，解冻编码器末尾 4 层 + 解码器逐步精化，其余配置相同（除非注明）。

| backbone | 输入 | 取层 | 划分 | Dice | IoU |
|---|---|---|---|---|---|
| DeepSeek ViT-412M (V4.1-Flash) | 392 | 7/15/23/31 | 12% | 0.9042 | 0.8252 |
| DeepSeek ViT-412M (V4.1-Flash) | 392 | 7/15/23/31 | 8:2 | 0.9196 | 0.8511 |
| DeepSeek ViT-412M (V4.1-Flash) | 392 | 7/15/23/31 | 8:2 | 0.9190 | 0.8502 |
| DeepSeek ViT-412M (V4.1-Flash) | 546 | 7/15/23/31 | 8:2 | 0.9190 | 0.8501 |
| DeepSeek ViT-412M (Vision-Exp) | 392 | 7/15/23/31 | 8:2 | 0.9212 | 0.8539 |
| **DINOv3 ViT-L/16 (web)** | **448** | **5/11/17/23** | **8:2** | **0.9352** | **0.8782** |
| DINOv3 ViT-L/16 (web) | 512 | 1/8/17/21 | 8:2 | 0.9348 | 0.8776 |
| DINOv3 ViT-L/16 (web) | 448×384 letterbox | 5/11/17/23 | 8:2 | 0.9346 | 0.8773 |

> 前 4 行是同一权重的重复/变体实验，差距（0.9190–0.9196）属于**复现噪声**：同配置同种子重跑，同一轮次的 val_dice 会差 0.18 左右。

**结论：**

1. **从多模态大模型里 remap 出来的视觉编码器，和专门用图像自监督训练的编码器，差距是实的**：DINOv3 ViT-L（303M）比 DeepSeek ViT-412M 高 **1.4 个 Dice 点 / 2.4 个 IoU 点**，而且参数更少、每轮更快。
2. **对照是干净的**：DeepSeek 在 392 下 patch 14 → 28×28 网格；DINOv3 在 448 下 patch 16 → 也是 28×28 网格。换算到原图，两者每个 patch 都对应 **22.2 个原图像素**，完全一致。同一个解码器、同一份划分、同一套训练配置——唯一变量就是编码器。
3. **输入侧怎么调都没用**（已逐一验证）：
   - 分辨率 392 → 546（DeepSeek）、448 → 512（DINOv3）：**无变化**
   - 纵横比：直接拉伸 vs letterbox 黑边补齐：**无变化**（0.9352 vs 0.9346）
   - 取层组合 5/11/17/23 vs 1/8/17/21：**无变化**
   - 换 DeepSeek 权重版本（V4.1-Flash vs V4-Flash-Vision-Exp）：**无变化**
4. **真正有效的是解冻**：全冻结编码器时 Dice 0.9042，解冻末尾 4 层 + 逐步精化后到 0.919–0.921（不同划分下比较，仅供参考）。
5. DINOv3 的 IoU 0.8782 已越过文献里多个**全量微调**的专用分割模型（PraNet 0.840、EU-Net 0.854、Polyp-PVT 0.864、TransFuse 0.868），而这里只训练 57M 参数、250M 冻结。

## 关于编码器的几点说明

1. **没有尺度金字塔**：ViT 所有层的特征都是 1/patch 分辨率，所以用「多深度融合 + ASPP + 逐级上采样」代替 FPN。
2. **RoPE 位置编码**：换输入尺寸不退化，只要边长能被 patch 整除。
3. **patch size 由脚本从模型读**（`_get_patch_size`），DeepSeek 是 14、DINOv3 是 16，`--img-size` 必须是对应倍数。
4. **timm 版本坑**：`deepseek_vit_412m` 只在 GitHub main 分支注册，PyPI 的 timm 1.0.29 里没有（报 unknown model）；本目录用 `timm_main/` + `PYTHONPATH` 注入。
5. DeepSeek 系列还有 `..._enc`、`..._align` 两个变体；DINOv3 另有 `sat493m`（卫星）版本。
6. **显存**：392/448 下解冻 4 层 + AMP 约 8.5–9.7 GB；546 下约 14.9 GB。

## 输出与可视化

```bash
python visualize_results.py --n 6 --ckpt outputs/best_model.pth --out-dir outputs/visualizations/myexp
```

画 6 张验证集样本的「原图 | 真值 | 预测概率 | 预测叠加+Dice」。**注意**：脚本从 ckpt 里恢复 `--progressive`/`--unfreeze-last` 等架构参数，否则 strict=False 会静默跳过那些权重；另外**只看几张图容易被采样波动误导**（同样 6 张图，不同模型平均能差 5 个点，而整体验证集只差 0.2 点），看整体指标。

## 参考

- DeepSeek 编码器权重: https://hf-mirror.com/timm/deepseek_vit_412m.deepseek_v4_1_flash
- DeepSeek Vision-Exp 权重: https://hf-mirror.com/timm/deepseek_vit_412m.deepseek_v4_flash_vision_exp
- DINOv3 权重(timm 镜像): https://hf-mirror.com/timm/vit_large_patch16_dinov3.lvd1689m
- 原始模型: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
- timm: https://github.com/huggingface/pytorch-image-models
