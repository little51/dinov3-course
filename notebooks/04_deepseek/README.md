# DeepSeek-V4.1-Flash 纯视觉编码器 — Kvasir-SEG 息肉分割

用 DeepSeek-V4.1-Flash 的**纯视觉编码器（ViT-412M）**当 backbone，在 Kvasir-SEG 上做息肉二分类分割。

这个编码器是 timm 对 DeepSeek-V4.1-Flash 视觉塔的原生 remap：**不含语言模型权重、也没有训练好的分类头**，本身只做图像特征提取；本目录把它当分割 backbone 用（冻结编码器 + 轻量解码器）。

## 目录结构

```bash
04_deepseek/
├── train_kvasir.py            # 训练脚本（含数据/模型/解码器/指标）
├── visualize_results.py       # 推理可视化脚本
├── model/                     # 编码器权重
│   └── model.safetensors      # 1571 MB，411.8M 参数
├── data/                      # 数据集
│   └── kvasir-seg/Kvasir-SEG/{images,masks}   # 1000 图 / 1000 掩码
└── outputs/                   # 训练输出（训练时产生）
    ├── best_model.pth         # 验证 Dice 最佳权重
    ├── last_model.pth
    ├── train_log.csv          # 每轮指标
    ├── history.json
    └── visualizations/        # 可视化结果
```

## 环境配置

```bash
conda create -n deepseek python=3.11 -y
conda activate deepseek

# torch（cu124；GTX 1060 是 sm_61，cu124 的轮子可用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 注意：PyPI 上最新版 timm（1.0.29）里还没有这个模型，必须装 GitHub main
pip install "git+https://github.com/huggingface/pytorch-image-models"
pip install numpy pillow tqdm matplotlib safetensors
```

> 验证环境：`python -c "import torch,timm;print(torch.__version__, torch.cuda.is_available(), timm.__version__)"`

## 模型准备

下载 timm remap 后的权重（1571 MB）放到 `model/`：

```bash
set HF_ENDPOINT=https://hf-mirror.com
curl -L -o model/model.safetensors \
  "https://hf-mirror.com/timm/deepseek_vit_412m.deepseek_v4_1_flash/resolve/main/model.safetensors"
```

## 数据准备

Kvasir-SEG 官方下载（44 MB）：

```bash
curl -L -o data/kvasir-seg.zip "https://datasets.simula.no/downloads/kvasir-seg.zip"
```

解压到 `data/kvasir-seg/`，结构为 `Kvasir-SEG/images/*.jpg` + `Kvasir-SEG/masks/*.jpg`（掩码是 JPEG，脚本里按 >127 二值化）。

划分：脚本按文件名排序后用固定种子随机划分，默认验证集占 12%（约 880/120），可用 `--split-json` 指定自己的划分文件。

## 训练

```bash
conda activate deepseek
cd notebooks/04_deepseek
# 用 python -u：conda run 会把子进程输出缓存到结束，看不到实时进度

# 冒烟测试（16 张图，1 轮，1 分钟内跑完）
python -u train_kvasir.py --epochs 1 --limit-train 16 --limit-val 8

# 正式训练：冻结编码器，只训解码器（默认会把特征缓存到 outputs/cache/）
python -u train_kvasir.py --epochs 80 --batch-size 12 --img-size 392 --lr 1e-3

# 全量微调编码器（显存紧张，开梯度检查点、降 batch）
python -u train_kvasir.py --finetune --grad-checkpoint --epochs 20 --batch-size 2 --lr 2e-5
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img-size` | 392 | **必须能被 14 整除**（patch=14），392=14×28 |
| `--out-indices` | 7,15,23,31 | 取哪 4 个 block 的中间特征（都是 1/14 分辨率） |
| `--batch-size` | 8 | 冻结编码器时可用 12；全量微调降到 2 |
| `--lr` | 1e-3 | 解码器学习率 |
| `--lr-encoder` | 2e-5 | `--finetune` 时的编码器学习率 |
| `--epochs` | 40 | 训练轮数 |
| `--mid-channels` | 256 | 解码器中间通道数 |
| `--finetune` | 关 | 打开则全量微调编码器（自动关缓存） |
| `--no-cache-features` | 关 | 关闭特征缓存 |
| `--prune-to` | 无 | 只保留前 N 个 block，省算力（如 16 → 大约快一倍） |
| `--amp` | 关 | 混合精度；**Pascal 卡 fp16 反而慢，默认不开** |
| `--limit-train/--limit-val` | 无 | 冒烟测试用 |


## 输出

- `outputs/train_log.csv` — 每轮 train_loss / val_dice / val_iou / 学习率
- `outputs/best_model.pth` — 验证 Dice 最佳的权重
- `outputs/history.json` — 训练曲线数据

## 可视化

```bash
python visualize_results.py --n 6
```

画 6 张验证集样本的「原图 | 真值 | 预测概率 | 预测叠加+Dice」，保存到 `outputs/visualizations/kvasir_predictions.png`。

## 架构概览

```
输入: (B, 3, 392, 392)
  │
  ├─ DeepSeek-V4 ViT 编码器（32 层 × 1024 宽，patch 14，RMSNorm + SwiGLU + 轴向 2D RoPE）
  │   └─ forward_intermediates @ blocks [7, 15, 23, 31]  → 4 × (B, 1024, 28, 28)
  │   └─ 冻结 / 微调可选
  │
  ├─ 1×1 投影（1024 → 256，各自 BN+GELU）
  │   └─ concat → (B, 1024, 28, 28)
  │
  ├─ 融合卷积 ×2 → ASPP（空洞率 1/3/6 + 全局池化）
  │   └─ 因为编码器只有 1/14 单尺度，多尺度上下文全靠这一层补
  │
  └─ 上采样 28 → 98 → 392（两级卷积）→ 1×1 分类头 → (B, 1, 392, 392)

损失: 0.5×BCEWithLogits + 0.5×SoftDice      指标: Dice / IoU
```

## 关于这个编码器的几点说明

1. **没有尺度金字塔**：所有 block 的特征都是 1/14 分辨率（`reduction` 恒为 patch_size），所以本脚本用「多深度融合 + ASPP + 逐级上采样」代替常见的 FPN 金字塔。
2. **RoPE 位置编码**（无绝对位置嵌入）：换输入尺寸不退化，只要边长能被 14 整除；`dynamic_img_pad=True` 可放宽这个限制。
3. **显存**：默认 546×546 时激活约 1.76G；本脚本用 392×392 且默认冻结编码器，6GB 显存可跑。
4. **timm 版本坑**：`deepseek_vit_412m*` 只在 GitHub main 分支注册，PyPI 的 timm 1.0.29 里没有（会报 unknown model）。
5. 同一系列还有 `..._enc`（原生编码器，带投影到 LLM 的 aligner）和 `..._align` 两个变体；分割用本目录这个纯编码器 + 自己的解码器最直接。

## 参考

- 编码器权重: https://hf-mirror.com/timm/deepseek_vit_412m.deepseek_v4_1_flash
- 原始模型: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- Kvasir-SEG: https://datasets.simula.no/kvasir-seg/
- timm: https://github.com/huggingface/pytorch-image-models
