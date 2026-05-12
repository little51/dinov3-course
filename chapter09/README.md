# 第九章 3D赋能：SAM 3D Body应用

> 本章介绍DINOv3在三维图像处理中的应用案例——SAM 3D Body，先讲解SAM3的图像分割实现，再详解使用SAM 3D Body进行三维人体网格重建的完整步骤。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter09/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-12*

---



## 🧪 扩展实验：SAM3 LoRA 微调（ISIC 2018 病灶分割）

> 完整记录从环境搭建、数据下载、LoRA 训练、效果验证到对比原始 SAM3 的全流程。
> 硬件：GTX 1060 6GB 实测通过，约 5 小时完成 1 轮训练。

### 1、实验概述

使用 **SAM3 Huge（8.44亿参数）** + **LoRA（仅训练 360万参数 = 0.43%）**，在 **ISIC 2018 皮肤病灶分割数据集** 上做微调，显著提升病灶轮廓分割精度。

**实验结果：**

| 指标 | 原始 SAM3 | LoRA 微调（1轮） |
|------|----------|----------------|
| 平均 Dice | **0.892** | **0.943** ↑ |
| 中位数 Dice | 0.911 | **0.951** |
| 提升样本占比 | — | **~80%** |

**技术选型：**

| 组件 | 选择 | 原因 |
|------|------|------|
| 基础模型 | SAM3 Huge（`facebook/sam3`） | 最新 SAM 架构，6GB 显存可推理 |
| 微调方法 | LoRA（r=8, targets=q/v/k/out） | 显存友好，只需额外 14MB 存储 |
| 损失函数 | BCE + Dice Loss（Dice 权重 5.0） | 病灶分割中 Dice 比 BCE 更敏感 |
| Prompt 策略 | Bounding Box（从 GT mask 计算） | 替代人工标注，实现全自动训练 |
| 数据集 | ISIC 2018 Task 1（2,200 张） | 皮肤病灶分割标准基准 |
| 输入分辨率 | 448×448 | 6GB 显存上限，SAM3 官方预训练分辨率 |

### 2、环境配置

```bash
# 创建虚拟环境
conda create -n dinov3_09a python=3.12 -y
# 激活虚拟环境
conda activate dinov3_09a
# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 安装依赖
pip install transformers==5.1.0 peft accelerate datasets pandas pyarrow pillow numpy
```

### 3、数据准备

数据集来自 HuggingFace：**[sergiomadrid/isic-melanoma-segmentation](https://huggingface.co/datasets/sergiomadrid/isic-melanoma-segmentation)**

```bash
# Windows (CMD)
set HF_ENDPOINT=https://hf-mirror.com
hf download sergiomadrid/isic-melanoma-segmentation --repo-type dataset --local-dir ./isic2018

# Linux/WSL
export HF_ENDPOINT=https://hf-mirror.com
hf download sergiomadrid/isic-melanoma-segmentation --repo-type dataset --local-dir ./isic2018
```

数据集目录结构：
```
isic2018/data/
├── train_00.parquet ~ 110 样本/文件，共 20 个文件（2,200 张）
├── train_01.parquet
├── ...
└── train_19.parquet
```

### 4、训练原理

SAM3 的微调面临两个核心挑战：

1. **显存瓶颈**：SAM3 Huge 8.44 亿参数，全量微调至少需要 24GB+ 显存
2. **Prompt 设计**：SAM3 原生支持 point/box/text prompt，训练时需提供这些 prompt 作为输入

**解决方案：**
- **LoRA（Low-Rank Adaptation）**：冻结 backbone，只插入 4 个低秩矩阵（q/k/v/out 投影层），参数量从 844M 降到 3.6M（0.43%）
- **自动 BBox Prompt**：从 GT mask 计算病灶的包围盒作为 box prompt，实现全自动训练

**损失函数设计：**
```
Loss = BCE(pred, gt) + 5.0 × (1 - Dice(pred, gt))
```
- BCE 提供像素级分类信号
- Dice Loss 专注于轮廓匹配，权重 5.0 放大其影响
- 从 200 个候选 mask 中选置信度 × Dice 最高的 top-5 做加权平均

### 5、参考代码

| 文件名 | 说明 |
|--------|------|
| `sam3_lora_isic.py` | SAM3 LoRA 训练主脚本，含数据加载、模型构建、训练循环 |
| `compare_sam3_vs_lora.py` | GT mask 作为 bbox prompt 的对比评估 |
| `compare_sam3_vs_lora_yolo.py` | YOLOv8n 检测 + SAM3 分割的端到端对比评估 |

### 6、运行方法

**训练：**
```bash
conda activate dinov3_09a
python sam3_lora_isic.py
```

**GT Mask 对比评估：**
```bash
python compare_sam3_vs_lora.py
```

**YOLO 端到端对比评估：**
```bash
pip install ultralytics
python compare_sam3_vs_lora_yolo.py
```

### 7、对比结果解读

输出图片为**三栏布局**：

| 第1栏 | 第2栏 | 第3栏 |
|-------|-------|-------|
| GT Mask（绿色覆盖） | **原始 SAM3** 预测（红色覆盖，显示 Dice） | **LoRA 微调** 预测（蓝色覆盖，显示 Dice） |
| 真实标注 | 零样本分割效果 | 1 轮微调后效果 |

底部有汇总行 `Dice: 0.835 → 0.940 ▲+0.1050`

### 8、总结

| 阶段 | 关键发现 |
|------|---------|
| **LoRA 足够有效** | 仅 0.43% 参数（3.6M），1 轮就能提升 5 个 Dice 点 |
| **原始 SAM3 已有基础** | 零样本 Dice = 0.89，但在困难样本上表现差（0.07） |
| **困难样本提升最大** | ~20% 样本原始 Dice < 0.80，微调后全部 > 0.90 |
| **6GB 显存可训练** | 关键：fp16 + LoRA + batch_size=1 + RoPE patch |
