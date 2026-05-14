# 第九章 3D赋能：SAM 3D Body应用

> 本章介绍DINOv3在三维图像处理中的应用案例——SAM 3D Body，先讲解SAM3的图像分割实现，再详解使用SAM 3D Body进行三维人体网格重建的完整步骤。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter09/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-13*

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
# 安装PyTorch（国内用清华镜像，避免被墙）
pip install torch==2.4.1 torchvision==0.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装依赖
pip install transformers==5.1.0 peft accelerate datasets pandas pyarrow pillow numpy ultralytics
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

### ★ YOLO 格式转换

上述 parquet 数据可直接训练 SAM3，但如果要训练 **YOLO 检测模型**（作为端到端对比的病灶检测前端），需要先转换为标准 YOLO 格式。转换工具包详见下文第 5 节。

```bash
# 一行命令完成转换（默认输出到 yolo_data_isic）
python prepare_yolo_data.py
```

输出目录结构：
```
yolo_data_isic
├── images/
│   ├── train/          1,980 张 .jpg（原始分辨率）
│   └── val/            110 张 .jpg
├── labels/
│   ├── train/          YOLO 格式 .txt（class cx cy w h 归一化）
│   └── val/
└── data.yaml           供 YOLO 训练的配置文件
```

**转换规则**：
| 步骤 | 说明 |
|------|------|
| 目标检测 | 将病灶分割任务简化为检测任务：从 GT mask 计算病灶 bbox |
| 标签格式 | `0 x_center y_center width height`（归一化到 [0,1]） |
| 单类别 | ISIC 2018 只有一个类别 `lesion`（class 0） |
| bbox 外扩 | mask 边缘外扩 5px（与 SAM3 训练一致） |
| 空 mask | 无病灶区域跳过（统计中记录跳过数） |
| 数据划分 | 同样是最后 10% 文件作为验证集，与 SAM3 保持对齐 |

### 4、SAM3 训练原理

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

### ★ YOLOv8n 检测训练

> YOLO 的作用是在对比实验中作为病灶检测前端，模拟真实 pipeline：
> 图片 → YOLO 检测病灶位置 → bbox → SAM3 分割。
> 避免使用 GT bbox 导致的高估，更贴近实际部署。

**训练参数：**

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | YOLOv8n（nano） | 6.3M 参数，轻量级检测器 |
| 输入分辨率 | 640×640 | 标准 YOLO 训练尺寸 |
| Epochs | 100 | 含 early stopping（patience=20） |
| Batch size | 16 | GTX 1060 6GB 可运行 |
| 增强策略 | mosaic 0.5 + 小旋转/缩放/翻转 | 保留病灶形态特征 |

**预期结果（GTX 1060，约 1 小时）：**
```
mAP@50:    ~0.85-0.90
mAP@50-95: ~0.65-0.70
```

训练完成后自动保存最佳权重至 `yolo_weights/best.pt`，供 `compare_sam3_vs_lora_yolo.py` 使用。

**训练命令：**
```bash
python train_yolo_isic.py   # 默认 GPU 0，查找 yolo_data_isic/data.yaml
python train_yolo_isic.py --data yolo_data_isic/data.yaml # 显式指定数据路径
python train_yolo_isic.py --device cpu                  # CPU 训练（慢）
python train_yolo_isic.py --epochs 50                   # 快速验证
python train_yolo_isic.py --batch-size 8                # 低显存模式
```

### 5、参考代码

| 文件名 | 说明 |
|--------|------|
| `sam3_lora_isic.py` | SAM3 LoRA 训练主脚本，含数据加载、模型构建、训练循环 |
| `compare_sam3_vs_lora.py` | GT mask 作为 bbox prompt 的对比评估 |
| `prepare_yolo_data.py` | 将 parquet 数据集转换为 YOLO 格式（images + labels + data.yaml） |
| `train_yolo_isic.py` | 在 ISIC 2018 上训练 YOLOv8n 检测模型 |
| `compare_sam3_vs_lora_yolo.py` | YOLOv8n 检测 + SAM3 分割的端到端对比评估 |

### 6、运行方法

**SAM3 LoRA 训练：**
```bash
conda activate dinov3_09a
python sam3_lora_isic.py
```

**数据准备（YOLO 格式）：**
```bash
conda activate dinov3_09a
pip install pandas pyarrow  # 如果尚未安装
python prepare_yolo_data.py
```

**YOLOv8n 训练：**

```bash
pip install ultralytics
python train_yolo_isic.py --data yolo_data_isic/data.yaml
```

**GT Mask 对比评估：**

```bash
python compare_sam3_vs_lora.py
```

**YOLO 端到端对比评估：**
```bash
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
