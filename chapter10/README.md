# 第十章 DINOv3图像分类：轻量级任务头训练

> 本章在CIFAR-10数据集上使用Timm框架进行DINOv3分类头训练，涵盖从训练到结果应用的基础流程，体现DINOv3在轻量级分类任务上的训练范式。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter10/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-24*

---



## 🧪 扩展实验：CHASEDB1 视网膜血管分割 — U-Net vs EUPE ConvNeXt Base 双模型对比

> 在 CHASEDB1 数据集上对比两种主流分割架构：传统 U-Net（resnet34 编码器）vs 自监督 ConvNeXt Base + DPT 头（冻结 backbone）。

### 1、实验说明

使用 CHASEDB1 视网膜图像数据集（28 张眼底照片），分别训练 U-Net（全参数微调）和 EUPE ConvNeXt Base + DPT 头（冻结 backbone），对比两者在视网膜血管分割任务上的表现。

U-Net 作为经典 CNN 分割基线，ConvNeXt + DPT 则展示自监督预训练特征在小样本医学分割场景中的迁移能力。

### 2、环境配置

```bash
# 创建 conda 环境
conda create -n dinov3_10a python=3.10 -y
conda activate dinov3_10a

# 安装 PyTorch（CUDA 12.4 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装其它依赖
pip install matplotlib pillow numpy tqdm segmentation-models-pytorch timm
```

### 3、数据准备

**CHASEDB1 数据集**（~22MB）

- 官网：https://blogs.kingston.ac.uk/retinal/chasedb1/
- Kaggle：https://www.kaggle.com/datasets/deepakat002/chase-db1-retinal-image-dataset
- HF 镜像：`Zomba/CHASE_DB1-retinal-dataset`（hf-mirror.com）

下载后按以下结构放置：

```
chasedb1/data/CHASEDB1/
├── train/
│   ├── input/          ← 训练原图（.png）
│   └── label/          ← 训练标签（_1stHO.png，二值血管掩码）
└── val/
    ├── input/          ← 验证原图
    └── label/          ← 验证标签
```

数据集划分（28 张图像，随机分割）：
- **训练集**：~20 张（Image_01~07）
- **验证集**：~8 张（Image_08~14）

**EUPE ConvNeXt-B 权重**（~430MB）

- HuggingFace：https://huggingface.co/nvidia/EUPE-ConvNeXt-B
- 国内镜像：https://hf-mirror.com/nvidia/EUPE-ConvNeXt-B

下载后放在 `chasedb1/models/EUPE-ConvNeXt-B.pt`。

**EUPE 代码库**

```bash
git clone https://github.com/nousr/EUPE.git chasedb1/EUPE
```

> ⚠️ 若 GitHub 被墙，`chasedb1/EUPE/hubconf.py` 已内置了 minimal 替代实现，可跳过克隆步骤。权重仍需要下载。

### 4、参考代码

`chasedb1/chasedb1_train.py`：双模型训练+对比可视化完整流程，包含：
- U-Net（resnet34 编码器，ImageNet 预训练，全参数微调）
- EUPE ConvNeXt Base + DPT 头（ConvNeXt 冻结，仅训练 DPT）
- 训练曲线、柱状图、样本可视化自动输出

### 5、运行方法

```bash
cd chapter10/chasedb1
conda activate dinov3_10a
python chasedb1_train.py
```

脚本已使用相对路径（基于 `_SCRIPT_DIR` 自动推导），无需手动配置路径。

---

### 📊 实验结果

| 模型 | 参数量 | 可训练 | mIoU | Vessel IoU |
|:-----|:------:|:------:|:----:|:----------:|
| **U-Net (resnet34)** | 24.4M | 24.4M（全参） | **0.8126** | **0.6534** |
| **EUPE ConvNeXt Base + DPT** | 93.7M | 6.7M（仅 DPT 头） | 0.8046 | 0.6383 |

**关键结论：**
- ConvNeXt 冻结 backbone 训练 DPT 头，参数量少但收敛慢
- mIoU 差距仅 **0.008**，Vessel IoU 差距 **0.015**
- 对于小数据集，全参训练的 U-Net 仍然是最实用的选择

---

### 📁 项目结构

```
chasedb1/
├── chasedb1_train.py              ← 训练 + 可视化
├── data/CHASEDB1/                 ← 数据集（需自行下载）
│   ├── train/input/
│   ├── train/label/
│   ├── val/input/
│   └── val/label/
├── outputs/                       ← 训练结果目录
│   ├── curves.png
│   ├── bar.png
│   ├── unet_best.pth
│   ├── convnext_best.pth
│   └── samples/
├── EUPE/                          ← EUPE 代码库（或 hubconf.py 替代）
├── models/
│   └── EUPE-ConvNeXt-B.pt         ← 预训练权重
└── requirements.txt               ← 依赖列表
```

#### 样本可视化布局

```
┌─────────────┬──────────────────┬─────────────────────┐
│   原图       │  U-Net 叠加🔵    │  ConvNeXt 叠加🟣    │
├─────────────┼──────────────────┼─────────────────────┤
│   真值       │  U-Net           │  ConvNeXt           │
└─────────────┴──────────────────┴─────────────────────┘
```

---

### 🧠 技术细节

#### U-Net (Part 1)

- **框架**：segmentation-models-pytorch（smp）
- **编码器**：resnet34（ImageNet 预训练）
- **输入**：1024×1024，三通道 RGB
- **批大小**：2
- **优化器**：AdamW（lr=3e-4, weight_decay=1e-4）
- **学习率**：CosineAnnealingLR
- **损失函数**：CrossEntropy + Dice Loss
- **早停**：连续 20 轮 mIoU 无提升
- **数据增强**：随机水平 + 垂直翻转

#### EUPE ConvNeXt Base + DPT (Part 2)

- **Backbone**：ConvNeXt Base（EUPE 自监督预训练权重）
- **冻结策略**：全部冻结，只训练 DPT 头
- **DPT 头结构**：4 级多尺度特征融合（1/4, 1/8, 1/16, 1/32）
- **DPT 头参数**：~6.7M 可训练参数
- **输入**：1024×1024
- **批大小**：1（显存限制）
- **优化器**：AdamW（lr=6e-5, weight_decay=0.01）
- **损失函数**：CrossEntropy + Dice Loss

#### 视觉特征维度

| 层级 | 空间尺寸 | ConvNeXt Base 通道 | 备注 |
|:----:|:--------:|:------------------:|:----:|
| 1/4 | 256×256 | 128 | 浅层细节 |
| 1/8 | 128×128 | 256 | |
| 1/16 | 64×64 | 512 | |
| 1/32 | 32×32 | 1024 | 深层语义 |

---

### 📈 输出说明

| 文件 | 说明 |
|:----|:-----|
| `curves.png` | mIoU 和 Vessel IoU 训练曲线，含终值参考线 |
| `bar.png` | 最终结果的柱状对比图 |
| `unet_best.pth` | U-Net 最佳 checkpoint（state_dict + metrics） |
| `convnext_best.pth` | ConvNeXt DPT 头最佳 checkpoint（head_state_dict + metrics） |
| `samples/sample_*.png` | 6 张验证集样本的对比可视化 |

#### 可视化解读

- **蓝色 🔵** = U-Net 预测 / 叠加
- **紫色 🟣** = ConvNeXt + DPT 预测 / 叠加
- 叠加图中预测区域以半透明颜色覆盖在原图上
- 关注 Vessel IoU（血管交并比）—— 这是视网膜分割的核心指标

---

### 📚 参考文献

- [CHASEDB1: A retinal image dataset](https://blogs.kingston.ac.uk/retinal/chasedb1/)
- [Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Liu et al. A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (ConvNeXt)
- [EUPE: Self-Supervised Vision Transformers for Segmentation](https://github.com/nousr/EUPE)
- [DPT: Vision Transformer for Dense Prediction](https://arxiv.org/abs/2103.13413)
