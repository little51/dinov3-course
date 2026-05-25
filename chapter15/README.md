# 第十五章 DINOv3蒸馏模型：轻量化图像分割

> 本章阐述使用DINOv3轻量化蒸馏模型和COCO128系列数据集完成全景分割、实例分割和前景分割等任务的训练与应用，并介绍语义分割掩码的自动标注方法。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter15/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-23*

---



## 🧪 扩展实验1：EUPE-ViT-B 遥感线性分割

> 使用冻结的 EUPE-ViT-B backbone + 可训练线性头，在 WHU Building 数据集上做语义分割。
> 目标：展示预训练模型的强大表征能力 —— 仅用 1×1 卷积 head（3074 个参数），3 个 epoch 即可达到接近全量训练的分割性能。

### 1、实验说明

EUPE（Efficient Universal Perception Encoder）是 Meta 2025 年发布的工作，通过知识蒸馏将 CLIP、DINOv3、SAM3 三个大模型的视觉能力压缩到一个 ViT-B/16（86M 参数）的学生模型中。本实验使用冻结的 EUPE-ViT-B backbone 提取 WHU Building 航拍图像特征，在其上训练一个极简线性分割头（仅 3074 个参数），验证蒸馏特征的语义理解能力。

### 2、环境配置

```bash
# 创建虚拟环境
conda create -n dinov3_15a python=3.12 -y
# 激活虚拟环境
conda activate dinov3_15a
# 安装 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 安装其他依赖
pip install timm pillow numpy opencv-python matplotlib segmentation-models-pytorch
```

### 3、克隆 EUPE 仓库

```bash
git clone https://github.com/facebookresearch/EUPE --depth=1
cd EUPE
pip install -r requirements.txt   # 安装 EUPE 的依赖（主要是 omegaconf, iopath）
cd ..
```

> ⚠️ **注意**：如果 `requirements.txt` 装不上 `iopath`，可以单独装 `pip install iopath iopath-fb`

### 4、数据准备

WHU Building 数据集解压到 `whu_building/` 目录，结构如下：

```
whu_building/
  ├── train/
  │   ├── image/      ← (*.tif 图像)
  │   └── label/      ← (*.tif 标签，0=背景，1=建筑)
  └── val/
      ├── image/
      └── label/
```

### 5、下载模型权重

```bash
set HF_ENDPOINT=https://hf-mirror.com
hf download introvoyz041/EUPE-ViT-B --local-dir models/eupe
```

### 6、参考代码

`eupe_sat_train.py`：完整的 EUPE 线性 probing 分割流程，包含 EUPE backbone 加载、冻结特征提取、线性分割头（Dropout+BN+Conv2d 1×1）训练、IoU 指标评估及四栏可视化。

### 7、运行方法

```bash
conda activate dinov3_15a
python eupe_sat_train.py
```

### 8、预期耗时

| 机器 | 训练总时间 | 备注 |
|------|-----------|------|
| GB10 (Ampere 20G) | ~3 分钟 | 已实测 |
| RTX 4090 | ~1.5 分钟 | 估算 |
| GTX 1060 (6G) | ~8 分钟 | batch_size=8 需下调 |
| CPU only | ~数小时 | 不推荐 |

如果显存不足（< 8GB），将脚本中 `BATCH_SIZE = 8` 改为 `BATCH_SIZE = 4` 或 `2`。

### 9、实验结果

| 指标 | EUPE-ViT-B |
|------|-----------|
| mIoU | **0.7660** |
| Building IoU | **0.5840** |
| Background IoU | **0.9479** |
| Pixel Acc | **0.9515** |

建筑类 IoU 达到 0.584，说明 EUPE 蒸馏特征对建筑结构已有很强的语义理解能力。

### 10、输出文件

运行完成后，`outputs/` 下生成：

| 文件 | 说明 |
|------|------|
| `EUPE_results.json` | 分割指标 |
| `eupe_best.pth` | 训练好的线性头（22KB，不含 backbone） |
| `EUPE_vis.png` | 分割结果可视化（四栏：原图\|分割\|标注\|叠加） |

## 🧪扩展实验2：EUPE与UNet分割对比试验

与实验1的区别在于：将线性头（最后1层）扩展到3、6、9、12层进行融合，这就的修改可以达到mIoU=0.9403的指标。

EUPE-ViT-B + DPT 在最终精度上胜出：mIoU 领先约 0.9 个百分点，Building IoU 领先约 1.6 个百分点。考虑到 EUPE 只用了 6.7M 可训练参数（U-Net 的 27%），参数效率的优势比较明显。

```bash
conda activate dinov3_15a
python eupe_vs_unet.py
```

