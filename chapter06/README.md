# 第六章 语义分割：像素级分类

> 本章聚焦基于DINOv3的语义分割实现，涵盖预训练模型配合任务头的分割方法、前景分割专用任务头的训练，以及主成分分析特征可视化的应用。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter06/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-05*

---



## 🧪 扩展实验：DINOv3 皮肤病变语义分割

第6章讲解了基于DINOv3的语义分割实现。本扩展实验在PH2数据集上进行皮肤镜图像病变分割，展示DINOv3作为固定特征提取器配合轻量解码器的完整流程。

### 1、实验说明

使用PH2数据集（200张768×560皮肤镜图像+像素级病变掩码），将DINOv3作为固定特征提取器，在其输出的patch特征上训练一个轻量解码器（4层Conv+Upsample），完成皮肤病变分割任务。

### 2、环境配置

```bash
# 创建虚拟环境
conda create -n dinov3_06a python=3.12 -y
# 激活虚拟环境
conda activate dinov3_06a
# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 验证PyTorch安装结果
python -c "import torch; print(torch.cuda.is_available())"
# 安装其他依赖
pip install timm matplotlib
```

### 3、数据准备

PH2数据集下载地址：
```
https://gitcode.com/open-source-toolkit/3f18e/blob/main/PH2%20Dataset.rar
```
下载解压后，将 `PH2Dataset/` 目录放入 `data/` 下，目录结构如下：
```
data/PH2Dataset/
└── PH2 Dataset images/
    ├── IMD240/
    │   ├── IMD240_Dermoscopic_Image/IMD240.bmp
    │   └── IMD240_lesion/IMD240_lesion.bmp
    ├── IMD242/
    └── ...
```

### 4、参考代码

`dinov3_ph2_seg.py`：语义分割完整流程，包含DINOv3特征提取、轻量解码器训练、BCE+Dice联合损失、IoU/Dice评估及分割结果可视化。

### 5、运行方法

```bash
# Windows (CMD)
set HF_ENDPOINT=https://hf-mirror.com
python dinov3_ph2_seg.py
```

```bash
# Linux/WSL
export HF_ENDPOINT=https://hf-mirror.com
python dinov3_ph2_seg.py
```
