# 第四章 特征提取：视觉基础表示

> 本章系统讲解DINOv3全局与局部特征的提取方法，解析特征向量结构，通过可视化展现特征与图像的关联，并给出图像相似度计算的实际应用示例。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter04/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-04

---



## 🧪 扩展实验：特征提取在医学影像上的应用

本书第4章讲解了DINOv3特征提取的原理、相似度比较和特征可视化等内容。本扩展实验展示特征提取的另一个用途：**特征 + 简单分类器 = 零标注分类**，以医学X光片分类为例。

### 1、实验说明

使用Chest X-Ray数据集（Kaggle公开，5863张，正常/肺炎二分类），将DINOv3作为固定特征提取器，在其输出的特征上训练一个逻辑回归分类器（Linear Probing），对比DINOv3与ResNet-50的特征质量。

### 2、环境配置

```bash
# 创建虚拟环境
conda create -n dinov3_04a python=3.12 -y
# 激活虚拟环境
conda activate dinov3_04a
# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 验证PyTorch安装结果
python -c "import torch; print(torch.cuda.is_available())"
# 安装其他依赖
pip install timm scikit-learn matplotlib seaborn
```

### 3、数据准备

```bash
# Chest X-Ray数据集网址：
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# 下载命令
curl -L -o chest-xray-pneumonia.zip https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia
```

下载后解压，目录结构如下：
```bash
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
```

### 4、参考代码

`dinov3_chestxray.py`：实验脚本，包含模型加载、特征提取、t-SNE可视化、分类对比全流程。

### 5、运行方法

```bash
# Windows (CMD)
set HF_ENDPOINT=https://hf-mirror.com
python dinov3_chestxray.py
```

```bash
# Linux/WSL
export HF_ENDPOINT=https://hf-mirror.com
python dinov3_chestxray.py
```

