# 第五章 零样本分类：文本与图像对齐

> 本章介绍基于DINOv3骨干模型实现零样本分类的方法——利用文本与图像特征的跨模态对齐能力，在无需图像-文本标签对训练数据的情况下完成开放词汇分类。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter05/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-05*

---



## 🧪 扩展实验：DINOv3与词向量零样本分类

第5章讲解了DINOv3骨干模型实现零样本分类的原理——通过文本与图像特征的跨模态对齐，无需图像-文本标签对。本扩展实验在CIFAR-100上完整演示：将DINOv3特征投影到GloVe词向量空间，在**80个已知类**上训练投影头，在**20个未见过的类**上做零样本分类。

### 1、实验说明

使用CIFAR-100数据集（100类，60000张32×32彩色图），将DINOv3特征通过线性投影头映射到GloVe-50维词向量空间，在80类上训练对齐，在其余20个未见类上直接做零样本分类，验证跨模态对齐的泛化能力。

### 2、环境配置

```bash
# 创建虚拟环境
conda create -n dinov3_05a python=3.12 -y
# 激活虚拟环境
conda activate dinov3_05a
# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 验证PyTorch安装结果
python -c "import torch; print(torch.cuda.is_available())"
# 安装其他依赖
pip install timm scikit-learn gensim numpy
```

### 3、数据准备

CIFAR-100 数据集由程序自动下载，若自动下载太慢，可按程序运行时的提示找到下载链接，手动下载后将文件放到 `data/` 目录下，再次运行脚本即可。

首次运行时，gensim 会自动下载 GloVe 词向量（`glove-wiki-gigaword-50`，约 170MB）。

### 4、参考代码

`dinov3_wordvec_zs.py`：零样本分类完整流程，包含数据加载、DINOv3特征提取、投影头训练、GloVe词向量对齐、零样本测试及各类准确率统计。

### 5、运行方法

```bash
# Windows (CMD)
set HF_ENDPOINT=https://hf-mirror.com
python dinov3_wordvec_zs.py
```

```bash
# Linux/WSL
export HF_ENDPOINT=https://hf-mirror.com
python dinov3_wordvec_zs.py
```
