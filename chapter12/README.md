# 第十二章 DINOv3语义分割：卫星遥感图像分割训练

> 本章基于DINOv3构建卫星遥感语义分割任务头，使用DeepGlobe数据集，系统讲解从数据准备、模型构建、训练调优到结果可视化的完整流程。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter12/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-15*

---



## 🧪 扩展实验：WHU 遥感建筑分割（DINOv3 + DPT Head）

基于冻结 DINOv3 sat493m backbone + DPT Head 的遥感建筑分割扩展实验，展示在高分辨率航空遥感数据上的语义分割完整流程。

### 1、实验说明

使用 WHU Building Dataset（航空遥感 0.3m 分辨率，二值建筑分割），将 DINOv3 作为固定特征提取器，在其输出的 4 层中间特征上训练 DPT Head，完成建筑区域分割。

| 项目 | 要求 |
|------|------|
| GPU | 6GB+ VRAM |
| 内存 | 8GB+ |
| 磁盘 | 10GB+ |
| Python | 3.10+ |
| CUDA | 11.8+ |

训练时 batch_size=8 约占用 4-5GB 显存，推理仅需 2-3GB。

### 2、环境配置

```bash
# 创建虚拟环境
conda create -n dinov3_12a python=3.11 -y
conda activate dinov3_12a
# 安装 PyTorch（CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 安装其他依赖
pip install timm>=1.0.0 numpy>=1.24 pillow>=10.0 tqdm>=4.60
# 验证
python -c "import torch, timm; print(f'torch {torch.__version__}, timm {timm.__version__}, CUDA={torch.cuda.is_available()}')"
```

### 3、数据准备

WHU Building Dataset（航空遥感 0.3m 分辨率，二值建筑分割）。

**下载地址：** http://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip

下载后命名为 `WHU_aerial_0.3m.zip`（约 **5.1GB**），保存到 `whu_download/` 目录下，然后解压整理：

```bash
python scripts/extract_whu.py
```

脚本会自动将 zip 解压为以下结构：

```plaintext
whu_building/
├── train/image/ + train/label/
├── val/image/   + val/label/
└── test/image/  + test/label/
```

### 4、参考代码

| 脚本 | 功能 |
|------|------|
| `scripts/extract_whu.py` | 数据集解压整理 |
| `scripts/train_dinov3_seg.py` | DINOv3 + DPT Head 训练 |
| `scripts/eval_dinov3_seg.py` | 模型评估与结果可视化 |

### 5、运行方法

训练策略：冻结 DINOv3 backbone，只训练 DPT 分割头。

```bash
python scripts/train_dinov3_seg.py ^
    --data_root whu_building ^
    --batch_size 8 ^
    --num_workers 0 ^
    --epochs 50 ^
    --save_dir dinov3_seg_whu
```

> 脚本首次运行时会从 HF 镜像站自动下载 DINOv3 backbone 权重（`vit_large_patch16_dinov3.sat493m`，约 **1.2GB**），下载后缓存到本地，后续不再重复下载。

训练完成后，`dinov3_seg_whu/` 目录下会生成：

| 文件 | 说明 |
|------|------|
| `best_model.pth` | 验证集 mIoU 最高的模型（推荐用于推理） |
| `final_model.pth` | 最终 epoch 的模型 |
| `training_history.json` | 完整训练日志 |

训练配置参考：

| 项目 | 值 |
|------|-----|
| 模型 | vit_large_patch16_dinov3.sat493m (303M) + DPT Head (6.95M) |
| 优化器 | AdamW (lr=6e-5, weight_decay=0.01) |
| 调度器 | CosineAnnealingLR (T_max=epochs) |
| 损失函数 | CrossEntropy + Dice Loss |
| 批大小 | 8（GTX 1060 6GB 经验值） |
| 训练耗时 | ~3-4 min/epoch，50 epoch 约 3-4 小时 |

### 6、模型评估

用训练好的 `best_model.pth` 在测试集上跑评估：

```bash
python scripts/eval_dinov3_seg.py
```

输出：

- `eval_results/test_results.json` — 数值指标（mIoU、Dice 等）
- `eval_results/sample_*.png` — 8 张随机采样的 2×2 网格可视化

### 7、目录结构

```plaintext
chapter12/
├── whu_building/                   ← WHU 数据集
├── dinov3_seg_whu/                 ← 训练产出的模型权重
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.json
├── eval_results/                   ← 评估产出
└── scripts/
    ├── train_dinov3_seg.py         ← 训练脚本
    ├── eval_dinov3_seg.py          ← 评估+可视化脚本
    └── extract_whu.py              ← 数据集解压脚本
```
