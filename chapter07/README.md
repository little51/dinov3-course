# 第七章 目标检测：定位与识别

> 本章讲解基于DINOv3骨干网络结合检测任务头实现目标检测的原理与实践，并提供一个使用蒸馏后轻量化模型完成目标检测的完整案例。

---

## 配套资源

本章的源码、数据和配套资源均包含在《视觉自监督模型DINOv3：原理、训练到部署》一书附带的二维码资源包（chapter07/）。

## 勘误与更新

<!-- 此处用于后续补充勘误、新增代码、补充说明等 -->

---

*最后更新：2026-05-17*

---



## 🧪 扩展实验：DINOv3 遥感小目标检测（零训练特征可视化）

> 基于冻结 DINOv3.sat493m 的中间层特征，通过简单阈值分割 + 连通域分析，在不经过任何训练的情况下，从高分辨率遥感图像中自动框出小目标候选区域。

### 1、实验说明

本实验展示 DINOv3 的一个令人惊讶的能力：**即便不做任何微调或训练，仅凭 backbone 中间层的特征范数，就能从遥感图像中定位到小目标（车辆、小型建筑等）。**

将 WHU Building Dataset 中的 512×512 遥感图像放大至 **1536×1536**（特征网格 96×96），用 DINOv3 第 19 层特征计算每个 patch 的范数，取前 3% 最强响应做连通域分析，共可检测出 **149 个候选区域**。

| 项目 | 要求 |
|------|------|
| GPU | 4GB+ VRAM（实测 1536×1536 仅需 **1.74GB**） |
| 内存 | 8GB+ |
| Python | 3.10+ |
| CUDA | 11.8+ |

> 无需训练，单次推理即可出结果。

### 2、环境配置

**推荐使用 Anaconda 创建独立环境：**

```bash
# 创建虚拟环境
conda create -n dinov3_07a python=3.11 -y
conda activate dinov3_07a

# 安装 PyTorch（CUDA 12.x，根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install timm>=1.0.0
pip install numpy>=1.24 pillow>=10.0 scikit-image>=0.22 matplotlib>=3.7 scikit-learn

# 验证
python -c "import torch, timm; print(f'torch {torch.__version__}, timm {timm.__version__}, CUDA={torch.cuda.is_available()}')"
```

> 首次运行会自动从 HF 镜像站下载 DINOv3 backbone 权重（`vit_large_patch16_dinov3.sat493m`，约 **1.2GB**），下载后缓存到本地，后续不再重复下载。
>
> 如果自动下载较慢，可设置 HF 镜像：`set HF_ENDPOINT=https://hf-mirror.com`（Windows）或 `export HF_ENDPOINT=https://hf-mirror.com`（Linux/Mac）。

### 3、目录结构

```plaintext
chapter07/
├── README.md                              ← 原书第 7 章 README（合并后含本实验说明）
├── dinov3_small_object_det.py             ← 主程序
├── test_samples/                          ← 测试图像
│   ├── 2253.tif                           ← WHU 原图 512×512
│   └── 2253_mask.tif                      ← 建筑标注（参考对比用）
└── output/                                ← 运行结果（自动生成）
    └── dinov3_small_object_result.png     ← 可视化结果图
```

### 4、运行方法

```bash
# 激活环境
conda activate dinov3_07a

# 运行（Windows 下在 Anaconda Prompt 里执行）
python dinov3_small_object_det.py
```

> **Windows 用户注意**：如果路径或编码有问题，请在 Anaconda Prompt 中执行，不要用 PowerShell 的 python。

### 5、运行结果

程序输出：

| 输出 | 说明 |
|------|------|
| `output/dinov3_small_object_result.png` | 3×4 网格可视化结果图 |
| 终端打印的统计信息 | 候选框数量、显存占用等 |

结果图包含：
- (a) 原始 WHU 图像（512×512）
- (b) 放大 2048×2048 + 自动框选结果（红=小目标，橙=中目标，蓝=建筑）
- (c) 建筑标注 GT（用于参考对比）
- (d) DINOv3 Layer 19 特征热力图
- (e)~ (h) 多层特征响应（Layer 3/11/15/19）
- (i) 前 1% 最强响应二值掩码
- (j)~ (l) 各层 PCA 特征空间可视化
- 统计信息面板

### 6、实验结果参考

在测试图像 **2253.tif**（建筑密度约 24%）上的实测结果：

| 指标 | 值 |
|------|-----|
| 输入尺寸 | 1536×1536 |
| 特征网格 | 96×96（DINOv3 patch=16） |
| 使用层 | Layer 19（深层语义） |
| 阈值 | 前 3% 分位数 |
| 显存占用 | **1.74 GB**（GTX 1060 6GB） |
| 检测候选区 | **149 个**（全部为小目标候选） |

149 个候选区均为"小目标"（面积 < 画面 0.5%），覆盖车辆、小建筑及更多人造物。

> 数值为参考值，不同图像结果会有所差异。

### 7、核心原理

```plaintext
输入图像 512×512
    ↓ LANCZOS 插值放大
放大图像 1536×1536
    ↓ DINOv3.sat493m (patch=16)
特征网格 96×96 (每 patch 1024 维)
    ↓ 取 Layer 19 的特征范数
响应图 96×96
    ↓ 取前 3% 分位数阈值
二值掩码 96×96
    ↓ 连通域分析
149 个候选框
    ↓ 按面积着色
红框（小目标）  / 橙框（中目标）  / 蓝框（建筑）
```

**为什么 DINOv3 能做到？**

DINOv3 通过自监督学习训练，其特征空间天然具有语义结构——同类物体（建筑、车辆）在特征空间中距离近。即便没有见过任何标注，其中间层的特征范数也能反映"这里有没有东西"，且越"突出"的小目标（如道路上亮色的车）响应越强。

### 8、局限性

- **无分类能力**：红框可能是车、小棚子或其他小物体，单纯靠特征范数无法区分
- **框不精准**：框是特征响应区域的外接矩形，非物体精确边界
- **需要合理阈值**：前 3% 分位数在本图上效果最佳，换图可能需要微调
- **有监督上限**：本实验仅证明"零训练能看到"，要准确检测需训练检测头
