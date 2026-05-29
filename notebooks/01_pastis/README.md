# PASTIS 时序语义分割 — ViT-L/16 distilled（SAT-493M，300M）	

## 目录结构

```bash
01_pastis/
├── train_pastis.py            # 训练脚本
├── visualize_results.py       # 可视化脚本
├── data/                      # PASTIS 数据（建议用迅雷下载）
│   └── PASTIS-R/
└── outputs/                   # 训练输出（训练时产生）
    ├── best_model.pth         # mIoU最佳模型权重
    ├── results.json           # 测试结果
    └── training.log           # 训练日志
```

## 环境配置

```bash
# 创建 conda 环境
conda create -n pastis python=3.10 -y
conda activate pastis

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm numpy torchgeo
```

## 数据准备

用迅雷下载 PASTIS 数据集（~29GB）：

```bash
https://zenodo.org/records/5012942/files/PASTIS.zip?download=1
```

下载后放到 `./data/PASTIS-R/` 目录，结构如下：

```shell
─data
│  └─PASTIS-R
│      ├─ANNOTATIONS
│      ├─DATA_S2
│      └─INSTANCE_ANNOTATIONS
```

## 训练

```bash
# 在项目根目录下运行
set HF_ENDPOINT=https://hf-mirror.com
python train_pastis.py
```

默认参数（`Config` 类中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 2 | 取决于 GPU 显存 |
| `EPOCHS` | 100 | 训练轮数，含早停 |
| `LR` | 1e-3 | 学习率 |
| `MAX_DATES` | 43 | 全时相 |
| `SEED` | 42 | 随机种子 |

## 输出

- `outputs/training.log` — 每轮 loss 和 mIoU
- `outputs/best_model.pth` — 最佳验证 mIoU 的模型权重
- `outputs/results.json` — 最终测试结果

## 可视化

```bash
# 在项目根目录下运行
python visualize_results.py
```

## 架构概览

```
输入: (B, 43, 10, 128, 128)
  │
  ├─ BandProjection: Conv2d(10→3, 1×1) + BN
  │
  ├─ (A) MultiLevelEncoder
  │   └─ DINOv3 ViT-L/16 sat493m (冻结)
  │       └─ forward_intermediates @ blocks [8, 16, 23]
  │       └─ 输出: 3层 × (B, 1024, 14, 14) + register tokens (B, 4, 1024)
  │
  ├─ (B) MultiScaleFPN
  │   └─ 28×28 / 14×14 / 7×7 金字塔
  │   └─ top-down 融合 → concat → (B, 256, 14, 14)
  │
  ├─ (C) RegLTAE
  │   └─ register → MLP → query bias
  │   └─ 43 时相 → 1 时相 (B, 196, 256)
  │
  └─ UpDecoder: 14×14 → 128×128 → 19 类
```

## 参考

- PASTIS: https://github.com/VSainteuf/pastis-gan
- DINOv3: https://github.com/facebookresearch/dinov3
- torchgeo: https://github.com/microsoft/torchgeo