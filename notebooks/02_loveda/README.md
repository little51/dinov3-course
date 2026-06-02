# 使用DINOv3对LoveDA进行语义分割

DINOv3 ViT-L sat493m **不 resize**，直接 1024×1024 原图输入，突破 224×224 的细粒度瓶颈。

## 环境

```bash
conda create -n loveda python=3.10 -y
conda activate loveda
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm torchgeo matplotlib numpy
```

## 数据

从飞浆 https://aistudio.baidu.com/datasetdetail/121200下载 Train.zip 和 Val.zip（**如果下载不了，可以联系我**），解压到 `./data/LoveDA/`：

```
data/LoveDA/
├── Train/  (Urban/ + Rural/)
└── Val/    (Urban/ + Rural/)
```

## 使用

```bash
set HF_ENDPOINT=https://hf-mirror.com
# 1、ASPP
# （1）训练
python train.py
# （2）可视化
python vis.py
# 2、简单 sum-fusion
# （1）训练
python train_simple.py
# （2）可视化
python vis_simple.py
```

## 参数

| 参数 | 默认 | 说明 |
|:---|:----|:----|
| BATCH_SIZE | 4 | 显存焦虑设 1 |
| EPOCHS | 30 | 训练轮数 |
| NUM_WORKERS | 0 | Windows 必须 0 |
| LR | 5e-4 | 学习率 |
