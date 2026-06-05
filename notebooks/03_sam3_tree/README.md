# SAM3 树计数 — Windows 安装与运行

## 环境配置

```powershell
conda create -n sam3_tree python=3.12 -y
conda activate sam3_tree
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install sam3 openai-clip opencv-python pillow psutil
```

## 下载模型

**sam3.pt**（~2.5GB）放脚本同目录：

https://hf-mirror.com/jetjodh/sam3/resolve/main/sam3.pt?download=true

## 虚拟 triton 模块（Windows 必备）

`sam3` 包依赖 `triton`，但 **triton 不支持 Windows**。树冠文本推理不需要 triton 的功能，用虚拟模块绕过即可。

把 `triton/` 文件夹放到 `sam3_tree.py` 同级，脚本会自动加载。

## 下载BPE 词表

https://hf-mirror.com/OpenGVLab/ViCLIP-B-16-hf/resolve/main/bpe_simple_vocab_16e6.txt.gz?download=true

## 文件结构

```
03_sam3_tree/
├── sam3_tree.py        # 主脚本（文本提示 → 树掩膜）
├── bpe_simple_vocab_16e6.txt.gz  # OpenAI CLIP 的 BPE 词表，用于文字→token 的编码
├── triton/             # 虚拟 triton 模块
│   ├── __init__.py
│   └── language.py
├── sam3.pt             # 模型权重
├── images/             # 测试图片
│   ├── sample_1.jpg
│   └── ...
└── output/             # 输出可视化
    ├── inst_sample_1.jpg
    └── ...
```

## 运行

```powershell
python sam3_tree.py
```

输出：每张图检测到的树冠数打印在终端，可视化图保存到 `output/`。


