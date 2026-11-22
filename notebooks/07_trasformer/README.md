# DINOv3做CIFAR-10分类

本例加载预训练DINOv3 ViT-S/16，冻结骨干，只训练一个10分类线性头。

流程：

```text
CIFAR-10图片（32×32）
        ↓ Resize到224×224
DINOv3骨干（冻结）
        ↓ CLS token，384维
Linear(384, 10)（训练）
        ↓ 10类预测
```

## 环境

建议环境名为`dinov3`：

```bash
conda create -n dinov3 python=3.11 -y
conda activate dinov3
pip install torch torchvision transformers
```

程序自动使用CUDA；没有GPU时退回CPU。

## 运行

```bash
conda activate dinov3
python train_dinov3_cifar10.py
```

第一次运行会下载：

- CIFAR-10到当前目录的`data/`
- `Fluxmire/dinov3-vits16-pretrain-lvd1689m`权重到Hugging Face缓存

脚本已设置`HF_ENDPOINT=https://hf-mirror.com`，模型走hf-mirror下载，不需要token。

默认训练5轮、batch size为64。每一轮结束都会输出测试集准确率：

```text
epoch 1: test accuracy = xx.xx%
...
epoch 5: test accuracy = xx.xx%
```

这是10000张测试图上的accuracy，不是训练集结果。

## 代码做了什么

`AutoModel.from_pretrained()`加载DINOv3，不带任务分类器。随后：

```python
for p in backbone.parameters():
    p.requires_grad = False
```

冻结骨干，所以不保存其梯度，也不更新权重。每张图经过ViT后，取第0个CLS token作为整张图的384维特征：

```python
cls_feature = backbone(pixel_values=images).last_hidden_state[:, 0]
```

最后只训练`Linear(384, 10)`。`torch.no_grad()`包住骨干前向，确保反向传播只更新分类头。

## 常调参数

```python
BATCH_SIZE = 64  # 显存不足改为32或16
EPOCHS = 5       # 想训练更久可改为10
```

不要一开始就解冻DINOv3。本例的目的就是测试固定预训练特征加线性头的迁移效果；跑通后再考虑解冻最后几个Transformer block。

## 常见问题

- 下载慢或失败：先检查`curl -I https://hf-mirror.com`；已下载的模型会复用缓存。
- 显存不足：减小`BATCH_SIZE`。
- GPU没启用：运行`python -c "import torch; print(torch.cuda.is_available())"`，应输出`True`。
- CIFAR-10原图很小，Resize到224×224不会产生新细节，因此结果不是DINOv3在高分辨率自然图像上的真实上限。

## Kvasir-SEG医学分割

`train_dinov3_kvasir.py`使用Transformers加载DINOv3 ViT-L/16，在Kvasir-SEG上训练息肉分割模型。

数据目录应为：

```text
data/Kvasir-SEG/
├── images/  # 内镜图像
└── masks/   # 与图像同名的二值掩码
```

运行：

```bash
conda activate dinov3
cd ~/dinov3-course/notebooks/07_trasformer
python train_dinov3_kvasir.py --data data/Kvasir-SEG
```

默认配置：输入448×448、batch size为12、最多200轮、验证Dice连续10轮不提高则早停。程序自动从hf-mirror下载`kryveil/dinov3-vitl16-pretrain-lvd1689m`权重。

模型会同时读取ViT-L的第6、12、18、24层特征，经过投影、融合、ASPP和原图细节分支后输出448×448掩码。骨干前20层冻结，仅解冻最后4层；编码器学习率为2e-5，分割解码器学习率为1e-3。

每轮输出训练损失与验证Dice：

```text
epoch 1: loss = x.xxxx, val dice = x.xxxx
```

Kvasir-SEG共1000张图，脚本按固定种子划分800张训练、200张验证。实测最佳验证Dice为0.9105，第9轮达到；第19轮因连续10轮未提升而早停。

若显存不足，可先将`--batch-size 12`改成`--batch-size 6`。
