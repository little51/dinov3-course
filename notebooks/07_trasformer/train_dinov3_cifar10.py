"""CIFAR-10 分类：加载冻结的DINOv3 Transformer骨干，只训练线性分类头。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Hugging Face 国内镜像，须在导入transformers前设置

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Fluxmire/dinov3-vits16-pretrain-lvd1689m"
BATCH_SIZE = 64
EPOCHS = 5

# CIFAR-10原图是32×32；DINOv3输入224×224，故先放大再按ImageNet统计量归一化。
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
train_set = datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         num_workers=4, pin_memory=True)

# AutoModel自动按模型配置加载DINOv3 ViT；不加载语言模型，也不使用预置分类器。
backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False

# 取CLS token作为整张图的表示，接一个10分类线性层。
head = nn.Linear(backbone.config.hidden_size, 10).to(DEVICE)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    head.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            cls_feature = backbone(pixel_values=images).last_hidden_state[:, 0]
        loss = loss_fn(head(cls_feature), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    head.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            cls_feature = backbone(pixel_values=images.to(DEVICE)).last_hidden_state[:, 0]
            prediction = head(cls_feature).argmax(1).cpu()
            correct += (prediction == labels).sum().item()
            total += labels.numel()
    print(f"epoch {epoch + 1}: test accuracy = {correct / total:.2%}")
