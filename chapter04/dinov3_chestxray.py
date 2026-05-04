import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
import timm

# ===== 配置 =====
DATA_ROOT = "chest_xray"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# ===== 1. 数据集 =====
class ChestXRayDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []; self.labels = []
        self.classes = ["NORMAL", "PNEUMONIA"]
        for label_idx, cls_name in enumerate(self.classes):
            for img_path in (Path(root) / cls_name).glob("*.jpeg"):
                self.samples.append(str(img_path))
                self.labels.append(label_idx)
        self.transform = transform or T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.samples[idx]).convert("RGB")), self.labels[idx]

# ===== 特征提取函数 =====
def extract_features(model, loader, is_vit=False):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            feat = model.forward_features(imgs)
            if is_vit:
                feat = feat[:, 0, :]          # ViT: CLS token
            else:
                feat = feat.mean([-2, -1])     # CNN: 全局平均池化
            feats.append(feat.cpu()); labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


if __name__ == '__main__':
    # ===== 2. 加载模型 =====
    print(f"Device: {DEVICE}")

    dinov3 = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True).to(DEVICE)
    dinov3.eval()
    for p in dinov3.parameters(): p.requires_grad = False

    resnet = timm.create_model("resnet50.a1_in1k", pretrained=True).to(DEVICE)
    resnet.eval()
    for p in resnet.parameters(): p.requires_grad = False

    # ===== 3. 加载数据 =====
    train_loader = DataLoader(ChestXRayDataset(os.path.join(DATA_ROOT,"train")),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(ChestXRayDataset(os.path.join(DATA_ROOT,"test")),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ===== 4. 提取特征 =====
    print("DINOv3 extracting...")
    d3_train, lbl_train = extract_features(dinov3, train_loader, is_vit=True)
    d3_test, lbl_test = extract_features(dinov3, test_loader, is_vit=True)
    print(f"  train {d3_train.shape}, test {d3_test.shape}")

    print("ResNet extracting...")
    r50_train, _ = extract_features(resnet, train_loader)
    r50_test, _ = extract_features(resnet, test_loader)
    print(f"  train {r50_train.shape}, test {r50_test.shape}")

    # ===== 5. t-SNE 可视化 =====
    np.random.seed(42)
    d3_tsne = torch.cat([d3_train[np.random.choice(len(d3_train), 300, replace=False)], d3_test])
    r50_tsne = torch.cat([r50_train[np.random.choice(len(r50_train), 300, replace=False)], r50_test])
    lbl_tsne = torch.cat([lbl_train[np.random.choice(len(lbl_train), 300, replace=False)], lbl_test])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (feats, title) in zip(axes, [
        (d3_tsne, "DINOv3 Features (t-SNE)"),
        (r50_tsne, "ResNet-50 ImageNet (t-SNE)")
    ]):
        emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(feats.numpy())
        colors = ["#4CAF50" if l==0 else "#F44336" for l in lbl_tsne[:len(emb)]]
        ax.scatter(emb[:,0], emb[:,1], c=colors, s=8, alpha=0.7)
        ax.set_title(title, fontsize=14); ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#4CAF50",label="NORMAL"),Patch(color="#F44336",label="PNEUMONIA")])
    plt.tight_layout()
    plt.savefig("tsne_comparison.png", dpi=200, bbox_inches="tight")
    print("\n✅ tsne_comparison.png")

    # ===== 6. 准确率对比 =====
    print("\n===== 准确率 =====")
    for name, train_f, test_f in [("DINOv3", d3_train, d3_test), ("ResNet-50", r50_train, r50_test)]:
        acc = accuracy_score(lbl_test.numpy(),
            LogisticRegression(max_iter=1000).fit(train_f.numpy(), lbl_train.numpy()).predict(test_f.numpy()))
        print(f"  {name:20s}: {acc*100:.1f}%")

    # ===== 7. 柱状图 =====
    d3_acc = accuracy_score(lbl_test.numpy(),
        LogisticRegression(max_iter=1000).fit(d3_train.numpy(), lbl_train.numpy()).predict(d3_test.numpy()))
    r50_acc = accuracy_score(lbl_test.numpy(),
        LogisticRegression(max_iter=1000).fit(r50_train.numpy(), lbl_train.numpy()).predict(r50_test.numpy()))
    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(["DINOv3", "ResNet-50"], [d3_acc*100, r50_acc*100], color=["#2196F3","#FF9800"], width=0.5)
    for b, v in zip(bars, [d3_acc*100, r50_acc*100]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)"); ax.set_ylim(0,105); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=200)
    print("✅ accuracy_comparison.png\n🎉 完成！")
