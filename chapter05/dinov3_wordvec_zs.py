import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
import timm
import numpy as np
from sklearn.metrics import accuracy_score
import gensim.downloader as api

# ===== 配置 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10
TRAIN_CLASSES = 80
TEST_CLASSES = 20

# ===== 1. 数据 =====
transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
full_train = CIFAR100(root="./data", train=True, download=True, transform=transform)
full_test = CIFAR100(root="./data", train=False, download=True, transform=transform)
all_classes = full_train.classes          # 100 个类名

# 80/20 拆分
np.random.seed(42)
perm = np.random.permutation(100)
train_cls_idx = perm[:80]
test_cls_idx = perm[80:]

train_idx = [i for i, (_, lbl) in enumerate(full_train) if lbl in set(train_cls_idx)]
test_idx = [i for i, (_, lbl) in enumerate(full_test) if lbl in set(test_cls_idx)]
train_loader = DataLoader(Subset(full_train, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(Subset(full_test, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_label_map = {old: new for new, old in enumerate(train_cls_idx)}
test_label_map = {old: new for new, old in enumerate(test_cls_idx)}

print(f"训练: {len(train_idx)} 张, 类: {[all_classes[i] for i in train_cls_idx[:5]]}...")
print(f"测试: {len(test_idx)} 张, 类: {[all_classes[i] for i in test_cls_idx]}")

# ===== 2. 模型 =====
print("\n加载 DINOv3...")
dinov3 = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True).to(DEVICE)
dinov3.eval()
for p in dinov3.parameters(): p.requires_grad = False

print("加载 GloVe 词向量（首次会自动下载，~170MB）...")
# glove-wiki-gigaword-50: 50维, 40万词, 轻量
glove = api.load("glove-wiki-gigaword-50")
VEC_DIM = 50    # GloVe 维度
print(f"  词向量维度: {VEC_DIM}, 词表大小: {len(glove.key_to_index)}")

# 投影头：DINOv3 384维 → GloVe 50维
proj = nn.Linear(384, VEC_DIM).to(DEVICE)
optimizer = torch.optim.Adam(proj.parameters(), lr=LR)

# ===== 3. 类名 → 词向量 =====
def class_to_vec(class_name, wv):
    """类名转词向量：下划线分隔，取平均"""
    words = class_name.replace("_", " ").split()
    vecs = [wv[w] for w in words if w in wv]
    if not vecs:
        return np.zeros(wv.vector_size)
    return np.mean(vecs, axis=0)

train_word_vecs = torch.tensor(
    np.array([class_to_vec(all_classes[i], glove) for i in train_cls_idx]),
    dtype=torch.float32
).to(DEVICE)   # (80, 50)

test_word_vecs = torch.tensor(
    np.array([class_to_vec(all_classes[i], glove) for i in test_cls_idx]),
    dtype=torch.float32
).to(DEVICE)   # (20, 50)

# ===== 4. 训练投影头 =====
print("\n训练投影头...")
proj.train()
for epoch in range(EPOCHS):
    losses = []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        labels_remap = torch.tensor([train_label_map[l.item()] for l in labels], device=DEVICE)

        with torch.no_grad():
            feat = F.normalize(dinov3.forward_features(imgs)[:, 0, :], dim=-1)  # (B, 384)
        proj_feat = F.normalize(proj(feat), dim=-1)                             # (B, 50)
        logits = proj_feat @ F.normalize(train_word_vecs, dim=-1).T             # (B, 80)

        loss = F.cross_entropy(logits, labels_remap)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"  Epoch {epoch+1:2d}: loss={np.mean(losses):.4f}")

# ===== 5. 零样本测试 =====
print("\n零样本测试（20个未见过的类）...")
proj.eval()
all_preds, all_gts = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        feat = F.normalize(dinov3.forward_features(imgs)[:, 0, :], dim=-1)
        proj_feat = F.normalize(proj(feat), dim=-1)
        logits = proj_feat @ F.normalize(test_word_vecs, dim=-1).T     # (B, 20)
        preds = logits.argmax(dim=1).cpu()
        true_labels = torch.tensor([test_label_map[l.item()] for l in labels])
        all_preds.extend(preds.numpy())
        all_gts.extend(true_labels.numpy())

acc = accuracy_score(all_gts, all_preds)
print(f"\n🎯 DINOv3 + 投影 → GloVe 词向量空间（零样本 20 类）: {acc*100:.1f}%")

# ===== 6. 按类看结果 =====
print("\n各类准确率:")
test_classes = [all_classes[i] for i in test_cls_idx]
for cls_name, cls_old_idx in zip(test_classes, test_cls_idx):
    cls_new_idx = test_label_map[cls_old_idx]
    cls_mask = np.array(all_gts) == cls_new_idx
    if cls_mask.sum() > 0:
        cls_acc = accuracy_score(np.array(all_gts)[cls_mask], np.array(all_preds)[cls_mask])
        print(f"  {cls_name:20s}: {cls_acc*100:.0f}%")

print("\n✅ 完成！")
