"""
DINOv3.sat493m + DPT Head — WHU Building 测试集评估与推理可视化
用法: python3 eval_dinov3_seg.py
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

# ── 路径（基于脚本所在目录自动定位） ──
BASE_DIR = Path(__file__).resolve().parent.parent          # chapter12/
CKPT = str(BASE_DIR / 'dinov3_seg_whu' / 'best_model.pth')
DATA_ROOT = str(BASE_DIR / 'whu_building')
SAVE_DIR = str(BASE_DIR / 'eval_results')
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# ── 加载 checkpoint ──
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
print(f'加载 checkpoint: epoch={ckpt["epoch"]}, best_mIoU={ckpt["best_mIoU"]:.4f}')
args = ckpt['args']

# ── 构建模型 ──
import timm
from timm.layers import Mlp

class DPTHead(torch.nn.Module):
    def __init__(self, embed_dim=1024, num_classes=2, fusion_dim=256):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.read_ops = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(embed_dim, fusion_dim, 1), torch.nn.GELU())
            for _ in range(4)
        ])
        self.refine = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1), torch.nn.GELU(),
                torch.nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1), torch.nn.GELU(),
            ) for _ in range(3)
        ])
        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(fusion_dim, num_classes, 1),
        )
        self.upsample2x = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, features):
        B = features[0].shape[0]
        grid_size = int((features[0].shape[1] - 5) ** 0.5)
        tokens = []
        for feat in features:
            patch = feat[:, 1:1+grid_size*grid_size, :]
            patch = patch.reshape(B, grid_size, grid_size, -1).permute(0, 3, 1, 2)
            patch = self.read_ops[len(tokens)](patch)
            tokens.append(patch)

        x = tokens[3]
        x = self.upsample2x(x)
        t2 = self.upsample2x(tokens[2])
        x = torch.cat([x, t2], dim=1)
        x = self.refine[2](x)
        x = self.upsample2x(x)
        t1 = self.upsample2x(self.upsample2x(tokens[1]))
        x = torch.cat([x, t1], dim=1)
        x = self.refine[1](x)
        x = self.upsample2x(x)
        t0 = self.upsample2x(self.upsample2x(self.upsample2x(tokens[0])))
        x = torch.cat([x, t0], dim=1)
        x = self.refine[0](x)
        x = self.output_conv(x)
        x = self.upsample2x(x)
        return x

class DINOSegModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.embed_dim = 1024
        self.head = DPTHead(self.embed_dim, num_classes)
        self.hook_block_ids = [5, 11, 17, 23]
        self.features = {}
        self.hook_handles = []
        for bid in self.hook_block_ids:
            def make_hook(name):
                def hook(_, __, output):
                    self.features[name] = output
                return hook
            handle = self.backbone.blocks[bid].register_forward_hook(make_hook(f'block_{bid}'))
            self.hook_handles.append(handle)

    def forward(self, x):
        self.features = {}
        _ = self.backbone(x)
        feat_list = [self.features[f'block_{bid}'] for bid in self.hook_block_ids]
        out = self.head(feat_list)
        return out

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.backbone.eval()
        return self

print('构建模型...')
model = DINOSegModel(num_classes=2)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f'模型参数: {sum(p.numel() for p in model.parameters()):,} (可训练: {sum(p.numel() for p in model.head.parameters()):,})')

# ── 标准化参数 ──
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

# ── 测试集 ──
test_img_dir = Path(DATA_ROOT) / 'test' / 'image'
test_lbl_dir = Path(DATA_ROOT) / 'test' / 'label'
img_paths = sorted(test_img_dir.glob('*'))
lbl_paths = sorted(test_lbl_dir.glob('*'))
print(f'测试集: {len(img_paths)} 张图片')

# ── 评估 ──
print('\n评估测试集 (224×224 sliding)...')
model.eval()

ious = []       # per-class IoU
accs = []       # per-image accuracy
precisions = [] # per-image precision (building class)
recalls = []    # per-image recall (building class)
f1s = []        # per-image F1 (building class)
dices = []      # per-image Dice (building class)
all_loss = []   # per-image combined loss

with torch.no_grad():
    for i, (img_p, lbl_p) in enumerate(zip(img_paths, lbl_paths)):
        if i % 200 == 0:
            print(f'  [{i}/{len(img_paths)}]')

        # 加载
        img = Image.open(img_p).convert('RGB')
        label = Image.open(lbl_p).convert('L')

        # 缩放到 224
        img_224 = img.resize((224, 224), Image.BILINEAR)
        label_224 = np.array(label.resize((224, 224), Image.NEAREST))
        label_t = torch.from_numpy(label_224).long().to(DEVICE)
        label_bin = (label_t > 0).long()

        # 预测
        img_t = torch.from_numpy(np.array(img_224)).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        img_t = (img_t - mean) / std

        logits = model(img_t)                       # [1, 2, 224, 224]
        pred = logits.argmax(dim=1).squeeze(0)      # [224, 224]

        # 计算损失 (for reference)
        ce_loss = F.cross_entropy(logits, label_bin.unsqueeze(0))
        dice_loss = 1 - (2 * ((F.softmax(logits, dim=1)[:, 1] * label_bin.float()).sum() + 1e-6) /
                         (F.softmax(logits, dim=1)[:, 1].sum() + label_bin.float().sum() + 1e-6))
        all_loss.append((ce_loss + dice_loss).item())

        # 逐类 IoU
        for c in [0, 1]:
            inter = ((pred == c) & (label_bin == c)).sum().item()
            union = ((pred == c) | (label_bin == c)).sum().item()
            ious.append((inter / union) if union > 0 else (1.0 if inter > 0 else 0.0))

        # Acc
        acc = (pred == label_bin).sum().item() / (224 * 224)
        accs.append(acc)

        # Building class metrics
        tp = ((pred == 1) & (label_bin == 1)).sum().item()
        fp = ((pred == 1) & (label_bin == 0)).sum().item()
        fn = ((pred == 0) & (label_bin == 1)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        # Dice (building class)
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        dices.append(dice)

# ── 汇总 ──
ious = np.array(ious).reshape(-1, 2)  # [N, 2]
miou_bg = ious[:, 0].mean()
miou_building = ious[:, 1].mean()
miou = (miou_bg + miou_building) / 2

results = {
    'num_samples': len(img_paths),
    'mIoU': round(float(miou), 4),
    'Bg_IoU': round(float(miou_bg), 4),
    'Building_IoU': round(float(miou_building), 4),
    'Accuracy': round(float(np.mean(accs)), 4),
    'Precision (building)': round(float(np.mean(precisions)), 4),
    'Recall (building)': round(float(np.mean(recalls)), 4),
    'F1 (building)': round(float(np.mean(f1s)), 4),
    'Dice (building)': round(float(np.mean(dices)), 4),
    'Avg_Loss': round(float(np.mean(all_loss)), 4),
}

print('\n═══ 测试集评估结果 (224×224) ═══')
print(f'  样本数:  {results["num_samples"]}')
print(f'  mIoU:     {results["mIoU"]:.4f}')
print(f'  背景 IoU: {results["Bg_IoU"]:.4f}')
print(f'  建筑 IoU: {results["Building_IoU"]:.4f}')
print(f'  准确率:   {results["Accuracy"]:.4f}')
print(f'  精确率:   {results["Precision (building)"]:.4f}')
print(f'  召回率:   {results["Recall (building)"]:.4f}')
print(f'  F1:       {results["F1 (building)"]:.4f}')
print(f'  Dice:     {results["Dice (building)"]:.4f}')

with open(f'{SAVE_DIR}/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 可视化 ──
print('生成推理可视化（筛选有建筑的图）...')
# 只选有建筑标签的图（避免空标签图显示全黑）
building_indices = []
for i, (img_p, lbl_p) in enumerate(zip(img_paths, lbl_paths)):
    lbl_arr = np.array(Image.open(lbl_p).convert('L').resize((224, 224), Image.NEAREST))
    if (lbl_arr > 0).sum() > 500:  # 至少 500 个建筑像素（~1%）
        building_indices.append(i)
random.seed(42)
sample_indices = random.sample(building_indices, min(8, len(building_indices)))
print(f'  有建筑的图: {len(building_indices)}/{len(img_paths)}, 从中抽取 {len(sample_indices)} 张')

def create_overlay(img_np, pred_mask, gt_mask, alpha=0.5):
    """创建 RGB 叠加图: 绿=TP, 红=FP, 蓝=FN, 白=TN(building)"""
    overlay = np.stack([img_np]*3, axis=-1) if img_np.ndim == 2 else img_np.copy()
    h, w = overlay.shape[:2]

    # Pred: 绿框/蓝框
    gt_bool = gt_mask > 0
    pred_bool = pred_mask > 0

    # TP (true positive): 预测正确且是建筑 → 半透明绿
    tp = gt_bool & pred_bool
    # FP (false positive): 预测是建筑但背景 → 红
    fp = ~gt_bool & pred_bool
    # FN (false negative): 实际是建筑但预测背景 → 蓝
    fn = gt_bool & ~pred_bool

    overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 200, 0]) * alpha
    overlay[fp] = overlay[fp] * (1 - alpha) + np.array([200, 0, 0]) * alpha
    overlay[fn] = overlay[fn] * (1 - alpha) + np.array([0, 0, 200]) * alpha

    return (overlay * 255).clip(0, 255).astype(np.uint8)

for idx, i in enumerate(sample_indices):
    img_p = img_paths[i]
    lbl_p = lbl_paths[i]

    # 原图 512×512
    img_orig = np.array(Image.open(img_p).convert('RGB')) / 255.0

    # 224 版本用于预测
    img_224 = Image.open(img_p).convert('RGB').resize((224, 224), Image.BILINEAR)
    img_224_np = np.array(img_224)
    label_224_np = np.array(Image.open(lbl_p).convert('L').resize((224, 224), Image.NEAREST))
    gt_bin = (label_224_np > 0).astype(np.uint8)

    # 预测
    img_t = torch.from_numpy(img_224_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    img_t = (img_t - mean) / std

    with torch.no_grad():
        logits = model(img_t)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # 叠加图
    overlay = create_overlay(img_224_np / 255.0, pred, gt_bin)

    # 拼接: 原图 | 标签 | 预测 | 叠加
    def to_rgb(arr):
        if arr.ndim == 2:
            return np.stack([arr * 255]*3, axis=-1).astype(np.uint8)
        return arr

    # 构建 2×2 网格
    def label_to_vis(mask):
        return np.stack([mask * 255]*3, axis=-1).astype(np.uint8)

    row1 = np.concatenate([img_224_np, label_to_vis(gt_bin)], axis=1)
    row2 = np.concatenate([label_to_vis(pred), overlay], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)

    # 添加标签文字
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    draw.text((5, 3), "Input", fill=(255,255,255), font=font)
    draw.text((229, 3), "Ground Truth", fill=(255,255,255), font=font)
    draw.text((5, 227), "Prediction", fill=(255,255,255), font=font)
    draw.text((229, 227), "Overlay (G=TP R=FP B=FN)", fill=(255,255,255), font=font)

    save_path = f'{SAVE_DIR}/sample_{i}.png'
    canvas_pil.save(save_path)
    print(f'  [{idx+1}/8] sample_{i}.png saved')

print(f'\n所有结果保存至: {SAVE_DIR}')
print('Done!')
