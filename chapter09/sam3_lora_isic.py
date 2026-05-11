# sam3_lora_isic.py — SAM3 LoRA 微调训练脚本
import os, time, torch, glob
import numpy as np
import pandas as pd
from PIL import Image
import io
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Sam3Model, Sam3Processor

# ═══════════════════════════════════
# 配置
# ═══════════════════════════════════
ISIC_DIR = r"isic2018"                   # ISIC 数据集路径
MODEL_NAME = "jetjodh/sam3"              # 或本地路径
OUTPUT_DIR = r"output/isic_lora"         # 输出目录
NUM_EPOCHS = 3                           # 训练轮数 
LR = 5e-4                                # 学习率
RESOLUTION = 448                         # 输入分辨率
VAL_RATIO = 0.1                          # 验证集比例
LORA_R = 8                               # LoRA rank
LORA_ALPHA = 16                          # LoRA alpha
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "out_proj"]

# RoPE Patch
from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding, Sam3ViTLayer

_original_rope_forward = Sam3ViTRotaryEmbedding.forward
def _dynamic_rope_forward(self):
    h = getattr(self, '_spatial_h', self.end_x)
    w = getattr(self, '_spatial_w', self.end_y)
    if isinstance(h, (list, tuple, torch.Size)): h = int(h[0])
    if isinstance(w, (list, tuple, torch.Size)): w = int(w[0])
    h, w = int(h), int(w)
    if h == self.end_x and w == self.end_y:
        return _original_rope_forward(self)
    device = self.rope_embeddings_cos.device
    freqs = 1.0 / (self.rope_theta ** (
        torch.arange(0, self.dim, 4, device=device)[:(self.dim // 4)].float() / self.dim
    ))
    flattened = torch.arange(h * w, device=device, dtype=torch.long)
    x_pos = (flattened % h).float() * self.scale
    y_pos = (flattened // h).float() * self.scale
    freqs_x = torch.outer(x_pos, freqs)
    freqs_y = torch.outer(y_pos, freqs)
    inv_freq = torch.cat([freqs_x, freqs_y], dim=-1)
    inv_freq = inv_freq.repeat_interleave(2, dim=-1)
    return inv_freq.cos(), inv_freq.sin()
Sam3ViTRotaryEmbedding.forward = _dynamic_rope_forward

_orig_layer_forward = Sam3ViTLayer.forward
def _patched_layer_forward(self, hidden_states, **kwargs):
    if self.window_size > 0:
        rotary_h = rotary_w = self.window_size
    else:
        rotary_h, rotary_w = hidden_states.shape[1], hidden_states.shape[2]
    if hasattr(self, 'rotary_emb'):
        self.rotary_emb._spatial_h = rotary_h
        self.rotary_emb._spatial_w = rotary_w
    return _orig_layer_forward(self, hidden_states, **kwargs)
Sam3ViTLayer.forward = _patched_layer_forward


# ═══════════════════════════════════
# 数据集类（直接从 parquet 读取）
# ═══════════════════════════════════
# 数据集类（直接从 parquet 读取，先调试数据格式）
# 数据集类（直接从 parquet 读取，处理 dict 格式）
# 数据集类（直接从 parquet 读取）
class ISICDataset(Dataset):
    def __init__(self, parquet_files):
        self.samples = []
        for pf in sorted(parquet_files):
            df = pd.read_parquet(pf)
            for i in range(len(df)):
                self.samples.append((pf, i))
        print(f"  {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pf, i = self.samples[idx]
        row = pd.read_parquet(pf).iloc[i]
        
        # 解码图片
        img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
        
        # 解码掩码
        mask = np.array(Image.open(io.BytesIO(row['mask']['bytes'])), dtype=np.uint8)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8) * 255

        # 从 GT mask 自动计算 bbox prompt
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            pad = 5
            bbox = [max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                    min(mask.shape[1], int(xs.max())+pad),
                    min(mask.shape[0], int(ys.max())+pad)]
        else:
            bbox = [0, 0, 10, 10]
        return img, bbox, [1], mask


# ═══════════════════════════════════
# 损失函数：BCE + Dice（top-5 加权）
# ═══════════════════════════════════
def compute_loss(outputs, gt_mask_np, dice_weight=5.0, smooth=1e-6):
    pred_masks = outputs.pred_masks[0].float()       # [200, H, W]
    scores = outputs.pred_logits[0].sigmoid()         # [200]
    Hp, Wp = pred_masks.shape[1:]

    gt_t = torch.from_numpy(gt_mask_np).float().to(pred_masks.device) / 255.0
    gt_r = torch.nn.functional.interpolate(
        gt_t.unsqueeze(0).unsqueeze(0),
        size=(Hp, Wp), mode="nearest"
    ).squeeze()

    probs = torch.sigmoid(pred_masks)
    inter = (probs * gt_r.unsqueeze(0)).sum(dim=(1, 2))
    total_pred = probs.sum(dim=(1, 2))
    total_gt = gt_r.sum()
    dice = (2.0 * inter + smooth) / (total_pred + total_gt + smooth)

    # 按 score×dice 选 top-5
    top_idx = (scores * dice).topk(min(5, len(scores))).indices
    total_loss = 0.0
    total_dice = 0.0
    weight_sum = 0.0

    for i in top_idx:
        w = max(scores[i].item(), 0.01)
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_masks[i], gt_r
        )
        d = dice[i]
        loss = loss_bce + dice_weight * (1.0 - d)
        total_loss += w * loss
        total_dice += w * d.item()
        weight_sum += w

    return total_loss / weight_sum, total_dice / weight_sum


# ═══════════════════════════════════
# 训练主函数
# ═══════════════════════════════════
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB")

    # 数据
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(ISIC_DIR, "data", "*.parquet")))
    n_val = max(1, int(len(parquet_files) * VAL_RATIO))
    print("训练集:")
    train_ds = ISICDataset(parquet_files[:-n_val])
    print("验证集:")
    val_ds = ISICDataset(parquet_files[-n_val:])

    # 模型 + LoRA
    processor = Sam3Processor.from_pretrained(MODEL_NAME)
    model = Sam3Model.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.1,
        target_modules=LORA_TARGETS, bias="none", task_type="FEATURE_EXTRACTION",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({trainable/total_p*100:.2f}%)")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    # Warmup（首次前向编译 CUDA kernel）
    print("[Warmup] ...")
    with torch.no_grad():
        w_img, w_bbox, w_bl, _ = train_ds[0]
        w_inputs = processor(images=[w_img], input_boxes=[[w_bbox]],
                             input_boxes_labels=[w_bl],
                             size={"height": RESOLUTION, "width": RESOLUTION},
                             return_tensors="pt")
        w_inputs = {k: v.half().to(device) if isinstance(v, torch.Tensor)
                    and v.dtype == torch.float32 and k != 'input_boxes_labels'
                    else v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in w_inputs.items()}
        _ = model(**w_inputs)
    print("[Warmup done]")

    # 训练循环
    print(f"\nTraining {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = total_dice = 0.0
        t0 = time.time()

        for step in range(len(train_ds)):
            img, bbox, bl, gt_mask = train_ds[step]
            inputs = processor(images=[img], input_boxes=[[bbox]],
                               input_boxes_labels=[bl],
                               size={"height": RESOLUTION, "width": RESOLUTION},
                               return_tensors="pt")
            inputs = {k: v.half().to(device) if isinstance(v, torch.Tensor)
                      and v.dtype == torch.float32 and k != 'input_boxes_labels'
                      else v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            outputs = model(**inputs)
            loss, dice = compute_loss(outputs, gt_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_dice += dice

            if (step + 1) % 100 == 0:
                print(f"  E{epoch+1} S{step+1}/{len(train_ds)} "
                      f"loss={loss.item():.4f} dice={dice:.4f}", flush=True)

        avg_loss = total_loss / len(train_ds)
        avg_dice = total_dice / len(train_ds)
        print(f">>> Epoch {epoch+1} | loss={avg_loss:.4f} | "
              f"dice={avg_dice:.4f} | {time.time()-t0:.1f}s")

        # 验证
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for v in range(len(val_ds)):
                img, bbox, bl, gt_mask = val_ds[v]
                inputs = processor(images=[img], input_boxes=[[bbox]],
                                   input_boxes_labels=[bl],
                                   size={"height": RESOLUTION, "width": RESOLUTION},
                                   return_tensors="pt")
                inputs = {k: v.half().to(device) if isinstance(v, torch.Tensor)
                          and v.dtype == torch.float32 and k != 'input_boxes_labels'
                          else v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                outputs = model(**inputs)
                _, dice = compute_loss(outputs, gt_mask)
                val_dice += dice
        print(f">>> Val dice={val_dice/len(val_ds):.4f}")

        # 保存 checkpoint
        save_path = f"{OUTPUT_DIR}/epoch_{epoch+1}"
        model.save_pretrained(save_path)
        print(f">>> Saved: {save_path}")

    print("Training complete!")


if __name__ == "__main__":
    train()