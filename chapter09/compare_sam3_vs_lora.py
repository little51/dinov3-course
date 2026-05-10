# compare_sam3_vs_lora.py
import os, glob, torch, numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import Sam3Model, Sam3Processor
import io

# 配置
ISIC_DIR = r"isic2018"
MODEL_NAME = "jetjodh/sam3"  
LORA_PATH = r"output/isic_lora/epoch_1"
OUTPUT_DIR = r"output/isic_compare"
RESOLUTION = 448
VAL_RATIO = 0.1  # 与训练保持一致

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

def run(model, processor, img, bbox, device):
    inputs = processor(images=[img], input_boxes=[[bbox]], input_boxes_labels=[[1]],
                       size={"height": RESOLUTION, "width": RESOLUTION}, return_tensors="pt")
    inputs = {k: v.half().to(device) if isinstance(v, torch.Tensor)
              and v.dtype == torch.float32 and k != 'input_boxes_labels'
              else v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.pred_logits[0].sigmoid()
    masks = outputs.pred_masks[0]
    best = scores.argmax()
    return torch.sigmoid(masks[best]).cpu().numpy(), scores[best].item()

def compute_dice(p, g):
    return (2. * (p * g).sum()) / (p.sum() + g.sum() + 1e-6)

def load_samples_from_parquet(parquet_files, num_samples=5):
    """从parquet文件加载验证样本"""
    samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # 每个文件取一个样本用于对比
        for i in range(min(len(df), 1)):  # 每个parquet取1个样本
            if len(samples) >= num_samples:
                return samples
            row = df.iloc[i]
            
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
            
            samples.append({
                'img': img,
                'mask': mask,
                'bbox': bbox,
                'filename': f"sample_{os.path.basename(pf)}_{i}"
            })
    
    return samples

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
processor = Sam3Processor.from_pretrained(MODEL_NAME)

# 加载验证样本（与训练代码一致）
parquet_files = sorted(glob.glob(os.path.join(ISIC_DIR, "data", "*.parquet")))
n_val = max(1, int(len(parquet_files) * VAL_RATIO))
val_parquets = parquet_files[-n_val:]  # 使用验证集
print(f"加载 {len(val_parquets)} 个验证文件")

samples = load_samples_from_parquet(val_parquets, num_samples=5)
print(f"加载了 {len(samples)} 个测试样本")

# 加载两个模型
print("Loading models...")
base = Sam3Model.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
lora_m = PeftModel.from_pretrained(
    Sam3Model.from_pretrained(MODEL_NAME, torch_dtype=torch.float16), LORA_PATH
).to(device)
base.eval(); lora_m.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 存储每个样本的delta
deltas = []

for idx, sample in enumerate(samples):
    img = sample['img']
    mask_np = sample['mask']
    bbox = sample['bbox']
    fid = sample['filename']
    
    gt_bin = (mask_np > 0).astype(np.uint8)
    # 获取原始预测和 LoRA 预测
    pred_r, sc_r = run(base, processor, img, bbox, device)
    pred_l, sc_l = run(lora_m, processor, img, bbox, device)
    
    Hp, Wp = pred_r.shape
    
    # 调整 GT 尺寸以匹配预测掩码
    gt_resized = np.array(Image.fromarray((gt_bin * 255).astype(np.uint8)).resize((Wp, Hp), Image.NEAREST)) > 0
    
    dr = compute_dice((pred_r > 0.5).astype(np.uint8), gt_resized.astype(np.uint8))
    dl = compute_dice((pred_l > 0.5).astype(np.uint8), gt_resized.astype(np.uint8))
    
    # 存储delta
    delta = dl - dr
    deltas.append(delta)
    
    # 调整图像尺寸
    img_resized = np.array(img.resize((Wp, Hp), Image.LANCZOS))
    
    def overlay(img_np, mask_bin, color, alpha=0.35):
        o = img_np.copy().astype(float)
        for c in range(3):
            o[:,:,c] = np.where(mask_bin > 0,
                                (1-alpha)*img_np[:,:,c] + alpha*color[c],
                                img_np[:,:,c])
        return o.astype(np.uint8)
    
    # 三栏：GT | 原始SAM3 | LoRA
    gap, col_w, title_h = 12, Wp, 28
    canvas = np.ones((title_h + Hp, col_w * 3 + gap * 2, 3), dtype=np.uint8) * 255
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    
    panels = [
        (overlay(img_resized, gt_resized.astype(np.uint8), (50, 200, 50), 0.3), f"GT"),
        (overlay(img_resized, (pred_r > 0.5).astype(np.uint8), (255, 50, 50), 0.35), f"原始 D={dr:.3f}"),
        (overlay(img_resized, (pred_l > 0.5).astype(np.uint8), (50, 100, 255), 0.35), f"LoRA D={dl:.3f}"),
    ]
    for i, (arr, txt) in enumerate(panels):
        x0 = i * (col_w + gap)
        draw.text((x0 + 5, 8), txt, fill=(0, 0, 0))
        pil_img.paste(Image.fromarray(arr), (x0, title_h))
    
    summary = f"Dice: {dr:.3f} → {dl:.3f}  {'▲' if delta > 0 else '▼'}{abs(delta):.4f}"
    big = Image.new("RGB", (col_w * 3 + gap * 2, title_h + Hp + 22), (255, 255, 255))
    big.paste(pil_img, (0, 0))
    draw2 = ImageDraw.Draw(big)
    draw2.text((10, title_h + Hp + 5), summary, fill=(0, 0, 0))
    
    big.save(os.path.join(OUTPUT_DIR, f"对比_{fid}_D{dr:.3f}to{dl:.3f}.png"))
    print(f"[{idx+1}] {fid}: Dice {dr:.3f} → {dl:.3f} (Δ={delta:+.4f})")

print(f"\n完成！结果保存在 {OUTPUT_DIR}/")
print(f"平均Dice提升: {np.mean(deltas):+.4f}")