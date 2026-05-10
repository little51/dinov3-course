# compare_sam3_vs_lora_yolo.py
"""
对比：YOLOv8n检测 → bbox → 原始SAM3 vs LoRA SAM3
真实pipeline：不用GT mask，YOLO先找病灶，再交给SAM分割
"""
import os, glob, torch, numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel
from transformers import Sam3Model, Sam3Processor
from ultralytics import YOLO
import io

# ── 配置 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISIC_DIR = os.path.join(BASE_DIR, "isic2018")
MODEL_PATH = "jetjodh/sam3"
LORA_PATH = os.path.join(BASE_DIR, "output/isic_lora/epoch_1")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "yolo_weights/best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESOLUTION = 448
NUM_SAMPLES = 6
VAL_RATIO = 0.1  # 与训练保持一致

# ── RoPE Patch ──
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
        torch.arange(0, self.dim, 4, device=device)[:(self.dim // 4)].float() / self.dim))
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


def run_sam(model, processor, img, bbox, device):
    """用给定bbox跑SAM推理"""
    inputs = processor(
        images=[img], input_boxes=[[bbox]], input_boxes_labels=[[1]],
        size={"height": RESOLUTION, "width": RESOLUTION}, return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32 and k != 'input_boxes_labels':
            inputs[k] = v.half()
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.pred_logits[0].sigmoid()
    masks = outputs.pred_masks[0]
    best = scores.argmax()
    pred = torch.sigmoid(masks[best]).cpu().numpy()
    return pred, scores[best].item()


def compute_dice(p_bin, g_bin):
    inter = (p_bin * g_bin).sum()
    return (2. * inter) / (p_bin.sum() + g_bin.sum() + 1e-6)


def load_samples_from_parquet(parquet_files, num_samples=6):
    """从parquet文件加载验证样本"""
    samples = []
    for pf in parquet_files:
        if len(samples) >= num_samples:
            break
        df = pd.read_parquet(pf)
        # 每个文件取一个样本
        for i in range(min(len(df), 1)):
            if len(samples) >= num_samples:
                break
            row = df.iloc[i]
            
            # 解码图片
            img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
            
            # 解码掩码
            mask = np.array(Image.open(io.BytesIO(row['mask']['bytes'])), dtype=np.uint8)
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8) * 255
            
            samples.append({
                'img': img,
                'mask': mask,
                'filename': f"{os.path.basename(pf).replace('.parquet', '')}_{i}"
            })
    
    return samples


# ── 主流程 ──
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 加载YOLO
print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)
print("  YOLO loaded ✓")

# 加载SAM模型
processor = Sam3Processor.from_pretrained(MODEL_PATH)
print("Loading SAM models...")
base = Sam3Model.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
lora = PeftModel.from_pretrained(
    Sam3Model.from_pretrained(MODEL_PATH, torch_dtype=torch.float16), LORA_PATH
).to(device)
base.eval(); lora.eval()
print("  SAM models loaded ✓")

# 从parquet加载验证集样本
parquet_files = sorted(glob.glob(os.path.join(ISIC_DIR, "data", "*.parquet")))
n_val = max(1, int(len(parquet_files) * VAL_RATIO))
val_parquets = parquet_files[-n_val:]  # 使用验证集
print(f"加载 {len(val_parquets)} 个验证文件")

samples = load_samples_from_parquet(val_parquets, num_samples=NUM_SAMPLES)
print(f"加载了 {len(samples)} 个测试样本")

print(f"\nEvaluating {len(samples)} samples: YOLO detect → SAM segment\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_dice_orig = []
all_dice_lora = []

for idx, sample in enumerate(samples):
    img = sample['img']
    mask_np = sample['mask']
    fid = sample['filename']
    gt_bin = (mask_np > 0).astype(np.uint8)

    # ── YOLO 检测 ──
    results = yolo(img, conf=0.25, iou=0.5, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        # 取置信度最高的检测框
        best_box = boxes[0]  # YOLO默认按置信度排序
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        conf = best_box.conf[0].item()
        # 转为整数bbox
        yolo_bbox = [int(x1), int(y1), int(x2), int(y2)]
        detected = True
    else:
        # YOLO没检测到 → 用整图框兜底
        W, H = img.size
        yolo_bbox = [0, 0, W, H]
        conf = 0.0
        detected = False

    # ── SAM 推理（都用YOLO的bbox）──
    pred_orig, sc_orig = run_sam(base, processor, img, yolo_bbox, device)
    pred_lora, sc_lora = run_sam(lora, processor, img, yolo_bbox, device)
    Hp, Wp = pred_orig.shape

    # GT mask 缩放到预测尺寸
    gt_r = np.array(Image.fromarray(mask_np).resize((Wp, Hp), Image.NEAREST)) > 0
    gt_bin_r = gt_r.astype(np.uint8)

    d_orig = compute_dice((pred_orig > 0.5).astype(np.uint8), gt_bin_r)
    d_lora = compute_dice((pred_lora > 0.5).astype(np.uint8), gt_bin_r)

    all_dice_orig.append(d_orig)
    all_dice_lora.append(d_lora)

    # ── 可视化：4栏 ──
    img_r = np.array(img.resize((Wp, Hp), Image.LANCZOS))

    # YOLO框坐标缩放到显示尺寸
    orig_w, orig_h = img.size
    scale_x = Wp / orig_w
    scale_y = Hp / orig_h
    vis_bbox = [
        int(yolo_bbox[0] * scale_x),
        int(yolo_bbox[1] * scale_y),
        int(yolo_bbox[2] * scale_x),
        int(yolo_bbox[3] * scale_y),
    ]

    def overlay(img_np, mask_bin, color, alpha=0.35):
        o = img_np.copy().astype(float)
        for c in range(3):
            o[:,:,c] = np.where(mask_bin > 0,
                                (1-alpha)*img_np[:,:,c] + alpha*color[c],
                                img_np[:,:,c])
        return o.astype(np.uint8)

    gt_vis = overlay(img_r, gt_bin_r, (50, 200, 50), 0.3)
    orig_vis = overlay(img_r, (pred_orig > 0.5).astype(np.uint8), (255, 50, 50), 0.35)
    lora_vis = overlay(img_r, (pred_lora > 0.5).astype(np.uint8), (50, 100, 255), 0.35)

    # 绘制YOLO框
    def draw_bbox(img_np, bbox, color=(255, 255, 0), label=""):
        from PIL import ImageDraw
        pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil)
        draw.rectangle(bbox, outline=color, width=4)
        if label:
            # 标签在框上方，确保不超出图片
            ty = max(0, bbox[1] - 18)
            draw.text((bbox[0]+3, ty), label, fill=(0, 0, 0), stroke_width=1, stroke_fill=(255,255,255))
        return np.array(pil)

    status = f"YOLO conf={conf:.2f}" if detected else "YOLO未检测到(用全图)"
    yolo_vis = draw_bbox(img_r.copy(), vis_bbox, (255, 255, 0), status)

    gap, col_w, title_h = 10, Wp, 28
    total_w = col_w * 4 + gap * 3
    canvas = np.ones((title_h + Hp, total_w, 3), dtype=np.uint8) * 255
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 12)
    except:
        font = ImageFont.load_default()

    panels = [
        (gt_vis, "GT"),
        (yolo_vis, f"YOLO框 {status}"),
        (orig_vis, f"SAM3 D={d_orig:.3f}"),
        (lora_vis, f"LoRA D={d_lora:.3f}"),
    ]
    for i, (arr, txt) in enumerate(panels):
        x0 = i * (col_w + gap)
        draw.text((x0 + 3, 6), txt, fill=(0, 0, 0), font=font)
        pil_img.paste(Image.fromarray(arr), (x0, title_h))

    delta = d_lora - d_orig
    arrow = "▲" if delta > 0 else "▼"
    summary = f"YOLO→SAM | Dice: {d_orig:.3f} → {d_lora:.3f}  {arrow}{abs(delta):.4f}"
    big = Image.new("RGB", (total_w, title_h + Hp + 22), (255, 255, 255))
    big.paste(pil_img, (0, 0))
    draw2 = ImageDraw.Draw(big)
    draw2.text((10, title_h + Hp + 5), summary, fill=(0, 0, 0), font=font)

    out_path = os.path.join(OUTPUT_DIR, f"yolo_sam_{fid}_D{d_orig:.3f}to{d_lora:.3f}.png")
    big.save(out_path)
    print(f"[{idx+1}] YOLO conf={conf:.2f} detected={detected} | Dice: {d_orig:.3f} → {d_lora:.3f} (Δ={delta:+.4f})")

# ── 汇总 ──
print(f"\n{'='*55}")
print(f"YOLOv8n → SAM3 对比汇总（{len(samples)}张）")
print(f"{'='*55}")
print(f"原始SAM3  平均Dice: {np.mean(all_dice_orig):.4f}")
print(f"LoRA微调  平均Dice: {np.mean(all_dice_lora):.4f}")
print(f"提升: {np.mean(all_dice_lora) - np.mean(all_dice_orig):+.4f}")
print(f"\n结果保存: {OUTPUT_DIR}/")
print(f"YOLO模型: {YOLO_WEIGHTS}")