import os, sys

fake_triton_dir =  "triton"
if os.path.isdir(fake_triton_dir):
    sys.path.insert(0, "./") 

import torch, cv2, numpy as np, PIL.Image as Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


SAM3_CKPT = "sam3.pt"
IMG_DIR    = "images"
OUT_DIR    = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# 加载模型
model = build_sam3_image_model(
    bpe_path="bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path=SAM3_CKPT,
    enable_segmentation=True,
    enable_inst_interactivity=False,
)
processor = Sam3Processor(model, resolution=1008, confidence_threshold=0.3)

# 处理每张图
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    fpath = os.path.join(IMG_DIR, fname)
    img_bgr = cv2.imread(fpath)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    state = processor.set_image(img_pil)
    state = processor.set_text_prompt("tree", state)

    masks = state.get("masks")
    if masks is None or len(masks) == 0:
        print(f"{fname}: 0 棵树")
        continue

    n = len(masks)
    print(f"{fname}: {n} 棵树")

    # 可视化
    vis = img_bgr.copy()
    for i in range(n):
        mask = masks[i].cpu().numpy().squeeze()
        np.random.seed(i)
        color = tuple(int(np.random.randint(50, 255)) for _ in range(3))
        mask_3c = np.stack([mask, mask, mask], axis=-1)
        vis = np.where(mask_3c, (vis * 0.5 + np.array(color, dtype=np.float32) * 0.5).astype(np.uint8), vis)

    cv2.imwrite(os.path.join(OUT_DIR, f"inst_{fname}"), vis)

print("Done!")
