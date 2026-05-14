# prepare_yolo_data.py
"""
将 ISIC 2018 parquet 数据集转换为 YOLO 训练格式。

输出目录（默认 yolo_data_isic）：
  images/train/     *.jpg
  images/val/       *.jpg
  labels/train/     *.txt  (YOLO格式: class x_center y_center w h 归一化)
  labels/val/       *.txt
  data.yaml
  stats.txt
"""
import os, glob, io, argparse
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_OUTPUT = os.path.expanduser("yolo_data_isic")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISIC_DIR = os.path.join(BASE_DIR, "isic2018")
VAL_RATIO = 0.1
PAD = 5


def main():
    parser = argparse.ArgumentParser(
        description="ISIC 2018 parquet → YOLO 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s                                      # 输出到 yolo_data_isic\n"
            "  %(prog)s --output /path/to/data               # 自定义输出路径\n"
            "  %(prog)s --val-ratio 0.2                      # 20% 验证集\n"
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"输出目录（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help=f"验证集比例（默认: {VAL_RATIO}）")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.abspath(args.output)

    # ── 目录准备 ──
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    train_img = os.path.join(OUTPUT_DIR, "images/train")
    train_lbl = os.path.join(OUTPUT_DIR, "labels/train")
    val_img = os.path.join(OUTPUT_DIR, "images/val")
    val_lbl = os.path.join(OUTPUT_DIR, "labels/val")

    # ── 获取 parquet 文件 ──
    parquet_files = sorted(glob.glob(os.path.join(ISIC_DIR, "data", "train-*.parquet")))
    if not parquet_files:
        print(f"❌ 未找到 parquet 文件: {ISIC_DIR}/data/train-*.parquet")
        return

    n_val = max(1, int(len(parquet_files) * args.val_ratio))
    train_files = parquet_files[:-n_val]
    val_files = parquet_files[-n_val:]
    print(f"找到 {len(parquet_files)} 个训练 parquet 文件")
    print(f"训练文件: {len(train_files)} 个, 验证文件: {len(val_files)} 个")
    print(f"输出目录: {OUTPUT_DIR}")

    # ── 处理函数 ──
    def process_file(parquet_path, img_dir, lbl_dir, prefix=""):
        """处理一个 parquet 文件：批量提取图片和 bbox"""
        df = pd.read_parquet(parquet_path)
        samples = empty = 0

        for i in range(len(df)):
            row = df.iloc[i]

            # 直接从 bytes 保存 JPEG（不解码再编码，大幅提速）
            jpeg_bytes = row["image"]["bytes"]
            # 用 PIL 只读尺寸信息，不 decode 像素
            img = Image.open(io.BytesIO(jpeg_bytes))
            img_w, img_h = img.size

            # 解码 mask 计算 bbox
            mask = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])), dtype=np.uint8)
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)
            if not np.any(mask):
                empty += 1
                continue

            ys, xs = np.where(mask)
            x1 = max(int(xs.min()) - PAD, 0)
            y1 = max(int(ys.min()) - PAD, 0)
            x2 = min(int(xs.max()) + PAD, img_w - 1)
            y2 = min(int(ys.max()) + PAD, img_h - 1)

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = max((x2 - x1) / img_w, 1e-6)
            bh = max((y2 - y1) / img_h, 1e-6)

            stem = f"{prefix}{i:05d}"
            # 直接写 JPEG 字节流，不要 PIL 重新压缩
            with open(os.path.join(img_dir, f"{stem}.jpg"), "wb") as f:
                f.write(jpeg_bytes)
            with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            samples += 1

        return samples, empty

    # ── 处理训练集 ──
    total_train = empty_train = 0
    print("\n▶ 处理训练集...")
    for idx, f in enumerate(train_files):
        fname = os.path.basename(f)
        prefix = f"{fname}_"
        samples, empty = process_file(f, train_img, train_lbl, prefix)
        total_train += samples
        empty_train += empty
        print(f"  [{idx+1}/{len(train_files)}] {fname}: {samples} 样本, {empty} 空mask")

    # ── 处理验证集 ──
    total_val = empty_val = 0
    print("\n▶ 处理验证集...")
    for idx, f in enumerate(val_files):
        fname = os.path.basename(f)
        prefix = f"{fname}_"
        samples, empty = process_file(f, val_img, val_lbl, prefix)
        total_val += samples
        empty_val += empty
        print(f"  [{idx+1}/{len(val_files)}] {fname}: {samples} 样本, {empty} 空mask")

    # ── 写 data.yaml ──
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    img_train_abs = os.path.join(OUTPUT_DIR, "images/train")
    img_val_abs = os.path.join(OUTPUT_DIR, "images/val")
    with open(yaml_path, "w") as f:
        f.write(f"# ISIC 2018 病灶检测数据集\n")
        f.write(f"train: {img_train_abs}\n")
        f.write(f"val: {img_val_abs}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['lesion']\n")
    print(f"\n  data.yaml -> {yaml_path}")

    # ── 统计 ──
    print(f"\n{'='*45}")
    print(f"✅ 完成！")
    print(f"  训练集: {total_train} 张")
    print(f"  验证集: {total_val} 张")
    print(f"  总计:   {total_train + total_val} 张")
    print(f"  空mask跳过: {empty_train + empty_val}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*45}")
    print(f"\n下一步: python {__file__} 完成后运行:")
    print(f"  python train_yolo_isic.py --data {OUTPUT_DIR}")
    print(f"  # 或复制到 chapter09 目录:")
    print(f"  cp -r {OUTPUT_DIR} {os.path.join(BASE_DIR, 'yolo_data')}")


if __name__ == "__main__":
    main()
