# train_yolo_isic.py
"""
在 ISIC 2018 病灶检测数据集上训练 YOLOv8n 检测模型。

运行前需先执行 prepare_yolo_data.py 生成 YOLO 格式数据集。
训练完成后自动保存最佳权重到 yolo_weights/best.pt。
"""
import os, sys, argparse, json
from pathlib import Path

# ── 配置 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.expanduser("yolo_data_isic")
DATA_YAML = os.path.join(DEFAULT_DATA_DIR, "data.yaml")
OUTPUT_WEIGHTS = os.path.join(BASE_DIR, "yolo_weights", "best.pt")
EXPORT_DIR = os.path.join(BASE_DIR, "yolo_weights")

# 训练参数
DEFAULT_MODEL = "yolov8n.pt"       # YOLOv8 nano 预训练权重（6.3M param）
DEFAULT_EPOCHS = 100               # 最大训练轮数
DEFAULT_BS = 16                    # batch size
DEFAULT_IMSZ = 640                 # 输入分辨率
DEFAULT_PATIENCE = 20              # early stopping


def check_data_ready(data_yaml):
    """检查数据集是否已准备好"""
    if not os.path.exists(data_yaml):
        print(f"❌ 未找到 data.yaml: {data_yaml}")
        print("请先运行: python prepare_yolo_data.py")
        return False

    data_dir = os.path.dirname(data_yaml)
    for sub in ["images/train", "images/val"]:
        p = os.path.join(data_dir, sub)
        if not os.path.isdir(p):
            print(f"❌ 目录不存在: {p}")
            return False
        n = len([f for f in os.listdir(p) if f.endswith((".jpg", ".png"))])
        if n == 0:
            print(f"❌ {sub} 中没有图片文件")
            return False
        print(f"  {sub}: {n} 张图片 ✓")

    return True


def train():
    parser = argparse.ArgumentParser(description="在 ISIC 2018 上训练 YOLOv8n 检测模型")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"YOLO 模型版本（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--data", default=DATA_YAML,
                        help="data.yaml 路径（默认从 prepare_yolo_data.py 输出自动查找）")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"训练轮数（默认: {DEFAULT_EPOCHS}）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BS,
                        help=f"Batch size（默认: {DEFAULT_BS}）")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMSZ,
                        help=f"输入分辨率（默认: {DEFAULT_IMSZ}）")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help=f"Early stopping patience（默认: {DEFAULT_PATIENCE}）")
    parser.add_argument("--device", default="0",
                        help="训练设备: '0' (GPU 0), 'cpu', '0,1' (多卡) 等")
    args = parser.parse_args()

    # ── 检查数据集 ──
    print("=" * 50)
    print("ISIC 2018 YOLOv8n 训练")
    print("=" * 50)
    print("\n▶ 检查数据集...")
    if not check_data_ready(args.data):
        sys.exit(1)

    # ── 导入 ultralytics ──
    try:
        from ultralytics import YOLO
        print("  ultralytics 已导入 ✓")
    except ImportError:
        print("❌ 需要安装 ultralytics: pip install ultralytics")
        sys.exit(1)

    # ── 创建输出目录 ──
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # ── 加载模型 ──
    print(f"\n▶ 加载模型: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("  YOLO 会自动下载，请确保网络通畅")
        print("  或指定本地路径: --model /path/to/yolov8n.pt")
        sys.exit(1)

    model_info = model.info()
    print(f"  模型: {model.model.yaml.get('yolo_version', 'YOLOv8')} "
          f"({sum(p.numel() for p in model.parameters()):,} 参数)")

    # ── 训练 ──
    print(f"\n▶ 开始训练...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Patience: {args.patience}\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=os.path.join(BASE_DIR, "runs"),
        name="yolo_isic",
        exist_ok=True,
        workers=4,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        augment=False,           # ISIC 病灶检测不需要复杂增强
        hsv_h=0.015,             # 少量色相增强
        hsv_s=0.4,               # 饱和度增强
        hsv_v=0.4,               # 亮度增强
        degrees=10.0,            # 小角度旋转
        translate=0.1,           # 小位移
        scale=0.5,               # 缩放范围
        fliplr=0.5,              # 水平翻转
        mosaic=0.5,              # 50% mosaic 增强
        mixup=0.0,               # 不打乱混合
        copy_paste=0.0,          # 不复制粘贴
        verbose=True,
    )

    # ── 提取最佳指标 ──
    best_epoch = results.results_dict.get("best_epoch", -1)
    best_map50 = results.results_dict.get("metrics/mAP50(B)", -1)
    best_map = results.results_dict.get("metrics/mAP50-95(B)", -1)
    print(f"\n{'='*45}")
    print("训练完成！")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  mAP@50:      {best_map50:.4f}")
    print(f"  mAP@50-95:   {best_map:.4f}")
    print(f"{'='*45}")

    # ── 复制最佳权重到固定位置 ──
    # Ultralytics 将最佳权重保存在 runs/detect/yolo_isic/weights/best.pt
    best_pt = os.path.join(BASE_DIR, "runs", "detect", "yolo_isic", "weights", "best.pt")
    if os.path.exists(best_pt):
        import shutil
        shutil.copy2(best_pt, OUTPUT_WEIGHTS)
        print(f"\n✅ 最佳权重已保存: {OUTPUT_WEIGHTS}")
        print(f"   文件大小: {os.path.getsize(OUTPUT_WEIGHTS) / 1024 / 1024:.1f} MB")

        # 同时保存训练报告
        report = {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "best_epoch": best_epoch,
            "mAP50": best_map50,
            "mAP50_95": best_map,
        }
        report_path = os.path.join(EXPORT_DIR, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📊 训练报告: {report_path}")
    else:
        print(f"⚠️ 未找到 best.pt: {best_pt}")

    print(f"\n下一步: python compare_sam3_vs_lora_yolo.py")


if __name__ == "__main__":
    train()
