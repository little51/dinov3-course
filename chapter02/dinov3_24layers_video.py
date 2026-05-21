#!/usr/bin/env python3
"""
DINOv3 ViT-L 24层语义层级可视化动画

对输入图像执行一次 DINOv3 ViT-L 前向传播，
提取全部 24 层的 patch 特征，PCA 降维到 RGB，
生成逐层递进的可视化视频。

用法:
  python dinov3_24layers_video.py [--image path/to/img.jpg] [--output out.mp4]

依赖: torch, timm, sklearn, matplotlib, pillow, ffmpeg
"""

import os, sys, argparse, urllib.request, warnings
import numpy as np
import torch
import timm
import matplotlib
matplotlib.use('Agg')
# 中文字体配置
import matplotlib.font_manager as fm
_zh_font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
fm.fontManager.addfont(_zh_font_path)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.decomposition import PCA
from PIL import Image

warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LAYERS = 24
FPS = 3                      # 帧率
LAYER_STAY = 10              # 每层 10 帧 ≈ 3.33 秒
END_PAUSE_SEC = 3            # 最后一帧停留秒数
END_PAUSE_FRAMES = END_PAUSE_SEC * FPS
TOTAL_FRAMES = N_LAYERS * LAYER_STAY + END_PAUSE_FRAMES
MODEL_NAME = 'vit_large_patch16_dinov3'
PREFIX_TOKENS = 5            # 1 cls + 4 register tokens

# 主题色（深色科技风）
BG_COLOR    = '#0F0F1A'
TITLE_COLOR = '#E0E0FF'
TEXT_COLOR  = '#C0C0E0'
ACCENT      = '#58C4DD'
ACCENT2     = '#FF6B6B'
PROGRESS_BG = '#2A2A3E'
STAGE_COLORS = ['#58C4DD', '#83C167', '#FFD93D', '#FF6B6B', '#BB86FC']
COLORMAP    = 'plasma'       # 特征图颜色映射

# 层级阶段划分（来自 dinov3_24layers.md）
STAGES = [
    ( 0,  4, '浅层：边缘 / 纹理',        '边缘检测、梯度方向、简单纹理',         '#58C4DD'),
    ( 5, 10, '中浅层：模式 / 部件',       '几何形状、局部结构、部件组合',         '#83C167'),
    (11, 15, '中深层：目标 / 语义',       '语义部件、目标局部、目标检测',         '#FFD93D'),
    (16, 20, '深层：类别 / 场景',         '类别区分、目标边界、场景理解',         '#FF6B6B'),
    (21, 23, '最深层：抽象语义',          '全局关系、高层概念、分类输入',          '#BB86FC'),
]

def get_stage_info(layer_idx):
    for i, (lo, hi, name, desc, color) in enumerate(STAGES):
        if lo <= layer_idx <= hi:
            return name, desc, color
    return '', '', ACCENT


def download_sample_image():
    """下载标准测试图"""
    url = 'https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/cats.png'
    dst = '/tmp/dinov3_sample.jpg'
    print(f'[下载] 测试图片 -> {dst}')
    try:
        urllib.request.urlretrieve(url, dst)
        return dst
    except Exception:
        # fallback: 创建一个简单测试图
        img = Image.new('RGB', (518, 518), (40, 40, 80))
        img.save(dst)
        return dst


def load_image(path):
    """加载并转为 RGB"""
    if path is None or not os.path.exists(path):
        path = download_sample_image()
    img = Image.open(path).convert('RGB')
    print(f'[图像] {path}  ({img.size[0]}x{img.size[1]})')
    return img


def extract_features(model, img, device, input_size=448):
    """单次前向传播（forward hook），提取全部 24 层的 patch 特征"""
    # 使用自定义输入尺寸（默认 448），自动插值位置编码
    from torchvision import transforms as T
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    transform = T.Compose([
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    # 注册 forward hook 捕获每个 block 的输出
    intermediate = [None] * N_LAYERS

    def make_hook(idx):
        def hook(module, inp, out):
            intermediate[idx] = out
        return hook

    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(tensor)  # 一次前向，hooks 自动填充 intermediate

    for h in hooks:
        h.remove()

    # 解析 patch token：跳过 cls(1) + register(4)
    patch_features = []
    for layer_output in intermediate:
        # layer_output shape: [1, prefix+patches, 1024]
        patches = layer_output[0, PREFIX_TOKENS:, :]  # [num_patches, 1024]
        num_patches = patches.shape[0]
        H = W = int(num_patches ** 0.5)
        patch_features.append(patches.cpu().numpy().reshape(H, W, 1024))

    print(f'[特征] 提取 {len(patch_features)} 层, 每层 {patch_features[0].shape}')
    return patch_features


def pca_reduce(layer_features):
    """对每层做独立 PCA 3 通道降维"""
    pca_maps = []
    for layer_idx, patches in enumerate(layer_features):
        H, W, D = patches.shape
        flat = patches.reshape(-1, D)
        pca = PCA(n_components=3, random_state=layer_idx)
        reduced = pca.fit_transform(flat)  # [P, 3]
        # 每个通道独立归一化到 [0, 1]
        for c in range(3):
            mn, mx = reduced[:, c].min(), reduced[:, c].max()
            reduced[:, c] = (reduced[:, c] - mn) / (mx - mn + 1e-8)
        pca_maps.append(reduced.reshape(H, W, 3))

    print(f'[PCA] 降维完成, 特征图尺寸 {pca_maps[0].shape[:2]}')
    return pca_maps


def make_video(orig_img, pca_maps, output_path, input_size=448):
    """生成动画视频"""
    H_fmap, W_fmap = pca_maps[0].shape[:2]

    # 原始图像缩放到模型输入尺寸
    orig_resized = orig_img.resize((input_size, input_size), Image.BICUBIC)

    # ── 创建 figure ──
    dpi = 120
    fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR, dpi=dpi)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08)

    # ── 左侧：原始图像 ──
    ax_left = fig.add_axes([0.03, 0.10, 0.42, 0.75])
    ax_left.imshow(orig_resized)
    ax_left.set_title('Input Image', fontsize=14, color=TEXT_COLOR, fontweight='bold')
    ax_left.axis('off')

    # ── 右侧：特征图 ──
    ax_right = fig.add_axes([0.52, 0.10, 0.42, 0.75])
    # ax_right is updated per frame

    # ── 顶部：标题 + 进度条 ──
    title_ax = fig.add_axes([0.03, 0.92, 0.94, 0.06])
    title_ax.axis('off')

    # ── 底部：进度条（24层显示） ──
    progress_ax = fig.add_axes([0.10, 0.03, 0.80, 0.03])
    progress_ax.set_xlim(0, N_LAYERS - 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    # 画阶段背景
    for i, (lo, hi, _, _, color) in enumerate(STAGES):
        progress_ax.fill_betweenx([0, 1], lo, hi + 1,
                                  color=color, alpha=0.25, edgecolor=None)
        # 阶段分隔线
        if lo > 0:
            progress_ax.axvline(lo, color='#FFFFFF', linewidth=0.3, alpha=0.5)
    # 刻度标签
    for i in range(N_LAYERS):
        if i % 5 == 0:
            progress_ax.text(i, -0.6, str(i), fontsize=7, color=TEXT_COLOR,
                            ha='center', va='top')

    stage_labels_bottom = []
    for i, (lo, hi, name, _, color) in enumerate(STAGES):
        mid = (lo + hi) / 2
        lbl = progress_ax.text(mid, 1.6, name.split('：')[1] if '：' in name else name,
                               fontsize=8, color=color, ha='center', va='bottom',
                               fontweight='bold', alpha=0.7)
        stage_labels_bottom.append(lbl)

    # 进度指示器（圆点）
    dot = progress_ax.plot(0, 0.5, 'o', color='#FFFFFF', markersize=10,
                          markeredgecolor='white', markeredgewidth=1, zorder=5)[0]

    # ── 初始化帧（第一帧） ──
    im = ax_right.imshow(pca_maps[0], cmap=COLORMAP, interpolation='bilinear')
    ax_right.axis('off')

    frame_idx = [0]  # mutable counter

    def update(frame):
        layer_idx = frame // LAYER_STAY  # 每层停留多帧
        layer_idx = min(layer_idx, N_LAYERS - 1)

        # 更新特征图
        im.set_array(pca_maps[layer_idx])

        # 更新右侧标题
        stage_name, stage_desc, stage_color = get_stage_info(layer_idx)
        ax_right.set_title(
            f'Layer {layer_idx}  —  {stage_name}',
            fontsize=13, color=stage_color, fontweight='bold'
        )

        # 更新底部进度点
        dot.set_xdata([layer_idx])

        # 更新顶部标题
        title_ax.clear()
        title_ax.axis('off')
        title_ax.text(
            0.5, 0.5,
            f'DINOv3 ViT-L / 24 Layers  —  逐层语义递进  (Layer {layer_idx} / {N_LAYERS - 1})',
            fontsize=16, color=TITLE_COLOR, fontweight='bold',
            ha='center', va='center', transform=title_ax.transAxes
        )
        # 小描述
        title_ax.text(
            0.5, -0.3, stage_desc,
            fontsize=10, color=stage_color, ha='center', va='top',
            transform=title_ax.transAxes, alpha=0.8, style='italic'
        )

        return im, dot,

    ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                                  interval=1000 // FPS, blit=False)

    print(f'[渲染] 视频 -> {output_path}  ({TOTAL_FRAMES} frames @ {FPS}fps)')
    ani.save(output_path, writer='ffmpeg', fps=FPS, dpi=dpi,
             bitrate=8000, metadata={'title': 'DINOv3 24 Layers',
                                     'artist': 'DINOv3 Book'})
    plt.close()
    print(f'[完成] {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='DINOv3 ViT-L 24层语义层级可视化动画')
    parser.add_argument('--image', '-i', default=None,
                        help='输入图像路径（默认自动下载测试图）')
    parser.add_argument('--output', '-o', default='dinov3_24layers.mp4',
                        help='输出视频路径')
    parser.add_argument('--size', '-s', type=int, default=448,
                        help='输入分辨率（默认 448，patch16 时 448→28x28 patch）')
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    print(f'═' * 55)
    print(f'  DINOv3 ViT-L 24层可视化成 {output_path}')
    print(f'  设备: {DEVICE.upper()}')
    print(f'═' * 55)

    # 1. 加载模型
    input_size = args.size
    patch_grid = input_size // 16
    print(f'[加载] DINOv3 ViT-L/16 ({input_size}x{input_size}, {patch_grid}x{patch_grid} patches) ...')
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True, num_classes=0
    ).to(DEVICE).eval()
    print(f'  参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    # 2. 加载图像
    img = load_image(args.image)

    # 3. 提取特征
    patch_features = extract_features(model, img, DEVICE, input_size)

    # 4. PCA 降维
    pca_maps = pca_reduce(patch_features)

    # 5. 生成视频
    make_video(img, pca_maps, output_path, input_size)


if __name__ == '__main__':
    main()
