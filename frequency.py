import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

def setup_academic_style():
    """设置学术绘图风格：Times New Roman + 极简白边"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['savefig.bbox'] = 'tight'
    # 进一步减小外边距
    plt.rcParams['savefig.pad_inches'] = 0.01

def standardize_image(data, target_size=(512, 512)):
    """强制对齐尺寸"""
    if isinstance(data, Image.Image):
        return ImageOps.fit(data, target_size, Image.LANCZOS)
    
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 2: tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3: tensor = tensor.unsqueeze(0)
    
    resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()

def robust_normalize(img, p_low=2, p_high=98):
    """抗噪百分位归一化：保证橙红色纹理清晰可见"""
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    clipped = np.clip(img, vmin, vmax)
    if vmax - vmin < 1e-8:
        return clipped
    return (clipped - vmin) / (vmax - vmin)

def compute_spectrum_then_resize(raw_feat, target_size=(512, 512), gamma=0.6):
    """先计算FFT再放大"""
    f_shift = np.fft.fftshift(np.fft.fft2(raw_feat))
    mag = np.log1p(np.abs(f_shift))
    mag = mag ** gamma
    return (mag - mag.min()) / (mag.max() - mag.min() + 1e-10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--npy_original", required=True)
    parser.add_argument("--npy_fgm", required=True)
    parser.add_argument("--save", default="Fig3_Tight_Vertical.pdf")
    args = parser.parse_args()

    setup_academic_style()
    size = (512, 512)

    # 1. 准备数据
    img_pil = Image.open(args.image).convert('RGB')
    img_show = standardize_image(img_pil, size)

    # Baseline
    raw_a = np.load(args.npy_original, allow_pickle=True).astype(np.float32)
    raw_a = raw_a[0] if raw_a.ndim > 2 else raw_a 
    feat_a_vis = robust_normalize(standardize_image(raw_a, size))
    spec_a_vis = compute_spectrum_then_resize(raw_a, size)

    # SOEP/FGM
    raw_b = np.load(args.npy_fgm, allow_pickle=True).astype(np.float32)
    raw_b = raw_b[0] if raw_b.ndim > 2 else raw_b
    feat_b_vis = robust_normalize(standardize_image(raw_b, size))
    spec_b_vis = compute_spectrum_then_resize(raw_b, size)

    # 2. 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    im_kwargs = dict(cmap='inferno', aspect='auto') 
    
    # 【调整点 1】将标题上移，贴紧图片底部
    label_y = -0.12 
    fs = 14
    fw = 'bold'

    # --- Row 1 ---
    axes[0,0].imshow(img_show, aspect='auto')
    axes[0,0].set_title("(a) Input Image (Drone View)", y=label_y, fontweight=fw, fontsize=fs)
    
    axes[0,1].imshow(feat_a_vis, **im_kwargs)
    axes[0,1].set_title("(b) Original Feature (HFER: 0.65)", y=label_y, fontweight=fw, fontsize=fs)
    
    axes[0,2].imshow(spec_a_vis, **im_kwargs)
    axes[0,2].set_title("(c) Original Spectrum", y=label_y, fontweight=fw, fontsize=fs)

    # --- Row 2 ---
    axes[1,0].imshow(img_show, aspect='auto')
    axes[1,0].set_title("(d) Input Image (Drone View)", y=label_y, fontweight=fw, fontsize=fs)
    
    axes[1,1].imshow(feat_b_vis, **im_kwargs)
    axes[1,1].set_title("(e) FGM Processed Feature (HFER: 0.73)", y=label_y, fontweight=fw, fontsize=fs)
    
    axes[1,2].imshow(spec_b_vis, **im_kwargs)
    axes[1,2].set_title("(f) FGM Processed Spectrum", y=label_y, fontweight=fw, fontsize=fs)

    # 3. 美化与保存
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')

    # 【调整点 2】极限压缩垂直间距：hspace=0.12
    # wspace=0.01 保持左右无缝
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.01, hspace=0.12)
    
    plt.savefig(args.save, dpi=300, format='pdf')
    print(f"[Success] Saved tight-layout figure to: {args.save}")

if __name__ == "__main__":
    main()
