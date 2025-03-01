#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# 创建自定义的二分类色图：背景为白色，目标为红色
binary_cmap = ListedColormap(['white', 'red'])

def visualize_all_channels(image, gt_mask, pr_mask, idx, save_path):
    """可视化所有通道的图像以及掩码并保存"""
    plt.figure(figsize=(15, 10))
    
    # 获取通道名称
    channel_names = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
    
    # 显示每个通道
    for i in range(5):
        plt.subplot(2, 3, i+1)
        # 旋转图像90度以便更好地显示
        img_data = np.rot90(image[i].cpu().numpy())
        plt.imshow(img_data, cmap='viridis')
        plt.title(f"Channel: {channel_names[i]}")
        plt.axis("off")
    
    # 显示真实掩码（背景为白色，目标为红色）
    plt.subplot(2, 3, 6)
    # 旋转掩码90度
    gt_data = np.rot90(gt_mask.cpu().numpy())
    plt.imshow(gt_data, cmap=binary_cmap, vmin=0, vmax=1)
    plt.title("Ground Truth")
    plt.axis("off")
    
    # 显示预测掩码 (如果有)
    if pr_mask is not None:
        plt.subplot(2, 3, 3)
        # 旋转预测掩码90度
        pr_data = np.rot90(pr_mask.cpu().numpy())
        plt.imshow(pr_data, cmap=binary_cmap, vmin=0, vmax=1)
        plt.title("Prediction")
        plt.axis("off")
    
    plt.suptitle(f"Sample {idx}")
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(save_path, f"all_channels_sample_{idx}.png"), dpi=300)
    plt.close() 