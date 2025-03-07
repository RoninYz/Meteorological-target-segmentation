#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# 创建自定义的二分类色图：背景为白色，目标为红色
binary_cmap = ListedColormap(['white', 'red'])

def visualize_all_channels(image, gt_mask, pr_mask, idx, save_path, dataset_name=None):
    """
    可视化所有通道的图像以及掩码并保存
    
    Args:
        image: 输入图像张量
        gt_mask: 真实掩码
        pr_mask: 预测掩码
        idx: 样本索引
        save_path: 保存路径
        dataset_name: 数据集名称（可选）
    """
    plt.figure(figsize=(15, 10))
    
    # 获取通道名称
    channel_names = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
    
    # 显示每个通道
    for i in range(5):
        plt.subplot(2, 3, i+1)
        # 旋转图像90度以便更好地显示
        img_data = np.rot90(image[i].cpu().numpy())
        plt.imshow(img_data, cmap='viridis')
        title = f"{dataset_name} - {channel_names[i]}" if dataset_name else f"Channel: {channel_names[i]}"
        plt.title(title, fontsize=10)
        plt.axis("off")
    
    # 显示真实掩码（背景为白色，目标为红色）
    plt.subplot(2, 3, 6)
    # 旋转掩码90度
    gt_data = np.rot90(gt_mask.cpu().numpy())
    plt.imshow(gt_data, cmap=binary_cmap, vmin=0, vmax=1)
    title = f"{dataset_name} - Ground Truth" if dataset_name else "Ground Truth"
    plt.title(title, fontsize=10)
    plt.axis("off")
    
    # 显示预测掩码 (如果有)
    if pr_mask is not None:
        plt.subplot(2, 3, 3)
        # 旋转预测掩码90度
        pr_data = np.rot90(pr_mask.cpu().numpy())
        plt.imshow(pr_data, cmap=binary_cmap, vmin=0, vmax=1)
        title = f"{dataset_name} - Prediction" if dataset_name else "Prediction"
        plt.title(title, fontsize=10)
        plt.axis("off")
    
    # 添加总标题，包含数据集信息
    suptitle = f"{dataset_name} - Sample {idx}" if dataset_name else f"Sample {idx}"
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    
    # 保存图像
    # 在文件名中添加数据集信息
    filename = f"{dataset_name}_all_channels_sample_{idx}.png" if dataset_name else f"all_channels_sample_{idx}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close() 