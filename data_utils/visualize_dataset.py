#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def parse_args():
    parser = argparse.ArgumentParser(description='THI数据集可视化工具')
    parser.add_argument('--data-txt', type=str, default='./data/THI/train.txt',
                        help='数据集txt文件路径（包含输入和标签路径）')
    parser.add_argument('--output-dir', type=str, default='./Result/visualize',
                        help='可视化结果保存目录')
    parser.add_argument('--dpi', type=int, default=300,
                        help='输出图像的DPI')
    return parser.parse_args()

def create_custom_colormap():
    """创建离散的颜色映射"""
    # 红色(0)，绿色(1)，白色(无效值)
    return ListedColormap([(1, 0, 0), (0, 0.8, 0), (1, 1, 1)])

def visualize_sample(input_path, label_path, output_path, dpi=300):
    """可视化单个样本的所有通道和标签"""
    try:
        # 读取输入数据
        data = np.load(input_path)
        channels = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
        
        # 读取标签
        label = np.load(label_path)
        
        # 创建图像布局 (2行3列)
        plt.figure(figsize=(18, 10))
        
        # 检查是否需要旋转（如果高大于宽）
        sample_shape = data[channels[0]].shape
        should_rotate = sample_shape[0] > sample_shape[1]
        
        # 创建组合的无效值掩码（任何通道中的NaN）
        combined_invalid_mask = np.any([np.isnan(data[ch]) for ch in channels], axis=0)
        
        # 可视化每个通道
        for idx, channel in enumerate(channels):
            ax = plt.subplot(2, 3, idx + 1)
            
            # 获取通道数据
            channel_data = data[channel]
            
            # 检测无效值
            invalid_mask = np.isnan(channel_data)
            
            # 处理无效值
            channel_data = np.nan_to_num(channel_data, nan=0.0)
            
            # 如果需要旋转
            if should_rotate:
                channel_data = np.rot90(channel_data)
                invalid_mask = np.rot90(invalid_mask)
            
            # 显示通道数据
            plt.imshow(channel_data, cmap='viridis')
            
            # 在无效值区域上叠加白色蒙版
            if np.any(invalid_mask):
                plt.imshow(invalid_mask, cmap='gray', alpha=0.5)
            
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"通道: {channel}")
            plt.axis('on')  # 显示坐标轴
            ax.spines['top'].set_visible(True)  # 显示上边框
            ax.spines['right'].set_visible(True)  # 显示右边框
            ax.spines['bottom'].set_visible(True)  # 显示下边框
            ax.spines['left'].set_visible(True)  # 显示左边框
            ax.set_xticks([])  # 不显示x轴刻度
            ax.set_yticks([])  # 不显示y轴刻度
        
        # 显示标签
        ax = plt.subplot(2, 3, 6)
        
        # 创建修改后的标签，将无效值区域设置为2（将显示为白色）
        modified_label = label.copy()
        if should_rotate:
            modified_label = np.rot90(modified_label)
            combined_invalid_mask = np.rot90(combined_invalid_mask)
        
        # 在无效值位置设置为2（白色）
        modified_label[combined_invalid_mask] = 2
        
        plt.imshow(modified_label, cmap=create_custom_colormap())
        plt.title("标签 (白色表示无效区域)")
        plt.axis('on')  # 显示坐标轴
        ax.spines['top'].set_visible(True)  # 显示上边框
        ax.spines['right'].set_visible(True)  # 显示右边框
        ax.spines['bottom'].set_visible(True)  # 显示下边框
        ax.spines['left'].set_visible(True)  # 显示左边框
        ax.set_xticks([])  # 不显示x轴刻度
        ax.set_yticks([])  # 不显示y轴刻度
        
        # 添加总标题
        plt.suptitle(f"样本: {Path(input_path).stem}")
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"已保存可视化结果到: {output_path}")
        return True
        
    except Exception as e:
        print(f"处理样本时出错 {input_path}: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取数据集文件列表
    with open(args.data_txt, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]
    
    print(f"开始处理数据集中的 {len(lines)} 个样本...")
    
    # 处理每个样本
    success_count = 0
    for input_path, label_path in lines:
        # 创建输出文件路径
        output_name = f"{Path(input_path).stem}_visualization.png"
        output_path = os.path.join(args.output_dir, output_name)
        
        # 可视化样本
        if visualize_sample(input_path, label_path, output_path, args.dpi):
            success_count += 1
    
    print(f"\n处理完成！")
    print(f"成功处理: {success_count}/{len(lines)} 个样本")
    print(f"可视化结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 