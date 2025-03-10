# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def parse_args():
    parser = argparse.ArgumentParser(description='查看NPY文件工具')
    parser.add_argument('--file', type=str, default='./data/THI/Label/Z1_20230805.npy',
                        help='NPY文件路径')
    return parser.parse_args()

def create_custom_colormap():
    """创建离散的颜色映射"""
    from matplotlib.colors import ListedColormap
    return ListedColormap([(1, 0, 0), (0, 0.8, 0)])  # 红色(0)和绿色(1)

def main():
    args = parse_args()
    
    # 加载NPY文件
    data = np.load(args.file)
    
    # 打印数组信息
    print(f"\n数组信息:")
    print(f"形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"唯一值: {np.unique(data)}")
    print(f"最小值: {np.min(data)}")
    print(f"最大值: {np.max(data)}")
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 显示数据（使用离散颜色映射）
    plt.imshow(data, cmap=create_custom_colormap())
    
    # 设置边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title("NPY文件内容")
    plt.show()

if __name__ == "__main__":
    main()