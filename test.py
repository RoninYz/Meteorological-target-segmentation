#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def collect_images(source_dir: str, target_dir: str):
    """
    收集所有子文件夹中的图片到一个目标文件夹

    Args:
        source_dir (str): 源目录路径（包含多个子文件夹）
        target_dir (str): 目标目录路径
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    subfolders.sort()  # 按文件夹名称排序
    
    # 使用tqdm显示进度
    with tqdm(total=len(subfolders), desc="处理进度") as pbar:
        for folder in subfolders:
            # 源文件路径
            source_path = os.path.join(source_dir, folder, "all_parameters_comparison.png")
            
            # 如果文件存在，则复制
            if os.path.exists(source_path):
                # 使用文件夹名作为新的文件名
                target_path = os.path.join(target_dir, f"{folder}.png")
                shutil.copy2(source_path, target_path)
            
            pbar.update(1)
    
    print(f"\n处理完成！")
    print(f"- 共处理了 {len(subfolders)} 个文件夹")
    print(f"- 所有图片已保存到: {os.path.abspath(target_dir)}")

def main():
    # 配置路径
    source_dir = "./data/THI_fuzzy_v2/plots"  # 源目录
    target_dir = "./data/THI_fuzzy_v2/all_plots"  # 目标目录
    
    try:
        collect_images(source_dir, target_dir)
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()