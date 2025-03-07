#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import argparse
from pathlib import Path


def get_matching_files(root_dir):
    """获取输入和标签文件夹中匹配的文件名"""
    # 获取输入和标签文件夹路径
    input_dir = os.path.join(root_dir, "Input")
    label_dir = os.path.join(root_dir, "Label")

    # 获取所有文件名(不含后缀)
    input_files = {Path(f).stem for f in os.listdir(input_dir)}
    label_files = {Path(f).stem for f in os.listdir(label_dir)}

    # 找出匹配的文件名
    matched_files = input_files.intersection(label_files)
    
    # 找出不匹配的文件名
    unmatched_inputs = input_files - label_files
    unmatched_labels = label_files - input_files

    if unmatched_inputs:
        print("以下输入文件缺少对应的标签:")
        print("\n".join(unmatched_inputs))
    
    if unmatched_labels:
        print("以下标签文件缺少对应的输入:")
        print("\n".join(unmatched_labels))

    return list(matched_files)


def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """划分数据集为训练集、验证集和测试集,并生成txt文件保存路径"""
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    
    # 获取输入和标签文件夹路径
    input_dir = os.path.join(root_dir, "Input")
    label_dir = os.path.join(root_dir, "Label")
    
    # 获取匹配的文件名
    matched_files = get_matching_files(root_dir)
    
    if not matched_files:
        print("没有找到匹配的输入和标签文件!")
        return

    # 设置随机种子
    random.seed(seed)
    random.shuffle(matched_files)

    # 计算每个集合的大小
    total = len(matched_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # 划分数据集
    train_files = matched_files[:train_size]
    val_files = matched_files[train_size:train_size + val_size]
    test_files = matched_files[train_size + val_size:]

    # 获取输入和标签文件的后缀
    input_suffix = Path(os.listdir(input_dir)[0]).suffix
    label_suffix = Path(os.listdir(label_dir)[0]).suffix

    # 生成txt文件
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        txt_path = os.path.join(root_dir, f"{split_name}.txt")
        with open(txt_path, 'w') as f:
            for file in files:
                input_file = os.path.abspath(os.path.join(input_dir, file + input_suffix))
                label_file = os.path.abspath(os.path.join(label_dir, file + label_suffix))
                f.write(f"{input_file}\t{label_file}\n")

    print(f"数据集划分完成！")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")


def main():
    parser = argparse.ArgumentParser(description='划分数据集为训练集、验证集和测试集')
    parser.add_argument('--root_dir', type=str, default='./data/THI_fuzzy/membership_results/clearsky', help='数据根目录(包含Input和Label子文件夹)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例 (默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')

    args = parser.parse_args()
    
    split_dataset(
        args.root_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )


if __name__ == "__main__":
    main()

"""
使用示例:

1. 基本用法:
python split_data.py ./data

2. 自定义划分比例:
python split_data.py ./data 0.8 0.1 0.1

3. 指定随机种子:
python split_data.py ./data 0.7 0.15 0.15 123

生成的txt文件格式:
每行包含输入文件和标签文件的绝对路径，用制表符分隔

./data/
├── train.txt
├── val.txt
└── test.txt
"""
