#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import argparse
import os
from pprint import pprint
import json

def load_checkpoint(ckpt_path):
    """
    加载并显示检查点文件的内容
    
    Args:
        ckpt_path: 检查点文件路径
    """
    print(f"\n正在加载检查点: {ckpt_path}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 从文件路径中提取信息
        dir_name = os.path.dirname(os.path.dirname(ckpt_path))
        folder_name = os.path.basename(dir_name)
        
        # 解析文件夹名称中的信息
        print("\n=== 训练配置信息 ===")
        if 'hyper_parameters' in checkpoint:
            hp = checkpoint['hyper_parameters']
            print(f"模型架构: {hp.get('arch', 'unknown')}")
            print(f"编码器: {hp.get('encoder_name', 'unknown')}")
            print(f"学习率: {hp.get('learning_rate', 'unknown')}")
            print(f"批次大小: {hp.get('batch_size', 'unknown')}")
        
        # 显示基本信息
        print("\n=== 训练状态 ===")
        if 'epoch' in checkpoint:
            print(f"当前轮次: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"总训练步数: {checkpoint['global_step']}")
        
        # 显示超参数
        if 'hyper_parameters' in checkpoint:
            print("\n=== 模型超参数 ===")
            pprint(checkpoint['hyper_parameters'])
        
        # 尝试加载metrics文件
        metrics_dir = os.path.join(dir_name, "*_metrics")
        # 使用glob查找metrics目录
        import glob
        metrics_dirs = glob.glob(metrics_dir)
        if metrics_dirs:
            metrics_dir = metrics_dirs[0]
            all_metrics_path = glob.glob(os.path.join(metrics_dir, "*_all_metrics.json"))
            
            if all_metrics_path:
                print("\n=== 训练指标 ===")
                with open(all_metrics_path[0], 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    print("\n数据集:", metrics.get('dataset', 'unknown'))
                    print("\n验证集指标:")
                    pprint(metrics['validation'])
                    print("\n测试集指标:")
                    pprint(metrics['test'])
        
        # 显示模型结构
        if 'state_dict' in checkpoint:
            print("\n=== 模型结构 ===")
            total_params = 0
            for key, value in checkpoint['state_dict'].items():
                if hasattr(value, 'shape'):
                    print(f"层: {key}")
                    print(f"形状: {value.shape}")
                    total_params += torch.prod(torch.tensor(value.shape)).item()
            print(f"\n总参数量: {total_params:,}")
        
        # 显示优化器状态
        if 'optimizer_states' in checkpoint:
            print("\n=== 优化器状态 ===")
            print(f"优化器类型: {checkpoint['optimizer_states'][0]['type']}")
            print(f"当前学习率: {checkpoint['optimizer_states'][0].get('lr', 'unknown')}")
            
    except Exception as e:
        print(f"加载检查点时出错: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='检查点文件查看器')
    parser.add_argument('--ckpt', type=str, default="./Train_Result/clearsky_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep1_bs4_p16mixed_20250305_1900/clearsky_best_model.ckpt", help='检查点文件路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt):
        print(f"错误：找不到检查点文件 {args.ckpt}")
        return
    
    load_checkpoint(args.ckpt)

if __name__ == "__main__":
    main() 