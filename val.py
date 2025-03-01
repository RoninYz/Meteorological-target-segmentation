#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import ThiModel
import argparse
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import matplotlib
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from data_utils import ThiDataset, get_validation_augmentation
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import datetime
import json
import csv

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割验证脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed/best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--val_data', type=str,
                        default="./data/THI/test.txt",
                        help='验证集文件列表路径')
    parser.add_argument('--output_dir', type=str, default='Evaluation_Result',
                        help='评估结果保存的根目录路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载器的工作线程数')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='可视化样本数量')
    return parser.parse_args()

def create_custom_colormap():
    """创建自定义的气象目标分割颜色映射"""
    # 背景为黑色，目标为红色
    colors = [(0, 0, 0), (1, 0, 0)]  # 黑色背景，红色目标
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=2)

def visualize_prediction(image, mask, prediction, save_path=None):
    """可视化原始图像、真实掩码和预测结果"""
    try:
        # 修改布局为三行三列，更平衡的比例
        plt.figure(figsize=(18, 12))
        
        # 获取通道名称
        channel_names = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
        
        # 检查图像尺寸
        _, h, w = image.shape
        should_rotate = h > w  # 如果高大于宽，需要旋转
        
        # 显示每个通道
        for i in range(5):
            plt.subplot(3, 3, i+1)
            
            # 如果需要旋转，则旋转图像使其宽大于高
            if should_rotate:
                # 旋转90度，使高小于宽
                display_img = np.rot90(image[i])
                plt.imshow(display_img, cmap='viridis')
                plt.title(f"通道: {channel_names[i]} (已旋转90°)")
            else:
                plt.imshow(image[i], cmap='viridis')
                plt.title(f"通道: {channel_names[i]}")
                
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")
        
        # 自定义颜色映射
        custom_cmap = create_custom_colormap()
        
        # 显示真实掩码
        plt.subplot(3, 3, 7)
        if should_rotate:
            # 旋转掩码以保持一致
            rotated_mask = np.rot90(mask)
            plt.imshow(rotated_mask, cmap=custom_cmap)
            plt.title("真实掩码 (已旋转90°)")
        else:
            plt.imshow(mask, cmap=custom_cmap)
            plt.title("真实掩码")
        plt.axis("off")
        
        # 显示预测掩码
        plt.subplot(3, 3, 8)
        if should_rotate:
            # 旋转预测结果以保持一致
            rotated_prediction = np.rot90(prediction)
            plt.imshow(rotated_prediction, cmap=custom_cmap)
            plt.title("预测结果 (已旋转90°)")
        else:
            plt.imshow(prediction, cmap=custom_cmap)
            plt.title("预测结果")
        plt.axis("off")
        
        # 添加错误图 (预测错误的地方标记为白色)
        plt.subplot(3, 3, 9)
        error_map = (prediction != mask).astype(np.uint8)
        if should_rotate:
            error_map = np.rot90(error_map)
        plt.imshow(error_map, cmap='gray')
        plt.title("预测错误区域")
        plt.axis("off")
        
        plt.suptitle("气象目标分割评估结果")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"可视化时出错: {str(e)}")
        # 使用英文备用方案
        plt.figure(figsize=(18, 12))
        
        # 检查图像尺寸
        _, h, w = image.shape
        should_rotate = h > w  # 如果高大于宽，需要旋转
        
        for i in range(5):
            plt.subplot(3, 3, i+1)
            
            # 如果需要旋转，则旋转图像使其宽大于高
            if should_rotate:
                # 旋转90度，使高小于宽
                display_img = np.rot90(image[i])
                plt.imshow(display_img, cmap='viridis')
                plt.title(f"Channel: {channel_names[i]} (Rotated 90°)")
            else:
                plt.imshow(image[i], cmap='viridis')
                plt.title(f"Channel: {channel_names[i]}")
                
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")
        
        plt.subplot(3, 3, 7)
        if should_rotate:
            plt.imshow(np.rot90(mask), cmap=custom_cmap)
            plt.title("Ground Truth (Rotated 90°)")
        else:
            plt.imshow(mask, cmap=custom_cmap)
            plt.title("Ground Truth")
        plt.axis("off")
        
        plt.subplot(3, 3, 8)
        if should_rotate:
            plt.imshow(np.rot90(prediction), cmap=custom_cmap)
            plt.title("Prediction (Rotated 90°)")
        else:
            plt.imshow(prediction, cmap=custom_cmap)
            plt.title("Prediction")
        plt.axis("off")
        
        plt.subplot(3, 3, 9)
        error_map = (prediction != mask).astype(np.uint8)
        if should_rotate:
            error_map = np.rot90(error_map)
        plt.imshow(error_map, cmap='gray')
        plt.title("Error Map")
        plt.axis("off")
        
        plt.suptitle("Meteorological Target Segmentation Evaluation")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def calculate_metrics(preds, masks):
    """
    计算各种评估指标
    Args:
        preds: 模型预测结果 (N, H, W)
        masks: 真实掩码 (N, H, W)
    Returns:
        字典包含各种指标
    """
    # 获取统计信息
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, masks, mode='multiclass', num_classes=2
    )
    
    # 计算IoU (Jaccard)
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
    
    # 计算按图像的IoU (每张图像的IoU平均)
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro-imagewise')
    
    # 计算Dice系数 (F1-score)
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
    
    # 计算准确率
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
    
    # 计算精确率
    precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro')
    
    # 计算召回率
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro')
    
    return {
        'iou': iou.item(),
        'per_image_iou': per_image_iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def save_metrics(metrics, filepath, format='json'):
    """
    保存评价指标到文件
    
    Args:
        metrics: 要保存的评价指标字典
        filepath: 保存文件路径
        format: 保存格式，'json' 或 'csv'
    """
    if format.lower() == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
    elif format.lower() == 'csv':
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['metric', 'value'])
            # 写入数据
            for metric_name, value in metrics.items():
                writer.writerow([metric_name, value])
    else:
        raise ValueError(f"不支持的格式: {format}，请使用 'json' 或 'csv'")
    
    print(f"评价指标已保存到 {filepath}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 找不到checkpoint文件: {args.checkpoint_path}")
        print("请确认checkpoint文件路径是否正确，以及文件是否存在。")
        return
    
    try:
        # 加载模型
        print(f"正在加载模型从: {args.checkpoint_path}")
        model = ThiModel.load_from_checkpoint(args.checkpoint_path)
        model = model.to(args.device)
        model.eval()
        
        # 创建更有组织的输出目录结构
        # 提取模型名称信息
        model_name = os.path.basename(os.path.dirname(args.checkpoint_path))
        
        # 添加时间戳
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        
        # 提取架构信息
        arch_info = f"{model.hparams.arch}_{model.hparams.encoder_name}" if hasattr(model, 'hparams') else model_name
        
        # 创建具有描述性的结果目录
        result_dir = os.path.join(args.output_dir, f"{arch_info}_evaluation_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 创建可视化结果的子目录
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 加载验证数据集
        valid_dataset = ThiDataset(
            args.val_data,
            augmentation=get_validation_augmentation(),
        )

        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True
        )
        
        print(f"验证集包含 {len(valid_dataset)} 个样本")
        
        # 存储所有批次的结果
        all_metrics = defaultdict(list)
        
        # 存储要可视化的样本
        vis_images = []
        vis_masks = []
        vis_preds = []
        
        # 在验证集上进行评估
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_loader, desc="评估中")):
                images, masks = batch
                
                # 将数据移动到设备上
                images = images.to(args.device)
                masks = masks.to(args.device)
                
                # 前向传播
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                # 计算评估指标
                metrics = calculate_metrics(preds.cpu(), masks.cpu())
                
                # 存储指标
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                # 存储一些样本用于可视化
                if len(vis_images) < args.visualize_samples:
                    for i in range(min(len(images), args.visualize_samples - len(vis_images))):
                        vis_images.append(images[i].cpu().numpy())
                        vis_masks.append(masks[i].cpu().numpy())
                        vis_preds.append(preds[i].cpu().numpy())
                        
                        # 达到所需样本数后停止收集
                        if len(vis_images) >= args.visualize_samples:
                            break
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        # 打印评估指标
        print("\n评估指标:")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # 保存评估指标到CSV和JSON文件
        metrics_csv_path = os.path.join(result_dir, "metrics.csv")
        metrics_json_path = os.path.join(result_dir, "metrics.json")
        
        save_metrics(avg_metrics, metrics_csv_path, format='csv')
        save_metrics(avg_metrics, metrics_json_path, format='json')
        
        # 可视化一些样本
        print(f"\n正在生成 {len(vis_images)} 个样本的可视化结果...")
        for i, (image, mask, pred) in enumerate(zip(vis_images, vis_masks, vis_preds)):
            vis_path = os.path.join(vis_dir, f"sample_{i+1}.png")
            visualize_prediction(image, mask, pred, save_path=vis_path)
        
        print(f"评估完成! 结果保存在: {result_dir}")
            
    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 使用示例:
# python val.py --checkpoint_path ./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep1_bs4_p16mixed/best_model.ckpt --val_data ./data/val.txt 