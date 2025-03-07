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
                        default="Result\Train\met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918\THI_best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--val_data', type=str,
                        default="./data/THI/test.txt",
                        help='验证集文件列表路径')
    parser.add_argument('--output_dir', type=str, default='Result/Evaluation',
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
    
    # 计算图片中提供的指标
    # 在这里，我们认为：
    # class 0 (target) = 降水区域
    # class 1 (background) = 非降水区域(杂波和噪声)
    
    # Ns = 真正例 (成功识别的降水区域) = tp[0]
    Ns = tp[0].sum().item()
    
    # Nf = 漏警 (将降水区域误分类为背景) = fn[0]
    Nf = fn[0].sum().item()
    
    # Ni = 虚警 (将背景误分类为降水区域) = fp[0]
    Ni = fp[0].sum().item()
    
    # NT = 总像素数
    NT = (tp[0] + tp[1] + fn[0] + fn[1] + fp[0] + fp[1] + tn[0] + tn[1]).sum().item()
    
    # 检测率 Pd = Ns / (Ns + Nf)
    Pd = Ns / (Ns + Nf) if (Ns + Nf) > 0 else 0
    
    # 虚警率 Pfa = Ni / (NT - Ns - Nf)
    Pfa = Ni / (NT - Ns - Nf) if (NT - Ns - Nf) > 0 else 0
    
    return {
        'iou': iou.item(),
        'per_image_iou': per_image_iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'detection_rate_Pd': Pd,
        'false_alarm_rate_Pfa': Pfa
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
    
    # 规范化路径，确保使用正斜杠
    args.checkpoint_path = os.path.normpath(args.checkpoint_path).replace('\\', '/')
    args.val_data = os.path.normpath(args.val_data).replace('\\', '/')
    args.output_dir = os.path.normpath(args.output_dir).replace('\\', '/')
    
    print(f"使用模型检查点: {args.checkpoint_path}")
    
    # 加载模型
    model = ThiModel.load_from_checkpoint(args.checkpoint_path)
    model.to(args.device)
    model.eval()
    
    # 打印模型信息
    print(f"模型信息:")
    print(f"- 架构: {model.hparams.arch}")
    print(f"- 主干网络: {model.hparams.encoder_name}")
    print(f"- 数据集: {model.hparams.dataset_name if hasattr(model.hparams, 'dataset_name') else '未知'}")
    if hasattr(model.hparams, 'calculate_membership'):
        print(f"- 隶属度计算方式: {model.hparams.calculate_membership}")
    
    # 读取模型参数，创建数据集
    calculate_membership = getattr(model.hparams, 'calculate_membership', 'none')
    polynomial_dir = getattr(model.hparams, 'polynomial_dir', None)
    height_bands = getattr(model.hparams, 'height_bands', None)
    
    # 创建验证数据集和数据加载器
    val_dataset = ThiDataset(
        args.val_data,
        augmentation=get_validation_augmentation(),
        calculate_membership=calculate_membership,
        polynomial_dir=polynomial_dir,
        height_bands=height_bands
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    dataset_name = model.hparams.dataset_name if hasattr(model.hparams, 'dataset_name') else 'unknown'
    membership_str = ''
    if hasattr(model.hparams, 'calculate_membership'):
        if model.hparams.calculate_membership == 'none':
            membership_str = 'raw_'
        elif model.hparams.calculate_membership == 'clearsky':
            membership_str = 'cs_'
        else:  # meteorological
            membership_str = 'met_'
            
    arch_info = f"{model.hparams.arch}_{model.hparams.encoder_name}"
    result_dir = os.path.join(
        args.output_dir,
        f"{membership_str}{dataset_name}_{arch_info}_eval_{timestamp}"
    )
    # 确保路径格式正确
    result_dir = os.path.normpath(result_dir).replace('\\', '/')
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"评估结果将保存到: {result_dir}")
    
    # 进行评估
    all_predictions = []
    all_masks = []
    sample_images = []
    sample_masks = []
    sample_predictions = []
    
    print(f"开始评估...")
    progress_bar = tqdm(val_loader, desc="评估中")
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(progress_bar):
            image = image.to(args.device)
            mask = mask.to(args.device)
            
            # 模型预测
            logits = model(image)
            pred_mask = logits.argmax(dim=1)
            
            # 收集用于计算指标的结果
            all_predictions.append(pred_mask.cpu())
            all_masks.append(mask.cpu())
            
            # 收集用于可视化的样本
            if i < args.visualize_samples:
                sample_images.append(image[0].cpu().numpy())
                sample_masks.append(mask[0].cpu().numpy())
                sample_predictions.append(pred_mask[0].cpu().numpy())
    
    # 合并批次结果用于计算指标
    all_predictions = torch.cat(all_predictions, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # 计算评估指标
    metrics = calculate_metrics(all_predictions, all_masks)
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"IoU (Jaccard): {metrics['iou']:.4f}")
    print(f"Per Image IoU: {metrics['per_image_iou']:.4f}")
    print(f"Dice Coefficient (F1): {metrics['dice']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"检测率 (Pd): {metrics['detection_rate_Pd']:.4f}")
    print(f"虚警率 (Pfa): {metrics['false_alarm_rate_Pfa']:.4f}")
    
    # 保存评估指标
    metrics_path = os.path.join(result_dir, "metrics.json")
    metrics_path = os.path.normpath(metrics_path).replace('\\', '/')
    save_metrics(metrics, metrics_path, 'json')
    
    metrics_csv_path = os.path.join(result_dir, "metrics.csv")
    metrics_csv_path = os.path.normpath(metrics_csv_path).replace('\\', '/')
    save_metrics(metrics, metrics_csv_path, 'csv')
    
    print(f"\n评估指标已保存到: {metrics_path}")
    
    # 可视化并保存样本图像
    print(f"\n可视化评估样本...")
    for i in range(len(sample_images)):
        img_name = f"sample_{i+1}"
        img_path = os.path.join(result_dir, f"{img_name}.png")
        img_path = os.path.normpath(img_path).replace('\\', '/')
        print(f"保存样本 {i+1}...")
        visualize_prediction(
            sample_images[i],
            sample_masks[i],
            sample_predictions[i],
            img_path
        )
    
    # 创建评估结果摘要表格
    summary_path = os.path.join(result_dir, "evaluation_summary.txt")
    summary_path = os.path.normpath(summary_path).replace('\\', '/')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("气象目标分割评估结果摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型: {arch_info}\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"隶属度计算: {getattr(model.hparams, 'calculate_membership', 'none')}\n\n")
        f.write("评估指标:\n")
        f.write(f"IoU (Jaccard): {metrics['iou']:.4f}\n")
        f.write(f"Dice Coefficient (F1): {metrics['dice']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"检测率 (Pd): {metrics['detection_rate_Pd']:.4f}\n")
        f.write(f"虚警率 (Pfa): {metrics['false_alarm_rate_Pfa']:.4f}\n\n")
        f.write(f"检测率定义: Pd = Ns / (Ns + Nf)\n")
        f.write(f"  其中 Ns 是成功识别的降水区域数量，Nf 是被错误分类为杂波和噪声的降水区域数量\n\n")
        f.write(f"虚警率定义: Pfa = Ni / (NT - Ns - Nf)\n")
        f.write(f"  其中 Ni 是被错误分类为降水的杂波和噪声区域数量，NT 是总像素数\n\n")
        f.write(f"说明: 在本评估中，类别0表示降水区域，类别1表示非降水区域(杂波和噪声)\n\n")
        f.write(f"评估时间: {timestamp}\n")
    
    print(f"\n评估完成! 结果保存在: {result_dir}")

if __name__ == "__main__":
    main()

# 使用示例:
# python val.py --checkpoint_path ./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep1_bs4_p16mixed/best_model.ckpt --val_data ./data/val.txt 