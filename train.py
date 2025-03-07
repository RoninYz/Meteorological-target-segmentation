#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from matplotlib import pyplot as plt
import torch
import warnings
from models import ThiModel
from data_utils import ThiDataset, get_training_augmentation, get_validation_augmentation
from utils import visualize_all_channels
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import numpy as np
from matplotlib.colors import ListedColormap
import datetime
import json
import csv

# 创建自定义的二分类色图：背景为黑色，目标为红色
binary_cmap = ListedColormap(['black', 'red'])

def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割训练脚本')
    
    # 添加网络架构参数 - 更新为包含所有支持的架构
    parser.add_argument('--arch', type=str, default='DeepLabV3Plus',
                        choices=['Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 
                                'PSPNet', 'PAN', 'LinkNet', 'MAnet', 'UPerNet', 'Segformer'],
                        help='分割模型架构')
    
    parser.add_argument('--encoder', type=str, default='mobilenet_v2',
                        choices=[
                            # ResNet 系列
                            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            
                            # ResNeXt 系列
                            'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
                            'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d',
                            
                            # EfficientNet 系列
                            'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 
                            'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                            'efficientnet-b6', 'efficientnet-b7',
                            
                            # MobileNet 系列
                            'mobilenet_v2',
                            
                            # DenseNet 系列
                            'densenet121', 'densenet169', 'densenet201', 'densenet161',
                            
                            # SE-ResNet 系列
                            'se_resnet50', 'se_resnet101', 'se_resnet152',
                            
                            # SE-ResNeXt 系列
                            'se_resnext50_32x4d', 'se_resnext101_32x4d',
                            
                            # SENet 系列
                            'senet154',
                            
                            # VGG 系列
                            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
                            'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                            
                            # DPN 系列
                            'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
                            
                            # Inception 系列
                            'inceptionv4',
                            
                            # Xception 系列
                            'xception',
                            
                            # Mix Vision Transformer (SegFormer) 系列
                            'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',
                            
                            # MobileOne 系列
                            'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4',
                            
                            # Timm 系列编码器
                            'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2',
                            'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5',
                            'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8',
                            'timm-efficientnet-l2',
                            'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1',
                            'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3',
                            'timm-tf_efficientnet_lite4',
                            'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d',
                        ],
                        help='主干网络')
    
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    
    # 添加恢复训练参数
    parser.add_argument('--resume-from', type=str, default=None, 
                        help='继续训练的检查点路径')
    
    # 修改数据集路径参数为单一的数据集根目录
    parser.add_argument('--data-dir', type=str, default="./data/THI", 
                        help='数据集根目录路径,目录下应包含train.txt、val.txt和test.txt')
    
    # 添加结果保存路径参数
    parser.add_argument('--output-dir', type=str, default="Result/Train", 
                        help='训练结果保存的根目录路径')
    
    # 添加训练精度参数
    parser.add_argument('--precision', type=str, default='16-mixed', 
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='训练精度: 32为全精度, 16-mixed为16位混合精度, bf16-mixed为bfloat16混合精度')
    
    # 添加隶属度计算相关参数
    parser.add_argument('--calculate-membership', type=str, default='meteorological',
                        choices=['none', 'clearsky', 'meteorological'],
                        help='隶属度计算方式: none为使用原始数据, clearsky为使用晴空隶属度, meteorological为使用气象目标隶属度')
    
    parser.add_argument('--polynomial-dir', type=str, 
                        default="./data_utils/export_membership/多项式拟合参数",
                        help='多项式拟合参数目录路径')
    
    parser.add_argument('--height-bands', type=str, default='0,100,50,150,125,180,100,200',
                        help='高度带列表，格式为逗号分隔的数字，每两个数字表示一个高度带的起止高度')
    
    return parser.parse_args()

def get_dataset_paths(data_dir):
    """
    从数据集根目录获取训练、验证和测试数据集的路径
    
    Args:
        data_dir (str): 数据集根目录路径
        
    Returns:
        tuple: 包含训练、验证和测试数据集路径的元组
    """
    train_txt = os.path.join(data_dir, "train.txt")
    val_txt = os.path.join(data_dir, "val.txt")
    test_txt = os.path.join(data_dir, "test.txt")
    
    # 检查文件是否存在
    for file_path, name in [(train_txt, "训练"), (val_txt, "验证"), (test_txt, "测试")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"错误：在{data_dir}目录下未找到{name}数据集文件 ({os.path.basename(file_path)})")
    
    return train_txt, val_txt, test_txt

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
    
    # 设置 Tensor Cores 优化
    torch.set_float32_matmul_precision('medium')
    
    # 忽略警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 获取数据集文件路径
    try:
        TRAIN_TXT, VAL_TXT, TEST_TXT = get_dataset_paths(args.data_dir)
        print(f"已找到数据集文件：")
        print(f"训练集：{TRAIN_TXT}")
        print(f"验证集：{VAL_TXT}")
        print(f"测试集：{TEST_TXT}")
    except FileNotFoundError as e:
        print(str(e))
        return

    # 获取数据集名称（使用目录的最后一级作为数据集名称）
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    # 获取当前时间
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # 解析高度带参数
    height_bands = []
    if args.height_bands:
        heights = [int(h) for h in args.height_bands.split(',')]
        height_bands = list(zip(heights[::2], heights[1::2]))

    # 创建数据集实例
    train_dataset = ThiDataset(
        TRAIN_TXT,
        augmentation=get_training_augmentation(),
        calculate_membership=args.calculate_membership,
        polynomial_dir=args.polynomial_dir if args.calculate_membership != 'none' else None,
        height_bands=height_bands if args.calculate_membership != 'none' else None
    )

    valid_dataset = ThiDataset(
        VAL_TXT,
        augmentation=get_validation_augmentation(),
        calculate_membership=args.calculate_membership,
        polynomial_dir=args.polynomial_dir if args.calculate_membership != 'none' else None,
        height_bands=height_bands if args.calculate_membership != 'none' else None
    )

    test_dataset = ThiDataset(
        TEST_TXT,
        augmentation=get_validation_augmentation(),
        calculate_membership=args.calculate_membership,
        polynomial_dir=args.polynomial_dir if args.calculate_membership != 'none' else None,
        height_bands=height_bands if args.calculate_membership != 'none' else None
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True
    )

    # 创建或加载模型
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"正在从检查点恢复训练: {args.resume_from}")
        # 从检查点加载模型（参数会自动从检查点恢复）
        model = ThiModel.load_from_checkpoint(args.resume_from)
        
        # 打印恢复的模型参数信息
        print(f"解码器架构: {model.hparams.arch}")
        print(f"主干网络: {model.hparams.encoder_name}")
        print(f"学习率: {model.hparams.learning_rate}")
        
        # 如果命令行提供了新参数，更新模型参数（可选）
        if args.lr != model.hparams.learning_rate:
            print(f"更新学习率: {model.hparams.learning_rate} -> {args.lr}")
            model.hparams.learning_rate = args.lr
        
        # 确保训练步骤输出被重置，避免在epoch_end时出错
        model.train_step_outputs = []
        model.valid_step_outputs = []
        model.test_step_outputs = []
        
        # 从检查点路径中提取原始文件夹名
        checkpoint_dir = os.path.dirname(os.path.dirname(args.resume_from))
        original_dir = os.path.basename(checkpoint_dir)
        
        # 创建新的结果保存路径，添加continued_前缀和时间戳
        precision_str = args.precision.replace('-', '')
        result_dir = os.path.join(
            args.output_dir, 
            f"continued_{dataset_name}_{original_dir}_lr{args.lr}_ep{args.epochs}_p{precision_str}_{timestamp}"
        )
    else:
        # 创建新模型实例
        model = ThiModel(
            arch=args.arch,
            encoder_name=args.encoder,
            in_channels=5,
            out_classes=2,
            encoder_weights="imagenet",
            learning_rate=args.lr,
            batch_size=args.batch_size,
            dataset_name=dataset_name,  # 添加数据集名称到模型参数
            calculate_membership=args.calculate_membership,
            polynomial_dir=args.polynomial_dir if args.calculate_membership != 'none' else None,
            height_bands=height_bands if args.calculate_membership != 'none' else None
        )
        # 保存超参数到模型
        model.save_hyperparameters()
        
        # 创建新的结果保存路径，包含数据集信息和时间戳
        precision_str = args.precision.replace('-', '')
        
        # 添加隶属度信息到目录名
        membership_str = ''
        if args.calculate_membership == 'none':
            membership_str = 'raw'
        elif args.calculate_membership == 'clearsky':
            membership_str = 'cs'
        else:  # meteorological
            membership_str = 'met'
            
        result_dir = os.path.join(
            args.output_dir, 
            f"{membership_str}_{dataset_name}_{args.arch}_{args.encoder}_lr{args.lr}_ep{args.epochs}_bs{args.batch_size}_p{precision_str}_{timestamp}"
        )
    
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 添加学习率调度器回调
    lr_scheduler = LearningRateMonitor(logging_interval='step')
    
    # 添加早停回调
    early_stopping = EarlyStopping(
        monitor='valid_dataset_iou',
        mode='max',
        patience=10,
        verbose=True
    )

    # 使用单个ModelCheckpoint回调，配置它同时保存最佳模型和其他检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(result_dir, "checkpoints"),
        filename=f'{dataset_name}_' + '{epoch:02d}-{valid_dataset_iou:.3f}',
        save_top_k=3,
        monitor='valid_dataset_iou',
        mode='max',
        save_last=True,  # 保存最后一个模型
        auto_insert_metric_name=True,  # 自动在文件名中插入指标名称
    )

    # 创建TensorBoard日志记录器，保存到特定参数命名的子文件夹
    logger = TensorBoardLogger(
        save_dir=result_dir,
        name="logs",
        default_hp_metric=False,
        version=f"{dataset_name}"  # 添加数据集名称作为版本标识
    )

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',  # 使用GPU
        devices=1,  # 使用1个GPU
        precision=args.precision,  # 使用命令行指定的精度
        callbacks=[
            checkpoint_callback, 
            lr_scheduler,
            early_stopping,
        ],
        logger=logger,  # 添加日志记录器
        log_every_n_steps=1,
        gradient_clip_val=1.0,  # 增大梯度裁剪值
        accumulate_grad_batches=1,  # 禁用梯度累积
        enable_model_summary=True,
        val_check_interval=0.5,  # 每半个 epoch 验证一次
    )

    # 开始训练，添加检查点路径参数
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.resume_from if args.resume_from and os.path.exists(args.resume_from) else None,
    )

    # 验证模型
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)
    
    # 保存验证指标
    # 将嵌套列表转换为单个字典
    valid_metrics_dict = valid_metrics[0] if valid_metrics and isinstance(valid_metrics, list) else valid_metrics
    
    # 创建指标保存目录
    metrics_dir = os.path.join(result_dir, f"{dataset_name}_metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 保存验证指标到文件
    valid_metrics_path = os.path.join(metrics_dir, f"{dataset_name}_validation_metrics.json")
    save_metrics(valid_metrics_dict, valid_metrics_path, format='json')
    
    # 同时保存为CSV格式方便Excel查看
    valid_metrics_csv_path = os.path.join(metrics_dir, f"{dataset_name}_validation_metrics.csv")
    save_metrics(valid_metrics_dict, valid_metrics_csv_path, format='csv')

    # 测试模型
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)
    
    # 保存测试指标
    # 将嵌套列表转换为单个字典
    test_metrics_dict = test_metrics[0] if test_metrics and isinstance(test_metrics, list) else test_metrics
    
    # 保存测试指标到文件
    test_metrics_path = os.path.join(metrics_dir, f"{dataset_name}_test_metrics.json")
    save_metrics(test_metrics_dict, test_metrics_path, format='json')
    
    # 同时保存为CSV格式方便Excel查看
    test_metrics_csv_path = os.path.join(metrics_dir, f"{dataset_name}_test_metrics.csv")
    save_metrics(test_metrics_dict, test_metrics_csv_path, format='csv')
    
    # 合并所有指标并保存一个综合报告
    all_metrics = {
        "dataset": dataset_name,
        "validation": valid_metrics_dict,
        "test": test_metrics_dict,
        "model_info": {
            "architecture": args.arch,
            "encoder": args.encoder,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "epochs": args.epochs,
            "calculate_membership": args.calculate_membership,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # 保存综合指标报告
    all_metrics_path = os.path.join(metrics_dir, f"{dataset_name}_all_metrics.json")
    with open(all_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"综合评价指标报告已保存到 {all_metrics_path}")

    # Fetch a batch from the test loader
    images, masks = next(iter(test_loader))

    # 创建图像保存目录
    imgs_dir = os.path.join(result_dir, f"{dataset_name}_test_images")
    os.makedirs(imgs_dir, exist_ok=True)

    # Switch the model to evaluation mode
    with torch.no_grad():
        model.eval()
        logits = model(images)  # Get raw logits from the model

    # Apply softmax to get class probabilities
    pr_masks = logits.softmax(dim=1)
    # Convert class probabilities to predicted class labels
    pr_masks = pr_masks.argmax(dim=1)

    # 保存图像样本（图像、真实掩码和预测掩码）
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx <= 4:  # 保存前5个样本
            plt.figure(figsize=(12, 6))

            # 选择第一个通道显示
            plt.subplot(1, 3, 1)
            # 旋转图像90度
            img_data = np.rot90(image[0].cpu().numpy())
            plt.imshow(img_data, cmap='viridis')
            plt.title(f"{dataset_name} - Channel 1 (Z1)")
            plt.axis("off")

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            gt_data = np.rot90(gt_mask.cpu().numpy())
            plt.imshow(gt_data, cmap=binary_cmap, vmin=0, vmax=1)
            plt.title(f"{dataset_name} - Ground truth")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 3, 3)
            pr_data = np.rot90(pr_mask.cpu().numpy())
            plt.imshow(pr_data, cmap=binary_cmap, vmin=0, vmax=1)
            plt.title(f"{dataset_name} - Prediction")
            plt.axis("off")

            # 保存图像
            plt.savefig(os.path.join(imgs_dir, f"{dataset_name}_sample_{idx}.png"), dpi=300)
            plt.close()   
            
            # 可选：保存所有通道图像
            visualize_all_channels(image, gt_mask, pr_mask, idx, imgs_dir, dataset_name)
        else:
            break

    # 训练完成后，复制最佳模型到结果目录根目录
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        # 复制最佳模型到结果目录，并在文件名中包含数据集信息
        best_model_save_path = os.path.join(result_dir, f"{dataset_name}_best_model.ckpt")
        torch.save(torch.load(best_model_path), best_model_save_path)
        print(f"最佳模型已保存到: {best_model_save_path}")

if __name__ == "__main__":
    main() 