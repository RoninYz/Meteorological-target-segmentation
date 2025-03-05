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
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'

def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割预测脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed/best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--input_path', type=str,
                        default="./data/THI_extension",
                        help='输入数据路径(单个npz文件或目录)')
    parser.add_argument('--output_dir', type=str, default='Predict_Result',
                        help='预测结果保存的根目录路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备 (cuda/cpu)')
    return parser.parse_args()

def create_custom_colormap():
    """创建离散的颜色映射"""
    from matplotlib.colors import ListedColormap
    # 红色(0)，绿色(1)，白色(无效值)
    return ListedColormap([(1, 0, 0), (0, 0.8, 0), (1, 1, 1)])

def load_data(file_path):
    """加载气象数据文件"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 找不到输入文件: {file_path}")
            return None
            
        # 读取npz格式文件
        data = np.load(file_path)
        
        # 将5个通道的数据堆叠在一起，与dataset.py中处理方式一致
        image = np.stack([
            data['Z1'],
            data['V1'],
            data['W1'],
            data['SNR1'],
            data['LDR']
        ], axis=0).astype(np.float32)
        
        return image
    except Exception as e:
        print(f"加载数据出错: {str(e)}")
        return None

def pad_to_divisible_by_32(image):
    """将图像填充到32的倍数，与dataset.py中保持一致，返回填充后的图像和原始尺寸"""
    _, h, w = image.shape
    
    # 检查是否需要padding
    if h % 32 == 0 and w % 32 == 0:
        return image, (h, w)
        
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    # 对图像进行填充 (channels, height, width)
    padded_image = np.pad(
        image,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0
    )
    
    return padded_image, (h, w)

def crop_prediction(prediction, original_size):
    """将预测结果裁剪回原始尺寸"""
    h, w = original_size
    return prediction[:h, :w]

def visualize_prediction(image, prediction, save_path=None):
    """可视化所有通道的图像和预测结果"""
    try:
        # 修改布局为两行三列，更平衡的比例
        plt.figure(figsize=(18, 10))
        
        # 获取通道名称
        channel_names = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
        
        # 检查图像尺寸
        _, h, w = image.shape
        should_rotate = h > w  # 如果高大于宽，需要旋转
        
        # 显示每个通道
        for i in range(5):
            plt.subplot(2, 3, i+1)
            
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
        
        # 显示预测掩码
        plt.subplot(2, 3, 6)
        if should_rotate:
            # 旋转预测结果以保持一致
            rotated_prediction = np.rot90(prediction)
            plt.imshow(rotated_prediction, cmap=custom_cmap)
            plt.title("预测结果 (已旋转90°)")
        else:
            plt.imshow(prediction, cmap=custom_cmap)
            plt.title("预测结果")
        plt.axis("off")
        
        plt.suptitle("气象目标分割预测结果")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"可视化时出错: {str(e)}")
        # 使用英文备用方案
        plt.figure(figsize=(18, 10))
        
        # 检查图像尺寸
        _, h, w = image.shape
        should_rotate = h > w  # 如果高大于宽，需要旋转
        
        for i in range(5):
            plt.subplot(2, 3, i+1)
            
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
        
        plt.subplot(2, 3, 6)
        if should_rotate:
            # 旋转预测结果以保持一致
            rotated_prediction = np.rot90(prediction)
            plt.imshow(rotated_prediction, cmap=custom_cmap)
            plt.title("Prediction (Rotated 90°)")
        else:
            plt.imshow(prediction, cmap=custom_cmap)
            plt.title("Prediction")
        plt.axis("off")
        
        plt.suptitle("Meteorological Target Segmentation Result")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def predict_single_file(model, file_path, output_dir, device, pbar=None):
    """对单个气象数据文件进行预测并可视化"""
    try:
        if pbar:
            pbar.set_description(f"处理文件: {Path(file_path).name}")
        
        # 创建处理步骤的进度条
        steps = ['准备目录', '加载数据', '数据预处理', '模型预测', '后处理', '保存结果']
        with tqdm(total=len(steps), desc="处理步骤", leave=False) as step_pbar:
            # 确保输出目录存在
            images_dir = os.path.join(output_dir, "images")
            arrays_dir = os.path.join(output_dir, "arrays")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(arrays_dir, exist_ok=True)
            step_pbar.update(1)
            
            # 获取文件名（不包含路径和扩展名）
            file_name = Path(file_path).stem
            
            # 加载数据
            image = load_data(file_path)
            if image is None:
                return None
            step_pbar.update(1)
            
            # 数据预处理
            invalid_mask = np.any(np.isnan(image), axis=0)
            image = np.nan_to_num(image, nan=0.0)
            
            # 对每个通道进行归一化
            for c in range(image.shape[0]):
                channel_data = image[c:c+1, :, :]
                channel_min = np.min(channel_data)
                channel_max = np.max(channel_data)
                channel_range = channel_max - channel_min
                if channel_range == 0:
                    channel_range = 1
                image[c:c+1, :, :] = (channel_data - channel_min) / channel_range
            
            padded_image, original_size = pad_to_divisible_by_32(image)
            step_pbar.update(1)
            
            # 模型预测
            x = torch.from_numpy(padded_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(x)
                padded_prediction = outputs.argmax(dim=1).squeeze().cpu().numpy()
            step_pbar.update(1)
            
            # 后处理
            prediction = crop_prediction(padded_prediction, original_size)
            prediction[invalid_mask] = 2
            step_pbar.update(1)
            
            # 保存结果
            output_path = os.path.join(images_dir, f"{file_name}_prediction.png")
            np_output_path = os.path.join(arrays_dir, f"{file_name}_prediction.npy")
            np.save(np_output_path, prediction)
            visualize_prediction(image, prediction, output_path)
            step_pbar.update(1)
            
            if pbar:
                pbar.update(1)
            
            return prediction
    except Exception as e:
        print(f"\n处理文件 {file_path} 时出错: {str(e)}")
        if pbar:
            pbar.update(1)
        return None

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 找不到checkpoint文件: {args.checkpoint_path}")
        print("请确认checkpoint文件路径是否正确，以及文件是否存在。")
        return
    
    try:
        # 加载模型（显示进度条）
        with tqdm(total=1, desc="加载模型") as pbar:
            print(f"正在加载模型从: {args.checkpoint_path}")
            model = ThiModel.load_from_checkpoint(args.checkpoint_path)
            model = model.to(args.device)
            model.eval()
            pbar.update(1)
        
        # 创建输出目录结构
        model_name = os.path.basename(args.checkpoint_path).split('.')[0]
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        arch_info = f"{model.hparams.arch}_{model.hparams.encoder_name}" if hasattr(model, 'hparams') else model_name
        
        result_dir = os.path.join(args.output_dir, f"{arch_info}_predict_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        images_dir = os.path.join(result_dir, "images")
        arrays_dir = os.path.join(result_dir, "arrays")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(arrays_dir, exist_ok=True)
        
        # 检查输入路径是文件还是目录
        if os.path.isfile(args.input_path):
            if args.input_path.endswith('.npz'):
                with tqdm(total=1, desc="处理文件") as pbar:
                    predict_single_file(model, args.input_path, result_dir, args.device, pbar)
            else:
                print(f"错误: 输入文件不是npz格式: {args.input_path}")
        elif os.path.isdir(args.input_path):
            # 首先计算需要处理的文件数量
            npz_files = [f for f in os.listdir(args.input_path) if f.endswith('.npz')]
            if not npz_files:
                print(f"警告: 在目录 {args.input_path} 中没有找到npz文件")
                return
                
            # 使用tqdm显示总体进度
            with tqdm(total=len(npz_files), desc="总体进度") as pbar:
                for filename in npz_files:
                    file_path = os.path.join(args.input_path, filename)
                    predict_single_file(model, file_path, result_dir, args.device, pbar)
                
            print(f"\n已完成对 {len(npz_files)} 个文件的预测，结果保存在 {result_dir}")
        else:
            print(f"错误: 输入路径不存在: {args.input_path}")
            
    except Exception as e:
        print(f"\n预测过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 使用示例:
# python predict.py --checkpoint_path checkpoints/best_model.ckpt --input_path ./data/test_sample.npz --output_dir ./predictions
