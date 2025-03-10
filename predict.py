#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models import ThiModel
import argparse
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pathlib import Path
import matplotlib
from tqdm import tqdm
import datetime
from data_utils.dataset import ThiDataset
import warnings

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割预测脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="Result\Train\met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918\THI_best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--input_path', type=str,
                        default="./data/THI_extension",
                        help='输入数据路径(单个npz文件或目录)')
    parser.add_argument('--output_dir', type=str, default='Result/Predict',
                        help='预测结果保存的根目录路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备 (cuda/cpu)')
    return parser.parse_args()

def create_custom_colormap():
    """创建离散的颜色映射"""
    # 红色(0)，绿色(1)，白色(无效值)
    return ListedColormap([(1, 0, 0), (0, 0.8, 0), (1, 1, 1)])

def get_file_list(input_path):
    """获取需要处理的文件列表"""
    if os.path.isfile(input_path):
        if input_path.endswith('.npz'):
            return [input_path]
        else:
            raise ValueError(f"输入文件不是npz格式: {input_path}")
    elif os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.npz')]
        if not files:
            raise ValueError(f"在目录 {input_path} 中没有找到npz文件")
        return files
    else:
        raise ValueError(f"输入路径不存在: {input_path}")

def visualize_prediction(image, prediction, save_path=None):
    """可视化所有通道的图像和预测结果"""
    try:
        plt.figure(figsize=(18, 10))
        
        channel_names = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
        
        _, h, w = image.shape
        should_rotate = h > w
        
        for i in range(5):
            plt.subplot(2, 3, i+1)
            
            if should_rotate:
                display_img = np.rot90(image[i])
                plt.imshow(display_img, cmap='viridis')
                plt.title(f"通道: {channel_names[i]} (已旋转90°)")
            else:
                plt.imshow(image[i], cmap='viridis')
                plt.title(f"通道: {channel_names[i]}")
                
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis("off")
        
        custom_cmap = create_custom_colormap()
        
        plt.subplot(2, 3, 6)
        if should_rotate:
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
        import traceback
        traceback.print_exc()

def process_single_file(file_path, model, dataset_processor, device, output_dir):
    """处理单个文件"""
    try:
        # 读取数据
        data = np.load(file_path)
        
        # 根据设置选择输入数据
        if dataset_processor.calculate_membership != 'none':
            # 使用隶属度作为输入
            degrees = dataset_processor.membership_calculator.calculate_membership(file_path)
            image = np.stack([
                degrees['Z1'],
                degrees['V1'],
                degrees['W1'],
                degrees['SNR1'],
                degrees['LDR']
            ], axis=0).astype(np.float32)
        else:
            # 使用原始数据作为输入
            try:
                image = np.stack([
                    data['Z1'],
                    data['V1'],
                    data['W1'],
                    data['SNR1'],
                    data['LDR']
                ], axis=0).astype(np.float32)
            except KeyError as e:
                print(f"数据文件中缺少必要的通道: {str(e)}")
                print(f"可用的通道: {list(data.keys())}")
                return
        
        # 获取原始尺寸
        original_size = (image.shape[1], image.shape[2])
        
        # 创建无效值掩码（在预处理之前）
        try:
            # 检查所有通道是否存在NaN值
            invalid_mask = np.zeros(original_size, dtype=bool)
            for channel_name in ['Z1', 'V1', 'W1', 'SNR1', 'LDR']:
                channel_data = data[channel_name]
                # 如果数据是三维的，取第一个时间步
                if len(channel_data.shape) == 3:
                    channel_data = channel_data[0]
                invalid_mask = invalid_mask | np.isnan(channel_data)
        except Exception as e:
            warnings.warn(f"创建无效值掩码时出错: {str(e)}")
            invalid_mask = None
        
        # 数据预处理
        image = dataset_processor._preprocess_data(image)
        
        # 进行padding
        padded_image, _ = dataset_processor._pad_to_divisible_by_32(image, None)
        
        # 转换为tensor并移到设备
        x = torch.from_numpy(padded_image).unsqueeze(0).to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = model(x)
            prediction = outputs.argmax(dim=1).squeeze().cpu().numpy()
        
        # 裁剪回原始尺寸
        prediction = dataset_processor._crop_to_original_size(prediction, original_size)
        
        # 应用无效值掩码
        if invalid_mask is not None:
            prediction[invalid_mask] = 2  # 2表示无效值
        
        print(prediction.shape)
        # 保存结果
        file_name = Path(file_path).stem
        images_dir = os.path.join(output_dir, "images")
        arrays_dir = os.path.join(output_dir, "arrays")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(arrays_dir, exist_ok=True)
        
        output_path = os.path.join(images_dir, f"{file_name}_prediction.png")
        np_output_path = os.path.join(arrays_dir, f"{file_name}_prediction.npy")
        
        # 保存预测数组
        np.save(np_output_path, prediction)
        
        # 可视化并保存图像
        visualize_prediction(image, prediction, output_path)
        
        return True
    except Exception as e:
        print(f"\n处理文件 {file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 找不到checkpoint文件: {args.checkpoint_path}")
        return
    
    try:
        # 加载模型
        print(f"正在加载模型从: {args.checkpoint_path}")
        model = ThiModel.load_from_checkpoint(args.checkpoint_path)
        model = model.to(args.device)
        model.eval()
        
        # 打印模型信息
        print(f"模型信息:")
        print(f"- 解码器架构: {model.hparams.arch}")
        print(f"- 主干网络: {model.hparams.encoder_name}")
        print(f"- 数据集: {model.hparams.dataset_name if hasattr(model.hparams, 'dataset_name') else '未知'}")
        print(f"- 隶属度计算方式: {model.hparams.calculate_membership}")
        
        # 使用模型中保存的隶属度参数
        calculate_membership = model.hparams.calculate_membership
        polynomial_dir = model.hparams.polynomial_dir
        height_bands = model.hparams.height_bands
        
        # 创建数据处理器
        dataset_processor = ThiDataset(
            txt_path="dummy.txt",  # 仅用于初始化
            calculate_membership=calculate_membership,
            polynomial_dir=polynomial_dir,
            height_bands=height_bands,
            is_prediction=True
        )
        
        # 创建输出目录
        dataset_name = model.hparams.dataset_name if hasattr(model.hparams, 'dataset_name') else 'unknown'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        arch_info = f"{model.hparams.arch}_{model.hparams.encoder_name}"
        
        # 添加隶属度信息到目录名
        membership_str = ''
        if model.hparams.calculate_membership == 'none':
            membership_str = 'raw'
        elif model.hparams.calculate_membership == 'clearsky':
            membership_str = 'cs'
        else:  # meteorological
            membership_str = 'met'
            
        result_dir = os.path.join(
            args.output_dir,
            f"{membership_str}_{dataset_name}_{arch_info}_predict_{timestamp}"
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # 获取需要处理的文件列表
        files = get_file_list(args.input_path)
        
        # 处理每个文件
        success_count = 0
        with tqdm(total=len(files), desc="处理进度") as pbar:
            for file_path in files:
                if process_single_file(file_path, model, dataset_processor, args.device, result_dir):
                    success_count += 1
                pbar.update(1)
        
        print(f"\n预测完成，成功处理 {success_count}/{len(files)} 个文件")
        print(f"结果保存在: {result_dir}")
            
    except Exception as e:
        print(f"\n预测过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 使用示例:
# python predict.py --checkpoint_path checkpoints/best_model.ckpt --input_path ./data/test_sample.npz --output_dir ./predictions
