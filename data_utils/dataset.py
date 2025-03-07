#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_utils.export_membership.membership_calculators import (
    ClearSkyMembershipCalculator,
    MeteorologicalMembershipCalculator
)

# 创建自定义的色图
binary_cmap = ListedColormap([(0, 0.8, 0), 'red'])
membership_cmap = plt.cm.jet

class ThiDataset(BaseDataset):
    CLASSES = ["background", "target"]
    
    def __init__(
        self, 
        txt_path, 
        augmentation=None,
        calculate_membership='none',  # 'none', 'clearsky', 'meteorological'
        polynomial_dir=None,
        height_bands=None,
        is_prediction=False  # 添加预测模式标志
    ):
        """
        初始化数据集
        
        Args:
            txt_path (str): 数据路径文件
            augmentation: 数据增强方法
            calculate_membership (str): 隶属度计算方式
                - 'none': 使用原始数据作为输入
                - 'clearsky': 使用晴空隶属度作为输入
                - 'meteorological': 使用气象目标隶属度作为输入
            polynomial_dir (str): 多项式拟合参数目录
            height_bands (list): 高度带列表
            is_prediction (bool): 是否为预测模式，如果是则跳过文件验证
        """
        # 如果不是预测模式，进行文件验证
        if not is_prediction:
            # 检查txt文件是否存在
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"数据集文件不存在: {txt_path}")
                
            # 读取txt文件中的数据路径
            with open(txt_path, 'r') as f:
                lines = [line.strip().split('\t') for line in f.readlines()]
                
            # 验证每行是否包含两列数据
            for i, line in enumerate(lines, 1):
                if len(line) != 2:
                    raise ValueError(f"数据集文件第{i}行格式错误: {line}")
                    
            self.input_paths, self.mask_paths = zip(*lines)
            
            # 验证所有文件是否存在
            for i, (input_path, mask_path) in enumerate(zip(self.input_paths, self.mask_paths)):
                if not os.path.exists(input_path):
                    raise FileNotFoundError(f"输入文件不存在: {input_path}")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
                    
            print(f"成功加载数据集，共{len(self.input_paths)}个样本")
        else:
            self.input_paths = []
            self.mask_paths = []
            
        self.augmentation = augmentation
        self.calculate_membership = calculate_membership.lower()
        
        # 验证隶属度计算方式参数
        if self.calculate_membership not in ['none', 'clearsky', 'meteorological']:
            raise ValueError("calculate_membership参数必须是'none'、'clearsky'或'meteorological'之一")
        
        # 初始化隶属度计算器
        if self.calculate_membership != 'none':
            if not polynomial_dir or not height_bands:
                raise ValueError("计算隶属度时必须提供polynomial_dir和height_bands参数")
            
            if self.calculate_membership == 'clearsky':
                self.membership_calculator = ClearSkyMembershipCalculator(
                    polynomial_dir, height_bands
                )
            else:  # meteorological
                self.membership_calculator = MeteorologicalMembershipCalculator(
                    polynomial_dir, height_bands
                )

    def _pad_to_divisible_by_32(self, image, mask):
        """将图像和掩码填充到32的倍数"""
        if len(image.shape) == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape
            
        # 检查是否需要padding
        if h % 32 == 0 and w % 32 == 0:
            return image, mask
            
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        
        pad_h = new_h - h
        pad_w = new_w - w
        
        # 对图像进行填充
        if len(image.shape) == 3:
            # (channels, height, width)
            padded_image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
        else:
            # (height, width)
            padded_image = np.pad(
                image,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
        
        # 对掩码进行填充
        if mask is not None:
            padded_mask = np.pad(
                mask,
                ((0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_mask = None
        
        return padded_image, padded_mask

    def _normalize_channel(self, channel_data):
        """对单个通道进行归一化
        
        Args:
            channel_data (np.ndarray): 形状为(H, W)的单通道数据
            
        Returns:
            np.ndarray: 归一化后的数据
        """
        # 处理无效值
        channel_data = np.nan_to_num(channel_data, 0)
        
        # 计算通道的最小值和最大值
        channel_min = np.min(channel_data)
        channel_max = np.max(channel_data)
        
        # 避免除以零
        channel_range = channel_max - channel_min
        if channel_range == 0:
            channel_range = 1
            
        # 归一化到[0,1]范围
        normalized_data = (channel_data - channel_min) / channel_range
        return normalized_data

    def _preprocess_data(self, image):
        """对输入数据进行预处理
        
        Args:
            image (np.ndarray): 形状为(C, H, W)的输入数据
            
        Returns:
            np.ndarray: 预处理后的数据
        """
        channels, height, width = image.shape
        processed_image = np.zeros_like(image, dtype=np.float32)
        
        # 对每个通道分别进行归一化
        for c in range(channels):
            processed_image[c] = self._normalize_channel(image[c])
            
        return processed_image
        
    def __getitem__(self, i):
        # 读取npz文件中的数据
        data = np.load(self.input_paths[i])
        
        # 根据设置选择输入数据
        if self.calculate_membership != 'none':
            # 使用隶属度作为输入
            degrees = self.membership_calculator.calculate_membership(self.input_paths[i])
            image = np.stack([
                degrees['Z1'],
                degrees['V1'],
                degrees['W1'],
                degrees['SNR1'],
                degrees['LDR']
            ], axis=0).astype(np.float32)
        else:
            # 使用原始数据作为输入
            image = np.stack([
                data['Z1'],
                data['V1'],
                data['W1'],
                data['SNR1'],
                data['LDR']
            ], axis=0).astype(np.float32)
        # 读取对应的标签文件，确保是整数类型
        try:
            mask_data = np.load(self.mask_paths[i])
            # 尝试不同的方式获取掩码数据
            if isinstance(mask_data, np.ndarray):
                mask = mask_data
           
            elif 'arr_0' in mask_data:
                mask = mask_data['arr_0']
            else:
                # 如果以上方式都不行，获取第一个数组
                mask = list(mask_data.values())[0]
            
            mask = mask.astype(np.int64)
        except Exception as e:
            print(f"加载掩码文件时出错: {self.mask_paths[i]}")
            print(f"错误信息: {str(e)}")
            raise
        
        # 数据预处理
        image = self._preprocess_data(image)
        
        # 进行padding
        image, mask = self._pad_to_divisible_by_32(image, mask)
        
        # 如果有数据增强，应用数据增强
        if self.augmentation:
            # 转换为(H, W, C)格式用于增强
            image_for_aug = image.transpose(1, 2, 0)
            sample = self.augmentation(image=image_for_aug, mask=mask)
            image = sample["image"].transpose(2, 0, 1)
            mask = sample["mask"]
            
        # 转换为PyTorch张量
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
            
        return image, mask

    def __len__(self):
        return len(self.input_paths)

    def _crop_to_original_size(self, padded_data, original_size):
        """将填充的数据裁剪回原始尺寸"""
        h, w = original_size
        return padded_data[:h, :w]

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        
        if name == "image":
            # 显示第一个通道，并旋转90度
            img_data = np.rot90(image[0])
            plt.imshow(img_data, cmap=membership_cmap if image.shape[0] == 5 else 'viridis')
            plt.colorbar()
        elif name == "mask":
            # 旋转掩码90度
            mask_data = np.rot90(image)
            plt.imshow(mask_data, cmap=binary_cmap, vmin=0, vmax=1)
    plt.show()

if __name__ == "__main__":
    # 测试代码
    height_bands = [(0, 100), (50, 150), (125, 180), (100, 200)]
    polynomial_dir = "./data_utils/export_membership/多项式拟合参数"
    
    # 创建使用晴空隶属度作为输入的数据集
    dataset_clearsky = ThiDataset(
        "./data/Thi/train.txt",
        calculate_membership='clearsky',
        polynomial_dir=polynomial_dir,
        height_bands=height_bands
    )
    
    # 创建使用气象目标隶属度作为输入的数据集
    dataset_meteorological = ThiDataset(
        "./data/Thi/train.txt",
        calculate_membership='meteorological',
        polynomial_dir=polynomial_dir,
        height_bands=height_bands
    )
    
    # 获取样本并显示
    sample_clearsky = dataset_clearsky[0]
    print("\n使用晴空隶属度作为输入的数据集:")
    print(f"Image shape: {sample_clearsky[0].shape}, Mask shape: {sample_clearsky[1].shape}")
    
    # 显示使用隶属度作为输入的结果
    visualize(
        image=sample_clearsky[0],
        mask=sample_clearsky[1]
    )
    
    # 获取气象目标样本并显示
    sample_meteorological = dataset_meteorological[0]
    print("\n使用气象目标隶属度作为输入的数据集:")
    print(f"Image shape: {sample_meteorological[0].shape}, Mask shape: {sample_meteorological[1].shape}")
    
    # 显示使用隶属度作为输入的结果
    visualize(
        image=sample_meteorological[0],
        mask=sample_meteorological[1]
    ) 