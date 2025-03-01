#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 创建自定义的二分类色图：背景为白色，目标为红色
binary_cmap = ListedColormap([(0, 0.8, 0), 'red'])

class ThiDataset(BaseDataset):
    CLASSES = ["background", "target"]
    
    def __init__(self, txt_path, augmentation=None):
        # 读取txt文件中的数据路径
        with open(txt_path, 'r') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            self.input_paths, self.mask_paths = zip(*lines)
            
        self.augmentation = augmentation

    def _pad_to_divisible_by_32(self, image, mask):
        """将图像和掩码填充到32的倍数"""
        _, h, w = image.shape
        
        # 检查是否需要padding
        if h % 32 == 0 and w % 32 == 0:
            return image, mask
            
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
        
        # 对掩码进行填充
        padded_mask = np.pad(
            mask,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )
        
        return padded_image, padded_mask
        
    def __getitem__(self, i):
        # 读取npz文件中的数据
        data = np.load(self.input_paths[i])
        # 将5个通道的数据堆叠在一起
        image = np.stack([
            data['Z1'],
            data['V1'],
            data['W1'],
            data['SNR1'],
            data['LDR']
        ], axis=0).astype(np.float32)  # 确保图像是float32类型
        
        # 读取对应的标签文件，确保是整数类型
        mask = np.load(self.mask_paths[i]).astype(np.int64)  # 使用int64类型
        
        # 进行padding
        image, mask = self._pad_to_divisible_by_32(image, mask)
        
        # 如果有数据增强，应用数据增强
        if self.augmentation:
            # 转换为(H, W, C)格式用于增强
            image_for_aug = image.transpose(1, 2, 0)
            sample = self.augmentation(image=image_for_aug, mask=mask)
            image = sample["image"].transpose(2, 0, 1)
            mask = sample["mask"]
            
        return image, mask.astype(np.int64)  # 再次确保mask是整数类型

    def __len__(self):
        return len(self.input_paths)

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
            # 只显示第一个通道作为示例，并旋转90度
            img_data = np.rot90(image[0])
            plt.imshow(img_data, cmap='viridis')
        else:
            # 旋转掩码90度
            mask_data = np.rot90(image)
            plt.imshow(mask_data, cmap=binary_cmap, vmin=0, vmax=1)
    plt.show()

if __name__ == "__main__":
    # 测试代码
    dataset = ThiDataset("./data/Thi/train.txt")
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    visualize(image=image, mask=mask) 