# 气象目标分割项目

## 📋 目录
- [项目简介](#-项目简介)
- [特性](#-特性)
- [环境配置](#-环境配置)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [详细指南](#-详细指南)
- [性能优化](#-性能优化)
- [常见问题](#-常见问题)
- [引用](#-引用)

## 📝 项目简介

本项目是一个专业的气象目标分割系统，基于深度学习技术，用于处理和分析气象雷达数据。系统能够自动识别和分割气象目标，为气象分析和预测提供重要支持。

### 🌟 特性

- **多模型支持**
  - 支持11种主流分割模型：Unet、UnetPlusPlus、DeepLabV3、DeepLabV3Plus等
  - 支持30+种编码器网络：ResNet系列、EfficientNet系列等
  
- **数据处理**
  - 支持多通道气象数据（Z1、V1、W1、SNR1、LDR）
  - 内置专业的数据预处理流程
  - 强大的数据增强功能

- **训练优化**
  - 支持混合精度训练（FP16/BF16）
  - 断点续训功能
  - 自动学习率调整
  - 多GPU训练支持

- **可视化功能**
  - 训练过程实时监控
  - 预测结果可视化
  - 性能指标图表展示

## 🛠 环境配置

### 系统要求
- Windows 10/11 x64 或 Linux
- NVIDIA GPU (建议8GB+显存)
- CUDA 11.7+
- Python 3.8+

### 安装步骤

1. **创建并激活虚拟环境**
```bash
# 使用conda创建环境
conda create -n meteo-seg python=3.8
conda activate meteo-seg

# 或使用venv
python -m venv meteo-seg
# Windows激活
.\\meteo-seg\\Scripts\\activate
# Linux激活
source meteo-seg/bin/activate
```

2. **安装PyTorch**
```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或使用pip（建议）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **安装项目依赖**
```bash
pip install -r requirements.txt
```

## 📁 项目结构

```
Meteorological-target-segmentation/
├── data/                       # 数据目录
│   └── THI/                   # 气象数据集
│       ├── Input/             # 输入数据
│       ├── Label/             # 标签数据
│       ├── train.txt          # 训练集列表
│       ├── val.txt            # 验证集列表
│       └── test.txt           # 测试集列表
├── models/                     # 模型定义
├── utils/                     # 工具函数
├── data_utils/                # 数据处理工具
├── Train_Result/              # 训练结果保存
├── Predict_Result/            # 预测结果保存
├── Visualization_Result/      # 可视化结果
├── train.py                   # 训练脚本
├── val.py                     # 验证脚本
├── predict.py                 # 预测脚本
└── requirements.txt           # 项目依赖
```

## 🚀 快速开始

### 1. 数据准备

#### 1.1 数据集结构
项目使用THI（Thunderstorm Hydrometeor Identification）数据集，数据集应按以下结构组织：

```
data/
└── THI/
    ├── Input/                 # 输入数据文件夹
    │   ├── case_001.npz      # 包含5个通道的气象数据
    │   ├── case_002.npz
    │   └── ...
    └── Label/                 # 标签数据文件夹
        ├── case_001.npy      # 对应的二分类标签
        ├── case_002.npy
        └── ...
```

#### 1.2 数据格式说明
- **输入数据**：`.npz`格式，每个文件包含5个通道
  - Z1: 雷达反射率因子
  - V1: 径向速度
  - W1: 谱宽
  - SNR1: 信噪比
  - LDR: 线性退偏比

  示例：
  ```python
  {
    'Z1': array([...], shape=(480, 360)),    # 雷达反射率数据
    'V1': array([...], shape=(480, 360)),    # 径向速度数据
    'W1': array([...], shape=(480, 360)),    # 谱宽数据
    'SNR1': array([...], shape=(480, 360)),  # 信噪比数据
    'LDR': array([...], shape=(480, 360))    # 线性退偏比数据
  }
  ```

- **标签数据**：`.npy`格式，二分类标签
  - 0: 背景
  - 1: 目标区域
  ```python
  array([[0, 0, 1, ...],
         [0, 1, 1, ...],
         ...], shape=(480, 360), dtype=np.uint8)
  ```

#### 1.3 数据集划分
使用`data_utils`目录下的数据集划分工具将数据集划分为训练集、验证集和测试集：

```bash
python data_utils/split_dataset.py --data-dir ./data/THI --train-ratio 0.7 --val-ratio 0.15
```

该命令将在`data/THI/`目录下生成三个文件：
- `train.txt`: 训练集文件列表
- `val.txt`: 验证集文件列表
- `test.txt`: 测试集文件列表

文件内容格式示例：
```
Input/case_001.npz    Label/case_001.npy
Input/case_002.npz    Label/case_002.npy
...
```

### 2. 运行方式

项目支持两种运行方式：

#### 2.1 命令行参数方式（推荐）
直接通过命令行参数配置运行参数，适合快速实验不同配置：

```bash
# 训练示例
python train.py --arch DeepLabV3Plus --encoder mobilenet_v2 --batch-size 4 --epochs 50

# 验证示例
python val.py --checkpoint_path "./Train_Result/best_model.ckpt" --val_data "./data/THI/val.txt"

# 预测示例
python predict.py --checkpoint_path "./Train_Result/best_model.ckpt" --input_path "./data/THI/Input/test.npz"
```

#### 2.2 修改默认参数方式
通过直接修改代码中的默认参数运行，适合固定配置的实验：

1. **训练参数配置**
修改 `train.py` 中的默认参数：
```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割训练脚本')
    
    # 修改这里的默认值
    parser.add_argument('--arch', type=str, default='DeepLabV3Plus',
                       choices=['Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 
                               'PSPNet', 'PAN', 'LinkNet', 'MAnet', 'UPerNet', 'Segformer'])
    parser.add_argument('--encoder', type=str, default='mobilenet_v2')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--precision', type=str, default='16-mixed')
    # ... 其他参数
    return parser.parse_args()
```

2. **验证参数配置**
修改 `val.py` 中的默认参数：
```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割验证脚本')
    parser.add_argument('--checkpoint_path', type=str,
                       default='./Train_Result/best_model.ckpt')
    parser.add_argument('--val_data', type=str,
                       default='./data/THI/val.txt')
    # ... 其他参数
    return parser.parse_args()
```

3. **预测参数配置**
修改 `predict.py` 中的默认参数：
```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割预测脚本')
    parser.add_argument('--checkpoint_path', type=str,
                       default='./Train_Result/best_model.ckpt')
    parser.add_argument('--input_path', type=str,
                       default='./data/THI/Input/test.npz')
    # ... 其他参数
    return parser.parse_args()
```

修改完默认参数后，直接运行相应脚本即可：
```bash
# 训练
python train.py

# 验证
python val.py

# 预测
python predict.py
```

## 📝 详细指南

### 训练参数说明

| 参数名 | 说明 | 默认值 | 可选值 |
|--------|------|--------|--------|
| arch | 模型架构 | DeepLabV3Plus | Unet, UnetPlusPlus... |
| encoder | 编码器 | mobilenet_v2 | resnet50, efficientnet-b0... |
| batch-size | 批次大小 | 4 | 根据显存设置 |
| epochs | 训练轮数 | 50 | 自定义 |
| lr | 学习率 | 1e-4 | 建议范围：1e-5~1e-3 |
| precision | 训练精度 | 16-mixed | 32, 16-mixed, bf16-mixed |

### 高级训练技巧

1. **学习率调整**
```bash
python train.py --lr 1e-4 --lr-scheduler cosine --warmup-epochs 5
```

2. **使用预训练模型**
```bash
python train.py --pretrained --weights "./pretrained/model.pth"
```

3. **多GPU训练**
```bash
python train.py --gpus 2 --strategy ddp
```

### 数据增强配置

在 `data_utils/augmentation.py` 中可以自定义数据增强策略：
- 随机旋转
- 随机翻转
- 随机裁剪
- 高斯噪声
- 亮度对比度调整

## 💡 性能优化

1. **内存优化**
- 使用混合精度训练
- 适当的batch size
- 梯度累积

2. **训练速度优化**
- 数据预加载
- 多进程数据加载
- GPU预热

3. **模型优化**
- 模型剪枝
- 知识蒸馏
- 量化

## ❓ 常见问题

1. **显存不足**
   - 减小batch size
   - 使用混合精度训练
   - 选择更轻量级的模型

2. **训练不收敛**
   - 检查学习率设置
   - 验证数据预处理
   - 尝试不同的优化器

3. **预测结果不理想**
   - 增加训练轮数
   - 调整数据增强策略
   - 尝试不同的模型架构

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{meteo-seg,
  title = {Meteorological Target Segmentation},
  year = {2024},
  author = {Ronin_yz},
  url = {https://github.com/yourusername/Meteorological-target-segmentation}
}
``` 