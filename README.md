# 气象目标分割项目

## 📋 目录
- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [数据集准备](#数据集准备)
- [隶属度计算](#隶属度计算)
- [训练模型](#训练模型)
- [模型预测](#模型预测)
- [模型评估](#模型评估)
- [修改默认参数](#修改默认参数)
- [工具函数说明](#工具函数说明)
- [常见问题](#常见问题)

## 项目简介

本项目是一个专业的气象目标分割系统，基于深度学习技术，用于处理和分析气象雷达数据。系统能够自动识别和分割气象目标，为气象分析和预测提供重要支持。

### 特性

- **多模型支持**
  - 支持多种主流分割模型：Unet、UnetPlusPlus、DeepLabV3、DeepLabV3Plus等
  - 支持30+种编码器网络：ResNet系列、EfficientNet系列等
  
- **数据处理**
  - 支持多通道气象数据（Z1、V1、W1、SNR1、LDR）
  - 隶属度计算支持：原始数据、晴空隶属度、气象目标隶属度
  - 专业的数据预处理流程

- **训练优化**
  - 支持混合精度训练（FP16/BF16）
  - 断点续训功能
  - 自动学习率调整
  - 多种评估指标：IoU、Dice、精确率、召回率、检测率、虚警率

- **可视化功能**
  - 训练过程实时监控
  - 预测结果可视化
  - 性能指标图表展示

## 环境配置

### 系统要求
- Windows 10/11 x64 或 Linux
- NVIDIA GPU (建议8GB+显存)
- CUDA 11.7+
- Python 3.8+

### 安装步骤

1. **创建并激活虚拟环境**
```bash
# 使用conda创建环境
conda create -n segmentation python=3.8
conda activate segmentation

# 或使用venv
python -m venv segmentation
# Windows激活
.\segmentation\Scripts\activate
# Linux激活
source segmentation/bin/activate
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

## 项目结构

```
Meteorological-target-segmentation/
├── data/                       # 数据目录
│   ├── THI/                   # 气象数据集
│   │   ├── Input/             # 输入数据
│   │   ├── Label/             # 标签数据
│   │   ├── train.txt          # 训练集列表
│   │   ├── val.txt            # 验证集列表
│   │   └── test.txt           # 测试集列表
│   └── THI_extension/         # 扩展数据集（仅用于预测）
├── data_utils/                # 数据处理工具
│   ├── export_membership/     # 隶属度计算模块
│   ├── dataset.py             # 数据集类定义
│   ├── augmentation.py        # 数据增强工具
│   ├── split_data.py          # 数据集划分工具
│   ├── view_npy.py            # NPY文件查看工具
│   └── visualize_dataset.py   # 数据集可视化工具
├── models/                     # 模型定义
│   ├── __init__.py            # 模型初始化
│   └── model.py               # 模型类定义
├── utils/                     # 工具函数
│   ├── checkpoint.py          # 检查点工具
│   ├── check_ckpt.py          # 检查点检查工具
│   └── visualization.py       # 可视化工具
├── Result/                    # 结果保存目录
│   ├── Train/                 # 训练结果
│   ├── Predict/               # 预测结果
│   └── Evaluation/            # 评估结果
├── train.py                   # 训练脚本
├── val.py                     # 验证评估脚本
├── predict.py                 # 预测脚本
└── requirements.txt           # 项目依赖
```

## 数据集准备

### 数据集结构
项目使用THI（气象目标）数据集，数据集应按以下结构组织：

- 输入数据：`npz`格式文件，包含多个通道（Z1、V1、W1、SNR1、LDR）
- 标签数据：`npy`格式文件，包含分割掩码（0表示降水区域，1表示背景）

### 数据集划分
数据集划分为训练集、验证集和测试集，通过文本文件指定：

- `train.txt`：训练数据列表
- `val.txt`：验证数据列表
- `test.txt`：测试数据列表

每个文本文件中的格式为：`输入文件路径 <Tab> 标签文件路径`

例如：
```
H:\Work Hub\Meteorological-target-segmentation\data\THI\Input\Z10_20230829.npz	H:\Work Hub\Meteorological-target-segmentation\data\THI\Label\Z10_20230829.npy
```

### 数据集可视化
可以使用项目提供的数据集可视化工具查看数据：

```bash
python data_utils/visualize_dataset.py --data-txt ./data/THI/train.txt --output-dir ./Result/visualize
```

## 隶属度计算

本项目支持三种数据输入方式：

1. **原始数据**：直接使用原始通道数据
2. **晴空隶属度**：计算每个像素属于晴空的隶属度
3. **气象目标隶属度**：计算每个像素属于气象目标的隶属度

隶属度计算基于多项式拟合参数，这些参数存储在 `data_utils/export_membership/多项式拟合参数` 目录中。

## 训练模型

### 基本用法

```bash
python train.py --arch DeepLabV3Plus --encoder mobilenet_v2 --batch-size 4 --epochs 10 --lr 0.0001 
```

### 完整参数说明

```bash
python train.py --help
```

主要参数包括：

- `--arch`：模型架构，如 `DeepLabV3Plus`、`Unet` 等
- `--encoder`：编码器网络，如 `mobilenet_v2`、`resnet34` 等
- `--batch-size`：批次大小
- `--epochs`：训练轮数
- `--lr`：学习率
- `--data-dir`：数据集根目录
- `--output-dir`：输出目录
- `--precision`：训练精度（32、16-mixed、bf16-mixed）
- `--calculate-membership`：隶属度计算方式（none、clearsky、meteorological）
- `--resume-from`：从检查点恢复训练

### 示例

训练使用晴空隶属度的DeepLabV3Plus模型：

```bash
python train.py --arch DeepLabV3Plus --encoder mobilenet_v2 --batch-size 4 --epochs 10 --lr 0.0001 --calculate-membership clearsky
```

训练使用气象目标隶属度的Unet模型：

```bash
python train.py --arch Unet --encoder resnet34 --batch-size 4 --epochs 10 --lr 0.0001 --calculate-membership meteorological
```

从检查点恢复训练：

```bash
python train.py --resume-from ./Result/Train/met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918/checkpoints/THI_last.ckpt --epochs 20
```

## 模型预测

### 基本用法

```bash
python predict.py --checkpoint_path ./Result/Train/met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918/checkpoints/THI_best_model.ckpt --input_path ./data/THI_extension
```

### 完整参数说明

```bash
python predict.py --help
```

主要参数包括：

- `--checkpoint_path`：模型检查点路径
- `--input_path`：输入数据路径（单个npz文件或目录）
- `--output_dir`：输出目录
- `--device`：使用的设备（cuda/cpu）

预测时不需要指定隶属度计算参数，因为这些参数已经保存在模型检查点中。

### 输出
预测结果将保存在指定的输出目录中，包括：

- `images/`：可视化结果图像
- `arrays/`：预测结果数组（npy文件）

## 模型评估

### 基本用法

```bash
python val.py --checkpoint_path ./Result/Train/met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918/checkpoints/THI_best_model.ckpt --val_data ./data/THI/test.txt
```

### 完整参数说明

```bash
python val.py --help
```

主要参数包括：

- `--checkpoint_path`：模型检查点路径
- `--val_data`：验证数据列表路径
- `--output_dir`：输出目录
- `--batch_size`：批次大小
- `--visualize_samples`：可视化样本数量

### 评估指标

评估脚本会计算多种性能指标：

- **IoU (Jaccard)**：交并比
- **Dice系数 (F1)**：精确率和召回率的调和平均数
- **准确率**：正确预测的像素比例
- **精确率**：真正例占所有预测为正例的比例
- **召回率**：真正例占所有实际正例的比例
- **检测率 (Pd)**：成功识别的降水区域比例，Pd = Ns / (Ns + Nf)
  - Ns：成功识别的降水区域数量
  - Nf：被错误分类为杂波和噪声的降水区域数量
- **虚警率 (Pfa)**：误报的杂波和噪声区域比例，Pfa = Ni / (NT - Ns - Nf)
  - Ni：被错误分类为降水的杂波和噪声区域数量
  - NT：总像素数

### 输出
评估结果将保存在指定的输出目录中，包括：

- 可视化样本图像
- `metrics.json`：评估指标JSON文件
- `metrics.csv`：评估指标CSV文件
- `evaluation_summary.txt`：评估结果摘要

## 修改默认参数

除了使用命令行参数外，您还可以直接修改脚本中的默认参数，这对于固定配置的实验特别有用。

### 1. 修改训练脚本默认参数

编辑 `train.py` 中的 `parse_args` 函数：

```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割训练脚本')
    
    # 修改这里的默认值
    parser.add_argument('--arch', type=str, default='DeepLabV3Plus',
                       choices=['Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 
                               'PSPNet', 'PAN', 'LinkNet', 'MAnet', 'UPerNet', 'Segformer'])
    parser.add_argument('--encoder', type=str, default='mobilenet_v2')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--calculate-membership', type=str, default='meteorological',
                       choices=['none', 'clearsky', 'meteorological'])
    # ... 其他参数
    return parser.parse_args()
```

### 2. 修改预测脚本默认参数

编辑 `predict.py` 中的 `parse_args` 函数：

```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割预测脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="Result/Train/met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918/THI_best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--input_path', type=str,
                        default="./data/THI_extension",
                        help='输入数据路径(单个npz文件或目录)')
    # ... 其他参数
    return parser.parse_args()
```

### 3. 修改评估脚本默认参数

编辑 `val.py` 中的 `parse_args` 函数：

```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割验证脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="Result/Train/met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918/THI_best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--val_data', type=str,
                        default="./data/THI/test.txt",
                        help='验证集文件列表路径')
    # ... 其他参数
    return parser.parse_args()
```

修改默认参数后，可以直接运行脚本而无需指定命令行参数：

```bash
python train.py
python predict.py
python val.py
```

## 工具函数说明

项目包含两个主要的工具目录：`data_utils` 和 `utils`，它们提供了多种实用功能。

### data_utils 目录

#### 1. dataset.py
- `ThiDataset` 类：核心数据集类，负责加载和预处理数据
  - 支持多通道数据处理
  - 支持隶属度计算
  - 提供数据填充和裁剪功能
  - 处理无效值（NaN）

#### 2. augmentation.py
- `get_training_augmentation()`：提供训练数据增强
  - 水平翻转
  - 垂直翻转
  - 旋转、缩放和平移
- `get_validation_augmentation()`：验证集数据处理
- `get_preprocessing()`：数据预处理函数

#### 3. split_data.py
- 数据集划分工具，将数据集分为训练集、验证集和测试集
- 用法：
  ```bash
  python data_utils/split_data.py --root_dir ./data/THI --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
  ```
- 生成 train.txt、val.txt 和 test.txt 文件

#### 4. view_npy.py
- NPY文件查看工具，用于可视化标签文件
- 用法：
  ```bash
  python data_utils/view_npy.py --file ./data/THI/Label/Z1_20230805.npy
  ```
- 显示数组信息和可视化结果

#### 5. visualize_dataset.py
- 数据集可视化工具，用于查看输入数据和标签
- 用法：
  ```bash
  python data_utils/visualize_dataset.py --data-txt ./data/THI/train.txt --output-dir ./Result/visualize
  ```

#### 6. export_membership
- 隶属度计算模块，包含两种隶属度计算器：
  - `ClearSkyMembershipCalculator`：晴空隶属度计算器
  - `MeteorologicalMembershipCalculator`：气象目标隶属度计算器

### utils 目录

#### 1. visualization.py
- `visualize_all_channels()`：可视化所有通道的图像和掩码
  - 支持旋转显示
  - 支持保存高质量图像
  - 显示真实掩码和预测掩码

#### 2. checkpoint.py
- `check_checkpoint()`：检查和显示检查点文件的内容
  - 显示模型结构
  - 显示超参数
  - 显示训练状态
- 用法：
  ```bash
  python utils/checkpoint.py ./Result/Train/met_THI_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep10_bs4_p16mixed_20250306_1918/checkpoints/THI_best_model.ckpt
  ```

#### 3. check_ckpt.py
- 检查点检查工具，提供更详细的检查点分析
- 显示模型层结构
- 显示参数数量
- 分析优化器状态

这些工具函数可以帮助您更好地理解数据、模型和训练过程，提高开发和调试效率。

## 常见问题

### 1. 路径错误
在Windows系统上，确保所有路径使用正斜杠(/)而不是反斜杠(\\)，或者使用Python的`os.path`函数处理路径。

### 2. 训练时显存不足
可以尝试：
- 减小批次大小（`--batch-size`）
- 使用混合精度训练（`--precision 16-mixed`）
- 选择更轻量级的编码器（如`mobilenet_v2`）

### 3. 模型加载错误
确保使用的是完整的检查点路径，检查是否包含了`*.ckpt`扩展名。 