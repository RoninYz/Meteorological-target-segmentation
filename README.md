# 气象目标分割项目文档

## 项目介绍

这是一个基于深度学习的气象目标分割项目，旨在对雷达气象数据进行自动化分割，实现对气象目标的精确识别。本项目基于 Segmentation Models PyTorch (SMP) 和 PyTorch Lightning 框架开发，支持多种先进的语义分割模型架构和编码器网络，能够处理多通道气象数据并实现高效训练和预测。

### 项目特点

1. **多种分割模型支持**：支持 Unet、UnetPlusPlus、DeepLabV3、DeepLabV3Plus、FPN、PSPNet、PAN、LinkNet、MAnet、UPerNet、Segformer 等多种先进的语义分割模型架构。

2. **丰富的主干网络**：支持 ResNet、ResNeXt、EfficientNet、MobileNet、DenseNet、SENet、VGG、Inception 等各系列主干网络。

3. **多通道数据处理**：专为气象数据设计，可处理包含 Z1、V1、W1、SNR1、LDR 等多个通道的雷达数据。

4. **混合精度训练**：支持 16 位和 32 位混合精度训练，提高训练效率。

5. **完整工作流**：包含数据处理、模型训练、验证评估和预测可视化的完整工作流程。

6. **数据增强**：集成了基于 Albumentations 的数据增强策略，提高模型泛化能力。

## 环境安装

### 依赖环境要求

本项目依赖以下主要库：

- Python 3.6+ 
- PyTorch 1.8+
- PyTorch Lightning 2.0+
- Segmentation Models PyTorch (SMP)
- Albumentations
- NumPy
- Matplotlib
- pandas
- tqdm

### 安装步骤

1. **创建虚拟环境**（推荐使用 conda）：

```bash
conda create -n meteor_seg python=3.8
conda activate meteor_seg
```

2. **安装 PyTorch**：

根据您的 CUDA 版本安装适当版本的 PyTorch，请访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取具体命令。例如，对于 CUDA 11.8：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. **安装其他依赖**：

```bash
pip install pytorch-lightning segmentation-models-pytorch albumentations pandas matplotlib tqdm
```

## 数据格式和处理

### 数据格式

- **输入数据**：npz 格式文件，包含 5 个通道（Z1、V1、W1、SNR1、LDR）的气象雷达数据。
- **标签数据**：npy 格式文件，包含二分类标签（0:背景，1:目标）。
- **数据列表**：train.txt、val.txt 和 test.txt 包含输入数据和对应标签的文件路径对。

### 数据结构

数据应组织为以下结构：

```
data/
  ├── THI/
  │   ├── Input/              # 包含 npz 格式的输入数据
  │   │   └── *.npz
  │   ├── Label/              # 包含 npy 格式的标签数据
  │   │   └── *.npy
  │   ├── train.txt           # 训练数据列表 
  │   ├── val.txt             # 验证数据列表
  │   └── test.txt            # 测试数据列表
```

数据列表文件中每行包含一对制表符分隔的路径：输入文件路径和对应的标签文件路径。

## 模型训练

### 运行方式

项目支持两种运行方式：

#### 1. 命令行参数方式

使用命令行参数可以灵活地配置训练参数，适合尝试不同的配置。

```bash
python train.py --arch DeepLabV3Plus --encoder mobilenet_v2 --batch-size 4 --epochs 10 --lr 1e-4
```

#### 2. 直接修改代码默认参数

除了使用命令行参数外，您也可以直接修改`train.py`中的默认参数值，然后直接运行代码。这种方式更简单直接，适合固定配置的实验。

打开`train.py`文件，找到`parse_args`函数，修改其中的默认参数：

```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割训练脚本')
    
    # 修改这里的默认值
    parser.add_argument('--arch', type=str, default='DeepLabV3Plus',
                        choices=['Unet', 'UnetPlusPlus', 'DeepLabV3', 'DeepLabV3Plus', ...],
                        help='分割模型架构')
    parser.add_argument('--encoder', type=str, default='mobilenet_v2', ...)
    parser.add_argument('--batch-size', type=int, default=4, ...)
    # ... 其他参数
```

修改完成后，直接运行：

```bash
python train.py
```

### 主要训练参数

- `--arch`：分割模型架构，可选值包括 Unet、UnetPlusPlus、DeepLabV3、DeepLabV3Plus、FPN、PSPNet、PAN、LinkNet、MAnet、UPerNet、Segformer
- `--encoder`：主干网络，支持多种选择如 resnet18、mobilenet_v2、efficientnet-b0 等
- `--batch-size`：批次大小
- `--epochs`：训练轮数
- `--lr`：学习率
- `--workers`：数据加载线程数
- `--resume-from`：继续训练的检查点路径
- `--train-data`：训练数据集路径
- `--val-data`：验证数据集路径
- `--test-data`：测试数据集路径
- `--output-dir`：训练结果保存的根目录路径
- `--precision`：训练精度，可选值包括 32、16-mixed、bf16-mixed

### 示例训练命令

训练一个使用 DeepLabV3Plus 架构和 MobileNetV2 主干网络的模型：

```bash
python train.py --arch DeepLabV3Plus --encoder mobilenet_v2 --batch-size 4 --epochs 20 --lr 1e-4 --workers 4 --precision 16-mixed
```

继续从现有检查点训练：

```bash
python train.py --arch DeepLabV3Plus --encoder mobilenet_v2 --batch-size 4 --epochs 10 --lr 5e-5 --resume-from "./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep20_bs4_p16mixed/checkpoints/best_model.ckpt"
```

## 模型评估

### 运行方式

评估模型同样支持两种方式：

#### 1. 命令行参数方式

```bash
python val.py --checkpoint_path PATH_TO_CHECKPOINT --val_data PATH_TO_TEST_DATA
```

#### 2. 直接修改代码默认参数

您可以直接修改`val.py`中的默认参数，然后直接运行：

```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割验证脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep1_bs4_p16mixed/best_model.ckpt",
                        help='模型检查点路径')
    # ... 修改其他默认参数
```

修改后直接运行：

```bash
python val.py
```

### 主要评估参数

- `--checkpoint_path`：模型检查点路径
- `--val_data`：验证集文件列表路径
- `--output_dir`：评估结果保存的根目录路径
- `--device`：使用的设备 (cuda/cpu)
- `--batch_size`：批次大小
- `--workers`：数据加载器的工作线程数
- `--visualize_samples`：可视化样本数量

### 示例评估命令

评估一个训练好的模型：

```bash
python val.py --checkpoint_path "./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep20_bs4_p16mixed/best_model.ckpt" --val_data "./data/THI/test.txt" --output_dir "./Evaluation_Result" --visualize_samples 10
```

## 模型预测

### 运行方式

同样支持两种运行方式：

#### 1. 命令行参数方式

```bash
python predict.py --checkpoint_path PATH_TO_CHECKPOINT --input_path PATH_TO_INPUT_DATA
```

#### 2. 直接修改代码默认参数

编辑`predict.py`文件中的默认参数：

```python
def parse_args():
    parser = argparse.ArgumentParser(description='气象目标分割预测脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default="./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep1_bs4_p16mixed/best_model.ckpt",
                        help='模型检查点路径')
    parser.add_argument('--input_path', type=str,
                        default="./data/THI/Input/Z1_20230805.npz",
                        help='输入数据路径(单个npz文件或目录)')
    # ... 修改其他默认参数
```

然后直接运行：

```bash
python predict.py
```

### 主要预测参数

- `--checkpoint_path`：模型检查点路径
- `--input_path`：输入数据路径(单个npz文件或目录)
- `--output_dir`：预测结果保存的根目录路径
- `--device`：使用的设备 (cuda/cpu)

### 示例预测命令

对单个文件进行预测：

```bash
python predict.py --checkpoint_path "./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep20_bs4_p16mixed/best_model.ckpt" --input_path "./data/THI/Input/Z1_20230805.npz" --output_dir "./Predict_Result"
```

对目录中的所有文件进行预测：

```bash
python predict.py --checkpoint_path "./Train_Result/DeepLabV3Plus_mobilenet_v2_lr0.0001_ep20_bs4_p16mixed/best_model.ckpt" --input_path "./data/THI/Input" --output_dir "./Predict_Result"
```

## 输出结果说明

### 训练结果

训练结果保存在 `Train_Result` 目录下，包括：

- `checkpoints/`：保存模型检查点文件
- `best_model.ckpt`：最佳模型文件
- TensorBoard 日志文件：记录训练过程中的各项指标

### 评估结果

评估结果保存在 `Evaluation_Result` 目录下，包括：

- `metrics.json`：包含各项评估指标（IoU、Dice系数、准确率等）
- `visualizations/`：包含原始图像、真实掩码和预测结果的可视化图像

### 预测结果

预测结果保存在 `Predict_Result` 目录下，包括：

- `images/`：包含可视化结果图像，显示多个通道数据和预测掩码
- `arrays/`：包含预测掩码的 NumPy 数组文件 (.npy)

## 性能指标

模型评估包括以下主要指标：

- **IoU (Intersection over Union)**：交并比，衡量预测掩码与真实掩码的重叠程度
- **Dice 系数**：等同于 F1 分数，评估分割质量
- **准确率 (Accuracy)**：分类正确的像素比例
- **精确率 (Precision)**：预测为目标的像素中实际为目标的比例
- **召回率 (Recall)**：实际为目标的像素中被正确预测为目标的比例

## 常见问题

1. **CUDA 内存不足**：
   - 减小批次大小 (--batch-size)
   - 使用混合精度训练 (--precision 16-mixed)
   - 选择较小的主干网络 (如 mobilenet_v2 代替 resnet101)

2. **训练不稳定**：
   - 减小学习率 (--lr)
   - 检查数据预处理和归一化操作
   - 查看是否有异常值或缺失值

3. **预测结果不理想**：
   - 增加训练轮数 (--epochs)
   - 尝试不同的模型架构和主干网络
   - 增加数据增强的多样性
   - 检查标签质量

## 高级用法

### 自定义数据集

如果您有自己的数据集，需要：

1. 将数据转换为适当的格式（npz 和 npy）
2. 创建数据列表文件（train.txt、val.txt、test.txt）
3. 根据需要修改 `data_utils/dataset.py` 中的数据加载逻辑

### 模型导出

训练好的模型可以导出为 ONNX 或 TorchScript 格式，方便在生产环境中部署。示例代码请参考 PyTorch 官方文档。

## 项目扩展

1. **增加新的模型架构**：可通过在 `models/model.py` 中扩展模型定义
2. **支持新的数据格式**：修改 `data_utils/dataset.py` 中的数据加载逻辑
3. **增加新的评估指标**：在 `val.py` 中添加新的评估函数

## 注意事项

- 请确保您有足够的 GPU 内存进行训练
- 对于大型数据集，推荐使用混合精度训练
- 在生产环境中部署前，请充分验证模型的性能和稳定性

## 参考资料

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- [Albumentations](https://albumentations.ai/)

## 联系方式

如有任何问题或建议，请联系项目维护者。

---

希望本文档能够帮助您了解和使用本气象目标分割项目。祝您使用愉快！ 