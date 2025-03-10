"""
气象目标隶属度计算模块

此模块提供两个独立的计算器类：
1. 晴空隶属度计算器
2. 气象目标隶属度计算器

主要功能：
1. 加载多通道气象数据
2. 加载多项式拟合参数
3. 分别计算晴空和气象目标隶属度矩阵
4. 批量处理数据
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from glob import glob
from tqdm import tqdm
from abc import ABC, abstractmethod

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

class BaseMembershipCalculator(ABC):
    """隶属度计算器基类"""
    
    def __init__(self, polynomial_dir: str, height_bands: list, target_type: str):
        """
        初始化隶属度计算器基类

        Args:
            polynomial_dir (str): 多项式拟合参数存储目录
            height_bands (list): 高度带列表，每个元素为(min_height, max_height)元组
            target_type (str): 目标类型，'ClearSky' 或 'Meteorological'
        """
        self.height_bands = height_bands
        self.target_type = target_type
        self.polynomials = self._load_all_polynomials(polynomial_dir)
        
    def _load_data(self, file_path: str) -> dict:
        """加载npz格式的气象数据"""
        with np.load(file_path) as data:
            return {k: v for k, v in data.items() if k != 'allow_pickle'}
    
    def _load_one_parameter_polynomials(self, directory: str) -> dict:
        """加载单个参数的多项式拟合参数"""
        polynomials = {}
        for height_min, height_max in self.height_bands:
            band_str = f"{height_min}-{height_max}"
            filename = f"polynomial_params_band_{band_str}_{self.target_type}_拟合参数.pkl"
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    coefficients, domain = pickle.load(f)
                    poly = Polynomial(coefficients)
                    poly.domain = domain
                    polynomials[band_str] = poly
            except FileNotFoundError:
                print(f"警告：找不到文件 {filepath}")
        return polynomials

    def _load_all_polynomials(self, directory: str) -> dict:
        """加载所有参数的多项式拟合参数"""
        parameters = ['SNR1', 'Z1', 'LDR', 'V1', 'W1']
        return {
            param: self._load_one_parameter_polynomials(
                os.path.join(directory, f'{param}多项式拟合参数')
            ) for param in parameters
        }

    def calculate_membership(self, file_path: str) -> dict:
        """计算给定数据的隶属度"""
        # 加载数据
        data_dict = self._load_data(file_path)
        
        # 初始化隶属度矩阵
        shape = (1440, 500)  # 标准数据形状
        membership_degrees = {param: np.zeros(shape) for param in self.polynomials.keys()}
        
        # 识别背景区域
        is_background = np.isnan(data_dict['Z1'])
        valid_indices = ~is_background
        
        # 按高度带计算隶属度
        for height_min, height_max in self.height_bands:
            # 创建高度带掩码
            height_mask = np.zeros_like(data_dict['Z1'])
            height_mask[:, height_min:height_max] = 1
            mask = valid_indices & height_mask.astype(bool)
            
            band_str = f"{height_min}-{height_max}"
            
            # 计算每个参数的隶属度
            for param, polynomials in self.polynomials.items():
                member = polynomials[band_str](data_dict[param][mask])
                membership_degrees[param][mask] = np.nan_to_num(member)
        
        return membership_degrees

    def analyze_membership(self, membership_degrees: dict):
        """分析隶属度结果"""
        parameters = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
        
        print(f"\n=== {self.target_type}隶属度分析结果 ===")
        for param in parameters:
            values = membership_degrees[param]
            
            # 计算有效值的掩码（非零值）
            valid_mask = values != 0
            valid_values = values[valid_mask]
            
            if len(valid_values) > 0:
                print(f"\n{param} 参数统计:")
                print(f"平均值={valid_values.mean():.3f}")
                print(f"最大值={valid_values.max():.3f}")
                print(f"最小值={valid_values.min():.3f}")
            else:
                print(f"\n{param} 参数: 无有效值")
    
    def save_membership_results(self, membership_degrees: dict, output_dir: str, base_name: str):
        """保存隶属度计算结果"""
        # 创建结果目录
        result_dir = os.path.join(output_dir, 'membership_results', self.target_type.lower())
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存隶属度
        result_path = os.path.join(result_dir, f'{base_name}_{self.target_type.lower()}.npz')
        np.savez_compressed(result_path, **membership_degrees)

    def process_file(self, input_file: str, output_dir: str):
        """处理单个文件"""
        try:
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            
            # 计算隶属度
            membership_matrix = self.calculate_membership(input_file)
            
            # 保存隶属度结果
            self.save_membership_results(membership_matrix, output_dir, base_name)
            
            # 分析结果
            print(f"\n处理文件: {base_name}")
            self.analyze_membership(membership_matrix)
            
            return membership_matrix
            
        except Exception as e:
            print(f"\n处理文件 {input_file} 时出错: {str(e)}")
            raise

    def process_directory(self, input_dir: str, output_dir: str):
        """批量处理目录中的所有npz文件"""
        # 获取所有npz文件
        input_files = glob(os.path.join(input_dir, '*.npz'))
        
        if not input_files:
            print(f"警告：在目录 {input_dir} 中未找到npz文件")
            return
        
        print(f"\n开始处理目录：{input_dir}")
        print(f"共找到 {len(input_files)} 个文件")
        
        # 创建输出目录结构
        os.makedirs(os.path.join(output_dir, 'membership_results', self.target_type.lower()), exist_ok=True)
        
        # 批量处理文件
        results = {}
        for file_path in tqdm(input_files, desc=f"计算{self.target_type}隶属度"):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            results[base_name] = self.process_file(file_path, output_dir)
            
        return results

class ClearSkyMembershipCalculator(BaseMembershipCalculator):
    """晴空隶属度计算器"""
    
    def __init__(self, polynomial_dir: str, height_bands: list):
        """
        初始化晴空隶属度计算器

        Args:
            polynomial_dir (str): 多项式拟合参数存储目录
            height_bands (list): 高度带列表
        """
        super().__init__(polynomial_dir, height_bands, 'ClearSky')

class MeteorologicalMembershipCalculator(BaseMembershipCalculator):
    """气象目标隶属度计算器"""
    
    def __init__(self, polynomial_dir: str, height_bands: list):
        """
        初始化气象目标隶属度计算器

        Args:
            polynomial_dir (str): 多项式拟合参数存储目录
            height_bands (list): 高度带列表
        """
        super().__init__(polynomial_dir, height_bands, 'Meteorological')

def visualize_membership_comparison(original_data: dict, clearsky_degrees: dict, 
                                 meteorological_degrees: dict, save_dir: str, base_name: str):
    """
    可视化隶属度结果和原始数据的对比

    Args:
        original_data (dict): 原始数据字典
        clearsky_degrees (dict): 晴空隶属度字典
        meteorological_degrees (dict): 气象目标隶属度字典
        save_dir (str): 保存目录
        base_name (str): 文件基础名称
    """
    parameters = ['Z1', 'V1', 'W1', 'SNR1', 'LDR']
    param_names = {
        'Z1': '雷达反射率因子',
        'V1': '径向速度',
        'W1': '谱宽',
        'SNR1': '信噪比',
        'LDR': '线性退偏比'
    }
    
    # 创建可视化目录
    plot_dir = os.path.join(save_dir, 'plots', base_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # 创建一个大图，5行3列
    fig, axes = plt.subplots(5, 3, figsize=(24, 30))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 为所有图设置相同的colormap范围
    for param_idx, param in enumerate(parameters):
        # 获取每个参数的数据范围
        vmin_orig = np.nanmin(original_data[param])
        vmax_orig = np.nanmax(original_data[param])
        
        # 绘制原始数据
        im0 = axes[param_idx, 0].imshow(original_data[param].T, 
                                      cmap='jet', 
                                      aspect='auto',
                                      vmin=vmin_orig,
                                      vmax=vmax_orig)
        axes[param_idx, 0].set_title(f'{param_names[param]} - 原始数据')
        plt.colorbar(im0, ax=axes[param_idx, 0])
        
        # 绘制晴空隶属度
        im1 = axes[param_idx, 1].imshow(clearsky_degrees[param].T, 
                                      cmap='jet', 
                                      aspect='auto',
                                      vmin=0,
                                      vmax=1)
        axes[param_idx, 1].set_title(f'{param_names[param]} - 晴空隶属度')
        plt.colorbar(im1, ax=axes[param_idx, 1])
        
        # 绘制气象目标隶属度
        im2 = axes[param_idx, 2].imshow(meteorological_degrees[param].T, 
                                      cmap='jet', 
                                      aspect='auto',
                                      vmin=0,
                                      vmax=1)
        axes[param_idx, 2].set_title(f'{param_names[param]} - 气象目标隶属度')
        plt.colorbar(im2, ax=axes[param_idx, 2])
        
        # 设置坐标轴标签
        for ax in axes[param_idx]:
            ax.set_xlabel('方位角')
            ax.set_ylabel('高度')
    
    # 设置总标题
    plt.suptitle(f'文件: {base_name}\n隶属度分析结果', y=0.95, fontsize=16)
    
    # 保存图像
    plt.savefig(os.path.join(plot_dir, f'membership_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()