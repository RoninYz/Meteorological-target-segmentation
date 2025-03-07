"""
隶属度计算主程序

此程序用于计算气象数据的晴空和气象目标隶属度。
使用两个独立的计算器分别计算两种隶属度。
"""

import os
import numpy as np
from tqdm import tqdm
from membership_calculators import (
    ClearSkyMembershipCalculator,
    MeteorologicalMembershipCalculator,
    visualize_membership_comparison
)

def process_data(input_dir: str, output_dir: str, polynomial_dir: str, height_bands: list):
    """
    处理气象数据，计算晴空和气象目标隶属度

    Args:
        input_dir (str): 输入数据目录
        output_dir (str): 输出结果目录
        polynomial_dir (str): 多项式拟合参数目录
        height_bands (list): 高度带列表
    """
    # 创建两个计算器实例
    clearsky_calculator = ClearSkyMembershipCalculator(polynomial_dir, height_bands)
    meteorological_calculator = MeteorologicalMembershipCalculator(polynomial_dir, height_bands)
    
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, 'membership_results', 'clearsky'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'membership_results', 'meteorological'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # 获取输入文件列表
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    if not input_files:
        print(f"警告：在目录 {input_dir} 中未找到npz文件")
        return
    
    print(f"\n开始处理目录：{input_dir}")
    print(f"共找到 {len(input_files)} 个文件")
    
    # 使用进度条显示处理进度
    with tqdm(total=len(input_files), desc="处理进度") as pbar:
        for filename in input_files:
            input_file = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                # 加载原始数据用于可视化
                with np.load(input_file) as data:
                    original_data = {k: v for k, v in data.items() if k != 'allow_pickle'}
                
                # 计算晴空隶属度
                clearsky_degrees = clearsky_calculator.calculate_membership(input_file)
                clearsky_calculator.save_membership_results(clearsky_degrees, output_dir, base_name)
                
                # 计算气象目标隶属度
                meteorological_degrees = meteorological_calculator.calculate_membership(input_file)
                meteorological_calculator.save_membership_results(meteorological_degrees, output_dir, base_name)
                
                # 可视化结果对比
                visualize_membership_comparison(
                    original_data,
                    clearsky_degrees,
                    meteorological_degrees,
                    output_dir,
                    base_name
                )
                
                # 分析结果
                print(f"\n处理文件: {base_name}")
                clearsky_calculator.analyze_membership(clearsky_degrees)
                meteorological_calculator.analyze_membership(meteorological_degrees)
                
            except Exception as e:
                print(f"\n处理文件 {filename} 时出错: {str(e)}")
                continue
                
            finally:
                pbar.update(1)
    
    print(f"\n处理完成！结果已保存至 {os.path.abspath(output_dir)} 目录")
    print(f"- 晴空隶属度结果：{os.path.join(output_dir, 'membership_results', 'clearsky')}")
    print(f"- 气象目标隶属度结果：{os.path.join(output_dir, 'membership_results', 'meteorological')}")
    print(f"- 可视化结果：{os.path.join(output_dir, 'plots')}")

def main():
    """主函数"""
    # 配置参数
    height_bands = [(0, 100), (50, 150), (125, 180), (100, 200)]
    polynomial_dir = "data_utils/export_membership/多项式拟合参数"
    input_dir = "./data/THI/Input"  # 输入目录
    output_dir = "./data/THI_fuzzy_v3"  # 输出目录
    
    # 处理数据
    process_data(input_dir, output_dir, polynomial_dir, height_bands)

if __name__ == "__main__":
    main()