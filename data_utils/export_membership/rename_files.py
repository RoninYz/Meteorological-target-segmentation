"""
重命名脚本

用于批量重命名隶属度结果文件，去除特定后缀。
"""

import os
from glob import glob
from tqdm import tqdm

def rename_files(directory: str, suffix: str = '_clearsky'):
    """
    重命名目录中的文件，去除指定后缀

    Args:
        directory (str): 文件目录
        suffix (str): 要去除的后缀
    """
    # 获取所有npz文件
    files = glob(os.path.join(directory, f'*{suffix}.npz'))
    
    if not files:
        print(f"警告：在目录 {directory} 中未找到包含后缀 {suffix} 的npz文件")
        return
    
    print(f"\n开始重命名文件：")
    print(f"目标目录：{directory}")
    print(f"要去除的后缀：{suffix}")
    print(f"共找到 {len(files)} 个文件")
    
    # 批量重命名
    for old_path in tqdm(files, desc="重命名进度"):
        # 获取文件名和目录
        directory = os.path.dirname(old_path)
        old_name = os.path.basename(old_path)
        
        # 构建新文件名
        new_name = old_name.replace(suffix, '')
        new_path = os.path.join(directory, new_name)
        
        try:
            # 如果新文件名已存在，则跳过
            if os.path.exists(new_path):
                print(f"跳过 {old_name}: 目标文件 {new_name} 已存在")
                continue
                
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"已重命名: {old_name} -> {new_name}")
            
        except Exception as e:
            print(f"重命名 {old_name} 时出错: {str(e)}")

def main():
    """主函数"""
    # 配置参数
    directory = "data/THI_fuzzy/membership_results/meteorological/Input"  # 晴空隶属度结果目录
    suffix = '_meteorological'  # 要去除的后缀
    
    # 执行重命名
    rename_files(directory, suffix)

if __name__ == "__main__":
    main() 