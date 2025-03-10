#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import sys
import pprint

def check_checkpoint(ckpt_path):
    """加载并显示checkpoint文件的内容结构"""
    try:
        # 加载checkpoint文件
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
        # 打印checkpoint的顶层键
        print("=" * 50)
        print(f"Checkpoint文件: {ckpt_path}")
        print("=" * 50)
        print("\n顶层键:")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        # 获取更多详细信息
        if 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            print("\n模型状态字典中的键数量:", len(model_state))
            print("\n模型层的前10个键:")
            for i, key in enumerate(list(model_state.keys())[:10]):
                shape = model_state[key].shape
                print(f"  {i+1}. {key}: 形状 {shape}")
        
        if 'hyper_parameters' in checkpoint:
            print("\n超参数:")
            hyper_params = checkpoint['hyper_parameters']
            pprint.pprint(hyper_params, indent=2)
            
        if 'optimizer_states' in checkpoint:
            print("\n优化器状态存在")
            print(f"优化器类型: {checkpoint['optimizer_states'][0].get('type', '未知')}")
        
        if 'epoch' in checkpoint:
            print(f"\n训练轮数: {checkpoint['epoch']}")
        
        if 'global_step' in checkpoint:
            print(f"全局步数: {checkpoint['global_step']}")
            
    except Exception as e:
        print(f"读取checkpoint文件时出错: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "Train_Result\clearsky_DeepLabV3Plus_mobilenet_v2_lr0.0001_ep1_bs4_p16mixed_20250305_1900\clearsky_best_model.ckpt"
    
    check_checkpoint(ckpt_path) 