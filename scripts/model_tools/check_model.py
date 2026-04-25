import torch
import sys
import os
import argparse

def check_model(model_path):
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    print(f"正在检查模型: {model_path}")
    try:
        # 强制加载到 CPU 以便检查
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("\n模型元数据:")
        if 'obs_dim' in state_dict:
            print(f"  - 观测维度 (obs_dim): {state_dict['obs_dim']}")
        if 'action_dim' in state_dict:
            print(f"  - 动作维度 (action_dim): {state_dict['action_dim']}")
        
        print("\n网络权重:")
        if 'model_state' in state_dict:
            model_weights = state_dict['model_state']
        elif 'model' in state_dict:
            model_weights = state_dict['model']
        else:
            model_weights = state_dict
            
        for key, value in model_weights.items():
            if hasattr(value, 'shape'):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: (非 Tensor 数据)")
                
        print("\n检查完成：模型文件格式似乎正确。")
        
    except Exception as e:
        print(f"检查模型时发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArmStudio 模型检查工具")
    parser.add_argument("model_path", type=str, nargs='?', default="outputs/checkpoints/vision/final_vision_policy.pth", help="模型文件路径")
    args = parser.parse_args()
    
    check_model(args.model_path)
