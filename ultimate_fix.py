import torch
import gymnasium as gym
import numpy as np
import os
import sys

# 确保可以导入 arm_studio
sys.path.append(os.path.abspath("."))

from models.piper.agent import PiperArm, PiperActionWrapper
import mani_skill.envs

def diagnostic_and_fix():
    print("开始诊断环境维度...")
    
    # 1. 创建环境探测维度
    try:
        env = gym.make("Empty-v1", obs_mode="state", robot_uids="piper_arm")
        obs, _ = env.reset()
        obs_dim = obs.shape[0]
        print(f"探测到环境观测维度 (obs_dim): {obs_dim}")
        env.close()
    except Exception as e:
        print(f"探测维度失败: {e}")
        obs_dim = 16 # 回退到默认
    
    action_dim = 7
    model_path = "policy.pi0"

    # 2. 生成匹配的模型
    print(f"正在生成模型: {model_path} (obs:{obs_dim} -> act:{action_dim})")
    
    state_dict = {
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'model': {
            '0.weight': torch.randn(256, obs_dim),
            '0.bias': torch.randn(256),
            '2.weight': torch.randn(256, 256),
            '2.bias': torch.randn(256),
            '4.weight': torch.randn(action_dim, 256),
            '4.bias': torch.randn(action_dim),
        }
    }
    
    torch.save(state_dict, model_path)
    print("--- 修复完成 ---")
    print("请按以下顺序执行：")
    print("1. 关闭所有终端")
    print("2. 终端 A 启动服务器: python inference/server.py --model policy.pi0")
    print("3. 终端 B 启动客户端: python scripts/run_inference.py --mode remote --addr localhost")

if __name__ == "__main__":
    diagnostic_and_fix()
