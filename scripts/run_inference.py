import sys
import os
import argparse
import numpy as np
import time
import torch

# 确保可以导入 arm_studio 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from models.piper.agent import PiperArm, PiperActionWrapper
from inference.client import InferenceClient
from inference.server import PI0Policy
import mani_skill.envs 

def main():
    parser = argparse.ArgumentParser(description="ArmStudio Inference Tool")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "remote"],
                        help="推理模式: 'local' (本地) 或 'remote' (远程)")
    parser.add_argument("--addr", type=str, default="localhost",
                        help="远程服务器地址 (仅 remote 模式)")
    parser.add_argument("--port", type=int, default=5555,
                        help="服务器端口")
    parser.add_argument("--model", type=str, default="policy.pi0",
                        help="本地模型路径 (仅 local 模式)")
    parser.add_argument("--ctrl_mode", type=str, default="joint", choices=["joint", "pose"],
                        help="控制空间: 'joint' 或 'pose'")
    parser.add_argument("--binary_gripper", action="store_true",
                        help="使用二值化夹爪控制")
    parser.add_argument("--render", action="store_true", default=True,
                        help="是否开启渲染")
    args = parser.parse_args()

    # 1. 初始化策略/客户端
    if args.mode == "remote":
        print(f"正在启动远程推理模式，连接至 {args.addr}:{args.port}...")
        policy = InferenceClient(host=args.addr, port=args.port)
    else:
        print(f"正在启动本地推理模式，加载模型 {args.model}...")
        policy = PI0Policy(model_path=args.model)

    # 2. 环境初始化
    control_mode = "pd_ee_pose" if args.ctrl_mode == "pose" else "pd_joint_pos"
    
    env = gym.make(
        "Empty-v1", 
        obs_mode="state",
        control_mode=control_mode,
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human" if args.render else None
    )
    
    # 统一应用 PiperActionWrapper
    env = PiperActionWrapper(env, binary_gripper=args.binary_gripper)

    obs, _ = env.reset()
    print("\n" + "="*50)
    print(f"模式: {args.mode.upper()} | 控制: {args.ctrl_mode.upper()}")
    print("="*50)
    print("正在运行推理循环，按 Ctrl+C 退出...")

    try:
        while True:
            if args.render:
                env.render()
                # 检查窗口是否关闭
                window = env.unwrapped.viewer.window if env.unwrapped.viewer else None
                if window and window.key_down('escape'):
                    break

            # 1. 获取推理动作
            # 消除 ManiSkill 自带的 Batch 维度 (1, 16) -> (16,)
            if len(obs.shape) > 1 and obs.shape[0] == 1:
                obs = obs[0]
                
            action = policy.act(obs)

            if action is None:
                print("警告: 未收到有效动作，跳过本帧")
                time.sleep(0.01)
                continue

            # 2. 执行物理步进
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    finally:
        if hasattr(policy, 'close'):
            policy.close()
        env.close()

if __name__ == "__main__":
    main()
