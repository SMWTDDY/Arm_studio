#!/usr/bin/env python3
"""调试观察结构和摄像机输出"""
import os
import sys
os.environ["SAPIEN_VULKAN_DEVICE"] = "0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import gymnasium as gym
import numpy as np
import torch
from environments.conveyor_env import PiperConveyorEnv

# 直接实例化环境
env = PiperConveyorEnv(
    obs_mode="rgb+state",
    robot_uids="piper_arm",
    control_mode="pd_joint_pos",
    render_mode="rgb_array"
)

print(f"✓ Environment created")
print(f"✓ Agent: {env.agent}")

# 重置环境
try:
    obs, info = env.reset()
    print(f"✓ Reset successful")
except Exception as e:
    print(f"✗ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查观察结构
print(f"\nObservation keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
if isinstance(obs, dict):
    for key, value in obs.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    print(f"    {subkey}:")
                    for subsubkey, subsubvalue in subvalue.items():
                        if hasattr(subsubvalue, 'shape'):
                            print(f"      {subsubkey}: shape={subsubvalue.shape}, dtype={subsubvalue.dtype}, type={type(subsubvalue).__name__}")
                        else:
                            print(f"      {subsubkey}: {type(subsubvalue).__name__}")
                elif hasattr(subvalue, 'shape'):
                    print(f"    {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}")
                else:
                    print(f"    {subkey}: {type(subvalue).__name__}")
        elif hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__}")

# 执行一个动作
print(f"\nTaking a step...")
action = np.array([0.5, -0.8, 0.5, 0, 0, 0, 0.5, 0.5], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print(f"✓ Step successful")

# 再查看一次数据
print(f"\nFirst camera (front_view) RGB data:")
front_rgb = obs["sensor_data"]["front_view"]["rgb"][0]  # [1, H, W, C] -> [H, W, C]
print(f"  Shape: {front_rgb.shape}")
print(f"  Type: {type(front_rgb).__name__}")
print(f"  Unique values: {torch.unique(front_rgb)[:10] if torch.is_tensor(front_rgb) else np.unique(front_rgb)[:10]}")

print(f"\nSecond camera (hand_camera) RGB data:")
hand_rgb = obs["sensor_data"]["hand_camera"]["rgb"][0]
print(f"  Shape: {hand_rgb.shape}")
print(f"  Type: {type(hand_rgb).__name__}")
print(f"  Unique values: {torch.unique(hand_rgb)[:10] if torch.is_tensor(hand_rgb) else np.unique(hand_rgb)[:10]}")

env.close()
print(f"✓ Environment closed")
