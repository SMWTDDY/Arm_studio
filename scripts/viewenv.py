import sys
import os
import argparse
import numpy as np

# 强制显卡渲染设备 (部分系统需要)
os.environ["SAPIEN_VULKAN_DEVICE"] = "0"

# 确保项目根目录在 Python 路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import mani_skill.envs
import models.piper.agent 
from environments.conveyor_env import PiperConveyorEnv
import torch

def main():
    # 创建环境
    env = gym.make(
        "PiperConveyor-v0",
        obs_mode="rgb+state",
        robot_uids="piper_arm",
        control_mode="pd_joint_pos",
        render_mode="human"
    )

    obs, _ = env.reset()
    
    # 自动调整预览视角
    if env.unwrapped.viewer:
        from mani_skill.utils.sapien_utils import look_at
        # 相机位置靠近工作区，对准 x=0.5 处的传送带
        viewer_pose = look_at(eye=[1.0, 1.0, 0.8], target=[0.5, 0, 0.1])
        
        if hasattr(viewer_pose, "sp"):
            viewer_pose = viewer_pose.sp
        try:
            if len(viewer_pose.shape) > 0:
                viewer_pose = viewer_pose[0]
        except:
            pass
            
        env.unwrapped.viewer.set_camera_pose(viewer_pose)

    try:
        while True:
            # 持续发送零动作
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            window = env.unwrapped.viewer.window if env.unwrapped.viewer else None
            if window and window.key_down('escape'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
