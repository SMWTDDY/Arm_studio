import sys
import os
import gymnasium as gym
import numpy as np
import time

# 自动将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.piper.agent import PiperArm, PiperActionWrapper
from teleop.get_pose import get_pose
import mani_skill.envs

def main():
    # 1. 创建环境，使用绝对位姿控制模式 (pd_ee_pose)
    env = gym.make(
        "Empty-v1", 
        obs_mode="state",
        control_mode="pd_ee_pose", # 此模式已在 agent.py 中设为绝对模式
        robot_uids="piper_arm",
        render_mode="human"
    )
    
    # 2. 应用包装器
    env = PiperActionWrapper(env, binary_gripper=True)
    
    env.reset()
    print("仿真环境已启动。正在进行绝对位姿控制...")

    # 3. 绝对目标位姿序列 [x, y, z, r, p, y]
    targets = [
        [0.15, 0.0, 0.35, 0.0, 1.57, 0.0],
        [0.15, 0.1, 0.25, 0.0, 1.57, 0.0],
        [0.15, -0.1, 0.25, 0.0, 1.57, 0.0],
        [0.25, 0.0, 0.15, 0.0, 1.57, 0.0],
    ]
    
    for i, target in enumerate(targets):
        print(f"移动到目标 {i+1}: {target}")
        
        # 4. 直接将绝对位姿作为动作发送！
        # 无需计算当前位姿偏差，控制器会在底层自动追踪这个坐标。
        action = np.array(target + [-1.0], dtype=np.float32)
        
        # 维持同一个动作一段时间，直到仿真机械臂追踪到达
        for _ in range(100):
            env.step(action)
            env.render()
            time.sleep(0.005)
            
    print("绝对位姿序列测试完成。")
    env.close()

if __name__ == "__main__":
    main()
