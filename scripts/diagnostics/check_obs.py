import gymnasium as gym
import sys
import os
import numpy as np

# 确保可以导入 arm_studio 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mani_skill.envs 
import environments.conveyor_env
from robot.piper.agent import PiperArm, PiperActionWrapper

def main():
    env = gym.make(
        "PiperConveyor-v0", 
        obs_mode="rgb+state",
        robot_uids="piper_arm",
        render_mode="rgb_array"
    )
    obs, _ = env.reset()
    print("Observation keys:", obs.keys())
    if 'sensor_data' in obs:
        print("Sensor data keys:", obs['sensor_data'].keys())
        for cam in obs['sensor_data']:
            print(f"Camera {cam} keys:", obs['sensor_data'][cam].keys())
    if 'sensor_param' in obs:
        print("Sensor param keys:", obs['sensor_param'].keys())
    
    env.close()

if __name__ == "__main__":
    main()
