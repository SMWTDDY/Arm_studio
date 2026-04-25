import gymnasium as gym
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mani_skill.envs
import environments.conveyor_env
import robot.piper.agent
import numpy as np

def main():
    env = gym.make(
        "PiperConveyor-v0",
        obs_mode="rgb+state",
        robot_uids="piper_arm",
        num_envs=1,
    )
    obs, _ = env.reset()
    print("Observation keys:", obs.keys())
    if "sensor_data" in obs:
        print("Sensor data keys:", obs["sensor_data"].keys())
        for cam_name, cam_data in obs["sensor_data"].items():
            print(f"Camera '{cam_name}' keys:", cam_data.keys())
            if "rgb" in cam_data:
                print(f"  RGB shape: {cam_data['rgb'].shape}, dtype: {cam_data['rgb'].dtype}")
    
    if "sensor_param" in obs:
        print("Sensor param keys:", obs["sensor_param"].keys())

    env.close()

if __name__ == "__main__":
    main()
