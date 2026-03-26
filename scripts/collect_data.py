import sys
import os
import argparse
import numpy as np
import time

# 确保可以导入 arm_studio 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from models.piper.agent import PiperArm, PiperActionWrapper
from teleop.keyboard_ik import KeyboardTeleop
from teleop.real_to_sim import RealToSimTeleop
from data.recorder import HDF5Recorder
import mani_skill.envs 

def main():
    parser = argparse.ArgumentParser(description="ArmStudio Data Collection")
    parser.add_argument("--teleop", type=str, default="real", choices=["keyboard", "real"],
                        help="Teleoperation mode.")
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "pose"],
                        help="Control space: 'joint' or 'pose'.")
    parser.add_argument("--binary_gripper", action="store_true",
                        help="Use binary gripper control (-1 or 1).")
    args = parser.parse_args()

    # 1. 环境初始化
    control_mode = "pd_ee_pose" if args.mode == "pose" else "pd_joint_pos"
    
    env = gym.make(
        "Empty-v1", 
        obs_mode="state",
        control_mode=control_mode,
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human"
    )
    
    # 统一应用 PiperActionWrapper 进行升维 (从 7 维到 8 维)
    env = PiperActionWrapper(env, binary_gripper=args.binary_gripper)

    # 2. 遥操作器初始化
    if args.teleop == "keyboard":
        teleop = KeyboardTeleop(env)
    else:
        teleop = RealToSimTeleop(can_name='can_master')

    # 3. 记录器初始化
    recorder = HDF5Recorder(robot="piper", mode=args.mode)

    obs, _ = env.reset()
    print("\n" + "="*50)
    print(f"模式: {args.mode.upper()} | 遥操作: {args.teleop.upper()} | 夹爪: {'Binary' if args.binary_gripper else 'Continuous'}")
    print("="*50)
    print("控制指南:")
    print("  [R] - 开始录制新轨迹")
    print("  [S] - 停止并保存当前轨迹")
    print("  [ESC] - 退出程序")
    print("="*50 + "\n")

    try:
        while True:
            env.render()
            window = env.unwrapped.viewer.window if env.unwrapped.viewer else None

            if window:
                if window.key_press('r'):
                    if not recorder.is_recording: recorder.start_episode()
                
                if window.key_press('s'):
                    if recorder.is_recording: recorder.save()

                if window.key_down('escape'):
                    if recorder.is_recording: recorder.save()
                    break

            # 1. 获取遥操作指令
            # 对齐用户选择的 mode：
            # 如果是 pose 模式，get_action 会调用 get_pose 计算坐标发给 pd_ee_pose
            # 如果是 joint 模式，get_action 会直接发角度给 pd_joint_pos
            action = teleop.get_action(mode=args.mode, use_binary_gripper=args.binary_gripper)

            # 2. 执行物理步进
            next_obs, reward, terminated, truncated, info = env.step(action)


            # 3. 录制数据
            if recorder.is_recording:
                recorder.add_step(obs, action, reward)

            obs = next_obs

    except KeyboardInterrupt:
        pass
    finally:
        if recorder.is_recording: recorder.save()
        if hasattr(teleop, 'close'): teleop.close()
        env.close()

if __name__ == "__main__":
    main()
