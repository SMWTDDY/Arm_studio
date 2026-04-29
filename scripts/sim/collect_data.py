import sys
import os
import argparse
import numpy as np
import time

# 确保可以导入 arm_studio 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gymnasium as gym
from robot.piper.agent import PiperArm, PiperActionWrapper
from robot.piper.pose_ik import BoundedPiperIK
from teleop.real_to_sim import RealToSimTeleop
from data.recorder import HDF5Recorder
import mani_skill.envs 
import environments.conveyor_env # 导入自定义传送带环境

import cv2
import sapien

def main():
    parser = argparse.ArgumentParser(description="ArmStudio Data Collection")
    parser.add_argument("--teleop", type=str, default="real", choices=["keyboard", "real"],
                        help="Teleoperation mode.")
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "pose"],
                        help="Control space: 'joint' or 'pose'.")
    parser.add_argument("--binary_gripper", action="store_true",
                        help="Use binary gripper control (-1 or 1).")
    parser.add_argument("--env_name", type=str, default="PiperConveyor-v0",
                        help="Environment UID to run.")
    parser.add_argument("--save-dir", type=str, default="datasets/piper",
                        help="Directory for collected HDF5 trajectories.")
    parser.add_argument("--can-name", type=str, default="can_mr",
                        help="Leader arm CAN name used by real-to-sim teleop.")
    parser.add_argument("--arm-side", type=str, default="right", choices=["left", "right"],
                        help="Sim Piper arm model to load; selects the calibrated wrist-camera URDF.")
    parser.add_argument("--robot-uids", type=str, default=None,
                        help="Override ManiSkill robot_uids directly.")
    args = parser.parse_args()

    # 1. 环境初始化
    use_pose_ik = args.mode == "pose" and args.teleop == "real"
    control_mode = "pd_joint_pos" if use_pose_ik else ("pd_ee_pose" if args.mode == "pose" else "pd_joint_pos")
    # arm-side 只影响加载哪套已标定的 URDF：right/left 的机械臂主体相同，
    # 但 wrist camera 的 hand_camera_joint 使用不同真实标定外参。
    robot_uids = args.robot_uids or f"piper_arm_{args.arm_side}"
    
    env = gym.make(
        args.env_name, 
        obs_mode="rgb+state", # 关键：开启视觉画面的采集
        control_mode=control_mode,
        robot_uids=robot_uids,
        num_envs=1,
        render_mode="human"
    )
    
    # 统一应用 PiperActionWrapper 进行升维 (从 7 维到 8 维)
    env = PiperActionWrapper(env, binary_gripper=args.binary_gripper)

    # 2. 遥操作器初始化
    teleop = RealToSimTeleop(can_name=args.can_name)
    time.sleep(0.2)

    # 3. 记录器初始化
    recorder = HDF5Recorder(robot="piper", mode=args.mode, backend="sim", save_dir=args.save_dir)
    pose_ik = BoundedPiperIK() if use_pose_ik else None
    
    # 显式初始化可视化窗口
    window_title = f"Multi-View (Front | {args.arm_side.title()} Hand)"
    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    obs, _ = env.reset()


    try:
        while True:
            env.render()
            
            # --- 二视角显示逻辑 (正前 | 手眼) ---
            if "sensor_data" in obs:
                def to_numpy(x):
                    if hasattr(x, "cpu"): return x.detach().cpu().numpy()
                    return x

                try:
                    # 获取两个相机的画面
                    front_img = to_numpy(obs["sensor_data"]["front_view"]["rgb"][0])
                    hand_img = to_numpy(obs["sensor_data"]["hand_camera"]["rgb"][0])
                    
                    # 转换 RGB -> BGR
                    front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                    hand_bgr = cv2.cvtColor(hand_img, cv2.COLOR_RGB2BGR)

                    if front_bgr.shape[0] != hand_bgr.shape[0]:
                        target_h = min(front_bgr.shape[0], hand_bgr.shape[0])
                        def resize_to_height(img, height):
                            scale = height / img.shape[0]
                            width = max(1, int(round(img.shape[1] * scale)))
                            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                        front_bgr = resize_to_height(front_bgr, target_h)
                        hand_bgr = resize_to_height(hand_bgr, target_h)
                    
                    # 拼接两个画面
                    combined_img = np.hstack([front_bgr, hand_bgr])
                    
                    # 整体缩放
                    combined_img = cv2.resize(combined_img, (0, 0), fx=1.2, fy=1.2)
                    
                    cv2.imshow(window_title, combined_img)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"[Display Error] {e}")
            # ----------------------

            window = env.unwrapped.viewer.window if env.unwrapped.viewer else None

            if window:
                if window.key_press('r'):
                    if not recorder.is_recording: 
                        recorder.start_episode()
                        print("开始录制轨迹。操作指南：使用遥操作控制机械臂，按 'e' 保存轨迹，按 'q' 抛弃轨迹，按 'escape' 退出")
                
                if window.key_press('e'):
                    if recorder.is_recording: 
                        recorder.save()
                
                if window.key_press('q'):
                    if recorder.is_recording:
                        recorder.is_recording = False
                        recorder.reset_buffers()
                        print("轨迹已抛弃")

                if window.key_down('escape'):
                    if recorder.is_recording: recorder.save()
                    break

            # 1. 获取遥操作指令
            if use_pose_ik:
                record_action = teleop.get_action(mode="pose", use_binary_gripper=args.binary_gripper)
                seed_action = teleop.get_action(mode="joint", use_binary_gripper=args.binary_gripper)
                ik_result = pose_ik.solve(record_action, seed_action[:6])
                sim_action = np.concatenate([ik_result["qpos"], [record_action[6]]]).astype(np.float32)
                if ik_result["pos_error"] > 0.01 or ik_result["rot_error"] > 0.1:
                    print(
                        f"[PoseIK Warning] pos={ik_result['pos_error']:.4f}m "
                        f"rot={ik_result['rot_error']:.4f}rad status={ik_result['status']}"
                    )
            else:
                sim_action = teleop.get_action(
                    mode=args.mode,
                    use_binary_gripper=args.binary_gripper,
                )
                record_action = sim_action

            # 2. 执行物理步进
            next_obs, reward, terminated, truncated, info = env.step(sim_action)

            # 3. 录制数据
            if recorder.is_recording:
                recorder.add_step(obs, record_action, reward)

            obs = next_obs

    except KeyboardInterrupt:
        pass
    finally:
        if recorder.is_recording: recorder.save()
        if hasattr(teleop, 'close'): teleop.close()
        env.close()

if __name__ == "__main__":
    main()
