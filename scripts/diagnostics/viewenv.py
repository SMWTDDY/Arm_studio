import sys
import os
import argparse
import numpy as np
import time
import cv2
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 强制显卡渲染设备
os.environ["SAPIEN_VULKAN_DEVICE"] = "0"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import mani_skill.envs
import robot.piper.agent 
from environments.conveyor_env import PiperConveyorEnv

def main():
    # 创建环境
    env = gym.make(
        "PiperConveyor-v0",
        obs_mode="rgb+state",
        robot_uids="piper_arm",
        control_mode="pd_joint_pos",
        render_mode="human"
    )

    print(f"\n[Env Info] Control Freq: {env.unwrapped.control_freq} Hz")
    print(f"[Env Info] Sim Freq: {env.unwrapped.sim_freq} Hz")

    obs, _ = env.reset()
    
    # 获取动作空间信息
    action_space = env.action_space
    print(f"[Action Space] Shape: {action_space.shape}, Low: {action_space.low}, High: {action_space.high}")
    
    # 定义几个有意义的关节位置（关节1-6是机械臂，关节7-8是夹爪）
    # 参考 keyframes: rest 位置
    pose_rest = np.array([0, -0.2, 0, 0, 0, 0, 0, 0])
    
    # 使用连续的余弦函数运动，幅度更大
    # 关节1-6: 幅度 1.5 弧度 (~86度)，不同相位
    # 关节7-8: 夹爪保持轻微开合
    amplitude = 1.5  # 关节运动幅度 (弧度)
    frequency = 0.05  # 运动频率
    
    # 创建输出目录用于保存图像
    output_dir = os.path.join("outputs", "frames", "viewenv")
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查 OpenCV GUI 是否可用
    has_gui = True
    try:
        cv2.namedWindow("Dual-View: Front | Hand-Eye Camera", cv2.WINDOW_AUTOSIZE)
    except cv2.error as e:
        print(f"\n[Warning] OpenCV GUI 不可用 (缺少 GTK+ 支持)")
        print(f"[Info] 将在 headless 模式下运行，图像帧会保存到 '{output_dir}/' 目录\n")
        has_gui = False

    last_log_time = time.time()
    frame_count = 0
    save_interval = 30  # 每30帧保存一次图像
    total_frame_count = 0
    start_time = time.time()

    try:
        while True:
            # --- 构造保持静止的 action ---
            # 1. 直接从底层机器人对象获取当前的关节绝对位置 (qpos)
            qpos = env.unwrapped.agent.robot.qpos
            
            # 2. ManiSkill3 默认返回 tensor，需转为 numpy
            if hasattr(qpos, "cpu"):
                qpos = qpos.detach().cpu().numpy()
            
            # 3. 处理多环境并行的情况，如果是 [1, 8] 的 shape 则取 [8]
            if qpos.ndim > 1:
                qpos = qpos[0]
                
            # 4. 组装符合 Action Wrapper 格式的 7维 动作
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[:6] = qpos[:6]  # 机械臂的 6 个关节设为当前的实际位置，维持不动
            action[6] = -1.0       # 夹爪指令（根据你的 Wrapper，-1.0 是默认松开状态）
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # ... 下方保留你原本的双视角显示逻辑
            
            # --- 双视角处理逻辑 - 始终执行 ---
            if "sensor_data" in obs:
                def to_numpy(x):
                    if hasattr(x, "cpu"): 
                        return x.detach().cpu().numpy()
                    return np.array(x)

                try:
                    # 获取两个相机的画面 (shape: [1, H, W, 3], uint8)
                    front_tensor = obs["sensor_data"]["front_view"]["rgb"]
                    hand_tensor = obs["sensor_data"]["hand_camera"]["rgb"]
                    
                    # 转换为 numpy 数组 [1, H, W, 3] -> [H, W, 3]
                    front_img = to_numpy(front_tensor[0]).astype(np.uint8)
                    hand_img = to_numpy(hand_tensor[0]).astype(np.uint8)
                    
                    # 转换 RGB -> BGR
                    front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                    hand_bgr = cv2.cvtColor(hand_img, cv2.COLOR_RGB2BGR)
                    
                    # 添加标签
                    cv2.putText(front_bgr, "Front View (External)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(hand_bgr, "Hand-Eye Camera (on Flange)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 拼接两个画面
                    combined_img = np.hstack([front_bgr, hand_bgr])
                    
                    # 缩放以便查看
                    combined_img = cv2.resize(combined_img, (0, 0), fx=1.0, fy=1.0)
                    
                    # 显示方式选择
                    if has_gui:
                        cv2.imshow("Dual-View: Front | Hand-Eye Camera", combined_img)
                        cv2.waitKey(1)
                    
                    # 定期保存图像快照
                    if frame_count % save_interval == 0:
                        frame_file = os.path.join(output_dir, f"frame_{total_frame_count:06d}.jpg")
                        cv2.imwrite(frame_file, combined_img)
                        t = time.time() - start_time
                        joint_status = f"Joints: {action[:3]} | t={t:.1f}"
                        print(f"\r[ViewEnv] 已保存: {os.path.basename(frame_file)} | {joint_status}", end="", flush=True)
                        
                except Exception as e:
                    pass
            
            frame_count += 1
            total_frame_count += 1
            
            # 更新 FPS 计数
            now = time.time()
            if frame_count % 100 == 0:
                elapsed = now - last_log_time
                fps = 100 / elapsed if elapsed > 0 else 0
                print(f"\r[ViewEnv] FPS: {fps:.1f} | 总帧数: {total_frame_count} | 连续余弦运动", end="", flush=True)
                last_log_time = now

            window = env.unwrapped.viewer.window if env.unwrapped.viewer else None
            if window and window.key_down('escape'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        if has_gui:
            cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    main()
