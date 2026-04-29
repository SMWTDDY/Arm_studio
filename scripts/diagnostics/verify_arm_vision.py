#!/usr/bin/env python3
"""
长时间运行机械臂视觉验证脚本
- 机械臂执行循环动作
- 持续保存双视角图像
- 显示实时统计
"""
import sys
import os
import numpy as np
import time
import cv2

os.environ["SAPIEN_VULKAN_DEVICE"] = "0"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import mani_skill.envs
import robot.piper.agent
from environments.conveyor_env import PiperConveyorEnv

def main(duration_seconds=300):
    """运行机械臂视觉验证"""
    print(f"\n🚀 启动长时间运行的机械臂视觉验证 (运行 {duration_seconds} 秒)")
    
    env = gym.make(
        "PiperConveyor-v0",
        obs_mode="rgb+state",
        robot_uids="piper_arm",
        control_mode="pd_joint_pos",
        render_mode="human"
    )

    print(f"✅ 环境已创建")
    print(f"   - 控制频率: {env.unwrapped.control_freq} Hz")
    print(f"   - 仿真频率: {env.unwrapped.sim_freq} Hz")
    print(f"   - 动作空间: {env.action_space.shape}")

    obs, _ = env.reset()
    
    # 定义关键位置
    poses = {
        "rest": np.array([0, -0.2, 0, 0, 0, 0, 0, 0]),
        "reach": np.array([0.5, -0.8, 0.5, 0, 0, 0, 0.5, 0.5]),
        "up": np.array([0, -1.5, 0.5, 0, 0, 0, 0, 0]),
        "sweep_left": np.array([-0.8, -1.2, 0.3, 0, 0, 0, 0, 0]),
        "sweep_right": np.array([0.8, -1.2, 0.3, 0, 0, 0, 0, 0]),
    }
    
    pose_sequence = list(poses.keys())
    action_idx = 0
    frame_per_pose = 60  # 每个位置持续 60 帧 (1 秒)
    frame_count_per_pose = 0
    
    output_dir = os.path.join("outputs", "frames", "viewenv")
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查 GUI
    has_gui = True
    try:
        cv2.namedWindow("🔍 双视角监控", cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        print("⚠️  OpenCV GUI 不可用，以 headless 模式运行")
        has_gui = False

    start_time = time.time()
    total_frames = 0
    saved_frames = 0
    save_interval = 30
    last_fps_update = start_time
    fps_frame_count = 0

    try:
        while time.time() - start_time < duration_seconds:
            # 获取当前目标位置
            target_pose_name = pose_sequence[action_idx]
            target_action = poses[target_pose_name].copy()
            
            # 步进环境
            obs, reward, terminated, truncated, info = env.step(target_action)
            env.render()
            
            # 处理双视角图像
            if "sensor_data" in obs:
                def to_numpy(x):
                    return x.detach().cpu().numpy() if hasattr(x, "cpu") else x

                try:
                    front_img = to_numpy(obs["sensor_data"]["front_view"]["rgb"][0])
                    hand_img = to_numpy(obs["sensor_data"]["hand_camera"]["rgb"][0])
                    
                    front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                    hand_bgr = cv2.cvtColor(hand_img, cv2.COLOR_RGB2BGR)
                    
                    # 添加信息标签
                    cv2.putText(front_bgr, f"Front: {target_pose_name}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(hand_bgr, f"Hand-Eye: {target_pose_name}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    combined = np.hstack([front_bgr, hand_bgr])
                    
                    if has_gui:
                        cv2.imshow("🔍 双视角监控", combined)
                        cv2.waitKey(1)
                    
                    # 保存帧
                    if total_frames % save_interval == 0:
                        frame_path = os.path.join(output_dir, f"frame_{total_frames:06d}.jpg")
                        cv2.imwrite(frame_path, combined)
                        saved_frames += 1
                        
                except Exception as e:
                    pass
            
            # 更新统计
            total_frames += 1
            frame_count_per_pose += 1
            fps_frame_count += 1
            
            # 切换到下一个位置
            if frame_count_per_pose >= frame_per_pose:
                action_idx = (action_idx + 1) % len(pose_sequence)
                frame_count_per_pose = 0
            
            # 更新 FPS 显示
            now = time.time()
            if now - last_fps_update >= 2.0:
                fps = fps_frame_count / (now - last_fps_update)
                elapsed = now - start_time
                remaining = duration_seconds - elapsed
                
                pose_name = pose_sequence[action_idx]
                print(f"\r[运行中] "
                      f"帧: {total_frames:6d} | "
                      f"FPS: {fps:.1f} | "
                      f"已保存: {saved_frames:3d} | "
                      f"当前位置: {pose_name:12s} | "
                      f"耗时: {elapsed:.1f}s / {duration_seconds:.1f}s | "
                      f"剩余: {remaining:.1f}s", 
                      end="", flush=True)
                
                last_fps_update = now
                fps_frame_count = 0
                
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断运行")
    except Exception as e:
        print(f"\n❌ 出现错误: {e}")
    finally:
        if has_gui:
            cv2.destroyAllWindows()
        env.close()
        
        elapsed = time.time() - start_time
        print(f"\n\n✅ 运行完成")
        print(f"   - 总帧数: {total_frames}")
        print(f"   - 已保存帧: {saved_frames}")
        print(f"   - 运行时间: {elapsed:.1f}s")
        print(f"   - 平均 FPS: {total_frames / elapsed:.1f}")
        print(f"   - 图像目录: {os.path.abspath(output_dir)}")
        print(f"\n💡 查看结果: python scripts/diagnostics/view_frames.py")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="机械臂长时间视觉验证")
    parser.add_argument("--duration", type=int, default=300, 
                       help="运行时长 (秒), 默认 300 秒 (5 分钟)")
    args = parser.parse_args()
    
    main(duration_seconds=args.duration)
