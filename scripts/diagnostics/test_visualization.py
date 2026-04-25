#!/usr/bin/env python3
"""验证摄像机可视化标记并生成图像"""
import os
import sys
import cv2
import numpy as np
os.environ["SAPIEN_VULKAN_DEVICE"] = "0"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from environments.conveyor_env import PiperConveyorEnv

# 创建环境
env = PiperConveyorEnv(
    obs_mode="rgb+state",
    robot_uids="piper_arm",
    control_mode="pd_joint_pos",
    render_mode="human"  # 使用human模式可以看到窗口
)

print("✅ 环境创建成功")
print("✅ 摄像机可视化标记已添加到场景中")
print("   - 红色大方块(10cm): 手眼摄像机位置 (link6法兰)")
print("   - 绿色粗棒(30cm): 法兰方向指示器 (沿Y轴正方向)")

# 重置环境
obs, _ = env.reset()
print("✅ 环境重置完成，可视化标记已定位")

# 生成一些测试图像
output_dir = os.path.join("outputs", "frames", "test_visualization")
os.makedirs(output_dir, exist_ok=True)

print("\n📸 生成测试图像...")
for i in range(3):
    # 执行不同动作
    if i == 0:
        action = np.array([0, -0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # rest
        pose_name = "rest"
    elif i == 1:
        action = np.array([0.5, -0.8, 0.5, 0, 0, 0, 0.5, 0.5], dtype=np.float32)  # reach
        pose_name = "reach"
    else:
        action = np.array([0, -1.5, 0.5, 0, 0, 0, 0, 0], dtype=np.float32)  # up
        pose_name = "up"
    
    obs, _, _, _, _ = env.step(action)
    env.render()
    
    # 获取图像
    front_img = obs["sensor_data"]["front_view"]["rgb"][0].cpu().numpy().astype(np.uint8)
    hand_img = obs["sensor_data"]["hand_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)
    
    # 转换为BGR
    front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
    hand_bgr = cv2.cvtColor(hand_img, cv2.COLOR_RGB2BGR)
    
    # 添加标签
    cv2.putText(front_bgr, f"Front View ({pose_name})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(hand_bgr, f"Hand Camera ({pose_name})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 拼接
    combined = np.hstack([front_bgr, hand_bgr])
    
    # 保存
    filename = os.path.join(output_dir, f"test_{pose_name}.jpg")
    cv2.imwrite(filename, combined)
    print(f"   ✅ 保存: {filename}")

print("\n💡 查看结果:")
print(f"   打开 {output_dir}/ 目录查看生成的图像")
print("   红色大方块应该在link6位置随机器人移动")
print("   绿色粗棒应该在link6前方0.2米处指示方向")

env.close()
