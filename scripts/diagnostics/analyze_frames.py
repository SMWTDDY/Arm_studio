#!/usr/bin/env python3
"""分析保存的帧，检查手眼摄像机是否真的在追踪"""
import cv2
import numpy as np
import os
from pathlib import Path

frame_dir = Path("outputs/frames/viewenv")

# 获取所有帧
frames = sorted(frame_dir.glob("frame_*.jpg"))[:10]  # 只看前 10 帧

print(f"Found {len(frames)} frames")

for frame_path in frames:
    # 读取帧
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"✗ Failed to read {frame_path.name}")
        continue
    
    height, width = img.shape[:2]
    mid_x = width // 2
    
    # 分割左右摄像机部分
    left_cam = img[:, :mid_x]       # 外部摄像机
    right_cam = img[:, mid_x:]      # 手眼摄像机
    
    # 计算每个摄像机的平均颜色和标准差
    left_mean = np.mean(left_cam, axis=(0, 1))
    left_std = np.std(left_cam, axis=(0, 1))
    right_mean = np.mean(right_cam, axis=(0, 1))
    right_std = np.std(right_cam, axis=(0, 1))
    
    # 计算直方图
    left_hist = cv2.calcHist([left_cam], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    right_hist = cv2.calcHist([right_cam], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # 比较直方图（使用交集方法）
    hist_compare = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_INTERSECT)
    
    print(f"\n{frame_path.name}:")
    print(f"  Left Camera  - Mean: {left_mean.astype(int)}, Std: {left_std.astype(int)}")
    print(f"  Right Camera - Mean: {right_mean.astype(int)}, Std: {right_std.astype(int)}")
    print(f"  Histogram similarity: {hist_compare:.2f}")
    
    # 检查右摄像机是否变化很大（方差）
    right_var = np.var(right_cam)
    print(f"  Right camera variance: {right_var:.2f}")
