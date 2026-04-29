#!/usr/bin/env python3
"""
查看 outputs/frames/viewenv 目录中保存的图像帧
支持遍历和统计
"""
import os
import sys
from pathlib import Path

def view_frames(frame_dir="outputs/frames/viewenv", display_count=10):
    """查看保存的图像帧"""
    frame_dir = Path(frame_dir)
    
    if not frame_dir.exists():
        print(f"❌ 目录不存在: {frame_dir}")
        print("请先运行: python scripts/diagnostics/viewenv.py")
        return
    
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    
    if not frames:
        print(f"❌ 没有找到图像文件在: {frame_dir}")
        return
    
    print(f"\n✅ 找到 {len(frames)} 个图像帧")
    print(f"📁 位置: {frame_dir.absolute()}")
    print(f"\n📸 最近的 {min(display_count, len(frames))} 个帧:\n")
    
    for i, frame in enumerate(frames[-display_count:]):
        size_kb = frame.stat().st_size / 1024
        print(f"  [{i+1}] {frame.name} ({size_kb:.1f} KB)")
    
    # 提供查看指令
    print(f"\n💡 查看图像的方法:")
    print(f"   1. 在文件管理器中打开: {frame_dir.absolute()}")
    print(f"   2. 使用命令行预览最新的帧:")
    print(f"      file {frame_dir / frames[-1].name}")
    print(f"   3. 生成视频:")
    print(f"      ffmpeg -framerate 30 -pattern_type glob -i '{frame_dir}/*.jpg' -c:v libx264 -pix_fmt yuv420p output.mp4")

def cleanup_frames(frame_dir="outputs/frames/viewenv", keep_latest=50):
    """清理旧帧，只保留最新的帧"""
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    
    if len(frames) <= keep_latest:
        print(f"✅ 帧数 ({len(frames)}) <= 保留数 ({keep_latest}), 无需清理")
        return
    
    remove_count = len(frames) - keep_latest
    for frame in frames[:-keep_latest]:
        frame.unlink()
    
    print(f"🧹 已删除 {remove_count} 个旧帧，保留最新 {keep_latest} 个")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_frames()
    else:
        view_frames()
