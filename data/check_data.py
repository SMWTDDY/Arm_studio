import h5py
import cv2
import numpy as np

def h5_to_video(h5_path, output_path, fps=30):
    with h5py.File(h5_path, 'r') as f:
        sensor_data = f['observation/sensor_data']
        
        # 使用 np.squeeze 去除多余的维度 
        # 将形状由 (N, 1, H, W, C) 转换为 (N, H, W, C)
        front_images = np.squeeze(sensor_data['front_view/rgb'][:])
        top_images = np.squeeze(sensor_data['top_view/rgb'][:])
        side_images = np.squeeze(sensor_data['side_view/rgb'][:])
        
        num_frames = front_images.shape[0]
        h, w, c = front_images[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w * 3, h))
        
        print(f"正在转换 {num_frames} 帧图像...")
        
        for i in range(num_frames):
            img_f = cv2.cvtColor(front_images[i], cv2.COLOR_RGB2BGR)
            img_t = cv2.cvtColor(top_images[i], cv2.COLOR_RGB2BGR)
            img_s = cv2.cvtColor(side_images[i], cv2.COLOR_RGB2BGR)
            
            combined_frame = np.hstack((img_f, img_t, img_s))
            out.write(combined_frame)
            
        out.release()
        print(f"转换完成！视频保存于: {output_path}")

if __name__ == "__main__":
    h5_path = 'datasets/piper_joint_recording_000.hdf5'
    video_path = 'datasets/video/trajectory_check.mp4'
    h5_to_video(h5_path, video_path)