import h5py
import numpy as np
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="View recorded HDF5 dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to the HDF5 file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    with h5py.File(args.file, "r") as f:
        print(f"Viewing: {args.file}")
        print(f"Total frames: {f.attrs['total_frames']}")
        print(f"Control mode: {f.attrs['control_mode']}")
        
        obs = f["observation"]
        
        # 自动识别图像键
        cam_names = list(obs["sensor_data"].keys())
        print(f"Found cameras: {cam_names}")

        for i in range(f.attrs["total_frames"]):
            frames = []
            for cam in cam_names:
                # [total_frames, num_envs, H, W, 3] -> [H, W, 3]
                img_rgb = obs["sensor_data"][cam]["rgb"][i]
                if img_rgb.ndim == 4: # [num_envs, H, W, 3]
                    img_rgb = img_rgb[0]
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                frames.append(img_bgr)
            
            # 拼接显示
            combined = np.hstack(frames)
            
            # 显示信息
            cv2.putText(combined, f"Frame: {i}/{f.attrs['total_frames']-1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Dataset Viewer (Press any key for next, 'q' to quit)", combined)
            key = cv2.waitKey(0) # 按键继续
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
