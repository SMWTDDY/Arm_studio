import h5py
import numpy as np
import os
import time
import copy

try:
    import torch
except ImportError:
    torch = None

class HDF5Recorder:
    def __init__(self, robot="piper", mode="joint", save_dir="datasets"):
        self.robot = robot
        self.mode = mode
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.is_recording = False
        self.reset_buffers()

    def reset_buffers(self):
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.timestamp_buffer = []
        self.start_time = None

    def start_episode(self):
        """开始新的轨迹录制，自动生成 episode_name"""
        self.reset_buffers()
        self.start_time = time.time()
        
        # 自动命名逻辑: piper_{mode}_recording_xxx
        idx = 0
        while idx < 1000:
            name = f"{self.robot}_{self.mode}_recording_{idx:03d}"
            path = os.path.join(self.save_dir, f"{name}.hdf5")
            if not os.path.exists(path):
                self.episode_name = name
                self.file_path = path
                break
            idx += 1
        
        self.is_recording = True
        print(f"\n[Recorder] 开始录制新轨迹: {self.episode_name}")

    def _to_numpy(self, x):
        """将数据转换为 numpy 格式，并确保在 CPU 上"""
        if x is None:
            return None
        
        # 处理 PyTorch 张量
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        
        # 递归处理字典
        if isinstance(x, dict):
            return {k: self._to_numpy(v) for k, v in x.items()}
        
        # 递归处理列表或元组
        if isinstance(x, (list, tuple)):
            return [self._to_numpy(v) for v in x]
            
        return x

    def add_step(self, obs, action, reward):
        if not self.is_recording:
            return
            
        # 转换并存储数据 (转换为 numpy 以节省 GPU 显存并确保兼容性)
        self.obs_buffer.append(copy.deepcopy(self._to_numpy(obs)))
        self.action_buffer.append(copy.deepcopy(self._to_numpy(action)))
        self.reward_buffer.append(copy.deepcopy(self._to_numpy(reward)))
        self.timestamp_buffer.append(time.time() - self.start_time)
        
    def save(self):
        """结束当前轨迹并保存"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if len(self.action_buffer) == 0:
            print("[Recorder] 数据为空，取消保存。")
            return
            
        print(f"[Recorder] 正在保存轨迹: {self.file_path} (共 {len(self.action_buffer)} 步)")
        
        # 堆叠所有步骤的数据
        stacked_obs = self._stack_obs(self.obs_buffer)
        
        with h5py.File(self.file_path, "w") as f:
            # 基础数据
            f.create_dataset("action", data=np.array(self.action_buffer, dtype=np.float32))
            f.create_dataset("reward", data=np.array(self.reward_buffer, dtype=np.float32))
            f.create_dataset("timestamp", data=np.array(self.timestamp_buffer, dtype=np.float32))
            
            # 递归保存观察值字典
            obs_group = f.create_group("observation")
            self._save_dict_to_hdf5(obs_group, stacked_obs)
            
            # 元数据
            f.attrs["total_frames"] = len(self.action_buffer)
            f.attrs["episode_name"] = self.episode_name
            f.attrs["control_mode"] = self.mode
            f.attrs["recorded_at"] = time.ctime()
            
        print(f"[Recorder] 保存完成！")

    def _stack_obs(self, obs_list):
        if len(obs_list) == 0: return None
        first = obs_list[0]
        
        if isinstance(first, dict):
            stacked = {}
            for k in first.keys():
                stacked[k] = self._stack_obs([obs[k] for obs in obs_list])
            return stacked
        else:
            # 此时应该是 numpy 数组或标量
            return np.array(obs_list)

    def _save_dict_to_hdf5(self, h5_group, d):
        if not isinstance(d, dict): return
        for k, v in d.items():
            if isinstance(v, dict):
                sub_group = h5_group.create_group(k)
                self._save_dict_to_hdf5(sub_group, v)
            else:
                # 图像数据识别与压缩 (RGB 图像通常是 uint8)
                if isinstance(v, np.ndarray) and len(v.shape) >= 4 and v.dtype == np.uint8:
                    # 启用 gzip 压缩，级别为 4 (平衡速度与压缩率)
                    h5_group.create_dataset(k, data=v, compression="gzip", compression_opts=4, chunks=True)
                else:
                    h5_group.create_dataset(k, data=v)
