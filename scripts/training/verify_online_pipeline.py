import os
import sys
import torch
import numpy as np
import h5py
import json
import glob
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset

# 1. 引入项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.data.dataset import ExpertDataset
from agent_factory.data.replaybuffer import FileReplayBuffer
from agent_factory.agents.registry import get_default_config, make_agent

def simulate_runner_output(save_dir, num_trajs=3):
    """严格模拟 BaseRunner._save_trajectory 的输出格式"""
    print(f"🎬 模拟 BaseRunner 产出数据到 {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_trajs):
        fpath = os.path.join(save_dir, f"piper_joint_sim_{i:03d}.hdf5")
        T = 20 # 模拟短轨迹
        with h5py.File(fpath, 'w') as f:
            f.create_dataset("actions", data=np.random.randn(T, 7).astype(np.float32))
            obs = f.create_group("obs")
            obs.create_dataset("rgb", data=np.zeros((T+1, 6, 224, 224), dtype=np.uint8))
            obs.create_dataset("state", data=np.random.randn(T+1, 7).astype(np.float32))
            
            f.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
            f.create_dataset("terminated", data=np.zeros(T, dtype=bool))
            f.create_dataset("truncated", data=np.zeros(T, dtype=bool))
            
            meta = f.create_group("meta")
            meta.create_dataset("env_cfg", data="dummy_yaml")
            f.attrs["success"] = (i == 0) # 只有第一条成功

def load_replay_files(buffer, folder):
    """模拟 ReplayBuffer 批量加载逻辑"""
    files = sorted(glob.glob(os.path.join(folder, "*.h5")))
    files.extend(sorted(glob.glob(os.path.join(folder, "*.hdf5"))))
    print(f"📥 正在加载 {len(files)} 个 Replay 文件...")
    
    trajectories = {
        "observations": [], "actions": [], "terminated": [], "rewards": []
    }
    
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            # 读取数据
            act = f["actions"][()]
            # 这里的 ReplayBuffer 预期 observations 是 list of dict (每帧一个字典)
            # 所以我们需要把 (T+1, C, H, W) 拆开
            rgb = f["obs/rgb"][()]
            state = f["obs/state"][()]
            
            obs_list = []
            for t in range(len(rgb)):
                obs_list.append({
                    "rgb": torch.from_numpy(rgb[t]),
                    "state": torch.from_numpy(state[t])
                })
            
            trajectories["observations"].append(obs_list)
            trajectories["actions"].append(act)
            trajectories["terminated"].append(f["terminated"][()])
            trajectories["rewards"].append(f["rewards"][()])
            
    buffer.push(trajectories)
    return len(files)

def verify_online_pipeline():
    # --- A. 准备模拟环境 ---
    online_dir = "datasets/verify_online_dummy"
    expert_path = "datasets/verify_expert_dummy.hdf5"
    
    # 模拟 Runner 产出
    simulate_runner_output(online_dir, num_trajs=3)
    
    # 模拟 Expert 产出
    from scripts.training.verify_training_pipeline import setup_dummy_dataset
    setup_dummy_dataset(expert_path)
    
    # --- B. 初始化配置 ---
    cfg = get_default_config("Diffusion_ITQC")
    cfg.dataset.expert.demo_path = expert_path
    cfg.env.control_mode = "joint_pos"
    cfg.env.proprio_dim = 7
    cfg.env.action_dim = 7
    cfg.actor.encoder.proprio_dim = 7
    cfg.critic.encoder.proprio_dim = 7
    cfg.actor.encoder.visual.in_channels = 3
    cfg.critic.encoder.visual.in_channels = 3
    cfg.env.num_cameras = 2
    cfg.train.batch_size = 4
    
    device = "cpu"
    
    # --- C. 数据集加载验证 ---
    print("\n[Step 1] Loading Expert and Replay...")
    expert_dataset = ExpertDataset(cfg=cfg, device=device)
    
    # 初始化 ReplayBuffer (FileReplayBuffer 会自动扫描 online_dir)
    replay_buffer = FileReplayBuffer(
        cfg=cfg,
        folder_path=online_dir
    )
    
    # 不需要 load_replay_files，因为 FileReplayBuffer 已经自动扫描了磁盘
    assert len(replay_buffer) > 0, "Replay buffer is empty! Check if simulator output files exist."
    print(f"✅ ReplayBuffer loaded via FileView. Slice count: {len(replay_buffer)}")

    # --- D. Agent 初始化 ---
    print("\n[Step 2] Initializing Agent...")
    agent = make_agent("Diffusion_ITQC", cfg)
    agent.fit_action_normalizer(expert_dataset.get_all_actions())

    # --- E. 验证训练情况 1: 仅 Replay ---
    print("\n[Step 3] Case 1: Training on ONLY ReplayBuffer...")
    loader_only_replay = DataLoader(replay_buffer, batch_size=4, shuffle=True)
    batch = next(iter(loader_only_replay))
    batch = agent._batch_to_device(batch)
    
    # 验证是否包含 value (在线数据通常没有预计算的 value，需要 Agent 处理或补充)
    # Note: ITQC 在处理 Replay 数据时通常会动态计算 V
    if "value" not in batch:
        batch["value"] = torch.zeros(batch["terminated"].shape)
        
    loss_c = agent.update_critic(batch, critic_type="Standard")
    loss_a = agent.update_actor(batch)
    print(f"✅ Loss computed successfully (Only Replay). Loss C: {loss_c['loss_q']:.4f}, Loss A: {loss_a['loss_actor']:.4f}")

    # --- F. 验证训练情况 2: Expert + Replay 混合 ---
    print("\n[Step 4] Case 2: Training on Expert + Replay (ConcatDataset)...")
    # 注意：ReplayBuffer 和 ExpertDataset 的 required_keys 必须对齐
    replay_buffer.required_keys = expert_dataset.required_keys
    
    combined_dataset = ConcatDataset([expert_dataset, replay_buffer])
    loader_mixed = DataLoader(combined_dataset, batch_size=4, shuffle=True)
    
    batch_mixed = next(iter(loader_mixed))
    batch_mixed = agent._batch_to_device(batch_mixed)
    
    loss_c_mix = agent.update_critic(batch_mixed, critic_type="Standard")
    loss_a_mix = agent.update_actor(batch_mixed)
    print(f"✅ Loss computed successfully (Mixed). Loss C: {loss_c_mix['loss_q']:.4f}, Loss A: {loss_a_mix['loss_actor']:.4f}")

    print("\n🎉 Online RL data pipeline verified! All systems GO.")

if __name__ == "__main__":
    try:
        verify_online_pipeline()
    finally:
        import shutil
        for p in ["datasets/verify_online_dummy", "datasets/verify_expert_dummy.hdf5"]:
            if os.path.exists(p):
                if os.path.isdir(p): shutil.rmtree(p)
                else: os.remove(p)
