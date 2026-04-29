import os
import sys
import torch
import numpy as np
import h5py
import json
from omegaconf import OmegaConf

# 1. 根目录引入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.data.dataset import ExpertDataset
from agent_factory.agents.registry import get_default_config, make_agent

def setup_dummy_dataset(path):
    print(f"Creating dummy flattened dataset at {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    meta_keys = {
        "obs": {
            "rgb": {"base": [480, 640, 3], "hand": [480, 640, 3]},
            "state": {"joint": [6], "gripper": [1]}
        },
        "action": {"arm": [6], "gripper": [1]}
    }
    
    with h5py.File(path, 'w') as f:
        meta = f.create_group("meta")
        meta.create_dataset("env_meta", data=json.dumps(meta_keys))
        
        for i in range(2):
            traj = f.create_group(f"traj_{i}")
            obs = traj.create_group("obs")
            # 模拟合并/转换后的格式: CHW 堆叠, uint8
            obs.create_dataset("rgb", data=np.zeros((50, 6, 224, 224), dtype=np.uint8))
            obs.create_dataset("state", data=np.random.randn(50, 7).astype(np.float32))
            
            traj.create_dataset("actions", data=np.random.randn(49, 7).astype(np.float32))
            traj.create_dataset("terminated", data=np.zeros(49, dtype=bool))
            traj.create_dataset("rewards", data=np.zeros(49, dtype=np.float32))
            # 新增: 模拟 Value (MC Return)
            traj.create_dataset("value", data=np.random.randn(49).astype(np.float32))
            traj.attrs["success"] = (i == 0)

def verify_pipeline():
    dummy_path = "datasets/verify_training_dummy.hdf5"
    setup_dummy_dataset(dummy_path)
    
    # 1. 获取默认配置
    cfg = get_default_config("Diffusion_ITQC")
    cfg.dataset.expert.demo_path = dummy_path
    cfg.env.control_mode = "joint_pos"
    
    # --- 模型配置 (维度对齐) ---
    cfg.env.proprio_dim = 7
    cfg.env.action_dim = 7
    cfg.actor.encoder.proprio_dim = 7
    cfg.critic.encoder.proprio_dim = 7
    cfg.actor.action_dim = 7
    
    # 核心修正：
    # 1. VisualEncoder 代表单相机提取器，in_channels 应该是 3
    cfg.actor.encoder.visual.in_channels = 3 
    cfg.critic.encoder.visual.in_channels = 3
    # 2. 我们需要在创建 Agent 时确保 StateEncoder 知道有 2 个相机
    cfg.env.num_cameras = 2 
    
    cfg.train.batch_size = 4
    
    print("\n[Step 1] Initializing Dataset...")
    dataset = ExpertDataset(cfg=cfg, device="cpu")
    print(f"Dataset slice count: {len(dataset)}")
    
    # 验证 getitem
    sample = dataset[0]
    assert "observations" in sample and "rgb" in sample["observations"], "Obs RGB missing!"
    assert sample["observations"]["rgb"].dtype == torch.uint8, "Dataset should return uint8!"
    assert sample["observations"]["state"].shape == (cfg.env.obs_horizon, 7), "State sequence shape error!"
    print("✅ Dataset __getitem__ structure verified.")

    print("\n[Step 2] Initializing Agent...")
    agent = make_agent("Diffusion_ITQC", cfg)
    
    # 手动触发归一化拟合
    print("Fitting action normalizer...")
    agent.fit_action_normalizer(dataset.get_all_actions())
    
    print("\n[Step 3] Simulating Training Step...")
    # 模拟 train_loop 逻辑
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)))
    
    # 搬运数据
    batch = agent._batch_to_device(batch)
    
    # 1. 验证 Preprocess (关键: 检查是否执行了 /255)
    # Note: update_critic 内部会调用 preprocess_obs
    print("Testing Critic Update...")
    loss_critic = agent.update_critic(batch, critic_type="Standard")
    assert "loss_q" in loss_critic, "Critic update failed!"
    print(f"Critic Loss Q: {loss_critic['loss_q']:.4f}")
    
    print("Testing Actor Update...")
    loss_actor = agent.update_actor(batch)
    assert "loss_actor" in loss_actor, "Actor update failed!"
    print(f"Actor Loss: {loss_actor['loss_actor']:.4f}")

    print("\n🎉 Pipeline verification complete! Dataset and Policy are perfectly aligned.")

if __name__ == "__main__":
    try:
        verify_pipeline()
    finally:
        if os.path.exists("datasets/verify_training_dummy.hdf5"):
            os.remove("datasets/verify_training_dummy.hdf5")
