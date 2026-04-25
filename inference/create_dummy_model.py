import os
import sys

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.DiffusionPolicy.model import DiffusionModel
from training.Diffusion_Training.training_config import TRAINING_CONFIG


def create_dummy_model(output_path="outputs/checkpoints/vision/final_vision_policy.pth"):
    print("正在构建虚假视觉模型结构...")
    config = TRAINING_CONFIG
    model = DiffusionModel(
        obs_dim=config.obs_dim,
        vision_feature_dim=config.vision_feature_dim,
        action_dim=config.continuous_action_dim,
        n_obs_steps=config.n_obs_steps,
        prediction_horizon=config.prediction_horizon,
        image_size=config.image_size,
        pretrained_vision=config.pretrained_vision,
        vision_weights_path=config.vision_weights_path,
    )

    checkpoint = {
        "model_state": model.state_dict(),
        "stats": {
            "state_mean": np.zeros(config.obs_dim, dtype=np.float32),
            "state_std": np.ones(config.obs_dim, dtype=np.float32),
            "action_mean": np.zeros(config.continuous_action_dim, dtype=np.float32),
            "action_std": np.ones(config.continuous_action_dim, dtype=np.float32),
        },
        "step": 0,
        "is_dummy": True,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"✅ 虚假模型已生成至: {output_path}")
    print(
        "模型参数: "
        f"obs_dim={config.obs_dim}, vision_feature_dim={config.vision_feature_dim}, "
        f"continuous_action_dim={config.continuous_action_dim}, horizon={config.prediction_horizon}"
    )


if __name__ == "__main__":
    create_dummy_model()
