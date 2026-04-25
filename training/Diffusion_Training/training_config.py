import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class TrainingConfig:
    # Data
    data_dirs: Tuple[str, ...] = (
        os.path.join(PROJECT_ROOT, "datasets", "unified_pose_all"),
    )
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "vision")
    allowed_control_modes: Tuple[str, ...] = ("joint", "pose")
    allowed_source_control_modes: Tuple[str, ...] = ()
    front_camera: str = "front_view"
    hand_camera: str = "hand_camera"
    image_size: int = 128
    val_ratio: float = 0.1
    seed: int = 42

    # Observation/action representation
    obs_dim: int = 16
    vision_feature_dim: int = 128
    pretrained_vision: bool = False
    vision_weights_path: str = ""
    continuous_action_dim: int = 9  # [x, y, z, rotation_6d]
    gripper_num_classes: int = 2
    action_representation: str = "xyz_rot6d_gripper_binary"
    gripper_threshold: float = 0.0
    gripper_open_value: float = -1.0
    gripper_close_value: float = 1.0

    # Sequence
    prediction_horizon: int = 32
    n_obs_steps: int = 3
    inference_stride: int = 8

    # Optimization
    batch_size: int = 64
    total_steps: int = 100_000
    save_every: int = 5_000
    log_every: int = 10
    num_workers: int = 2
    diffusion_lr: float = 1e-4
    vision_lr: float = 3e-5
    weight_decay: float = 1e-6
    min_lr: float = 1e-6
    gripper_loss_weight: float = 0.2
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.995

    # Diffusion
    num_train_timesteps: int = 100
    num_inference_steps: int = 20
    sample_clip: float = 4.0

    # Resume
    resume_path: str = ""


TRAINING_CONFIG = TrainingConfig()
