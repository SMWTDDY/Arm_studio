from collections import deque
from dataclasses import replace
import os
import sys

import numpy as np
import torch
from diffusers import DDPMScheduler
from torchvision import transforms

from .action_codec import decode_sequence_to_pose_actions
from .model import DiffusionModel
from .state_codec import project_state_row

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.Diffusion_Training.training_config import TRAINING_CONFIG


class DiffusionVisionPolicy:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        num_inference_steps=None,
        use_ema=True,
    ):
        self.config = self._load_checkpoint_config(checkpoint_path)
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.prediction_horizon = self.config.prediction_horizon
        self.num_inference_steps = num_inference_steps or self.config.num_inference_steps
        self.n_obs_steps = self.config.n_obs_steps
        self.use_ema = use_ema
        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.stats = None

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(
            f"[Policy] device={self.device} inference_steps={self.num_inference_steps} "
            f"obs_steps={self.n_obs_steps} pred_horizon={self.prediction_horizon}"
        )
        self._build_model()
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"权重文件不存在: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_dict = checkpoint.get("config")
        if not config_dict:
            return TRAINING_CONFIG

        valid = {field: value for field, value in config_dict.items() if hasattr(TRAINING_CONFIG, field)}
        if "allowed_control_modes" in valid and isinstance(valid["allowed_control_modes"], list):
            valid["allowed_control_modes"] = tuple(valid["allowed_control_modes"])
        if "data_dirs" in valid and isinstance(valid["data_dirs"], list):
            valid["data_dirs"] = tuple(valid["data_dirs"])
        return replace(TRAINING_CONFIG, **valid)

    def _build_model(self):
        self.model = DiffusionModel(
            obs_dim=self.config.obs_dim,
            vision_feature_dim=self.config.vision_feature_dim,
            action_dim=self.config.continuous_action_dim,
            n_obs_steps=self.config.n_obs_steps,
            prediction_horizon=self.config.prediction_horizon,
            image_size=self.config.image_size,
            pretrained_vision=self.config.pretrained_vision,
            vision_weights_path=self.config.vision_weights_path,
        ).to(self.device)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"权重文件不存在: {path}")

        print(f"[Policy] loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state" in checkpoint:
            state_key = "ema_model_state" if self.use_ema and "ema_model_state" in checkpoint else "model_state"
            state = checkpoint[state_key]
            model_state = self.model.state_dict()
            model_state.update({k: v for k, v in state.items() if k in model_state})
            self.model.load_state_dict(model_state)
            print(f"[Policy] loaded {state_key}")
        elif "vision_encoder" in checkpoint and "model" in checkpoint:
            raise RuntimeError(
                "This is a legacy checkpoint for the old UNet layout. "
                "Please retrain with training/Diffusion_Training/train_diffusion_vision.py."
            )
        else:
            self.model.load_state_dict(checkpoint)

        if "stats" in checkpoint:
            self.stats = {
                key: torch.as_tensor(value, device=self.device, dtype=torch.float32)
                for key, value in checkpoint["stats"].items()
            }
            print("[Policy] normalizer stats loaded")
        else:
            raise RuntimeError("Checkpoint is missing normalizer stats.")

        self.model.eval()
        print("[Policy] checkpoint ready")

    def _normalize_state(self, state):
        state_mean = self.stats["state_mean"]
        state_std = self.stats["state_std"]
        return (state - state_mean) / state_std

    def _unnormalize_action(self, action):
        mean = self.stats["action_mean"].view(1, -1, 1)
        std = self.stats["action_std"].view(1, -1, 1)
        return action * std + mean

    def _process_view(self, obs, view_name):
        img = obs["sensor_data"][view_name]["rgb"]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.1 else img.astype(np.uint8)
        return self.transform(img)

    @torch.no_grad()
    def act(self, obs_dict):
        self.obs_buffer.append(obs_dict)
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.append(obs_dict)

        states, fronts, hands = [], [], []
        for obs in self.obs_buffer:
            state = obs["state"]
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()
            state_tensor = torch.from_numpy(
                project_state_row(state, self.config.obs_dim)
            ).to(self.device)
            states.append(self._normalize_state(state_tensor))
            fronts.append(self._process_view(obs, self.config.front_camera))
            hands.append(self._process_view(obs, self.config.hand_camera))

        states_tensor = torch.stack(states).unsqueeze(0)
        front_tensor = torch.stack(fronts).unsqueeze(0).to(self.device)
        hand_tensor = torch.stack(hands).unsqueeze(0).to(self.device)
        obs_cond = self.model.make_obs_cond(states_tensor, front_tensor, hand_tensor)

        normalized_actions = torch.randn(
            (1, self.config.continuous_action_dim, self.prediction_horizon),
            device=self.device,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            noise_pred = self.model.predict_noise(normalized_actions, timestep, obs_cond)
            normalized_actions = self.noise_scheduler.step(
                noise_pred, timestep, normalized_actions
            ).prev_sample

        normalized_actions = torch.clamp(
            normalized_actions,
            -self.config.sample_clip,
            self.config.sample_clip,
        )
        continuous = self._unnormalize_action(normalized_actions)[0].transpose(0, 1)

        gripper_logits = self.model.predict_gripper_logits(obs_cond)[0]
        gripper_labels = (torch.sigmoid(gripper_logits) > 0.5).long()

        pose_actions = decode_sequence_to_pose_actions(
            continuous.detach().cpu().numpy(),
            gripper_labels.detach().cpu().numpy(),
            open_value=self.config.gripper_open_value,
            close_value=self.config.gripper_close_value,
        )
        return pose_actions.T.astype(np.float32)
