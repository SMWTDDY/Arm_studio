import math

import torch
import torch.nn as nn

from .vision_encoder import MultiViewEncoder


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, channels, cond_dim, kernel_size=5, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        self.act = nn.Mish()

    def forward(self, x, cond):
        scale, shift = self.cond_proj(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return x + h


class ConditionalTemporalModel(nn.Module):
    def __init__(self, action_dim, cond_dim, hidden_dim=256, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        block_cond_dim = cond_dim + time_dim
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [
                ConditionalResidualBlock1D(hidden_dim, block_cond_dim, dilation=d)
                for d in (1, 2, 4, 8, 1, 2, 4, 8)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, action_dim, 1),
        )

    def forward(self, noisy_actions, timesteps, obs_cond):
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(noisy_actions.shape[0])
        elif timesteps.shape[0] == 1 and noisy_actions.shape[0] > 1:
            timesteps = timesteps.expand(noisy_actions.shape[0])

        time_emb = self.time_mlp(timesteps.to(noisy_actions.device))
        cond = torch.cat([obs_cond, time_emb], dim=1)

        h = self.input_proj(noisy_actions)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        obs_dim=16,
        vision_feature_dim=224,
        action_dim=5,
        n_obs_steps=3,
        prediction_horizon=32,
        image_size=224,
        hidden_dim=256,
        pretrained_vision=False,
        vision_weights_path="",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.vision_feature_dim = vision_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.prediction_horizon = prediction_horizon
        self.image_size = image_size

        self.vision_encoder = MultiViewEncoder(
            feature_dim=vision_feature_dim,
            pretrained=pretrained_vision,
            weights_path=vision_weights_path,
        )
        self.obs_cond_dim = (obs_dim + vision_feature_dim * 2) * n_obs_steps

        self.noise_net = ConditionalTemporalModel(
            action_dim=action_dim,
            cond_dim=self.obs_cond_dim,
            hidden_dim=hidden_dim,
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(self.obs_cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, prediction_horizon),
        )

    def get_vision_features(self, front, hand):
        bsz = front.shape[0]
        f = front.view(bsz * self.n_obs_steps, 3, self.image_size, self.image_size)
        h = hand.view(bsz * self.n_obs_steps, 3, self.image_size, self.image_size)
        features = self.vision_encoder(f, h)
        return features.view(bsz, -1)

    def make_obs_cond(self, states, fronts, hands):
        vision_features = self.get_vision_features(fronts, hands)
        vision_per_frame = vision_features.view(states.shape[0], self.n_obs_steps, -1)
        obs_per_frame = torch.cat([states, vision_per_frame], dim=2)
        return obs_per_frame.reshape(states.shape[0], -1)

    def predict_noise(self, noisy_actions, timesteps, obs_cond):
        return self.noise_net(noisy_actions, timesteps, obs_cond)

    def predict_gripper_logits(self, obs_cond):
        return self.gripper_head(obs_cond)
