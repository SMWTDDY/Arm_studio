import argparse
import glob
import os
import sys
from dataclasses import asdict, replace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.DiffusionPolicy.action_codec import encode_pose_to_continuous, gripper_to_label
from models.DiffusionPolicy.model import DiffusionModel
from models.DiffusionPolicy.state_codec import project_state_batch
from teleop.get_pose import get_pose
from training.Diffusion_Training.training_config import TRAINING_CONFIG, TrainingConfig


def infer_control_mode(path, attrs):
    mode = attrs.get("control_mode")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if mode in ("joint", "pose"):
        return mode

    name = os.path.basename(path).lower()
    if "joint" in name:
        return "joint"
    if "pose" in name:
        return "pose"
    raise ValueError(f"Cannot infer control_mode for {path}")


def expand_data_paths(data_dirs):
    paths = []
    for item in data_dirs:
        item = os.path.abspath(item)
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.hdf5"))))
        else:
            matches = sorted(glob.glob(item))
            paths.extend(matches if matches else [item])
    return list(dict.fromkeys(paths))


def gripper_value(row):
    if row.shape[0] <= 6:
        return 1.0
    if row.shape[0] == 7:
        return float(row[6])
    return float(np.mean(row[6:]))


def encode_actions(actions, control_mode, config):
    continuous = np.zeros((actions.shape[0], config.continuous_action_dim), dtype=np.float32)
    gripper = np.zeros((actions.shape[0],), dtype=np.float32)

    for i, row in enumerate(actions):
        if control_mode == "joint":
            pose = get_pose(row[:6])
        elif control_mode == "pose":
            pose = row[:6]
        else:
            raise ValueError(f"Unsupported control_mode={control_mode}")

        continuous[i] = encode_pose_to_continuous(pose)
        gripper[i] = gripper_to_label(gripper_value(row), config.gripper_threshold)

    return continuous, gripper


class TrajectoryStore:
    def __init__(self, config):
        self.config = config
        self.episodes = []
        self._load()

    def _load(self):
        paths = expand_data_paths(self.config.data_dirs)
        if not paths:
            raise FileNotFoundError(f"No HDF5 files found in {self.config.data_dirs}")

        print("Loading trajectories:")
        for path in tqdm(paths):
            if not os.path.exists(path):
                print(f"[Skip] missing: {path}")
                continue
            try:
                episode = self._load_one(path)
            except Exception as exc:
                print(f"\n[Skip] {path}: {exc}")
                continue
            if episode is not None:
                self.episodes.append(episode)

        if not self.episodes:
            raise RuntimeError("No valid episodes loaded.")
        print(f"Loaded {len(self.episodes)} valid episodes.")

    def _load_one(self, path):
        with h5py.File(path, "r") as f:
            actions = np.asarray(f["action"], dtype=np.float32)
            attrs = dict(f.attrs)
            control_mode = infer_control_mode(path, attrs)
            if control_mode not in self.config.allowed_control_modes:
                return None
            source_mode = attrs.get("source_control_mode")
            if isinstance(source_mode, bytes):
                source_mode = source_mode.decode("utf-8")
            if self.config.allowed_source_control_modes and source_mode not in self.config.allowed_source_control_modes:
                return None
            states = project_state_batch(f["observation/state"][:], self.config.obs_dim)

            front_key = f"observation/sensor_data/{self.config.front_camera}/rgb"
            hand_key = f"observation/sensor_data/{self.config.hand_camera}/rgb"
            if front_key not in f or hand_key not in f:
                raise KeyError(f"missing required cameras: {self.config.front_camera}, {self.config.hand_camera}")

            n_images = min(f[front_key].shape[0], f[hand_key].shape[0])

        n = min(len(actions), len(states), n_images)
        if n <= self.config.prediction_horizon:
            return None

        with h5py.File(path, "r") as f:
            if "action_continuous" in f and f["action_continuous"].shape[1] == self.config.continuous_action_dim:
                continuous = np.asarray(f["action_continuous"][:n], dtype=np.float32)
            else:
                continuous, _ = encode_actions(actions[:n], control_mode, self.config)

            if "gripper_label" in f:
                gripper = np.asarray(f["gripper_label"][:n], dtype=np.float32)
            else:
                _, gripper = encode_actions(actions[:n], control_mode, self.config)

        return {
            "path": path,
            "control_mode": control_mode,
            "source_control_mode": source_mode,
            "front_key": front_key,
            "hand_key": hand_key,
            "states": states[:n],
            "continuous": continuous,
            "gripper": gripper,
        }


def split_episodes(num_episodes, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_episodes)
    rng.shuffle(indices)
    val_count = max(1, int(round(num_episodes * val_ratio))) if num_episodes > 1 else 0
    val_ids = set(indices[:val_count].tolist())
    train_ids = [i for i in range(num_episodes) if i not in val_ids]
    val_ids = sorted(val_ids)
    if not train_ids:
        train_ids, val_ids = [0], []
    return train_ids, val_ids


def compute_stats(episodes, episode_ids):
    states = np.concatenate([episodes[i]["states"] for i in episode_ids], axis=0)
    actions = np.concatenate([episodes[i]["continuous"] for i in episode_ids], axis=0)

    return {
        "state_mean": states.mean(axis=0).astype(np.float32),
        "state_std": np.maximum(states.std(axis=0), 1e-6).astype(np.float32),
        "action_mean": actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(actions.std(axis=0), 1e-6).astype(np.float32),
    }


class WindowedVisualDataset(Dataset):
    def __init__(self, episodes, episode_ids, stats, config, train=True):
        self.episodes = episodes
        self.stats = stats
        self.config = config
        self._h5_files = {}
        self.indices = []
        for ep_idx in episode_ids:
            ep_len = len(episodes[ep_idx]["continuous"])
            for start in range(ep_len - config.prediction_horizon):
                self.indices.append((ep_idx, start))

        color_aug = []
        if train:
            color_aug = [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02)]
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                *color_aug,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def normalize(self, x, key):
        return (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]

    def _file(self, ep):
        path = ep["path"]
        h5 = self._h5_files.get(path)
        if h5 is None:
            h5 = h5py.File(path, "r")
            self._h5_files[path] = h5
        return h5

    def _read_rgb(self, ep, key, frame_idx):
        img = self._file(ep)[key][frame_idx]
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        return img.astype(np.uint8, copy=False)

    def close(self):
        for h5 in self._h5_files.values():
            h5.close()
        self._h5_files.clear()

    def __del__(self):
        self.close()

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        ep = self.episodes[ep_idx]
        obs_ids = np.clip(np.arange(start - self.config.n_obs_steps + 1, start + 1), 0, None)
        action_ids = slice(start, start + self.config.prediction_horizon)

        states = self.normalize(ep["states"][obs_ids], "state")
        actions = self.normalize(ep["continuous"][action_ids], "action")
        gripper = ep["gripper"][action_ids]

        fronts = torch.stack([self.transform(self._read_rgb(ep, ep["front_key"], i)) for i in obs_ids])
        hands = torch.stack([self.transform(self._read_rgb(ep, ep["hand_key"], i)) for i in obs_ids])

        return {
            "states": torch.from_numpy(states).float(),
            "continuous_action": torch.from_numpy(actions).float().transpose(0, 1),
            "gripper": torch.from_numpy(gripper).float(),
            "fronts": fronts,
            "hands": hands,
        }


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for key, value in state.items():
            if key not in self.shadow:
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.detach().clone() for k, v in state.items()}


def build_model(config, device):
    return DiffusionModel(
        obs_dim=config.obs_dim,
        vision_feature_dim=config.vision_feature_dim,
        action_dim=config.continuous_action_dim,
        n_obs_steps=config.n_obs_steps,
        prediction_horizon=config.prediction_horizon,
        image_size=config.image_size,
        pretrained_vision=config.pretrained_vision,
        vision_weights_path=config.vision_weights_path,
    ).to(device)


def build_optimizer(model, config):
    vision_params = list(model.vision_encoder.parameters())
    vision_param_ids = {id(p) for p in vision_params}
    other_params = [p for p in model.parameters() if id(p) not in vision_param_ids]
    return torch.optim.AdamW(
        [
            {"params": vision_params, "lr": config.vision_lr},
            {"params": other_params, "lr": config.diffusion_lr},
        ],
        weight_decay=config.weight_decay,
    )


def batch_loss(model, batch, noise_scheduler, device, config, use_amp):
    actions = batch["continuous_action"].to(device)
    states = batch["states"].to(device)
    fronts = batch["fronts"].to(device)
    hands = batch["hands"].to(device)
    gripper = batch["gripper"].to(device)

    with autocast(device_type=device.type, enabled=use_amp):
        obs_cond = model.make_obs_cond(states, fronts, hands)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            config.num_train_timesteps,
            (actions.shape[0],),
            device=device,
        ).long()
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = model.predict_noise(noisy_actions, timesteps, obs_cond)
        diffusion_loss = nn.functional.mse_loss(noise_pred, noise)
        gripper_logits = model.predict_gripper_logits(obs_cond)
        gripper_loss = nn.functional.binary_cross_entropy_with_logits(gripper_logits, gripper)
        loss = diffusion_loss + config.gripper_loss_weight * gripper_loss

    return loss, diffusion_loss.detach(), gripper_loss.detach()


@torch.no_grad()
def evaluate(model, dataloader, noise_scheduler, device, config):
    if dataloader is None:
        return None
    model.eval()
    losses = []
    diffusion_losses = []
    gripper_losses = []
    use_amp = device.type == "cuda"
    for batch in dataloader:
        loss, diffusion_loss, gripper_loss = batch_loss(
            model, batch, noise_scheduler, device, config, use_amp
        )
        losses.append(float(loss.detach().cpu()))
        diffusion_losses.append(float(diffusion_loss.cpu()))
        gripper_losses.append(float(gripper_loss.cpu()))
    model.train()
    if not losses:
        return None
    return {
        "loss": float(np.mean(losses)),
        "diffusion": float(np.mean(diffusion_losses)),
        "gripper": float(np.mean(gripper_losses)),
    }


def save_checkpoint(path, model, ema, optimizer, scheduler, scaler, stats, step, best_val_loss, config):
    torch.save(
        {
            "model_state": model.state_dict(),
            "ema_model_state": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "stats": stats,
            "step": step,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
        },
        path,
    )


def load_resume(path, model, ema, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if "ema_model_state" in checkpoint:
        ema.load_state_dict(checkpoint["ema_model_state"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("best_val_loss", np.inf))


def plot_loss(loss_history, output_dir, current_step):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, alpha=0.3, color="blue", label="Raw Loss")
    if len(loss_history) > 100:
        smooth = np.convolve(loss_history, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(99, len(loss_history)), smooth, color="red", label="Smoothed Loss (100 avg)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve (Step {current_step})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Piper vision diffusion policy.")
    parser.add_argument("--data-dir", action="append", dest="data_dirs", help="HDF5 directory or glob. Can be repeated.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--source-control-mode", action="append", dest="source_control_modes",
                        help="只训练指定 source_control_mode，可重复传参，如 --source-control-mode pose")
    parser.add_argument("--prediction-horizon", type=int, default=None)
    parser.add_argument("--inference-stride", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    return parser.parse_args()


def config_from_args(args):
    config = TRAINING_CONFIG
    updates = {}
    if args.data_dirs:
        updates["data_dirs"] = tuple(args.data_dirs)
    if args.output_dir:
        updates["output_dir"] = os.path.abspath(args.output_dir)
    if args.resume:
        updates["resume_path"] = os.path.abspath(args.resume)
    if args.total_steps is not None:
        updates["total_steps"] = args.total_steps
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.save_every is not None:
        updates["save_every"] = args.save_every
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    if args.obs_dim is not None:
        updates["obs_dim"] = args.obs_dim
    if args.source_control_modes:
        updates["allowed_source_control_modes"] = tuple(args.source_control_modes)
    if args.prediction_horizon is not None:
        updates["prediction_horizon"] = args.prediction_horizon
    if args.inference_stride is not None:
        updates["inference_stride"] = args.inference_stride
    if args.num_inference_steps is not None:
        updates["num_inference_steps"] = args.num_inference_steps
    return replace(config, **updates) if updates else config


def train():
    config = config_from_args(parse_args())
    os.makedirs(config.output_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type != "cuda":
        print("[Warning] CUDA is not available. Training will run on CPU and can be extremely slow.")

    store = TrajectoryStore(config)
    train_ids, val_ids = split_episodes(len(store.episodes), config.val_ratio, config.seed)
    stats = compute_stats(store.episodes, train_ids)

    train_dataset = WindowedVisualDataset(store.episodes, train_ids, stats, config, train=True)
    val_dataset = WindowedVisualDataset(store.episodes, val_ids, stats, config, train=False) if val_ids else None
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": use_amp,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        if val_dataset is not None
        else None
    )

    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_steps, eta_min=config.min_lr
    )
    scaler = GradScaler(enabled=use_amp)
    ema = EMA(model, config.ema_decay)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    current_step = 0
    best_val_loss = np.inf
    if config.resume_path:
        current_step, best_val_loss = load_resume(
            config.resume_path, model, ema, optimizer, scheduler, scaler, device
        )
        print(f"Resumed from {config.resume_path} at step {current_step}")

    print(
        f"Training start | device={device} | episodes={len(store.episodes)} "
        f"train_windows={len(train_dataset)} val_windows={len(val_dataset) if val_dataset else 0}"
    )
    print(
        f"Action={config.action_representation} | continuous_dim={config.continuous_action_dim} "
        f"| gripper=binary"
    )

    loss_history = []
    interval_losses = []
    interval_diffusion = []
    interval_gripper = []

    model.train()
    while current_step < config.total_steps:
        for batch in train_loader:
            if current_step >= config.total_steps:
                break

            loss, diffusion_loss, gripper_loss = batch_loss(
                model, batch, noise_scheduler, device, config, use_amp
            )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            current_step += 1
            loss_value = float(loss.detach().cpu())
            diffusion_value = float(diffusion_loss.cpu())
            gripper_value_loss = float(gripper_loss.cpu())
            loss_history.append(loss_value)
            interval_losses.append(loss_value)
            interval_diffusion.append(diffusion_value)
            interval_gripper.append(gripper_value_loss)

            if current_step % config.log_every == 0:
                lrs = [group["lr"] for group in optimizer.param_groups]
                print(
                    f"\rStep {current_step}/{config.total_steps} "
                    f"loss={loss_value:.5f} diffusion={diffusion_value:.5f} "
                    f"gripper={gripper_value_loss:.5f} lr={lrs[0]:.2e}/{lrs[1]:.2e}",
                    end="",
                    flush=True,
                )

            if current_step % config.save_every == 0:
                val_metrics = evaluate(model, val_loader, noise_scheduler, device, config)
                avg_loss = float(np.mean(interval_losses))
                avg_diff = float(np.mean(interval_diffusion))
                avg_grip = float(np.mean(interval_gripper))
                print(
                    f"\nCheckpoint {current_step}: "
                    f"interval_train_loss={avg_loss:.6f} "
                    f"diffusion={avg_diff:.6f} gripper={avg_grip:.6f}"
                )
                if val_metrics is not None:
                    print(
                        f"Validation: loss={val_metrics['loss']:.6f} "
                        f"diffusion={val_metrics['diffusion']:.6f} "
                        f"gripper={val_metrics['gripper']:.6f}"
                    )
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        save_checkpoint(
                            os.path.join(config.output_dir, "best_val.pth"),
                            model,
                            ema,
                            optimizer,
                            scheduler,
                            scaler,
                            stats,
                            current_step,
                            best_val_loss,
                            config,
                        )
                        print(f"Saved best_val.pth (val_loss={best_val_loss:.6f})")

                plot_loss(loss_history, config.output_dir, current_step)
                save_checkpoint(
                    os.path.join(config.output_dir, f"policy_vision_step_{current_step}.pth"),
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    scaler,
                    stats,
                    current_step,
                    best_val_loss,
                    config,
                )
                interval_losses.clear()
                interval_diffusion.clear()
                interval_gripper.clear()

    save_checkpoint(
        os.path.join(config.output_dir, "final_vision_policy.pth"),
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
        stats,
        current_step,
        best_val_loss,
        config,
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
