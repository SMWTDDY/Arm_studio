# Training And Data Usage

This document records the current recommended workflow for Piper pose-action datasets and vision diffusion policy training.

## 1. Current Policy Contract

The dataset-facing action is always a 7D pose action:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

The training-facing continuous action is 9D:

```text
[x, y, z, rotation_6d]
```

The gripper is not trained as a continuous regression target. It is trained separately as a binary label:

```text
0 = open
1 = close
```

The policy checkpoint therefore stores:

```text
model_state
ema_model_state
optimizer
scheduler
scaler
stats
step
best_val_loss
```

At inference time, the model decodes 9D continuous action back to 6D pose, thresholds the binary gripper output, then executes:

```text
pose -> BoundedPiperIK -> pd_joint_pos
```

This avoids ManiSkill's unstable built-in `pd_ee_pose` IK branch for Piper.

## 2. Important Configuration

Main file:

```text
training/Diffusion_Training/training_config.py
```

Important defaults:

```python
image_size = 128
vision_feature_dim = 128
continuous_action_dim = 9
action_representation = "xyz_rot6d_gripper_binary"
prediction_horizon = 32
n_obs_steps = 3
inference_stride = 8
batch_size = 64
save_every = 5000
log_every = 10
num_workers = 2
pretrained_vision = False
```

`pretrained_vision=False` means the ResNet18 vision encoder will not download weights from the internet. To use local ImageNet weights, set:

```python
vision_weights_path = "/path/to/resnet18-f37072fd.pth"
```

## 3. Prepare Unified Pose Dataset

Use this when your dataset directory contains a mix of old joint recordings and new pose recordings.

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/build_unified_pose_dataset.py \
  "datasets/*.hdf5" \
  "datasets/auto_collected_real/*.hdf5" \
  --output-dir datasets/unified_pose_all \
  --prefix piper_pose_unified
```

The output files are named like:

```text
datasets/unified_pose_all/piper_pose_unified_0000.hdf5
datasets/unified_pose_all/piper_pose_unified_0001.hdf5
...
```

Each converted file contains:

```text
action
action_continuous
gripper_label
observation/state
observation/sensor_data/front_view/rgb
observation/sensor_data/hand_camera/rgb
```

A manifest is written to:

```text
datasets/unified_pose_all/manifest.csv
```

## 4. Build Image Cache

Lazy HDF5 image loading avoids memory explosion, but training can still be slow because every batch reads and resizes images. Build a resized cache before serious training:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/build_training_image_cache.py \
  datasets/unified_pose_all \
  --output-dir datasets/unified_pose_all_128 \
  --image-size 128 \
  --compression lzf
```

Recommended compression choices:

```text
lzf    good default, faster random reads than gzip
none   fastest reads, larger files
gzip   smaller files, slower random reads
```

Train from the cache directory:

```text
datasets/unified_pose_all_128
```

Do not point training at both the original and cached directories at the same time, or the same demonstrations will be duplicated.

## 5. Validate Dataset

Check pose/FK/IK consistency:

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/validate_pose_dataset.py \
  "datasets/unified_pose_all/*.hdf5"
```

If a solver reaches max iteration but final position and rotation errors are inside threshold, it is treated as acceptable.

## 6. Export Camera Videos

To quickly inspect whether front and hand cameras are recorded correctly:

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/export_hdf5_videos.py \
  "datasets/unified_pose_all/*.hdf5" \
  --output-dir outputs/dataset_videos/pose_camera_samples \
  --fps 30 \
  --max-frames 300
```

The exported videos concatenate:

```text
front_view | hand_camera
```

## 7. Train

First check CUDA:

```bash
nvidia-smi
```

Then:

```bash
/home/lebinge/miniforge3/envs/SL/bin/python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Start training:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/lebinge/miniforge3/envs/SL/bin/python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --total-steps 100000 \
  --save-every 5000 \
  --num-workers 2
```

Useful overrides:

```bash
--batch-size 32
--batch-size 64
--num-workers 0
--num-workers 2
--num-workers 4
```

If GPU utilization is low, increase `num_workers` carefully. If system RAM becomes unstable, reduce it.

Every `save_every` steps, training prints the average training loss since the previous checkpoint and writes a checkpoint.

## 8. Resume Training

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/lebinge/miniforge3/envs/SL/bin/python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --resume outputs/checkpoints/vision/checkpoint_5000.pth \
  --total-steps 100000 \
  --save-every 5000 \
  --num-workers 2
```

## 9. Inference

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/run_inference.py \
  --mode local \
  --model outputs/checkpoints/vision/final_vision_policy.pth \
  --env environments/conveyor_env.py \
  --ctrl_mode pose \
  --binary_gripper \
  --target_fps 60 \
  --inference_stride 8
```

Frequency notes:

```text
target_fps        target env.step control frequency
inference_stride  run policy once every N env steps
```

If training data was collected at 60 Hz, keep `target_fps=60` when possible. If inference cannot run at 60 Hz, use temporal action aggregation with `inference_stride` instead of slowing the environment step rate.

## 10. Troubleshooting

### Training Tries To Download ResNet18

The default config avoids network downloads:

```python
pretrained_vision = False
```

If you set it to `True`, torchvision may try to download weights. In offline environments, use `vision_weights_path` instead.

### CPU Training Is Very Slow

If training prints:

```text
[Warning] CUDA is not available. Training will run on CPU and can be extremely slow.
```

fix NVIDIA driver/CUDA visibility first. CPU training for this model is only useful for smoke tests.

### HDF5 Loading Is Slow

Use the 128 image cache:

```text
datasets/unified_pose_all_128
```

Try `--compression none` if disk space is available and random read speed still bottlenecks training.

### Memory Usage Is Too High

Use lazy HDF5 loading with the cache directory. Do not preload all RGB arrays into RAM.

Reduce:

```text
batch_size
num_workers
```

### Matplotlib Cache Warning

Use:

```bash
MPLCONFIGDIR=/tmp/matplotlib
```

before the training command.
