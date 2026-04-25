# Arm Studio

Arm Studio is a Piper robotic arm workspace for ManiSkill simulation, real-to-sim teleoperation, dataset collection, pose-action standardization, and vision diffusion policy training.

The recommended control/data path is:

```text
record pose action -> train pose policy -> decode xyz + rot6d -> pose action -> BoundedPiperIK -> pd_joint_pos
```

Actions stored for training use:

```text
action = [x, y, z, roll, pitch, yaw, gripper]
```

During training, pose rotation is encoded as 6D rotation and gripper is trained as a binary classifier:

```text
continuous_action = [x, y, z, rotation_6d]
gripper_label = 0/1
```

## Environment

Use the existing conda environment if available:

```bash
conda activate SL
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For training-only machines, this smaller list is also available:

```bash
pip install -r training/Diffusion_Training/requirements.txt
```

Before training, confirm CUDA is visible:

```bash
nvidia-smi

/home/lebinge/miniforge3/envs/SL/bin/python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

The training script will run on CPU if CUDA is not available, but it will be very slow.

## Repository Layout

```text
robot/                         Piper agent, action wrapper, bounded IK
environments/                  ManiSkill custom environments
teleop/                        Real arm and keyboard teleoperation
data/                          Data collection and dataset viewers
scripts/                       User-facing commands and utilities
scripts/data_tools/            Dataset conversion, validation, cache, video export
models/DiffusionPolicy/        Vision diffusion policy model and inference policy
training/Diffusion_Training/   Training script and training_config.py
inference/                     Local/remote inference helpers
docs/                          Usage notes and repository documentation
outputs/                       Checkpoints, debug images, exported videos
datasets/                      Local HDF5 datasets and generated caches
```

More detail is in [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md). Chinese day-to-day usage notes are in [docs/USAGE.md](docs/USAGE.md).

## Data Preparation

Build one unified pose dataset from mixed joint and pose recordings:

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/build_unified_pose_dataset.py \
  "datasets/*.hdf5" \
  "datasets/auto_collected_real/*.hdf5" \
  --output-dir datasets/unified_pose_all \
  --prefix piper_pose_unified
```

Build a resized image cache for faster vision training:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/build_training_image_cache.py \
  datasets/unified_pose_all \
  --output-dir datasets/unified_pose_all_128 \
  --image-size 128 \
  --compression lzf
```

Validate pose replay quality:

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/validate_pose_dataset.py \
  "datasets/unified_pose_all/*.hdf5"
```

Export a few HDF5 videos for camera checks:

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/export_hdf5_videos.py \
  "datasets/unified_pose_all/*.hdf5" \
  --output-dir outputs/dataset_videos/pose_camera_samples \
  --fps 30 \
  --max-frames 300
```

## Training

Main configuration lives in:

```text
training/Diffusion_Training/training_config.py
```

Start training from the cached dataset:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/lebinge/miniforge3/envs/SL/bin/python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --total-steps 100000 \
  --save-every 5000 \
  --num-workers 2
```

Checkpoints and the loss plot are written to:

```text
outputs/checkpoints/vision/
```

Resume training:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/lebinge/miniforge3/envs/SL/bin/python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --resume outputs/checkpoints/vision/checkpoint_5000.pth \
  --total-steps 100000 \
  --save-every 5000 \
  --num-workers 2
```

## Inference

Run local inference with the trained checkpoint:

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

The policy outputs pose actions. At execution time, pose is converted through `BoundedPiperIK` and sent to ManiSkill as `pd_joint_pos`.

## More Documentation

Full workflow notes are in [docs/TRAINING_AND_DATA_USAGE.md](docs/TRAINING_AND_DATA_USAGE.md). Chinese usage notes are in [docs/USAGE.md](docs/USAGE.md).
