# 使用说明

这份文档记录当前推荐的 Piper 机械臂数据、训练和推理流程。

## 1. 推荐链路

当前推荐统一使用 pose 数据训练：

```text
采集 pose action -> 统一数据 -> 生成 128 图像缓存 -> 训练视觉 diffusion policy -> 推理输出 pose -> BoundedPiperIK -> pd_joint_pos 执行
```

HDF5 中保存的标准动作：

```text
action = [x, y, z, roll, pitch, yaw, gripper]
```

训练时内部使用：

```text
action_continuous = [x, y, z, rotation_6d]
gripper_label = 0/1
```

也就是说，位姿连续部分是 9 维，夹爪是单独二分类。

## 2. 环境准备

进入项目目录：

```bash
cd /home/lebinge/Desktop/Arm_studio
```

激活环境：

```bash
conda activate SL
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果机器只用于训练，也可以安装训练专用依赖：

```bash
pip install -r training/Diffusion_Training/requirements.txt
```

训练前检查 CUDA：

```bash
nvidia-smi
```

再检查 PyTorch 是否能看到 GPU：

```bash
/home/lebinge/miniforge3/envs/SL/bin/python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

如果输出不是 `True`，训练会退回 CPU，非常慢。

## 3. 统一数据格式

如果目录里混有旧 joint 数据和新 pose 数据，先统一成 pose 数据：

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/build_unified_pose_dataset.py \
  "datasets/*.hdf5" \
  "datasets/auto_collected_real/*.hdf5" \
  --output-dir datasets/unified_pose_all \
  --prefix piper_pose_unified
```

输出目录：

```text
datasets/unified_pose_all
```

输出文件会统一命名为：

```text
piper_pose_unified_0000.hdf5
piper_pose_unified_0001.hdf5
...
```

并生成：

```text
datasets/unified_pose_all/manifest.csv
```

## 4. 生成训练图像缓存

为了避免训练时每个 batch 都临时 resize 图像，建议先生成 128 图像缓存：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python scripts/data_tools/build_training_image_cache.py \
  datasets/unified_pose_all \
  --output-dir datasets/unified_pose_all_128 \
  --image-size 128 \
  --compression lzf
```

推荐压缩方式：

```text
lzf    推荐默认值，读取速度和体积比较平衡
none   读取最快，但占用硬盘更大
gzip   文件更小，但训练随机读较慢
```

正式训练时推荐使用：

```text
datasets/unified_pose_all_128
```

不要同时把 `datasets/unified_pose_all` 和 `datasets/unified_pose_all_128` 都传给训练脚本，否则会重复训练同一批数据。

## 5. 验证数据

验证 pose/FK/IK 是否对齐：

```bash
python scripts/data_tools/validate_pose_dataset.py \
  "datasets/unified_pose_all/*.hdf5"
```

如果 IK solver 状态是 max iteration，但最终误差已经合格，不会被误判为失败。

## 6. 导出视频检查摄像头

导出前视角和手眼相机拼接视频：

```bash
/home/lebinge/miniforge3/envs/SL/bin/python scripts/data_tools/export_hdf5_videos.py \
  "datasets/unified_pose_all/*.hdf5" \
  --output-dir outputs/dataset_videos/pose_camera_samples \
  --fps 30 \
  --max-frames 300
```

视频格式：

```text
front_view | hand_camera
```

## 7. 训练

主要训练配置文件：

```text
training/Diffusion_Training/training_config.py
```

推荐训练命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/yons/miniconda3/envs/SL/bin/python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --total-steps 100000 \
  --prediction-horizon 16 \
  --inference-stride 4 \
  --num-inference-steps 16 \
  --save-every 5000 \
  --num-workers 4 \
  --output-dir outputs/checkpoints/vision_h16
```

常用参数：

```text
--batch-size      覆盖 batch_size
--num-workers     DataLoader 进程数
--save-every      checkpoint 保存间隔
--resume          从 checkpoint 恢复训练
--prediction-horizon  训练/推理动作窗口长度
--inference-stride    推理时动作重规划间隔
--num-inference-steps diffusion 去噪步数
```

checkpoint 输出目录：

```text
outputs/checkpoints/vision
```

每到 `save_every` 步，训练会打印上一个 checkpoint 到当前 checkpoint 之间的平均训练损失。

## 8. 恢复训练

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/yons/miniconda3/envs/SL/bin/python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --resume outputs/checkpoints/vision_h16/policy_vision_step_5000.pth \
  --total-steps 100000 \
  --prediction-horizon 16 \
  --inference-stride 4 \
  --num-inference-steps 16 \
  --save-every 5000 \
  --num-workers 4 \
  --output-dir outputs/checkpoints/vision_h16
```

## 9. 推理

```bash
/home/yons/miniconda3/envs/SL/bin/python scripts/run_inference.py \
  --mode local \
  --model outputs/checkpoints/vision_h16/final_vision_policy.pth \
  --env environments/conveyor_env.py \
  --ctrl_mode pose \
  --binary_gripper \
  --target_fps 24 \
  --inference_stride 4 \
  --num-inference-steps 16 \
  --ready-start
```

关键参数：

```text
--target_fps          推荐 24；当前实时推理更容易稳定达到
--inference_stride    每隔多少个 env.step 重新推理一次
--ready-start         推理前先进入采集时使用的 ready qpos
--ctrl_mode pose      模型输出 pose，再用 BoundedPiperIK 转成 joint 执行
--binary_gripper      使用二值夹爪
```

## 10. 常见问题

### 训练速度特别慢

先看训练开头是否出现：

```text
[Warning] CUDA is not available. Training will run on CPU and can be extremely slow.
```

如果出现，先修 CUDA/驱动，不要继续 CPU 正式训练。

### 数据加载慢

先生成并使用：

```text
datasets/unified_pose_all_128
```

如果还是慢，可以尝试缓存时使用：

```bash
--compression none
```

### 内存爆了

不要预加载全部 RGB 图像。当前训练脚本已经使用 lazy HDF5 loading。降低：

```text
batch_size
num_workers
```

### ResNet18 下载失败

默认配置已经关闭联网下载：

```python
pretrained_vision = False
```

如果需要本地预训练权重，在 `training_config.py` 中设置：

```python
vision_weights_path = "/path/to/resnet18-f37072fd.pth"
```

### Matplotlib 缓存目录报错

训练命令前加：

```bash
MPLCONFIGDIR=/tmp/matplotlib
```
