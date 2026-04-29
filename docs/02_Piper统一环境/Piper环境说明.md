# Piper 统一环境说明

`agent_infra/Piper_Env` 是 Piper 机械臂面向 VLA 训练、部署和专家介入采集的 infra 层。当前目标不是复刻 `Realman_Env` 的全部风格，而是保留它清晰分层与 `meta_keys` 语义，同时让 Piper 的单臂、双臂、多品牌相机和数据 pipeline 更容易被二次开发者调用。

## 1. 分层契约

当前主链路为：

```text
SinglePiperEnv / DualPiperEnv
  -> PiperCameraWrapper
  -> PiperEnv
  -> PiperBaseEnv
  -> BaseRobotEnv
```

- `PiperBaseEnv` 只处理机械臂核心能力：连接 follower/master SDK、读取本体状态、下发 follower action、构建 `obs/state/...` 和 `action/...`。
- `PiperEnv` 在 core env 上增加 teleop：后台读取 master、按 `T` 切换 `tele_enabled`、维护 `obs/under_control/...`，并在 `step()` 内把专家动作写入 `info["actual_action"]`。
- `PiperCameraWrapper` 只处理相机：根据 config 中的 `camera_type / serial_number / role` 启动相机，并追加 `obs/rgb/...`。
- `SinglePiperEnv` 和 `DualPiperEnv` 是最终用户入口：VLA 训练、部署、采集脚本优先调用这两个类。

## 2. Meta Keys 字段约定

单臂 `meta_keys` 形态：

```text
obs/state/joint_pos        (6,)
obs/state/joint_vel        (6,)
obs/state/ee_pose          (6,)
obs/state/gripper_pos      (1,)
obs/under_control/single   (1,)
obs/rgb/<camera_role>      (3, H, W)
action/arm                 (6,)
action/gripper             (1,)
```

双臂会自动加前缀：

```text
obs/state/left_joint_pos
obs/state/right_joint_pos
action/left_arm
action/right_arm
```

`joint` 模式下 `action/*_arm` 表示 6 维绝对关节角。`pose` 模式下 `action/*_arm` 表示 6 维绝对末端位姿 `[x, y, z, roll, pitch, yaw]`。`delta_pose` 模式下 `action/*_arm` 表示相对当前 follower 末端位姿的 6 维增量。`relative_pose_chunk` 模式下 `action/*_arm` 表示展平后的多步 6D 位姿增量，默认 `relative_pose_chunk_size=8`，因此单臂 arm action 维度为 `48`。

`action/*_gripper` 与 `obs/state/*_gripper_pos` 均使用 Piper SDK 的真实夹爪宽度，单位为米，范围为 `0.0 ~ 0.1`。infra 不再做 `0~1` 归一化，也不再做 `0.5/0.8` 这类隐式比例缩放。

遥操时 master 夹爪读数会经过短时跳变过滤，避免夹爪反馈偶发丢包、回零或超范围值导致 follower 夹爪突然开合。夹爪状态读取优先使用 `get_gripper_status()` 的实际反馈；拿不到可靠反馈时会回退到上一条已下发的夹爪命令，而不是把 idle 时的 `0.0` 当成闭合命令。相关配置在 `common` 下：

```yaml
master_gripper_jump_threshold_m: 0.04
master_gripper_confirm_frames: 6
master_gripper_zero_epsilon_m: 0.001
master_gripper_zero_guard_threshold_m: 0.03
```

其中 `master_gripper_zero_*` 用于处理主臂不动时 SDK 可能短暂返回精确 `0.0` 的情况：如果 follower 当前夹爪明显处于打开状态，infra 会把这类直接归零读数视为异常并保持当前夹爪宽度；真实闭合动作只要经过中间宽度，仍会正常下发到 `0.0`。

如果需要诊断 master 夹爪原始信号与过滤后信号：

```bash
python3 scripts/piper/test_master_gripper_signal.py \
  -cfg agent_infra/Piper_Env/Config/dual_piper_config.yaml
```

## 3. CAN 配置

Piper 专用 CAN 脚本已经放入：

```text
scripts/piper/can/find_all_can_port.sh
scripts/piper/can/can_activate.sh
scripts/piper/can/can_config.sh
```

先枚举 USB-CAN 物理接口：

```bash
bash scripts/piper/can/find_all_can_port.sh
```

当前推荐的双臂语义化接口名是短名，因为 Linux 网卡名最多 15 个字符：

```bash
sudo bash scripts/piper/can/can_activate.sh can_sl 1000000 1-1:1.0
sudo bash scripts/piper/can/can_activate.sh can_ml 1000000 1-3:1.0
sudo bash scripts/piper/can/can_activate.sh can_sr 1000000 1-9:1.0
sudo bash scripts/piper/can/can_activate.sh can_mr 1000000 1-10:1.0
```

对应 `dual_piper_config.yaml`：

```yaml
common:
  robot_model: "piper"
  firmware_version: "v183"  # default / v183 / v188

robots:
  left:
    master_can: "can_ml"
    follower_can: "can_sl"
  right:
    master_can: "can_mr"
    follower_can: "can_sr"
```

单臂默认使用左侧主从臂：

```yaml
common:
  robot_model: "piper"
  firmware_version: "v183"  # default / v183 / v188

robots:
  single:
    master_can: "can_ml"
    follower_can: "can_sl"
```

`pyAgxArm >= 1.0.0` 增加了固件版本驱动分支。Piper 系列固件建议按下面规则选择：

```text
default  <= S-V1.8-2
v183     S-V1.8-3 ~ S-V1.8-7
v188     >= S-V1.8-8
```

2026-04-26 实机只读探针读到右主臂固件为 `S-V1.8-7`，因此当前四臂调试推荐先使用 `firmware_version: "v183"`。如果更换机械臂或升级固件，再按上表调整。

## 4. Camera Configuration

相机通过 config 的 `cameras.nodes` 注入，不需要在训练代码里手动拼图像字段。

RealSense 示例：

```yaml
cameras:
  nodes:
    - serial_number: "346522076330"
      camera_type: "realsense"
      role: "base_camera"
    - serial_number: "419622073056"
      camera_type: "realsense"
      role: "wrist_camera"
  source_resolution: [640, 480]
  crop_resolution: [224, 224]
  fps: 30
```

Orbbec 示例：

```yaml
cameras:
  nodes:
    - serial_number: "ORBBEC_SERIAL_HERE"
      camera_type: "orbbec"
      camera_backend: "auto"
      camera_format: "MJPG"
      auto_exposure: true
      gain: 0
      enable_depth: true
      depth_required: false
      role: "base_camera"
      fallback_color: "black"
  source_resolution: [640, 480]
  crop_resolution: [224, 224]
  fps: 30
  enable_depth: true
  depth_required: false
```

仿真相机约定：

- `front_view` 是 ManiSkill 原始外部固定主视角，暂时不参与真实背景对齐前的位姿调整。
- `hand_camera` 是 ManiSkill 原始腕部相机，挂载在 Piper URDF 的 `hand_cam` 固定坐标系上；`hand_cam` 由 `link6 -> hand_camera_joint` 提供，仿真 `CameraConfig` 保持 `mount=hand_cam, pose=sapien.Pose()`，真实外参写入 URDF 固定关节。
- right hand 使用 `robot/piper/piper_assets/urdf/piper_description_with_camera_right.urdf`，当前来自 `outputs/camera_calibration/mr_sr_hand_eye_v2/result.json`：`xyz="0.013686479 0.069638234 0.038517091"`、`rpy="0.010509198 -1.200507973 -1.583916315"`。
- left hand 使用 `robot/piper/piper_assets/urdf/piper_description_with_camera_left.urdf`，当前来自 `outputs/camera_calibration/left_hand_eye/result_021_tsai.json`：`xyz="0.015040480 0.072650551 0.039562038"`、`rpy="-0.052107272 -1.229275696 -1.516695972"`。
- 单臂统一环境默认把 `front_view -> base_camera`、`hand_camera -> wrist_camera`，默认模型是 right hand；双臂统一环境分别用 `piper_arm_left` 和 `piper_arm_right`，再映射为 `base_camera/left_hand_camera/right_hand_camera`。

### 4.1 棋盘格手眼标定

如果需要让仿真 `hand_camera` 尽量对齐真实手眼相机，使用打印棋盘格做 eye-in-hand 标定。拍摄时棋盘纸固定不动，移动从臂到 12~20 个不同姿态，每个姿态保存一张手眼相机图片，并记录同一时刻从臂末端位姿 `base_T_link6`：

右主臂 `MR` 操控右从臂 `SR` 采集样本：

```bash
conda run --no-capture-output -n SL python scripts/piper/camera/capture_mr_sr_chessboard.py \
  --master-can can_mr \
  --follower-can can_sr \
  --camera-serial CH2592100GJ \
  --output-dir outputs/camera_calibration/mr_sr_hand_eye
```

运行后拖动 `MR`，脚本会实时下发关节目标给 `SR`，让 `SR` 手眼相机从不同距离和角度看到同一张固定棋盘纸；按 `s` 保存当前照片和 `SR` 实际法兰位姿，按 `q` 或 `Esc` 退出。脚本会写入 `outputs/camera_calibration/mr_sr_hand_eye/images/` 和 `outputs/camera_calibration/mr_sr_hand_eye/samples.csv`。

```csv
image,x,y,z,roll,pitch,yaw
sample_000.jpg,0.320,0.010,0.210,3.141,0.020,1.570
sample_001.jpg,0.300,-0.030,0.240,3.050,0.100,1.450
```

其中 `image` 可写相对 `samples.csv` 所在目录的图片路径；`x/y/z` 单位为米，`roll/pitch/yaw` 单位为弧度。棋盘格参数使用“内角点数量”，例如纸上是 10x7 个格子，则内角点是 `9x6`。

离线求解：

```bash
python3 scripts/piper/camera/calibrate_hand_eye_chessboard.py \
  --samples-csv outputs/camera_calibration/mr_sr_hand_eye/samples.csv \
  --image-dir outputs/camera_calibration/mr_sr_hand_eye \
  --pattern 9x6 \
  --square-size 0.025 \
  --preview-dir outputs/camera_calibration/mr_sr_hand_eye/previews \
  --output outputs/camera_calibration/mr_sr_hand_eye/result.json
```

脚本会输出三类坐标系：`flange_T_camera_optical` 是 OpenCV 标定结果；`flange_T_sapien_camera` 是 SDK/pose-mode 法兰下的 SAPIEN 相机；`urdf_link6_T_sapien_camera` 是补偿 URDF/SAPIEN `link6` 与 SDK 法兰固定差异后的结果。把 `urdf_link6_T_sapien_camera.urdf_origin` 写入对应侧的 `piper_description_with_camera_left/right.urdf` 与 `.xacro` 的 `hand_camera_joint`；仿真 `CameraConfig` 继续 `mount=hand_cam, pose=sapien.Pose()`，不要再在 `CameraConfig.pose` 里叠加一次外参。

枚举 RealSense：

```bash
python3 - <<'PY'
from agent_infra.Piper_Env.Camera.realsense_camera import get_connected_realsense_serials
print(get_connected_realsense_serials())
PY
```

枚举 Orbbec：

```bash
python3 - <<'PY'
from agent_infra.Piper_Env.Camera.orbbec_camera import (
    get_ignored_orbbec_usb_candidates,
    get_connected_orbbec_serials,
    get_orbbec_usb_devices,
)
print(get_connected_orbbec_serials())
print(get_orbbec_usb_devices())
print(get_ignored_orbbec_usb_candidates())
PY
```

标定左右手相机角色：

```bash
python3 scripts/piper/camera/calibrate_orbbec_roles.py
```

单独检查 Orbbec RGB/depth 帧：

```bash
python3 scripts/piper/camera/check_orbbec_depth.py --backend v4l2
python3 scripts/piper/camera/check_orbbec_depth.py --backend sdk
```

当前默认采集路径优先使用 V4L2 RGB fallback，不强制依赖 `pyorbbecsdk`。原 `Piper_test` 中的 Orbbec SDK 源码已迁入：

```text
third_party/pyorbbecsdk/
```

需要验证 SDK depth 时，再从该目录安装或构建 `pyorbbecsdk`。

如果 Orbbec 返回空列表，优先检查：

```bash
lsusb | grep -iE "orbbec|2bc5"
dmesg | tail -n 80
```

Orbbec 常见问题通常是 USB 线、供电、udev rules、SDK extension 路径或设备权限。`find_all_can_port.sh` 只用于 CAN，不用于相机枚举。

如果输出类似 `Invalid descriptor index`、`Failed to query USB device serial number` 或 `SDK 枚举子进程退出码: 1`，说明 `pyorbbecsdk` 在 SDK 层没有拿到设备序列号。此时先看 `get_orbbec_usb_devices()` 是否能看到明确的 Orbbec 相机设备。该函数会优先读取 `/sys/bus/usb/devices`，即使 `lsusb` 在当前环境不可用也可以做系统 USB 层诊断。

注意：当前 USB-CAN 适配器可能也显示为 `2bc5`，例如 `bytewerk candleLight USB to CAN adapter`。这类设备不是 Orbbec 相机，已经由 `get_ignored_orbbec_usb_candidates()` 单独列出并从相机候选里过滤。

## 5. Data Collection With Cameras

真实数据采集入口优先使用 `scripts/piper/single` 或 `scripts/piper/dual`：

```bash
bash scripts/piper/single/collect_h5.sh
bash scripts/piper/dual/collect_h5.sh
bash scripts/piper/single/collect_lerobot.sh
bash scripts/piper/dual/collect_lerobot.sh
```

所有 Piper 真实/仿真采集默认写入：

```text
datasets/piper/<task_name>/
```

Piper 仿真采集入口在 `scripts/piper/sim`，当前委托给可工作的 `scripts/sim/collect_data.py`：

```bash
bash scripts/piper/sim/collect_h5.sh --arm-side right --mode joint
bash scripts/piper/sim/collect_h5.sh --arm-side left --mode joint
bash scripts/piper/sim/collect_lerobot.sh --arm-side right --mode joint
```

`--arm-side right` 加载 `piper_arm_right` 和右腕相机 URDF，默认读取 `can_mr`；`--arm-side left` 加载 `piper_arm_left` 和左腕相机 URDF，默认读取 `can_ml`。旧的 `--arm-mode right_hand/left_hand` 会被兼容映射到同一逻辑。

`scripts/piper/sim/collect_lerobot.sh` 会先使用 RealToSimTeleop 仿真采集 ArmStudio H5，再把本次新增 H5 自动转换为 LeRobot。默认目录为：

```text
datasets/piper/piper_single_<control_mode>_sim_lerobot_task/h5_raw
datasets/piper/piper_single_<control_mode>_sim_lerobot_task/lerobot
```

`collect_unified.py` 目前仅作为统一配置 dry-run/协议检查工具，不作为日常采集入口：

```bash
PYTHONPATH=. python3 scripts/piper/collect_unified.py \
  -cfg agent_infra/Piper_Env/Config/unified_real_dual.yaml \
  --dry-config
```

日常采集不要去掉 `--dry-config` 调用这个实验入口，避免和真实/仿真正式脚本产生参数语义冲突。

CAN 激活脚本会按当前 USB 硬件地址恢复四路接口名称，并在激活完成后自动把 `can_sl`、`can_sr` 设置为从臂模式：

```bash
bash scripts/piper/can/can_quick_activate.sh
```

单臂和双臂采集脚本默认不会开启遥操，避免启动后机械臂立即跟随主臂。进入采集界面后按 `T` 开启或关闭遥操；如果采集时从臂不跟主臂动，优先确认终端里已经显示 `Teleoperation Toggle: ON`。单臂脚本顶部可直接修改：

```bash
MASTER_CAN="can_ml"
SLAVE_CAN="can_sl"
```

也可以在启动脚本时直接覆盖 CAN 接口：

```bash
bash scripts/piper/single/collect_h5.sh \
  --master can_ml \
  --slave can_sl
```

或者直接调用 recorder。若确实希望启动后立即开启遥操，可以显式加 `--teleop-on-start`：

```bash
python3 -m agent_infra.Piper_Env.Record.recorder \
  -m h5 \
  -cfg agent_infra/Piper_Env/Config/piper_config.yaml \
  --master can_ml \
  --slave can_sl \
  --teleop-on-start
```

双臂 `--master` 和 `--slave` 按 `left right` 顺序传入，例如 `--master can_ml can_mr --slave can_sl can_sr`。

后处理脚本都带默认路径，同时支持命令行覆盖。比如直接把单条 raw H5 转成 LeRobot：

```bash
bash scripts/piper/dual/convert_h5_to_lerobot.sh \
  -i datasets/piper/piper_dual_joint_real_h5_task/h5_raw/piper_joint_real_000.hdf5
```

LeRobot 转 merged H5 时，输入可以是 LeRobot 根目录，也可以是误传的 `meta/` 子目录；postprocess 会自动回退到父级数据集根目录：

```bash
bash scripts/piper/dual/convert_lerobot_to_h5.sh \
  -i datasets/piper/piper_dual_joint_real_lerobot_task/lerobot
```

新的 LeRobot 录制会在数据集根目录写入 `env_meta.json`。如果旧数据没有该文件，转换器会尝试从 `meta/info.json` 的 LeRobot features 推断 Piper single/dual 的 `env_meta`。

如果当前 shell 没有激活 `SL` 环境，可以显式指定 Python：

```bash
PYTHON_BIN=/home/lebinge/miniforge3/envs/SL/bin/python \
  bash scripts/piper/dual/convert_h5_to_lerobot.sh \
  -i datasets/piper/piper_dual_joint_real_h5_task/h5_raw/piper_joint_real_000.hdf5
```

只要 config 里配置了 `cameras.nodes`，并且通过 `SinglePiperEnv` 或 `DualPiperEnv` 创建环境，`PiperCameraWrapper` 会自动把图像写入：

```text
obs/rgb/<role>
```

H5 中会保存为：

```text
/obs/rgb/base_camera
/obs/rgb/wrist_camera
```

LeRobot 中会保存为：

```text
observation.images.<role>
```

Depth 默认开启。`PiperCameraWrapper` 会在 meta 和 obs 中追加：

```text
obs/depth/<role>   (1, H, W), uint16
```

当前配置中 Orbbec 使用 `camera_backend: "auto"`：如果 `enable_depth: true`，会优先尝试 SDK 以获取 RGB+depth；如果 SDK 不稳定或没有深度帧，会回退到 V4L2 RGB，并给 depth 写入零值 fallback。没有深度的相机或暂时不想录 depth 时，可以在全局或单个 camera node 中关闭：

```yaml
camera_type: "orbbec"
camera_backend: "auto"
enable_depth: false
depth_required: false
```

如果希望 depth 缺失时在日志中更明显，可以设置 `depth_required: true`；当前仍会使用零值 fallback 保持 obs 结构稳定。

当前 depth 优先建议写入 H5。LeRobot 写入路径暂时只保存 `obs/rgb`、`obs/state`、`action` 与 `under_control`，如果 config 打开了 depth，recorder 会提示并跳过 LeRobot depth 字段，避免生成不兼容的数据集。

注意：LeRobot 数据集写入成功不等于当前 Python 环境一定能解码视频。如果 `ds[0]` 报 `torchcodec/ffmpeg` 相关错误，说明是 LeRobot 视频解码依赖问题；可以先用 `ds.hf_dataset[0]` 检查 state/action/meta，或后续修复 `torchcodec` 与 FFmpeg 版本兼容。

采集任务名建议包含控制模式。当前脚本默认生成：

```text
piper_single_joint_real_h5_task
piper_dual_joint_real_h5_task
piper_single_joint_real_lerobot_task
piper_dual_joint_real_lerobot_task
piper_single_joint_sim_h5_task
piper_dual_joint_sim_h5_task
```

单条 H5/HDF5 轨迹文件名统一为：

```text
<robot>_<control_mode>_<backend>_<trajectory_id>.hdf5
```

例如：

```text
piper_joint_sim_000.hdf5
piper_joint_real_000.hdf5
piper_pose_real_001.hdf5
```

采集脚本默认不打开 OpenCV 预览窗口，避免 Qt 字体 warning 干扰终端日志。如果确实需要画面预览，可以在 recorder 命令后加：

```bash
--preview
```

## 6. Unified Smoke Test

统一 real/sim 配置入口已经新增，协议说明见：

```text
docs/02_Piper统一环境/统一协议.md
```

只检查统一配置与预期 `meta_keys`，不创建真实机械臂连接：

```bash
python3 scripts/piper/test_unified_env.py \
  -cfg agent_infra/Piper_Env/Config/unified_real_single.yaml \
  --dry-config
```

Arm_studio 单臂仿真走同一个入口：

```bash
python3 scripts/piper/test_unified_env.py \
  -cfg agent_infra/Piper_Env/Config/unified_sim_single.yaml \
  --steps 3
```

Arm_studio 双臂仿真协议入口：

```bash
python3 scripts/piper/test_unified_env.py \
  -cfg agent_infra/Piper_Env/Config/unified_sim_dual.yaml \
  --dry-config
```

当前双臂仿真先由两个单臂 ManiSkill 场景组合，对外输出真实双臂一致的
`left_*` / `right_*` state/action key；后续可替换为同一物理场景内的双臂任务。

统一测试入口：

```bash
python3 scripts/piper/test_piper_pipeline.py \
  --arm-mode single \
  --camera-mode without \
  --function teleop \
  --steps 20
```

常用组合：

```bash
python3 scripts/piper/test_piper_pipeline.py --arm-mode single --camera-mode with --function collect --start-cameras
python3 scripts/piper/test_piper_pipeline.py --arm-mode dual --camera-mode without --function all --steps 10
python3 scripts/piper/test_piper_pipeline.py --arm-mode dual --camera-mode with --function replay --replay-input path/to/traj.h5 --replay-format h5 --start-cameras
```

默认不会执行 `reset()`，避免真实机械臂突然回到初始位姿。如果确认安全，可以显式添加：

```bash
--reset-before-test
```

双臂遥操 obs/action 联调入口：

```bash
python3 scripts/piper/test_teleop_obs_action.py \
  --dual \
  --master can_ml can_mr \
  --slave can_sl can_sr \
  --no-reset \
  --print-every 30
```

这里 `--master` 和 `--slave` 的两个参数按 `left right` 顺序匹配，即 `can_ml/can_sl` 对应左臂，`can_mr/can_sr` 对应右臂。

## 7. Script Layout

正式入口：

```text
scripts/piper/single/
scripts/piper/dual/
scripts/piper/test_piper_pipeline.py
scripts/piper/can/find_all_can_port.sh
scripts/piper/can/can_activate.sh
scripts/piper/can/can_config.sh
scripts/piper/camera/calibrate_orbbec_roles.py
scripts/piper/camera/check_orbbec_depth.py
scripts/piper/camera/orbbec_sdk_device_probe.py
```

旧的顶层 `collect_h5.sh / collect_lerobot.sh / replay.sh` 已删除，避免与 `single/dual` 新入口混淆。
