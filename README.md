# Arm Studio

`Arm_studio` 是当前项目根目录，用于 Piper 机械臂的真实遥操、ManiSkill/SAPIEN 仿真控制、数据采集、回放、训练和推理。原 `Piper_test` 的代码能力已经融合到本仓库，后续运行与开发不再依赖外层 `Piper_test/` 目录。

当前推荐主链路：

```text
统一 config
  -> make_unified_piper_env()
  -> real/sim + single/dual 环境
  -> 统一 obs/action 协议
  -> H5/LeRobot 数据
  -> diffusion / VLA-style 策略训练
  -> 真实或仿真部署
```

训练侧当前主要使用 pose action：

```text
action = [x, y, z, roll, pitch, yaw, gripper]
```

训练时位姿旋转会转成 6D rotation，夹爪作为二分类标签：

```text
continuous_action = [x, y, z, rotation_6d]
gripper_label = 0/1
```

## 文档入口

项目文档已经统一收敛到：

```text
docs/
```

建议先读：

```text
docs/文档索引.md
docs/05_毕设规划/代码掌控地图.md
docs/05_毕设规划/毕设完成路线图.md
docs/01_项目总览/合并状态说明.md
docs/01_项目总览/目录约定.md
docs/02_Piper统一环境/统一协议.md
```

## 目录概览

```text
agent_infra/                    Piper 真实/仿真环境、相机、录制、回放
agent_factory/                  训练 agent、数据集、环境 factory、runner
scripts/                        唯一脚本根：Piper、仿真、数据、诊断、训练工具
robot/                          Piper 仿真 agent、动作 wrapper、IK
environments/                   ManiSkill 自定义环境
teleop/                         真实机械臂和键盘遥操
data/                           旧数据采集和数据查看工具，新数据不要写这里
third_party/                    可选外部 SDK 源码快照，如 pyorbbecsdk
models/DiffusionPolicy/         diffusion policy 模型与推理 policy
training/Diffusion_Training/    diffusion 训练脚本和训练配置
inference/                      本地/远程推理辅助
docs/                           中文文档中心
outputs/                        checkpoint、debug 图、导出视频
datasets/                       本地 HDF5/LeRobot 数据和训练缓存
```

## 快速检查

不连接硬件时，可以先检查统一配置是否能被正确读取：

```bash
PYTHONPATH=. python3 scripts/piper/test_unified_env.py \
  -cfg agent_infra/Piper_Env/Config/unified_real_dual.yaml \
  --dry-config

PYTHONPATH=. python3 scripts/piper/test_unified_env.py \
  -cfg agent_infra/Piper_Env/Config/unified_sim_dual.yaml \
  --dry-config
```

统一配置 dry-run 工具也支持检查：

```bash
PYTHONPATH=. python3 scripts/piper/collect_unified.py \
  -cfg agent_infra/Piper_Env/Config/unified_real_dual.yaml \
  --dry-config
```

Piper 真实/仿真采集默认写入 `datasets/piper`；训练和推理输出默认写入
`outputs/`。例如仿真采集：

```bash
bash scripts/piper/sim/collect_h5.sh --arm-mode single --no-preview
```

真实硬件只读诊断入口：

```bash
PYTHONPATH=. python3 scripts/piper/diagnose_real_setup.py --no-video-read
```

旧 Arm_studio HDF5 转统一 raw H5：

```bash
PYTHONPATH=. python3 scripts/data_tools/convert_legacy_armstudio_h5.py \
  training/Diffusion_Training/data/*.hdf5 \
  --output-dir datasets/converted_unified_h5
```

## 训练入口

主要训练配置：

```text
training/Diffusion_Training/training_config.py
```

训练命令示例：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python training/Diffusion_Training/train_diffusion_vision.py \
  --data-dir datasets/unified_pose_all_128 \
  --total-steps 100000 \
  --save-every 5000 \
  --num-workers 2
```

## 当前开发原则

新的开发只进入 `Arm_studio`。`Piper_test` 已由用户在别处备份，本仓库不再把它作为运行来源；后续新增功能应优先走统一入口、统一 config 和统一数据协议，避免真实、仿真、训练三套链路继续分裂。`3DGS` 暂时仍作为外层兄弟目录保留，等进入 RoboSimGS/3DGS 阶段再纳入根目录规划。

依赖安装按用途拆分：

```text
requirements-base.txt
requirements-hardware.txt
requirements-sim.txt
requirements-train.txt
```
