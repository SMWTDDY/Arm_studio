# ArmStudio & Api_test 兼容性说明文档

本文档详细说明了 `arm_studio` 库与 `Piper_Sdk/Api_test` 库之间的兼容性设计、功能对齐以及冗余部分说明。

## 1. 数据格式对齐 (Naming & Format)
- **存储结构**：`HDF5Recorder` 已更新，不再使用嵌套的 `data/obs` 结构，而是直接对齐 `Api_test` 的 HDF5 键名：
  - `action`: (N, 7) 的动作序列。
  - `observation.state`: (N, 19) 的状态序列（包含 6关节+6速度+6位姿+1夹爪）。
  - `timestamp`: 相对起始时间的秒数。
- **命名规范**：文件命名由 `trajectory_*.hdf5` 改为 `piper_recording_YYYYMMDD_HHMMSS.hdf5`，与 `Api_test` 的 `piper_joint_recording_*` 风格一致。

## 2. 控制模式对齐 (Joint & Pose)
- **位姿计算**：引入了 `teleop/get_pose.py`（正向运动学 FK），主臂读取关节角度后会自动计算法兰位姿。
- **模式切换**：`RealToSimTeleop.get_action()` 支持 `mode="joint"` 或 `mode="pose"`，分别对应 Api_test 的关节控制和位姿控制。
- **仿真对齐**：当使用 `pose` 模式时，仿真环境自动切换为 `pd_ee_delta_pose` 控制器。

## 3. 夹爪控制逻辑 (Binary & Continuous)
- **二值化控制**：保留了 `BinaryGripperWrapper`，适用于强化学习等需要离散/二值夹爪动作的场景。
- **连续控制**：新增了 `--continuous` 参数，允许直接记录并控制物理宽度 (0.01m ~ 0.1m)，对齐 `Api_test` 中的 `move_gripper(width)`。
- **映射系数**：在处理连续控制时，参考了 `Api_test` 的 `0.5` 映射系数逻辑。

## 4. 冗余与相似功能说明 (Redundancy List)
以下文件或代码块在两个库中功能相似，为保证本库独立运行而保留：
- **`teleop/get_pose.py`**: 与 `Api_test/piper_infra/Env/get_pose.py` 逻辑完全一致。
- **`teleop/real_to_sim.py`**: 其内部的 `_update_loop` 逻辑与 `Api_test/single_piper_env.py` 中的 `_master_read_thread` 极其相似，但封装为独立的 Teleop 类。
- **`models/piper/agent.py`**: 这里的 `PiperArm` 定义是专为 ManiSkill 仿真的 URDF 加载设计的，而 `Api_test` 中更多是针对实体的 `PiperTeleopEnv` 包装。

## 5. 快速开始
```bash
# 关节空间记录（默认二值夹爪）
bash scripts/record.sh -m joint

# 位姿空间记录（连续夹爪控制）
bash scripts/record.sh -m pose --continuous
```
