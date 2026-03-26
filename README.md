# ArmStudio - Piper 机械臂仿真遥操与数据采集系统

ArmStudio 是一个专为 Piper 机械臂设计的轻量级仿真与数据采集框架。它实现了真实机械臂与 ManiSkill 仿真环境的深度对齐，支持高精度的轨迹录制，生成的 HDF5 数据完全兼容 `Api_test` 及 `LeRobot` 规范。

## ✨ 核心特性

- **双空间同步**：支持关节空间（Joint）与笛卡尔位姿空间（Pose）的实时同步遥操。
- **高精度对齐**：正向运动学（FK）解算直接基于 URDF 原生参数构建，确保仿真与现实骨架一致。
- **自动化录制**：支持单次运行录制多条轨迹，文件名自动按 `000-999` 序列递增。
- **灵活控制**：集成 `PiperActionWrapper` 自动处理 7D 意图到 8D 物理控制的映射。
- **夹爪模式**：支持连续宽度控制（对齐实机）与二值化控制（对齐强化学习）。

## 🚀 快速开始

### 1. 环境准备
确保已激活 Conda 环境并安装必要依赖：
```bash
conda activate SL
pip install -r requirements.txt
```

### 2. 启动采集
项目使用 `scripts/record.sh` 作为统一入口，支持透传所有 Python 参数：

```bash
# 默认：关节模式 + 连续夹爪
bash scripts/record.sh --mode joint

# 进阶：位姿模式 + 二值化夹爪
bash scripts/record.sh --mode pose --binary_gripper
```

### 3. 录制快捷键
程序启动并进入仿真画面后，使用以下键位控制流程：
- **`[R]`**：**开始录制**新轨迹（会在控制台提示当前文件名）。
- **`[S]`**：**停止并保存**当前轨迹（自动保存至 `datasets/`）。
- **`[ESC]`**：安全保存并退出程序。

## 📂 目录结构

- **`scripts/`**: 包含 `record.sh` 入口及 `collect_data.py` 核心采集逻辑。
- **`teleop/`**: 
  - `real_to_sim.py`: 基于 `pyAgxArm` 的真机数据流解析。
  - `get_pose.py`: 基于 URDF 骨架的高精度正向运动学解算。
- **`data/`**: `recorder.py` 负责 HDF5 序列化，支持自动命名。
- **`models/`**: 定义 `PiperArm` 的仿真 Agent、控制器配置及 URDF 资产。
- **`test/`**: 包含位姿对比、CAN 通信等调试工具。

## 🛠 兼容性与规范

本项目严格遵循以下对齐标准：
1. **命名规范**：`piper_{mode}_recording_xxx.hdf5`。
2. **数据结构**：
   - `observation.state`: 19 维向量（6关节 + 6速度 + 6位姿 + 1夹爪）。
   - `action`: 7 维控制向量。
   - `timestamp`: 步进时间戳。
3. **硬件库**：已全面转向 `pyAgxArm`，弃用旧版 `piper_sdk`。

详细设计说明请参阅：[COMPATIBILITY.md](COMPATIBILITY.md)
