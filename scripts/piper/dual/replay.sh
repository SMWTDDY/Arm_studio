#!/bin/bash
# Piper 双臂轨迹回放启动脚本

CONFIG_PATH="agent_infra/Piper_Env/Config/dual_piper_config.yaml"
traj_PATH="datasets/piper/piper_dual_joint_real_h5_task/h5_raw/piper_joint_real_000.hdf5"
CTRL_MODE="joint"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动双臂轨迹回放..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.replay -cfg "$CONFIG_PATH" -i "$traj_PATH" -ctrl "$CTRL_MODE" "$@"
