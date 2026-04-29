#!/bin/bash
# Piper 双臂 LeRobot -> merged H5 转换脚本

CTRL_MODE="joint"
TASK_NAME="piper_dual_${CTRL_MODE}_real_lerobot_task"
INPUT_DIR="datasets/piper/${TASK_NAME}/lerobot"
OUTPUT_H5="datasets/piper/${TASK_NAME}/merged/piper_${CTRL_MODE}_real_from_lerobot.hdf5"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 双臂 LeRobot 转 merged H5..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.postprocess lerobot_to_h5 \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_H5" \
  "$@"
