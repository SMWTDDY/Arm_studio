#!/bin/bash
# Piper 单臂 merged H5 -> LeRobot 转换脚本

CTRL_MODE="joint"
TASK_NAME="piper_single_${CTRL_MODE}_real_h5_task"
INPUT_H5="datasets/piper/${TASK_NAME}/merged/piper_${CTRL_MODE}_real_merged.hdf5"
OUTPUT_DIR="datasets/piper/${TASK_NAME}/lerobot_converted"
TASK_DESCRIPTION="single piper ${CTRL_MODE} converted from h5"
VCODEC="h264"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 单臂 merged H5 转 LeRobot..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.postprocess h5_to_lerobot \
  -i "$INPUT_H5" \
  -o "$OUTPUT_DIR" \
  --task-description "$TASK_DESCRIPTION" \
  --vcodec "$VCODEC" \
  "$@"
