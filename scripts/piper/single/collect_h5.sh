#!/bin/bash
# Piper 单臂 H5 专家示教采集启动脚本

CTRL_MODE="joint"
TASK_NAME="piper_single_${CTRL_MODE}_real_h5_task"
CONFIG_PATH="agent_infra/Piper_Env/Config/piper_config.yaml"
MASTER_CAN="can_ml"
SLAVE_CAN="can_sl"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEBUG_EVERY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --master)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --master 需要 1 个 CAN 名称。"
        exit 2
      fi
      MASTER_CAN="$2"
      shift 2
      ;;
    --slave)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --slave 需要 1 个 CAN 名称。"
        exit 2
      fi
      SLAVE_CAN="$2"
      shift 2
      ;;
    --debug-every)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --debug-every 需要 1 个整数。"
        exit 2
      fi
      DEBUG_EVERY="$2"
      shift 2
      ;;
    *)
      echo "[Launcher] 未知参数: $1"
      echo "用法: bash $0 [--master can_ml] [--slave can_sl] [--debug-every 10]"
      exit 2
      ;;
  esac
done

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动单臂 H5 录制模式..."
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.recorder \
  -m h5 \
  -t "$TASK_NAME" \
  -ctrl "$CTRL_MODE" \
  -cfg "$CONFIG_PATH" \
  --backend real \
  --master "$MASTER_CAN" \
  --slave "$SLAVE_CAN" \
  --debug-every "$DEBUG_EVERY"
