#!/bin/bash
# Piper 仿真 H5 数据采集启动脚本
#
# This intentionally delegates to scripts/sim/collect_data.py, which is the
# working ArmStudio/ManiSkill simulation collector.

CTRL_MODE="${CTRL_MODE:-joint}"
EXTRA_ARGS=()
SAVE_DIR="${SAVE_DIR:-datasets/piper}"
CAN_NAME="${CAN_NAME:-}"
ARM_SIDE="${ARM_SIDE:-right}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm-mode)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --arm-mode 需要 left 或 right。"
        exit 2
      fi
      case "$2" in
        left|left_hand)
          ARM_SIDE="left"
          ;;
        right|right_hand|single)
          ARM_SIDE="right"
          ;;
        *)
          echo "[Launcher] 仿真采集当前一次只启动一只 Piper；--arm-mode 请使用 left 或 right。"
          exit 2
          ;;
      esac
      shift 2
      ;;
    --arm-side)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --arm-side 需要 left 或 right。"
        exit 2
      fi
      case "$2" in
        left|right)
          ARM_SIDE="$2"
          ;;
        *)
          echo "[Launcher] --arm-side 只支持 left 或 right。"
          exit 2
          ;;
      esac
      shift 2
      ;;
    --mode|--control|-ctrl)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] $1 需要控制模式：joint 或 pose。"
        exit 2
      fi
      CTRL_MODE="$2"
      shift 2
      ;;
    --save-dir|--root-dir)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] $1 需要目录路径。"
        exit 2
      fi
      SAVE_DIR="$2"
      shift 2
      ;;
    --can-name)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --can-name 需要 CAN 名称。"
        exit 2
      fi
      CAN_NAME="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CAN_NAME" ]]; then
  if [[ "$ARM_SIDE" == "left" ]]; then
    CAN_NAME="can_ml"
  else
    CAN_NAME="can_mr"
  fi
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动 Piper 仿真 H5 采集: side=${ARM_SIDE}, mode=${CTRL_MODE}, can=${CAN_NAME}, save_dir=${SAVE_DIR}"
"$PYTHON_BIN" scripts/sim/collect_data.py \
  --arm-side "$ARM_SIDE" \
  --mode "$CTRL_MODE" \
  --save-dir "$SAVE_DIR" \
  --can-name "$CAN_NAME" \
  "${EXTRA_ARGS[@]}"
