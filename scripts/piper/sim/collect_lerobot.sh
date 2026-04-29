#!/bin/bash
# Piper 仿真 LeRobot 数据采集启动脚本
#
# 工作链路：
#   1. 使用 scripts/sim/collect_data.py 进行 RealToSimTeleop 仿真 H5 采集。
#   2. 将本次新增的 ArmStudio H5 自动转换成一个 LeRobot dataset。

set -e

CTRL_MODE="${CTRL_MODE:-joint}"
ROOT_DIR="${ROOT_DIR:-datasets/piper}"
TASK_NAME="${TASK_NAME:-}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-}"
VCODEC="${VCODEC:-h264}"
CAN_NAME="${CAN_NAME:-}"
ARM_SIDE="${ARM_SIDE:-right}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRA_ARGS=()

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
    --root-dir)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --root-dir 需要目录路径。"
        exit 2
      fi
      ROOT_DIR="$2"
      shift 2
      ;;
    --save-dir)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --save-dir 需要目录路径。"
        exit 2
      fi
      H5_DIR="$2"
      shift 2
      ;;
    -t|--task)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] $1 需要任务名。"
        exit 2
      fi
      TASK_NAME="$2"
      shift 2
      ;;
    --task-description)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --task-description 需要描述文本。"
        exit 2
      fi
      TASK_DESCRIPTION="$2"
      shift 2
      ;;
    --vcodec)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[Launcher] --vcodec 需要编码器名称。"
        exit 2
      fi
      VCODEC="$2"
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

if [[ -z "$TASK_NAME" ]]; then
  TASK_NAME="piper_single_${CTRL_MODE}_sim_lerobot_task"
fi
if [[ -z "$TASK_DESCRIPTION" ]]; then
  TASK_DESCRIPTION="single piper ${CTRL_MODE} sim teleoperation"
fi
if [[ -z "${H5_DIR:-}" ]]; then
  H5_DIR="${ROOT_DIR}/${TASK_NAME}/h5_raw"
fi
if [[ -z "$CAN_NAME" ]]; then
  if [[ "$ARM_SIDE" == "left" ]]; then
    CAN_NAME="can_ml"
  else
    CAN_NAME="can_mr"
  fi
fi
LEROBOT_DIR="${ROOT_DIR}/${TASK_NAME}/lerobot"

mkdir -p "$H5_DIR" "$LEROBOT_DIR"
STAMP_FILE="/tmp/armstudio_collect_lerobot_${$}.stamp"
touch "$STAMP_FILE"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "[Launcher] 启动 Piper 仿真 LeRobot 采集链路: side=${ARM_SIDE}, mode=${CTRL_MODE}, can=${CAN_NAME}"
echo "[Launcher] H5 临时目录: ${H5_DIR}"
echo "[Launcher] LeRobot 输出目录: ${LEROBOT_DIR}"
echo "[Launcher] 采集阶段按键：r 开始，e 保存，q 丢弃，escape 退出。退出后会自动转换本次新增 H5。"

"$PYTHON_BIN" scripts/sim/collect_data.py \
  --arm-side "$ARM_SIDE" \
  --mode "$CTRL_MODE" \
  --save-dir "$H5_DIR" \
  --can-name "$CAN_NAME" \
  "${EXTRA_ARGS[@]}"

mapfile -t NEW_H5_FILES < <(find "$H5_DIR" -maxdepth 1 -type f \( -name "*.h5" -o -name "*.hdf5" \) -newer "$STAMP_FILE" | sort)

if [[ ${#NEW_H5_FILES[@]} -eq 0 ]]; then
  echo "[Launcher] 本次没有新增 H5 轨迹，跳过 LeRobot 转换。"
  exit 0
fi

echo "[Launcher] 本次新增 ${#NEW_H5_FILES[@]} 条 H5 轨迹，开始转换为 LeRobot..."
CONVERT_INPUT_DIR="/tmp/armstudio_collect_lerobot_${$}_h5"
mkdir -p "$CONVERT_INPUT_DIR"
for h5_file in "${NEW_H5_FILES[@]}"; do
  ln -sf "$(readlink -f "$h5_file")" "${CONVERT_INPUT_DIR}/$(basename "$h5_file")"
done
"$PYTHON_BIN" -m agent_infra.Piper_Env.Record.postprocess h5_to_lerobot \
  -i "$CONVERT_INPUT_DIR" \
  -o "$LEROBOT_DIR" \
  --task-description "$TASK_DESCRIPTION" \
  --vcodec "$VCODEC"
