#!/bin/bash

# 数据收集脚本 (简化版)
set -e

# 颜色输出函数
print_green() { printf "\033[0;32m%s\033[0m\n" "$1"; }
print_red() { printf "\033[0;31m%s\033[0m\n" "$1"; }

print_green "========================================"
print_green "   ArmStudio 数据收集工具"
print_green "========================================"

# 检查 Python 环境
PYTHON_EXE=$(which python 2>/dev/null || which python3 2>/dev/null)
if [ -z "$PYTHON_EXE" ]; then
    print_red "错误: 未找到 Python 环境。"
    exit 1
fi

# 启动 Python 脚本，直接透传所有命令行参数
# 例如: bash scripts/record.sh --teleop real --mode joint --binary_gripper
"$PYTHON_EXE" scripts/collect_data.py "$@"
