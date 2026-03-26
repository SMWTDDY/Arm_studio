#!/bin/bash

# 推理脚本
# 支持两种模式：远程推理 和 本地推理

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --mode MODE     推理模式 (默认: local)"
    echo "                      local   - 本地推理模式"
    echo "                      remote  - 远程推理模式（连接到云端服务器）"
    echo "  -a, --addr ADDR      远程服务器地址 (仅 remote 模式需要)"
    echo "  -p, --port PORT      远程服务器端口 (默认: 5000)"
    echo "  -s, --server         启动推理服务器模式"
    echo "  -h, --help           显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                          # 默认本地推理模式"
    echo "  $0 -m local                # 显式指定本地推理"
    echo "  $0 --mode remote             # 使用远程推理模式"
    echo "  $0 -m remote -a 192.168.1.100  # 指定服务器地址"
    echo "  $0 -s                        # 启动推理服务器"
}

# 默认参数
MODE="local"
SERVER_ADDR="localhost"
SERVER_PORT="5000"
START_SERVER=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -a|--addr)
            SERVER_ADDR="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -s|--server)
            START_SERVER=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 '$1'${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# 启动服务器模式
if [[ "$START_SERVER" == "true" ]]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   ArmStudio 推理服务器${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}启动推理服务器...${NC}"
    echo -e "监听端口: ${BLUE}$SERVER_PORT${NC}"
    echo ""
    
    python inference/server.py --port $SERVER_PORT
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}推理服务器已停止${NC}"
    else
        echo ""
        echo -e "${RED}推理服务器异常退出${NC}"
        exit 1
    fi
    
    exit 0
fi

# 验证推理模式
if [[ "$MODE" != "local" && "$MODE" != "remote" ]]; then
    echo -e "${RED}错误: 无效的模式 '$MODE'${NC}"
    echo "有效模式: local, remote"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   ArmStudio 推理工具${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}当前模式: $MODE${NC}"

if [[ "$MODE" == "remote" ]]; then
    echo -e "${BLUE}服务器地址: $SERVER_ADDR:$SERVER_PORT${NC}"
fi
echo ""

# 设置环境变量
export ARM_STUDIO_MODE=$MODE
export ARM_STUDIO_SERVER_ADDR=$SERVER_ADDR
export ARM_STUDIO_SERVER_PORT=$SERVER_PORT

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi

# 运行推理脚本
echo "启动推理..."
echo ""

if [[ "$MODE" == "remote" ]]; then
    python scripts/run_inference.py --mode remote --addr $SERVER_ADDR --port $SERVER_PORT
else
    python scripts/run_inference.py --mode local
fi

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}推理完成！${NC}"
else
    echo ""
    echo -e "${RED}推理失败！${NC}"
    exit 1
fi
