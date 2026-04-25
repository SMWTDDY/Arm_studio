import zmq
import pickle
import numpy as np
import time
import os
import argparse
import signal
import sys

# 确保可以导入项目根目录下的 models 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.DiffusionPolicy import DiffusionVisionPolicy

# 默认模型路径
MODEL_PATH = "outputs/checkpoints/vision/final_vision_policy.pth"

def start_inference_server(port=5555, model_path=MODEL_PATH, device='cuda'):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    try:
        socket.bind(f"tcp://*:{port}")
    except zmq.ZMQError as e:
        print(f"绑定端口 {port} 失败: {e}")
        return

    print(f"\n" + "="*50)
    print(f"ArmStudio 推理服务器已启动")
    print(f"监听地址: tcp://*:{port}")
    print(f"加载模型: {model_path}")
    print("="*50 + "\n")

    try:
        # 使用重构后的 DiffusionPolicy 类
        policy = DiffusionVisionPolicy(model_path, device=device)
        print("模型加载并初始化完成")
    except Exception as e:
        print(f"初始化策略模型失败: {e}")
        return

    # 信号处理，优雅退出
    def signal_handler(sig, frame):
        print("\n正在关闭服务器...")
        socket.close()
        context.term()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("等待客户端请求...")
    while True:
        try:
            # 接收观测数据
            message = socket.recv()
            
            # 使用 pickle 反序列化
            obs = pickle.loads(message)
            
            # 推理
            start_time = time.time()
            action = policy.act(obs)
            inference_time = (time.time() - start_time) * 1000
            
            # 发送动作数据
            socket.send(pickle.dumps(action))
            
            # print(f"推理耗时: {inference_time:.2f}ms")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理请求错误: {e}")
            try:
                socket.send(pickle.dumps({"error": str(e)}))
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArmStudio 推理服务器")
    parser.add_argument("--port", type=int, default=5555, help="监听端口")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="使用设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    start_inference_server(port=args.port, model_path=args.model, device=args.device)
