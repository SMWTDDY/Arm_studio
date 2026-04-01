import zmq
import pickle
import numpy as np
import time
import torch
import os
import argparse
import signal
import sys

# 默认模型路径
MODEL_PATH = "policy.pi0"

def load_model(model_path, device='cuda'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 尝试加载到指定设备
    try:
        state_dict = torch.load(model_path, map_location=device)
        return state_dict
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 降级尝试加载到 CPU
        print("尝试降级加载到 CPU...")
        return torch.load(model_path, map_location='cpu')

class PI0Policy:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"使用设备: {self.device}")
        self.state_dict = load_model(model_path, self.device)
        self._build_model()
        
    def _build_model(self):
        # 默认维度，根据实际模型 state_dict 调整
        obs_dim = self.state_dict.get('obs_dim', 14)
        action_dim = self.state_dict.get('action_dim', 7)
        
        print(f"构建模型: obs_dim={obs_dim}, action_dim={action_dim}")
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim)
        )
        
        if 'model' in self.state_dict:
            self.net.load_state_dict(self.state_dict['model'])
        else:
            # 兼容直接保存 state_dict 的情况
            self.net.load_state_dict(self.state_dict)
            
        self.net.to(self.device)
        self.net.eval()
        
    @torch.no_grad()
    def act(self, obs):
        obs = np.array(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        action = self.net(obs_tensor).cpu().numpy()[0]
        return action.astype(np.float32)

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
        policy = PI0Policy(model_path, device)
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
            
            # 使用 pickle 反序列化 (生产环境建议使用 msgpack 或 protobuf)
            obs = pickle.loads(message)
            
            # 推理
            start_time = time.time()
            action = policy.act(obs)
            inference_time = (time.time() - start_time) * 1000
            
            # 发送动作数据
            socket.send(pickle.dumps(action))
            
            # 打印调试信息（可选，根据需要调整频率）
            # print(f"推理耗时: {inference_time:.2f}ms")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"处理请求错误: {e}")
            # 发送错误响应，避免客户端死锁
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
