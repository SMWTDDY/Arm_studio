import zmq
import pickle
import numpy as np
import time

class InferenceClient:
    """
    ArmStudio 推理客户端，负责与远程推理服务器通信。
    """
    def __init__(self, host="localhost", port=5555, timeout=1000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # 设置超时 (单位: 毫秒)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.setsockopt(zmq.LINGER, 0) # 立即关闭不等待
        
        self.address = f"tcp://{host}:{port}"
        print(f"连接到推理服务器: {self.address}")
        self.socket.connect(self.address)
        
    def act(self, obs):
        """
        发送观测数据并接收推理结果。
        """
        try:
            # 序列化并发送
            self.socket.send(pickle.dumps(obs))
            
            # 等待响应
            message = self.socket.recv()
            
            # 反序列化
            action = pickle.loads(message)
            
            if isinstance(action, dict) and "error" in action:
                print(f"服务器端推理错误: {action['error']}")
                return None
                
            return action
            
        except zmq.Again:
            print(f"请求超时: {self.address}")
            # 超时后重置 socket 避免状态混乱
            self._reset_socket()
            return None
        except Exception as e:
            print(f"通信错误: {e}")
            return None
            
    def _reset_socket(self):
        """重置连接以恢复状态机"""
        self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)

    def close(self):
        self.socket.close()
        self.context.term()

if __name__ == "__main__":
    # 简单的测试逻辑
    client = InferenceClient(host="localhost", port=5555)
    # 关键修复：发送 16 维观测数据以匹配服务器
    dummy_obs = np.zeros(16, dtype=np.float32)
    print("发送测试请求...")
    action = client.act(dummy_obs)
    print(f"收到动作: {action}")
    client.close()
