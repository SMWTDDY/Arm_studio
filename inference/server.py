import zmq
import pickle
import numpy as np
import time

def start_inference_server(port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"推理服务端已启动，监听端口 {port}...")

    # TODO: 在此处加载你的 PyTorch/HuggingFace 模型网络
    # model = load_model("my_policy.pth").to('cuda')

    while True:
        # 1. 接收来自本地仿真的 obs
        message = socket.recv()
        obs = pickle.loads(message)
        
        # 2. 执行模型推理前向传播 (此处用随机 action 占位)
        # action = model(obs)
        # 占位：生成一个 7 维随机动作
        action = np.random.uniform(-0.1, 0.1, size=(7,)).astype(np.float32)
        action[6] = 1.0 # 保持夹爪开
        
        # 3. 将 Action 传回本地
        socket.send(pickle.dumps(action))

if __name__ == "__main__":
    start_inference_server()