import gymnasium as gym
import numpy as np
import threading
import time
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig
import mani_skill.envs  

class BinaryGripperWrapper(gym.ActionWrapper):
    def __init__(self, env, threshold=0.0):
        super().__init__(env)
        self.threshold = threshold
        import gymnasium.spaces as spaces
        low = np.array([-np.inf]*6 + [-1.0])
        high = np.array([np.inf]*6 + [1.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        gripper_val = 1.0 if action[6] > self.threshold else -1.0
        real_action = np.append(action[:6], [gripper_val, gripper_val])
        return real_action.astype(np.float32)

@register_agent()
class PiperArmJoint(BaseAgent):
    uid = "piper_arm_joint"
    urdf_path = "piper_assets/urdf/piper_description.urdf" 
    urdf_config = dict()

    @property
    def _sensor_configs(self):
        return []

    @property
    def _controller_configs(self):
        arm_pd = PDJointPosControllerConfig(
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            lower=None, upper=None,
            stiffness=1000, damping=100,
            normalize_action=False,
            drive_mode="force"
        )
        gripper_pd = PDJointPosControllerConfig(
            joint_names=["joint7", "joint8"],
            lower=0.0, upper=0.035, 
            stiffness=500, damping=50,
            normalize_action=True,
            drive_mode="force"
        )
        return dict(pd_joint_pos=dict(arm=arm_pd, gripper=gripper_pd))

# --- 真机通信后台线程 ---
class RealArmInterface:
    def __init__(self):
        self.latest_action = np.zeros(7, dtype=np.float32)
        self.latest_action[6] = -1.0 # 默认夹爪关闭
        self.running = True
        
        # TODO: 在此处初始化你的 Piper SDK
        # self.robot = PiperSDK.connect()
        
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            try:
                # TODO: 替换为实际 SDK 读取关节角度的代码
                # real_angles = self.robot.get_joint_angles() # 返回长度为 6 的 list
                # gripper_status = self.robot.get_gripper() # 1.0 或 -1.0
                
                # 下面是模拟数据，测试时机械臂会待在原地
                real_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                gripper_status = -1.0 
                
                self.latest_action[:6] = real_angles
                self.latest_action[6] = gripper_status
            except Exception as e:
                pass
            time.sleep(0.01) # 维持约 100Hz 读取频率

    def get_action(self):
        return self.latest_action.copy()

    def close(self):
        self.running = False
        self.thread.join()

def main():
    env = gym.make(
        "Empty-", 
        obs_mode="state",
        control_mode="pd_joint_pos", 
        robot_uids="piper_arm_joint",
        num_envs=1,
        render_mode="human"
    )
    env = BinaryGripperWrapper(env)
    obs, _ = env.reset()
    
    print("\n启动真机遥操作监听...")
    real_arm = RealArmInterface()
    
    try:
        while True:
            # 从后台线程获取最新的真机物理位姿，直接灌入仿真
            action = real_arm.get_action()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    except KeyboardInterrupt:
        print("\n退出程序")
    finally:
        real_arm.close()

if __name__ =="__main__":
    main()