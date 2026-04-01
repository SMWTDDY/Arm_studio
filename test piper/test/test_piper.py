import gymnasium as gym
import numpy as np
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig
import mani_skill.envs  

class BinaryGripperWrapper(gym.ActionWrapper):
    def __init__(self, env, threshold=0.0):
        super().__init__(env)
        self.threshold = threshold
        
        # 强制向外声明动作空间为 7 维 (6手臂 + 1夹爪意图)
        import gymnasium.spaces as spaces
        low = np.array([-np.inf]*6 + [-1.0])
        high = np.array([np.inf]*6 + [1.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        # 拦截 7 维输入，二值化第 7 位
        gripper_val = 1.0 if action[6] > self.threshold else -1.0
        
        # 扩维成 8 维。因为 URDF 中 joint8 已旋转 180 度，输入相同的正值即可实现相向闭合
        real_action = np.append(action[:6], [gripper_val, gripper_val])
        return real_action

@register_agent()
class PiperArm(BaseAgent):
    uid = "piper_arm"
    urdf_path = "models/piper/piper_assets/urdf/piper_description.urdf" 
    urdf_config = dict()

    @property
    def _sensor_configs(self):
        return []

    @property
    def _controller_configs(self):
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        arm_pd = PDJointPosControllerConfig(
            joint_names=arm_joint_names,
            lower=None, upper=None,
            stiffness=100, damping=10,
            normalize_action=False,
            drive_mode="force"
        )
        
        gripper_joint_names = ["joint7", "joint8"]
        # 恢复统一的正向标量限位
        gripper_pd = PDJointPosControllerConfig(
            joint_names=gripper_joint_names,
            lower=0.0, upper=0.035, 
            stiffness=500, damping=50,
            normalize_action=True,
            drive_mode="force"
        )
        
        return dict(pd_joint_pos=dict(arm=arm_pd, gripper=gripper_pd))

def main():
    env = gym.make(
        "Empty", 
        obs_mode="state",
        control_mode="pd_joint_pos", 
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human"
    )
    
    env = BinaryGripperWrapper(env)
    
    obs, _ = env.reset()
    print("环境加载成功，按 ESC 可退出渲染窗口。")
    
    import time
    import math
    step_count = 0
    
    while True:
        target_angle = math.sin(step_count * 0.05) * 0.35
        # 外部测试输入保持严格的 7 维
        a = np.ones(7, dtype=np.float32)
        action = a * target_angle
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step_count += 1
        time.sleep(0.01)

if __name__ =="__main__":
    main()