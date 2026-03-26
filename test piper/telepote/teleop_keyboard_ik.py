import gymnasium as gym
import numpy as np
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDEEPoseControllerConfig
import mani_skill.envs  

class BinaryGripperWrapper(gym.ActionWrapper):
    def __init__(self, env, threshold=0.0):
        super().__init__(env)
        self.threshold = threshold
        import gymnasium.spaces as spaces
        # 声明 7 维动作空间：6维末端增量(dx,dy,dz,dr,dp,dyaw) + 1维夹爪意图
        low = np.array([-np.inf]*6 + [-1.0])
        high = np.array([np.inf]*6 + [1.0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        gripper_val = 1.0 if action[6] > self.threshold else -1.0
        real_action = np.append(action[:6], [gripper_val, gripper_val])
        return real_action.astype(np.float32)

@register_agent()
class PiperArmIK(BaseAgent):
    uid = "piper_arm_ik"
    urdf_path = "test piper/piper_assets/urdf/piper_description.urdf"
    urdf_config = dict()

    @property
    def _sensor_configs(self):
        return []

    @property
    def _controller_configs(self):
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        # 使用逆运动学控制器
        # 使用逆运动学控制器
        
    @property
    def _controller_configs(self):
        arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        
        # 1. 手臂的逆运动学 (IK) 控制器配置
        arm_ik = PDEEPoseControllerConfig(
            joint_names=arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            ee_link="link6", # 确保这是你的末端 link
            urdf_path=self.urdf_path,
            stiffness=1000, damping=100,
            normalize_action=False,
            use_delta=True, # 增量控制模式
            drive_mode="force"
        )
        

        gripper_joint_names = ["joint7", "joint8"]
        gripper_pd = PDJointPosControllerConfig(
            joint_names=gripper_joint_names,
            lower=0.0, upper=0.035, 
            stiffness=500, damping=50,
            normalize_action=True,
            drive_mode="force"
        )
        
        # 3. 组合暴露出名为 pd_ee_delta_pose 的复合控制器
        return dict(pd_ee_delta_pose=dict(arm=arm_ik, gripper=gripper_pd))
        

def main():
    env = gym.make(
        "Empty", 
        obs_mode="state",
        control_mode="pd_ee_delta_pose", 
        robot_uids="piper_arm_ik",
        num_envs=1,
        render_mode="human"
    )
    env = BinaryGripperWrapper(env)
    obs, _ = env.reset()
    
    print("\n=== 完整 6D 末端控制指南 ===")
    print("【平移】 I/K:前后 | J/L:左右 | U/O:上下")
    print("【旋转】 Q/E:翻滚(Roll) | W/S:俯仰(Pitch) | A/D:偏航(Yaw)")
    print("【夹爪】 F: 切换开/合")
    print("============================\n")
    
    gripper_state = -1.0
    action = np.zeros(7, dtype=np.float32)
    
    # 平移和旋转的步长需分开设置
    pos_step = 0.005  # 每次平移 5mm
    rot_step = 0.02   # 每次旋转约 1.1 度 (不可设太大，否则 IK 极易崩溃)
    
    while True:
        env.render()
        viewer = env.unwrapped.viewer
        action[:6] = 0.0 # 每一帧重置增量，松开按键即停止运动
        
        if viewer is not None:
            window = viewer.window
            
            # --- 1. 末端平移控制 (X, Y, Z) ---
            if window.key_down('i'): action[0] = pos_step
            if window.key_down('k'): action[0] = -pos_step
            if window.key_down('j'): action[1] = pos_step
            if window.key_down('l'): action[1] = -pos_step
            if window.key_down('u'): action[2] = pos_step
            if window.key_down('o'): action[2] = -pos_step
            
            # --- 2. 末端旋转控制 (Roll, Pitch, Yaw) ---
            if window.key_down('q'): action[3] = rot_step
            if window.key_down('e'): action[3] = -rot_step
            if window.key_down('w'): action[4] = rot_step
            if window.key_down('s'): action[4] = -rot_step
            if window.key_down('a'): action[5] = rot_step
            if window.key_down('d'): action[5] = -rot_step
            
            # --- 3. 夹爪控制 ---
            if window.key_press('f'):
                gripper_state = -gripper_state
                
        action[6] = gripper_state
        obs, reward, terminated, truncated, info = env.step(action)

if __name__ =="__main__":
    main()