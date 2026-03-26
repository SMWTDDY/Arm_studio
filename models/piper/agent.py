import gymnasium as gym
import numpy as np
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDEEPoseControllerConfig

class PiperActionWrapper(gym.ActionWrapper):
    """
    统一的动作包装器：负责将 7 维输入映射为仿真器接受的格式。
    """
    def __init__(self, env, binary_gripper=True, threshold=0.0):
        super().__init__(env)
        self.binary_gripper = binary_gripper
        self.threshold = threshold
        import gymnasium.spaces as spaces
        
        # 探测当前环境的控制模式
        self.control_mode = env.unwrapped.control_mode
        
        if binary_gripper:
            low = np.array([-np.inf]*6 + [-1.0], dtype=np.float32)
            high = np.array([np.inf]*6 + [1.0], dtype=np.float32)
        else:
            low = np.array([-np.inf]*6 + [0.0], dtype=np.float32)
            high = np.array([np.inf]*6 + [0.1], dtype=np.float32)
            
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        
        # 1. 提取 6 轴指令
        # 注意：如果 collect_data 强制发的是 joint，但仿真开了 pose，这里会错。
        # 所以我们假设 collect_data 已经发了正确模式的数据。
        arm_cmd = action[:6]
        
        # 2. 夹爪逻辑 (保持一致)
        if self.binary_gripper:
            gripper_val = 1.0 if action[6] > self.threshold else -1.0
        else:
            width = np.clip(action[6] * 0.5, 0.0, 0.035)
            gripper_val = (width / 0.035) * 2.0 - 1.0
            
        # 3. 构造 8 维指令
        real_action = np.append(arm_cmd, [gripper_val, gripper_val])
        return real_action.astype(np.float32)

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
        arm_pd = PDJointPosControllerConfig(
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            lower=None, upper=None,
            stiffness=1000, damping=100,
            normalize_action=False,
            drive_mode="force"
        )
        
        arm_ik = PDEEPoseControllerConfig(
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            ee_link="link6",
            pos_lower=-0.8, pos_upper=0.8,
            rot_lower=-3.14, rot_upper=3.14,
            urdf_path=self.urdf_path,
            stiffness=1000, damping=100,
            normalize_action=False,
            use_delta=False,
            drive_mode="force"
        )

        gripper_pd = PDJointPosControllerConfig(
            joint_names=["joint7", "joint8"],
            lower=0.0, upper=0.035, 
            stiffness=500, damping=50,
            normalize_action=True,
            drive_mode="force"
        )
        
        return dict(
            pd_joint_pos=dict(arm=arm_pd, gripper=gripper_pd),
            pd_ee_pose=dict(arm=arm_ik, gripper=gripper_pd)
        )
