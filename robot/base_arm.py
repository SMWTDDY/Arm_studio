import gymnasium as gym
import numpy as np
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.sensors.camera import CameraConfig
import sapien.core as sapien

class BaseArm(BaseAgent):
    """
    ArmStudio 机械臂基类。
    提供了通用的传感器配置（侧视和末端相机）以及抽象的控制器框架。
    """
    
    # 子类必须覆盖以下属性
    uid = None
    urdf_path = None
    urdf_config = dict()

    @property
    def _sensor_configs(self):
        """
        定义通用的多视角相机。子类如果位姿不同可以覆盖。
        """
        return [
            # 侧视相机 - 默认位置
            CameraConfig(
                uid="side_view",
                pose=sapien.Pose(p=[0.5, 0.4, 0.4], q=[0, 0.707, 0, 0.707]),
                width=224, height=224, fov=1.0, near=0.01, far=10
            ),
            # 末端/手部相机 - 默认指向机器人前方
            CameraConfig(
                uid="wrist_view",
                pose=sapien.Pose(p=[0.4, 0, 0.3], q=[0.5, -0.5, 0.5, -0.5]),
                width=224, height=224, fov=1.0, near=0.01, far=5
            )
        ]

    @property
    def _controller_configs(self):
        """
        子类必须实现此方法以定义具体的 PD 控制器。
        """
        raise NotImplementedError("子类必须在 _controller_configs 中定义机械臂和夹爪的控制逻辑")

class BaseArmActionWrapper(gym.ActionWrapper):
    """
    通用的动作包装器基类。
    用于处理动作空间的映射（如 7 维到 8 维）以及夹爪的二值化转换。
    """
    def __init__(self, env, binary_gripper=True, threshold=0.0):
        super().__init__(env)
        self.binary_gripper = binary_gripper
        self.threshold = threshold
        self.control_mode = env.unwrapped.control_mode
