import gymnasium as gym
import numpy as np
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDEEPoseControllerConfig
from robot.base_arm import BaseArm, BaseArmActionWrapper
from mani_skill.sensors.camera import CameraConfig
import sapien.core as sapien

class PiperActionWrapper(BaseArmActionWrapper):
    def __init__(self, env, binary_gripper=True, threshold=0.0):
        super().__init__(env, binary_gripper, threshold)
        import gymnasium.spaces as spaces
        if binary_gripper:
            low = np.array([-np.inf]*6 + [-1.0], dtype=np.float32)
            high = np.array([np.inf]*6 + [1.0], dtype=np.float32)
        else:
            low = np.array([-np.inf]*6 + [0.0], dtype=np.float32)
            high = np.array([np.inf]*6 + [0.035], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        action = np.array(action, dtype=np.float32)
        arm_cmd = action[:6]
        if self.binary_gripper:
            gripper_val = 0.035 if action[6] > self.threshold else 0.0
        else:
            gripper_val = np.clip(action[6], 0.0, 0.035)
        return np.append(arm_cmd, [gripper_val, gripper_val]).astype(np.float32)

    def move_to_p(self, now_pose, now_joint_angle, target_pose):
        '''
        根据当前末端执行器的位姿和本体感知，输出从当前状态到目标位姿的动作指令
        
        Args:
            now_pose: 当前末端执行器的位姿 (sapien.Pose对象)
            now_joint_angle: 当前机械臂关节角度 (numpy array, 长度6)
            target_pose: 目标位姿 [x, y, z, rx, ry, rz, gripper_action] (numpy array)
        
        Returns:
            next_action: 下一步动作指令 (numpy array, 长度7)
        '''
        # 提取目标位置和旋转、夹爪动作
        target_pos = target_pose[:3].flatten()  # [x, y, z]
        target_rot = target_pose[3:6].flatten()  # [rx, ry, rz]
        gripper_action = float(target_pose[6])
        
        # 获取当前末端执行器位置
        current_pos = now_pose.p
        if hasattr(current_pos, 'detach'):
            current_pos = current_pos.detach().cpu().numpy()
        if isinstance(current_pos, (list, tuple)):
            current_pos = np.array(current_pos)
        current_pos = np.array(current_pos).flatten()
        
        # 计算位置增量：目标位置 - 当前位置
        pos_delta = (target_pos - current_pos).flatten()
        
        # pd_ee_pose 控制模式：返回位置增量、目标旋转和夹爪动作
        next_action = np.concatenate([
            pos_delta,       # 位置增量 [dx, dy, dz]
            target_rot,      # 目标旋转增量 [rx, ry, rz]
            [gripper_action] # 夹爪动作
        ]).astype(np.float32)
        
        return next_action
    
@register_agent()
class PiperArm(BaseArm):
    uid = "piper_arm"
    urdf_path = "robot/piper/piper_assets/urdf/piper_description.urdf" 

    @property
    def _sensor_configs(self):
        """
        设置双视角相机配置。
        参考：外部摄像机 + 腕部手眼摄像机
        """
        # 查找 link6 (法兰端点)
        mount_link = None
        if hasattr(self, "robot") and self.robot is not None:
            try:
                mount_link = self.robot.find_link_by_name("link6")
            except:
                pass
        
        cameras = [
            # 1. 外部主视角 (固定在世界坐标系)
            CameraConfig(
                uid="front_view",
                pose=sapien.Pose(p=[0, 0.6, 0.2], q=[0.707, 0, 0, -0.707]), 
                width=320, height=240, fov=1.0, near=0.01, far=10
            ),
        ]
        
        # 2. 手眼摄像机 (挂载在第6关节法兰上，指向法兰夹爪末端)
        if mount_link is not None:
            cameras.append(CameraConfig(
                uid="hand_camera",
                mount=mount_link,
                # 相对于 link6 的位置：法兰中心
                # 四元数：指向法兰夹爪末端 (局部 Y 轴正方向)
                pose=sapien.Pose(p=[-0.075,0,0.035], q=[0.866,0,-0.5,0]),
                width=320, height=240, fov=1.2, near=0.01, far=5
            ))
        else:
            # 备选方案：如果找不到 link6，使用固定摄像机
            cameras.append(CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0.3, 0, 0.3], q=[0.707, 0, 0, -0.707]), 
                width=320, height=240, fov=1.2, near=0.01, far=5
            ))
        
        return cameras

    @property
    def keyframes(self):
        return dict(
            rest=dict(
                qpos=np.array([0, -0.2, 0, 0, 0, 0, 0, 0]),
                qvel=np.array([0] * 8),
            )
        )

    @property
    def _controller_configs(self):
        arm_pd = PDJointPosControllerConfig(
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            lower=None, upper=None, stiffness=1000, damping=100, normalize_action=False, drive_mode="force"
        )
        arm_ik = PDEEPoseControllerConfig(
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            ee_link="link6", pos_lower=-0.8, pos_upper=0.8, rot_lower=-3.14, rot_upper=3.14,
            urdf_path=self.urdf_path, stiffness=1000, damping=100, normalize_action=False, use_delta=False, drive_mode="force"
        )
        gripper_pd = PDJointPosControllerConfig(
            joint_names=["joint7", "joint8"],
            lower=0.0, upper=0.035, stiffness=500, damping=50, normalize_action=False, drive_mode="force"
        )
        return dict(
            pd_joint_pos=dict(arm=arm_pd, gripper=gripper_pd),
            pd_ee_pose=dict(arm=arm_ik, gripper=gripper_pd)
        )
