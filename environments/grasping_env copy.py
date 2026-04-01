import sapien
import torch
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env

# 使用装饰器注册环境，命名规范通常为 "名称-v版本号"
@register_env("PiperConveyorGrasp-v0", max_episode_steps=200)
class PiperConveyorGraspEnv(BaseEnv):
    # 定义支持的奖励模式
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    
    def __init__(self, *args, robot_uids="piper", **kwargs):
        # 这里的 robot_uids 是 ManiSkill 内部管理智能体所需的参数
        self.robot_uids = robot_uids
        super().__init__(*args, **kwargs)
        
    def _load_scene(self, options: dict):
        """
        场景加载：只在环境初始化时调用一次。
        在这里放置静态物体、光源和机械臂模型。
        """
        # 1. 基础环境
        self.scene.add_ground(altitude=0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 0.5, -1], [0.8, 0.8, 0.8])
        
        # 2. 导入 Piper 机械臂
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.piper_robot = loader.load("models/piper/piper_assets/urdf/piper_description.urdf")
        self.piper_robot.set_pose(sapien.Pose([0, 0, 0]))
        
        # 3. 创建流水线 (Kinematic)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_sizes=[0.6, 0.15, 0.05])
        builder.add_box_visual(half_sizes=[0.6, 0.15, 0.05], color=[0.3, 0.3, 0.3])
        self.conveyor = builder.build_kinematic(name="conveyor")
        self.conveyor.set_pose(sapien.Pose([0.4, 0, 0.05]))
        
        # 4. 创建抓取目标方块 (Dynamic)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_sizes=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_sizes=[0.02, 0.02, 0.02], color=[0.8, 0.2, 0.2])
        self.target_block = builder.build(name="target_block")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        回合初始化：每次调用 env.reset() 时触发。
        用于随机化物体位置、机械臂初始姿态（Domain Randomization 在这里做）。
        """
        # ManiSkill 3 是张量化的，所有的位姿操作建议使用 torch 张量
        with torch.device(self.device):
            # 将方块重置到流水线起点的随机位置
            b = len(env_idx) # 并行环境数量
            start_x = torch.rand(b) * 0.1 + 0.8
            start_y = torch.rand(b) * 0.1 - 0.05
            start_z = torch.ones(b) * 0.2
            
            # 这部分需要配合 ManiSkill 的 Actor 封装 API 来设置位姿，
            # 纯 sapien.Pose 无法处理并行的 tensor
            # 伪代码：self.target_block.set_pose(torch.stack([start_x, start_y, start_z], dim=1))
            pass

    def evaluate(self):
        """
        状态评估：每一步都会调用，判断任务是否成功。
        对于抓取任务，通常判断物体的高度是否大于某个阈值。
        """
        # 假设物体 Z 坐标大于 0.3 视为抓取成功
        # is_grasped = self.target_block.pose.p[:, 2] > 0.3
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        
    def _get_obs_extra(self, info: dict):
        """
        定义除了图像和本体感受器之外的额外状态观测输入。
        """
        return dict()