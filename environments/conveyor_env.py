import numpy as np
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
import torch
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 必须导入机器人模型以注册
import models.piper.agent 

@register_env("PiperConveyor-v0", max_episode_steps=1000)
class PiperConveyorEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        self.belt_velocity = np.array([0.15, 0, 0]) # 调慢一点，方便抓取
        self.items = []
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        # 1. 创建地面
        self.ground = actors.build_box(
            self.scene,
            half_sizes=[5.0, 5.0, 0.01],
            color=[0.2, 0.2, 0.2, 1], # 深灰背景
            name="ground",
            body_type="static",
        )
        self.ground.set_pose(sapien.Pose([0, 0, -0.01]))
        
        # 2. 创建传送带 (跑道)
        # 长度 1.2m (half=0.6), 宽度 0.4m (half=0.2)

        # 近端边缘要在 x=0.15，所以中心要在 0.15 + 0.6 = 0.75
        self.conveyor = actors.build_box(
            self.scene,
            half_sizes=[2, 0.2, 0.02],
            color=[0.8, 0.1, 0.1, 1], # 亮红色
            name="conveyor",
            body_type="kinematic",
        )
        self.conveyor.set_pose(sapien.Pose([0.75, 0, 0.02]))

        # 3. 预创建物品池
        self.items = []
        for i in range(10):
            item = actors.build_box(
                self.scene,
                half_sizes=[0.02, 0.02, 0.02], # 4cm 的方块
                color=[0.1, 0.1, 0.8, 1], # 蓝色
                name=f"item_{i}",
                body_type="dynamic",
            )
            item.set_pose(sapien.Pose([10, i * 0.5, -10])) 
            self.items.append(item)

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([1, 1, -1], [1.2, 1.2, 1.2])
        # 将点光源移到传送带正上方
        self.scene.add_point_light([0.75, 0, 1.0], [2, 2, 2])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self.items:
            return
        for i, item in enumerate(self.items):
            item.set_pose(sapien.Pose([10, i * 0.5, -10]))
            item.set_linear_velocity([0, 0, 0])
        self._respawn_item(0)

    def _respawn_item(self, idx: int):
        if idx < len(self.items):
            item = self.items[idx]
            # 放在传送带近端起点 (x=0.15 稍微往里一点)
            item.set_pose(sapien.Pose([-2, np.random.uniform(-0.1, 0.1), 0.1]))
            item.set_linear_velocity([0, 0, 0])

    def step(self, action):
        if len(self.items) > 0 and self.elapsed_steps[0] > 0 and self.elapsed_steps[0] % 200 == 0:
            next_idx = (int(self.elapsed_steps[0]) // 200) % len(self.items)
            self._respawn_item(next_idx)

        for item in self.items:
            p = item.pose.p
            px, py, pz = p[0, 0].item(), p[0, 1].item(), p[0, 2].item()
            # 动力范围：x 在 [0.15, 1.35] 之间
            if -2 <= px <= 2 and -0.25 < py < 0.25 and 0.03 < pz < 0.2:
                item.set_linear_velocity(self.belt_velocity)
            # 传送出界回收
            if px > 2 or px <-2 :
                item.set_pose(sapien.Pose([10, 0, -10]))
        return super().step(action)

    def evaluate(self): return {}
    def get_reward(self, **kwargs): return torch.zeros(self.num_envs, device=self.device)
    def compute_dense_reward(self, **kwargs): return torch.zeros(self.num_envs, device=self.device)
    def compute_normalized_dense_reward(self, **kwargs): return torch.zeros(self.num_envs, device=self.device)
