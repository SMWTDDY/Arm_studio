import numpy as np
import sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation as R
import torch
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 必须导入机器人模型以注册
import robot.piper.agent 

@register_env("PiperConveyor-v0")
class PiperConveyorEnv(BaseEnv):
    # 正确的设置频率方式：定义为类属性
    control_freq = 32
    sim_freq = 64

    def __init__(self, *args, **kwargs):
        self.belt_velocity = np.array([0.1, 0, 0]) # 调慢一点，方便抓取
        self.item_spawn_interval = 400
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
            initial_pose=sapien.Pose([0, 0, -0.01]),
        )
        
        # 2. 添加灯光 (改用点光源照射工作区)
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_point_light([0, -0.25, 1.5], [1.5, 1.5, 1.5]) # 在机器人正上方 1.5m 处

        # 3. 创建传送带 (跑道)
        # 长度 2.4m (half=2), 宽度 0.4m (half=0.2)

        # 近端边缘要在 x=0.15，所以中心要在 0.15 + 0.6 = 0.75
        self.conveyor = actors.build_box(
            self.scene,
            half_sizes=[2, 0.2, 0.02],
            color=[0.8, 0.1, 0.1, 1], # 亮红色
            name="conveyor",
            body_type="static",
            initial_pose=sapien.Pose([0, 0, 0]),
        )

        # 3. 预创建物品池
        self.items = []
        for i in range(10):
            hidden_pose = sapien.Pose([10, i * 0.5, -10])
            item = actors.build_box(
                self.scene,
                half_sizes=[0.022, 0.022, 0.022], # 4cm 的方块
                color=[0.1, 0.1, 0.8, 1], # 蓝色
                name=f"item_{i}",
                body_type="dynamic",
                initial_pose=hidden_pose,
            )
            self.items.append(item)

        # 4. 创建堆叠区
        self.area = actors.build_box(
            self.scene,
            half_sizes = [0.1,0.1,0.01],
            color = [0.2,0.1,0.2,1],
            name="area",
            body_type="static",
            initial_pose=sapien.Pose([0.25,-0.3,0.01]),
        )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self.items:
            return
        for i, item in enumerate(self.items):
            item.set_pose(sapien.Pose([10, i * 0.5, -10]))
            item.set_linear_velocity([0, 0, 0])
        self._respawn_item(0)
        if hasattr(self, "agent") and self.agent is not None:
            # 1. 设置位置 (x, y, z) 
            # 比如把它放在传送带的一侧 (你的传送带中心在 y=1，设机械臂在 y=-0.5)
            target_position = [0.0, -0.25, 0.0] 

            # 2. 设置朝向 (四元数 [w, x, y, z])
            euler_angles = [0, 0, 90] # 分别对应绕 X, Y, Z 轴的旋转角度(度)
            
            # scipy 生成的是 [x, y, z, w] 格式，而 sapien 期望的是 [w, x, y, z]
            quat_xyzw = R.from_euler('xyz', euler_angles, degrees=True).as_quat() 
            target_quaternion = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

            # 3. 应用位姿
            self.agent.robot.set_pose(sapien.Pose(p=target_position, q=target_quaternion))
        else:
            # 如果 agent 尚未准备好，跳过设置位姿
            return
        
        # 4. 更新摄像机可视化标记
        self._update_camera_visualization()

    def _update_camera_visualization(self):
        """更新手眼摄像机的可视化标记 - 简化版本"""
        if not hasattr(self, 'agent') or self.agent is None:
            return
            
        try:
            # hand_camera now follows the URDF hand_cam optical mount.
            hand_cam = self.agent.robot.find_link_by_name("hand_cam")
            camera_world_pose = hand_cam.pose
            
            # 获取 hand_camera 的实际世界坐标位姿
            sensor_configs = getattr(self.agent, '_sensor_configs', [])
            hand_camera_config = None
            for config in sensor_configs:
                if getattr(config, 'uid', None) == 'hand_camera':
                    hand_camera_config = config
                    break

            if hand_camera_config is not None and getattr(hand_camera_config, 'mount', None) is not None:
                camera_world_pose = camera_world_pose * hand_camera_config.pose
            
        except Exception as e:
            # 如果出错，打印错误但不隐藏标记
            print(f"可视化更新错误: {e}")
            pass




    def _respawn_item(self, idx: int):
        if idx < len(self.items):
            item = self.items[idx]
            # 放在传送带近端起点 (x=0.15 稍微往里一点)
            item.set_pose(sapien.Pose([-0.5, np.random.uniform(-0.1, 0.1), 0.1]))
            item.set_linear_velocity([0, 0, 0])

    def step(self, action):
        spawn_interval = int(getattr(self, "item_spawn_interval", 400))
        if len(self.items) > 0 and spawn_interval > 0 and self.elapsed_steps[0] > 0 and self.elapsed_steps[0] % spawn_interval == 0:
            next_idx = (int(self.elapsed_steps[0]) // spawn_interval) % len(self.items)
            self._respawn_item(next_idx)

        for item in self.items:
            p = item.pose.p
            px, py, pz = p[0, 0].item(), p[0, 1].item(), p[0, 2].item()
            # 动力范围：x 在 [0.15, 1.35] 之间
            if -2 <= px <= 2 and -0.20 < py < 0.20 and 0.03 < pz < 0.2:
                item.set_linear_velocity(self.belt_velocity)
            # 传送出界回收
            if px > 2 or px <-2 :
                item.set_pose(sapien.Pose([10, 0, -10]))
        
        # 更新摄像机可视化标记
        self._update_camera_visualization()
        
        return super().step(action)

    def evaluate(self): return {}
    def get_reward(self, **kwargs): return torch.zeros(self.num_envs, device=self.device)
    def compute_dense_reward(self, **kwargs): return torch.zeros(self.num_envs, device=self.device)
    def compute_normalized_dense_reward(self, **kwargs): return torch.zeros(self.num_envs, device=self.device)
