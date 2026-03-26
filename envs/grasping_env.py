import gymnasium as gym
import numpy as np
import torch
import sapien
import math
import time

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import PDJointPosControllerConfig

# ==========================================
# 1. 机械臂动作空间封装 (Wrapper)
# ==========================================
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

# ==========================================
# 2. 机械臂智能体注册 (Agent)
# ==========================================
@register_agent()
class PiperArm(BaseAgent):
    uid = "piper_arm"
    # 确保此处相对路径与你本地一致
    urdf_path = "piper_assets/urdf/piper_description.urdf" 
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

# ==========================================
# 3. 流水线抓取环境定义 (Environment)
# ==========================================
@register_env("PiperConveyorGrasp-v0", max_episode_steps=500)
class PiperConveyorGraspEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["dense", "sparse"]
    
    def __init__(self, *args, robot_uids="piper_arm", **kwargs):
        self.robot_uids = robot_uids
        super().__init__(*args, **kwargs)
        
    def _load_scene(self, options: dict):
        """场景静态元素加载"""
        # 基础光影与地面
        self.scene.add_ground(altitude=0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 0.5, -1], [0.8, 0.8, 0.8])
        
        # 创建流水线 (Kinematic Actor)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.6, 0.15, 0.05])
        builder.add_box_visual(half_size=[0.6, 0.15, 0.05], color=[0.3, 0.3, 0.3])
        self.conveyor = builder.build_kinematic(name="conveyor")
        # 放置在机械臂前方，高度为 0.05，表面高度为 0.1
        self.conveyor.set_pose(sapien.Pose([0.4, 0, 0.05]))
        
        # 创建抓取目标方块 (Dynamic Actor)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[0.8, 0.2, 0.2])
        self.target_block = builder.build(name="target_block")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """回合重置时的逻辑"""
        with torch.device(self.device):
            # 将方块重置到流水线的最远端上方
            self.target_block.set_pose(sapien.Pose([0.9, 0, 0.15]))
            # 清除方块残留的速度
            self.target_block.set_linear_velocity(torch.zeros((len(env_idx), 3)))
            self.target_block.set_angular_velocity(torch.zeros((len(env_idx), 3)))
            
            # 初始化机械臂到一个默认的观察姿态
            # [0, 0, 0, 0, 0, 0, 0, 0] 对应 6个臂关节 + 2个夹爪关节
            initial_qpos = torch.zeros((len(env_idx), 8), device=self.device)
            self.agent.robot.set_qpos(initial_qpos)

    def evaluate(self):
        """评估任务是否成功"""
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        
    def _get_obs_extra(self, info: dict):
        return dict()

# ==========================================
# 4. 主程序与流水线逻辑控制
# ==========================================
def main():
    # 实例化自定义环境
    env = gym.make(
        "PiperConveyorGrasp-v0", 
        obs_mode="state",
        control_mode="pd_joint_pos", 
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human"
    )
    
    # 套上动作封装器
    env = BinaryGripperWrapper(env)
    
    obs, _ = env.reset()
    print("环境加载成功，按 ESC 可退出渲染窗口。")
    
    step_count = 0
    conveyor_velocity = torch.tensor([[-0.15, 0.0, 0.0]], device=env.unwrapped.device)

    while True:
        # --- 1. 流水线物理传送逻辑 ---
        # 获取底层未被 Wrapper 封装的 target_block 状态
        block = env.unwrapped.target_block
        pos = block.pose.p[0] # 取出当前坐标 [X, Y, Z]
        
        # 如果方块落在流水线表面范围内，施加线速度模拟传送
        if 0.1 <= pos[2] <= 0.13 and -0.2 < pos[0] < 1.0:
            block.set_linear_velocity(conveyor_velocity)
            
        # 如果方块掉出边界，重置环境（方块重新回到起点）
        if pos[0] < -0.2 or pos[2] < 0.05:
            env.reset()
            step_count = 0

        # --- 2. 机械臂控制逻辑 ---
        target_angle = math.sin(step_count * 0.05) * 0.35
        # 测试 7 维输入：前 6 维动，第 7 维控制夹爪开合
        action = np.array([1, 1, 1, 1, 1, 1, 1]) * target_angle
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        step_count += 1
        time.sleep(0.01)

if __name__ =="__main__":
    main()