import sys
import os
import numpy as np
import time
import torch
import gymnasium as gym
from pyAgxArm import create_agx_arm_config, AgxArmFactory

# 自动将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.piper.agent import PiperArm, PiperActionWrapper
from teleop.get_pose import get_pose
import mani_skill.envs

def main():
    # 1. 初始化真实主臂
    print(">>> 正在连接真实主臂...")
    cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_master")
    real_robot = AgxArmFactory.create_arm(cfg)
    real_robot.connect()
    real_robot.set_leader_mode()
    
    # 2. 初始化仿真环境 (无头模式，不显示画面)
    print(">>> 正在初始化仿真器 IK 引擎...")
    env = gym.make(
        "Empty-v1", 
        obs_mode="state",
        control_mode="pd_ee_pose", 
        robot_uids="piper_arm"
    )
    env = PiperActionWrapper(env, binary_gripper=True)
    obs, _ = env.reset()

    print("\n" + "="*85)
    print(f"{'关节角闭环还原测试 (Real -> FK -> Sim_IK -> Sim_Joints)':^85}")
    print("="*85)

    try:
        while True:
            # 3. 获取实机当前关节角 (J_real)
            mja = real_robot.get_leader_joint_angles()
            if mja is None:
                time.sleep(0.01)
                continue
            real_joints = np.array(mja.msg, dtype=np.float32)
            
            # 4. 通过 get_pose 计算绝对位姿 (Pose_target)
            target_pose = get_pose(real_joints)
            
            # 5. 将该位姿喂给仿真器的 IK 控制器 (Action)
            action = np.concatenate([target_pose, [-1.0]])
            
            # 6. 仿真步进：IK 算法在底层将 Pose_target 转为 J_sim
            obs, _, _, _, _ = env.step(action)
            
            # 7. 提取解算后的关节角 (J_sim)
            if isinstance(obs, torch.Tensor):
                obs_np = obs.detach().cpu().numpy().flatten()
            else:
                obs_np = np.array(obs).flatten()
            
            sim_joints = obs_np[0:6]
            
            # 8. 实时对比打印
            os.system('clear')
            print(f"输入位姿 (由 get_pose 计算): \n{np.round(target_pose, 4)}")
            print("-" * 85)
            print(f"{'轴号':<6} | {'实机原始角(J_real)':<20} | {'仿真逆解角(J_sim)':<20} | {'偏差(误差)':<15}")
            print("-" * 85)
            
            for i in range(6):
                rj = real_joints[i]
                sj = sim_joints[i]
                diff = rj - sj
                print(f"J{i+1:<5} | {rj:<20.6f} | {sj:<20.6f} | {diff:<15.6f}")
            
            print("-" * 85)
            total_error = np.linalg.norm(real_joints - sim_joints)
            print(f"所有关节总向量偏差: {total_error:.6f} rad")
            print(">>> 如果偏差巨大，说明仿真 URDF 的零位或坐标轴定义与 DH 参数不一致。")
            print("-" * 85)
            print("按 Ctrl+C 退出")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        env.close()

if __name__ == "__main__":
    main()
