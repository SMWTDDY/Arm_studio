import numpy as np
import time
import threading
from pyAgxArm import create_agx_arm_config, AgxArmFactory
from teleop.get_pose import get_pose

class RealToSimTeleop:
    """
    将真实 Piper 机械臂的关节空间或位姿数据同步到仿真环境中。
    兼容 Api_test 的控制逻辑。
    """
    def __init__(self, can_name='can_master', gripper_threshold=0.015):
        # 1. 初始化 pyAgxArm 配置
        self.cfg = create_agx_arm_config(robot="piper", comm="can", channel=can_name)
        self.robot = AgxArmFactory.create_arm(self.cfg)
        
        # 2. 连接并激活机械臂
        print(f"正在连接并激活主臂: {can_name} ...")
        self.robot.connect()
        
        # 必须循环使能，直到机械臂确认进入工作状态
        retry_count = 0
        while not self.robot.is_ok() and retry_count < 20:
            self.robot.enable_arm() 
            time.sleep(0.1)
            retry_count += 1
            print(f"尝试唤醒机械臂... ({retry_count}/20)")

        if not self.robot.is_ok():
            print("警告: 机械臂未响应 (is_ok=False)，请检查硬件连线或波特率。")
        
        # 3. 设置为 Leader 模式 (主臂拖拽模式)
        self.robot.set_leader_mode()
        
        # 4. 初始化夹爪
        self.effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        
        self.gripper_threshold = gripper_threshold
        self.running = True
        
        # 缓存最新的 action [j1..j6, g] 或 [x..yaw, g]
        self._latest_joints = np.zeros(7, dtype=np.float32)
        self._latest_pose = np.zeros(7, dtype=np.float32)
        
        # 默认夹爪关闭
        self._latest_joints[6] = -1.0
        self._latest_pose[6] = -1.0
        
        # 启动后台监听线程
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"RealToSimTeleop 已就绪。")

    def _update_loop(self):
        """后台线程：持续读取真实机械臂状态"""
        while self.running:
            try:
                # 1. 读取关节角 (弧度)
                joint_data = self.robot.get_leader_joint_angles()
                
                # 2. 读取夹爪状态 (宽度)
                gripper_data = self.effector.get_gripper_ctrl_states()
                
                if joint_data is not None:
                    # 获取关节角度
                    joints = np.array(joint_data.msg, dtype=np.float32)
                    self._latest_joints[:6] = joints
                    
                    # 通过 get_pose 计算法兰位姿
                    pose = get_pose(joints)
                    self._latest_pose[:6] = pose
                
                if gripper_data is not None:
                    width = gripper_data.msg.width
                    # 映射夹爪宽度到二进制 (-1 or 1) 供某些控制器使用
                    self._latest_joints[6] = 1.0 if width > self.gripper_threshold else -1.0
                    self._latest_pose[6] = 1.0 if width > self.gripper_threshold else -1.0
                    
                    # 兼容 Api_test: 将原始宽度信息记录在某些字段（可选，如果需要连续控制）
                    # 在 get_action 中根据需要返回
                
            except Exception:
                pass
            
            time.sleep(0.005) # 200Hz

    def get_action(self, mode="joint", use_binary_gripper=True):
        """
        返回当前最新的 7 维 Action
        Args:
            mode: "joint" (返回关节角) 或 "pose" (返回位姿)
            use_binary_gripper: 是否返回二值化后的夹爪控制信号
        """
        if mode == "pose":
            action = self._latest_pose.copy()
        else:
            action = self._latest_joints.copy()
            
        # 如果不使用二值化，则返回物理宽度 (m)
        if not use_binary_gripper:
            # 重新读取一次物理宽度（或从缓存中提取，这里简化处理）
            gripper_data = self.effector.get_gripper_ctrl_states()
            if gripper_data is not None:
                action[6] = gripper_data.msg.width
        
        return action

    def close(self):
        self.running = False
        print("RealToSimTeleop 已关闭。")

if __name__ == "__main__":
    teleop = RealToSimTeleop(can_name='can_master')
    try:
        while True:
            action_j = teleop.get_action(mode="joint")
            action_p = teleop.get_action(mode="pose")
            print(f"\rJoint: {np.round(action_j[:6], 2)} | Pose: {np.round(action_p[:6], 2)} | G: {action_j[6]}", end="")
            time.sleep(0.1)
    except KeyboardInterrupt:
        teleop.close()
