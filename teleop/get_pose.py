import numpy as np
from scipy.spatial.transform import Rotation as R

def get_pose(joint_angles: list) -> np.ndarray:
    """
    基于 piper_description.urdf 计算末端位姿 (Base -> Gripper Base)。
    """
    if len(joint_angles) < 6:
        return np.zeros(6, dtype=np.float32)

    def get_joint_transform(xyz, rpy, q=0.0, axis=[0, 0, 1]):
        mat = np.eye(4)
        mat[:3, 3] = xyz
        mat[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
        
        joint_rot = np.eye(4)
        if q != 0.0:
            joint_rot[:3, :3] = R.from_rotvec(np.array(axis) * q).as_matrix()
            
        return mat @ joint_rot

    # 1-6轴旋转关节
    T1 = get_joint_transform([0, 0, 0.123], [0, 0, 0], joint_angles[0])
    T2 = get_joint_transform([0, 0, 0], [1.5708, -0.1359, -3.1416], joint_angles[1])
    T3 = get_joint_transform([0.28503, 0, 0], [0, 0, -1.7939], joint_angles[2])
    T4 = get_joint_transform([-0.021984, -0.25075, 0], [1.5708, 0, 0], joint_angles[3])
    T5 = get_joint_transform([0, 0, 0], [-1.5708, 0, 0], joint_angles[4])
    T6 = get_joint_transform([8.8259E-05, -0.091, 0], [1.5708, 0, 0], joint_angles[5])
    
    # 显式添加末端夹爪基座的固定变换 (应对未来可能修改 URDF 的情况)
    T_gripper_base = get_joint_transform([0, 0, 0], [0, 0, 0])

    # 链式变换
    T_total = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T_gripper_base
    
    xyz = T_total[:3, 3]
    rpy = R.from_matrix(T_total[:3, :3]).as_euler('xyz')
    
    return np.concatenate([xyz, rpy]).astype(np.float32)