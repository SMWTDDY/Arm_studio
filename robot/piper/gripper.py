import numpy as np


PIPER_GRIPPER_MAX_WIDTH = 0.035


def piper_gripper_width_to_qpos(width):
    """Map one physical Piper gripper width to the signed finger joints."""
    width = float(np.clip(width, 0.0, PIPER_GRIPPER_MAX_WIDTH))
    return np.array([width, -width], dtype=np.float32)


def piper_gripper_qpos_to_width(qpos):
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 2:
        return 0.0
    return float(np.clip((qpos[0] - qpos[1]) * 0.5, 0.0, PIPER_GRIPPER_MAX_WIDTH))
