import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

from teleop.get_pose import get_pose


DEFAULT_URDF = os.path.join(
    os.path.dirname(__file__), "piper_assets/urdf/piper_description_with_camera_right.urdf"
)


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_joint_limits(urdf_path=DEFAULT_URDF):
    fallback = np.array(
        [
            [-2.618, 2.618],
            [0.0, 3.14],
            [-2.967, 0.0],
            [-1.745, 1.745],
            [-1.22, 1.22],
            [-2.0944, 2.0944],
        ],
        dtype=np.float64,
    )
    if not os.path.exists(urdf_path):
        return fallback

    root = ET.parse(urdf_path).getroot()
    limits = []
    for i in range(1, 7):
        joint = root.find(f"./joint[@name='joint{i}']")
        if joint is None:
            return fallback
        limit = joint.find("limit")
        if limit is None:
            return fallback
        limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
    return np.array(limits, dtype=np.float64)


def pose_error(current_pose, target_pose):
    err = np.asarray(current_pose[:6], dtype=np.float64) - np.asarray(target_pose[:6], dtype=np.float64)
    err[3:6] = wrap_angle(err[3:6])
    return err


class BoundedPiperIK:
    def __init__(
        self,
        joint_limits=None,
        pos_weight=100.0,
        rot_weight=1.0,
        joint_weight=0.0,
        max_nfev=80,
    ):
        self.joint_limits = load_joint_limits() if joint_limits is None else np.asarray(joint_limits, dtype=np.float64)
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.joint_weight = joint_weight
        self.max_nfev = max_nfev

    def solve(self, target_pose, seed_qpos):
        target_pose = np.asarray(target_pose[:6], dtype=np.float64)
        seed_qpos = np.asarray(seed_qpos[:6], dtype=np.float64)
        seed_qpos = np.clip(seed_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])

        def residual(qpos):
            err = pose_error(get_pose(qpos), target_pose)
            return np.concatenate(
                [
                    err[:3] * self.pos_weight,
                    err[3:6] * self.rot_weight,
                    (qpos - seed_qpos) * self.joint_weight,
                ]
            )

        result = least_squares(
            residual,
            seed_qpos,
            bounds=(self.joint_limits[:, 0], self.joint_limits[:, 1]),
            max_nfev=self.max_nfev,
            diff_step=1e-4,
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
        )
        err = pose_error(get_pose(result.x), target_pose)
        return {
            "qpos": result.x.astype(np.float32),
            "success": bool(result.success),
            "status": int(result.status),
            "pos_error": float(np.linalg.norm(err[:3])),
            "rot_error": float(np.linalg.norm(err[3:6])),
            "cost": float(result.cost),
        }
