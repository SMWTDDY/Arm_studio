import numpy as np
from scipy.spatial.transform import Rotation as R


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix_to_6d(matrix):
    matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    return matrix[:, :2].reshape(6, order="F").astype(np.float32)


def rotation_6d_to_matrix(rotation_6d):
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64).reshape(6)
    a1 = rotation_6d[:3]
    a2 = rotation_6d[3:6]

    b1_norm = np.linalg.norm(a1)
    if b1_norm < 1e-8:
        b1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        b1 = a1 / b1_norm

    a2_orth = a2 - np.dot(b1, a2) * b1
    b2_norm = np.linalg.norm(a2_orth)
    if b2_norm < 1e-8:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(b1, fallback)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        a2_orth = fallback - np.dot(b1, fallback) * b1
        b2_norm = np.linalg.norm(a2_orth)
    b2 = a2_orth / b2_norm
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def encode_pose_to_continuous(pose6):
    pose6 = np.asarray(pose6, dtype=np.float32)
    rot = R.from_euler("xyz", pose6[3:6]).as_matrix()
    return np.concatenate([pose6[:3], rotation_matrix_to_6d(rot)]).astype(np.float32)


def gripper_to_label(value, threshold=0.0):
    return np.float32(1.0 if float(value) > threshold else 0.0)


def gripper_label_to_value(label, open_value=-1.0, close_value=1.0):
    return close_value if int(label) > 0 else open_value


def decode_continuous_to_pose_action(continuous, gripper_value):
    continuous = np.asarray(continuous, dtype=np.float32)
    rot = rotation_6d_to_matrix(continuous[3:9])
    rpy = R.from_matrix(rot).as_euler("xyz").astype(np.float32)
    rpy = wrap_angle(rpy)
    return np.array(
        [
            continuous[0],
            continuous[1],
            continuous[2],
            rpy[0],
            rpy[1],
            rpy[2],
            gripper_value,
        ],
        dtype=np.float32,
    )


def decode_sequence_to_pose_actions(
    continuous_sequence,
    gripper_labels,
    open_value=-1.0,
    close_value=1.0,
):
    continuous_sequence = np.asarray(continuous_sequence, dtype=np.float32)
    gripper_labels = np.asarray(gripper_labels).reshape(-1)
    actions = []
    for i, continuous in enumerate(continuous_sequence):
        actions.append(
            decode_continuous_to_pose_action(
                continuous,
                gripper_label_to_value(gripper_labels[i], open_value, close_value),
            )
        )
    return np.stack(actions, axis=0)
