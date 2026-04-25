import numpy as np

from teleop.get_pose import get_pose


def flatten_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 3 and state.shape[1] == 1:
        state = state[:, 0, :]
    elif state.ndim == 1:
        state = state[None, :]
    return state.reshape(state.shape[0], -1).astype(np.float32)


def project_state_row(state_row, obs_dim):
    row = np.asarray(state_row, dtype=np.float32).reshape(-1)

    if obs_dim == 16:
        if row.shape[0] < 16:
            raise ValueError(f"state dim {row.shape[0]} is smaller than obs_dim=16")
        return row[:16].astype(np.float32)

    if obs_dim == 19:
        if row.shape[0] >= 19:
            return row[:19].astype(np.float32)
        if row.shape[0] < 16:
            raise ValueError(f"state dim {row.shape[0]} is too small to build obs_dim=19")

        qpos8 = row[:8]
        qvel8 = row[8:16]
        ee_pose6 = get_pose(qpos8[:6]).astype(np.float32)
        gripper = np.array([float(np.mean(qpos8[6:8]))], dtype=np.float32)
        return np.concatenate([qpos8[:6], qvel8[:6], ee_pose6, gripper]).astype(np.float32)

    if row.shape[0] < obs_dim:
        raise ValueError(f"state dim {row.shape[0]} is smaller than obs_dim={obs_dim}")
    return row[:obs_dim].astype(np.float32)


def project_state_batch(states, obs_dim):
    flat = flatten_state(states)
    return np.stack([project_state_row(row, obs_dim) for row in flat], axis=0).astype(np.float32)
