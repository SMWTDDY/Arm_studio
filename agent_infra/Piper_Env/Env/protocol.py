"""Shared Piper env/data protocol helpers.

The unified protocol is intentionally small: every real or simulated Piper
environment exposes nested obs/action dictionaries plus ``meta_keys`` that
describe their shapes. Recorders and adapters can then stay backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np


PIPER_CONTROL_MODES = ("joint", "pose", "delta_pose", "relative_pose_chunk")
PIPER_ARM_MODES = ("single", "dual")
PIPER_BACKENDS = ("real", "sim")


def arm_names_for_mode(arm_mode: str) -> List[str]:
    if arm_mode == "single":
        return ["single"]
    if arm_mode == "dual":
        return ["left", "right"]
    raise ValueError(f"Unsupported arm_mode={arm_mode!r}; expected one of {PIPER_ARM_MODES}.")


def arm_prefix(arm_name: str, arm_names: Sequence[str]) -> str:
    return f"{arm_name}_" if len(arm_names) > 1 else ""


def build_meta_keys(
    arm_names: Sequence[str],
    control_mode: str = "joint",
    camera_roles: Iterable[str] | None = None,
    depth_roles: Iterable[str] | None = None,
    image_shape: Sequence[int] = (3, 224, 224),
    depth_shape: Sequence[int] = (1, 224, 224),
    include_under_control: bool = True,
    relative_pose_chunk_size: int = 8,
) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    if control_mode not in PIPER_CONTROL_MODES:
        raise ValueError(
            f"Unsupported control_mode={control_mode!r}; expected one of {PIPER_CONTROL_MODES}."
        )

    arm_dim = 6
    if control_mode == "relative_pose_chunk":
        arm_dim = int(relative_pose_chunk_size) * 6

    meta: Dict[str, Dict[str, Dict[str, tuple]]] = {
        "obs": {"state": {}},
        "action": {},
    }

    arm_names = list(arm_names)
    for name in arm_names:
        prefix = arm_prefix(name, arm_names)
        meta["obs"]["state"].update(
            {
                f"{prefix}joint_pos": (6,),
                f"{prefix}joint_vel": (6,),
                f"{prefix}ee_pose": (6,),
                f"{prefix}gripper_pos": (1,),
            }
        )
        meta["action"].update(
            {
                f"{prefix}arm": (arm_dim,),
                f"{prefix}gripper": (1,),
            }
        )

    if include_under_control:
        meta["obs"]["under_control"] = {name: (1,) for name in arm_names}

    camera_roles = list(camera_roles or [])
    if camera_roles:
        meta["obs"]["rgb"] = {role: tuple(image_shape) for role in camera_roles}

    depth_roles = list(depth_roles or [])
    if depth_roles:
        meta["obs"]["depth"] = {role: tuple(depth_shape) for role in depth_roles}

    return meta


def flatten_by_meta(values: Mapping[str, Any], meta_section: Mapping[str, Sequence[int]]) -> np.ndarray:
    parts = []
    for key in meta_section.keys():
        if key not in values:
            raise KeyError(f"Missing key {key!r} for flatten_by_meta.")
        parts.append(np.asarray(values[key], dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def split_action_by_meta(
    action_vector: Any,
    action_meta: Mapping[str, Sequence[int]],
) -> Dict[str, np.ndarray]:
    vec = np.asarray(action_vector, dtype=np.float32).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    cursor = 0
    for key, shape in action_meta.items():
        shape = tuple(int(x) for x in shape)
        size = int(np.prod(shape))
        out[key] = vec[cursor:cursor + size].reshape(shape)
        cursor += size
    if cursor != vec.shape[0]:
        raise ValueError(f"Action vector length {vec.shape[0]} does not match meta size {cursor}.")
    return out


def ensure_under_control(obs: MutableMapping[str, Any], arm_names: Sequence[str]) -> MutableMapping[str, Any]:
    obs.setdefault("under_control", {})
    for name in arm_names:
        obs["under_control"].setdefault(name, np.array([False], dtype=np.bool_))
    return obs
