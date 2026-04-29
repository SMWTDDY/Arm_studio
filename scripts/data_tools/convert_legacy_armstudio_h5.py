#!/usr/bin/env python3
"""Convert old Arm_studio HDF5 files into unified Piper raw H5 layout."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPER_ENV_DIR = PROJECT_ROOT / "agent_infra" / "Piper_Env"
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from agent_infra.Piper_Env.Env.protocol import build_meta_keys
from agent_infra.Piper_Env.Env.utils.get_pose import get_pose


def _expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            matches = sorted(p.glob("*.hdf5")) + sorted(p.glob("*.h5"))
        else:
            matches = [Path(x) for x in sorted(glob.glob(str(p)))]
        paths.extend(m for m in matches if m.is_file())
    return list(dict.fromkeys(paths))


def _read_control_mode(h5: h5py.File, path: Path, override: str | None) -> str:
    if override:
        return override
    mode = h5.attrs.get("control_mode")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if mode in {"joint", "pose"}:
        return str(mode)
    name = path.name.lower()
    if "pose" in name:
        return "pose"
    return "joint"


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


def _rgb_to_chw(rgb: np.ndarray) -> np.ndarray:
    rgb = _squeeze_batch(rgb)
    if rgb.ndim != 4:
        raise ValueError(f"Expected RGB video [T,H,W,3], got {rgb.shape}")
    if rgb.shape[-1] == 3:
        rgb = rgb.transpose(0, 3, 1, 2)
    return rgb.astype(np.uint8, copy=False)


def _state_from_legacy(h5: h5py.File, action: np.ndarray, control_mode: str) -> Dict[str, np.ndarray]:
    obs_state = None
    if "observation/state" in h5:
        obs_state = _squeeze_batch(h5["observation/state"][()]).astype(np.float32)

    if obs_state is not None and obs_state.ndim == 2 and obs_state.shape[1] >= 6:
        joint_pos = obs_state[:, :6].astype(np.float32)
        if obs_state.shape[1] >= 14:
            joint_vel = obs_state[:, 8:14].astype(np.float32)
        else:
            joint_vel = np.zeros_like(joint_pos)
    elif control_mode == "joint":
        joint_pos = action[:, :6].astype(np.float32)
        joint_vel = np.zeros_like(joint_pos)
    else:
        joint_pos = np.zeros((action.shape[0], 6), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)

    if control_mode == "pose":
        ee_pose = action[:, :6].astype(np.float32)
    else:
        ee_pose = np.stack([get_pose(row).astype(np.float32) for row in joint_pos], axis=0)

    if action.shape[1] >= 7:
        gripper = action[:, 6:7].astype(np.float32)
    elif obs_state is not None and obs_state.shape[1] >= 8:
        gripper = np.mean(obs_state[:, 6:8], axis=1, keepdims=True).astype(np.float32)
    else:
        gripper = np.zeros((action.shape[0], 1), dtype=np.float32)

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "ee_pose": ee_pose,
        "gripper_pos": gripper,
    }


def _write_group(group: h5py.Group, values: Dict[str, np.ndarray]):
    for key, value in values.items():
        compression = {"compression": "gzip", "compression_opts": 4} if value.ndim >= 3 else {}
        group.create_dataset(key, data=value, **compression)


def convert_one(src_path: Path, dst_path: Path, control_override: str | None = None):
    with h5py.File(src_path, "r") as src:
        if "action" not in src or "observation" not in src:
            raise ValueError(f"{src_path} does not look like an old Arm_studio HDF5 file.")
        action = np.asarray(src["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] < 6:
            raise ValueError(f"{src_path}: action must be [T, >=6], got {action.shape}")
        control_mode = _read_control_mode(src, src_path, control_override)
        state = _state_from_legacy(src, action, control_mode)

        rgb: Dict[str, np.ndarray] = {}
        if "observation/sensor_data/front_view/rgb" in src:
            rgb["base_camera"] = _rgb_to_chw(src["observation/sensor_data/front_view/rgb"][()])
        if "observation/sensor_data/hand_camera/rgb" in src:
            rgb["right_hand_camera"] = _rgb_to_chw(src["observation/sensor_data/hand_camera/rgb"][()])

    action_group = {
        "arm": action[:, :6].astype(np.float32),
        "gripper": (
            action[:, 6:7].astype(np.float32)
            if action.shape[1] >= 7
            else state["gripper_pos"].astype(np.float32)
        ),
    }
    env_meta = build_meta_keys(
        ["single"],
        control_mode=control_mode,
        camera_roles=list(rgb.keys()),
        image_shape=next(iter(rgb.values())).shape[1:] if rgb else (3, 240, 320),
        include_under_control=True,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_path, "w") as dst:
        obs = dst.create_group("obs")
        _write_group(obs.create_group("state"), state)
        obs.create_group("under_control").create_dataset(
            "single",
            data=np.zeros((action.shape[0], 1), dtype=np.bool_),
        )
        if rgb:
            _write_group(obs.create_group("rgb"), rgb)
        _write_group(dst.create_group("action"), action_group)
        meta = dst.create_group("meta")
        meta.create_dataset("env_meta", data=json.dumps(env_meta))
        dst.attrs["source_path"] = str(src_path)
        dst.attrs["source_format"] = "armstudio_legacy_hdf5"
        dst.attrs["control_mode"] = control_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input .hdf5/.h5 files, directories, or globs.")
    parser.add_argument("--output-dir", default="datasets/converted_unified_h5")
    parser.add_argument("--control-mode", choices=["joint", "pose"], default=None)
    parser.add_argument("--prefix", default="piper")
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input HDF5 files found.")

    output_dir = Path(args.output_dir)
    for idx, src_path in enumerate(paths):
        control_mode = args.control_mode or ("pose" if "pose" in src_path.stem else "joint")
        dst_path = output_dir / f"{args.prefix}_{control_mode}_sim_{idx:03d}.hdf5"
        convert_one(src_path, dst_path, args.control_mode)
        print(f"[OK] {src_path} -> {dst_path}")


if __name__ == "__main__":
    main()
