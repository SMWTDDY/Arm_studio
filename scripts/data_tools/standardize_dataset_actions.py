#!/usr/bin/env python3
import argparse
import glob
import os

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from teleop.get_pose import get_pose


def infer_control_mode(path, attrs, actions):
    mode = attrs.get("control_mode")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    if mode in ("pose", "joint"):
        return mode

    name = os.path.basename(path).lower()
    if "pose" in name:
        return "pose"
    if "joint" in name:
        return "joint"

    # Last-resort heuristic: joint recordings are often named/marked poorly,
    # but both joint and pose are 7D, so do not guess silently.
    raise ValueError(
        f"Cannot infer control_mode for {path}. Set the HDF5 attr or rename with pose/joint."
    )


def gripper_value(action_row, mode):
    if action_row.shape[0] <= 6:
        return 1.0
    if action_row.shape[0] == 7:
        value = float(action_row[6])
    else:
        value = float(np.mean(action_row[6:]))

    if mode == "binary":
        return 1.0 if value > 0 else -1.0
    return value


def to_pose_actions(path, actions, control_mode, gripper_mode):
    if control_mode == "pose":
        pose_actions = actions[:, :7].astype(np.float32, copy=True)
        if gripper_mode == "binary" and pose_actions.shape[1] >= 7:
            pose_actions[:, 6] = np.where(pose_actions[:, 6] > 0, 1.0, -1.0)
        return pose_actions

    if control_mode != "joint":
        raise ValueError(f"Unsupported control_mode={control_mode!r} in {path}")

    converted = np.zeros((actions.shape[0], 7), dtype=np.float32)
    for i, row in enumerate(actions):
        converted[i, :6] = get_pose(row[:6]).astype(np.float32)
        converted[i, 6] = gripper_value(row, gripper_mode)
    return converted


def copy_with_new_actions(src_path, dst_path, pose_actions, source_mode):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src.keys():
            if key == "action":
                continue
            src.copy(key, dst)

        dst.create_dataset("action", data=pose_actions, dtype=np.float32)

        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["control_mode"] = "pose"
        dst.attrs["source_control_mode"] = source_mode
        dst.attrs["standardized_action"] = "pose_xyz_rpy_gripper"


def convert_file(src_path, output_dir, gripper_mode):
    with h5py.File(src_path, "r") as f:
        actions = np.asarray(f["action"], dtype=np.float32)
        attrs = dict(f.attrs)

    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"{src_path}: action must be [T, >=6], got {actions.shape}")

    source_mode = infer_control_mode(src_path, attrs, actions)
    pose_actions = to_pose_actions(src_path, actions, source_mode, gripper_mode)
    dst_path = os.path.join(output_dir, os.path.basename(src_path))
    copy_with_new_actions(src_path, dst_path, pose_actions, source_mode)
    return source_mode, actions.shape, pose_actions.shape, dst_path


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return list(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Convert mixed Piper joint/pose HDF5 recordings into one pose-action dataset."
    )
    parser.add_argument("paths", nargs="+", help="HDF5 files or globs, e.g. datasets/*.hdf5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gripper-mode",
        choices=["passthrough", "binary"],
        default="passthrough",
        help="Use binary if you want all gripper actions normalized to -1/1.",
    )
    args = parser.parse_args()

    for src_path in expand_paths(args.paths):
        if not os.path.exists(src_path):
            print(f"[Skip] missing: {src_path}")
            continue
        source_mode, old_shape, new_shape, dst_path = convert_file(
            src_path, args.output_dir, args.gripper_mode
        )
        print(f"[OK] {src_path} ({source_mode}, {old_shape}) -> {dst_path} {new_shape}")


if __name__ == "__main__":
    main()
