#!/usr/bin/env python3
import argparse
import csv
import glob
import os

import h5py
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from models.DiffusionPolicy.action_codec import encode_pose_to_continuous, gripper_to_label
from teleop.get_pose import get_pose


def expand_inputs(inputs, output_dir):
    paths = []
    output_dir = os.path.abspath(output_dir)
    for item in inputs:
        item = os.path.abspath(item)
        if os.path.isdir(item):
            matches = sorted(glob.glob(os.path.join(item, "*.hdf5")))
        else:
            matches = sorted(glob.glob(item))
        for path in matches:
            abs_path = os.path.abspath(path)
            if abs_path.startswith(output_dir + os.sep):
                continue
            paths.append(abs_path)
    return list(dict.fromkeys(paths))


def infer_control_mode(path, attrs):
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
    raise ValueError(f"Cannot infer control_mode for {path}")


def read_gripper(row, threshold=0.0):
    if row.shape[0] <= 6:
        value = 1.0
    elif row.shape[0] == 7:
        value = float(row[6])
    else:
        value = float(np.mean(row[6:]))
    return 1.0 if value > threshold else -1.0


def to_pose_actions(path, actions, control_mode, threshold):
    pose_actions = np.zeros((actions.shape[0], 7), dtype=np.float32)
    if control_mode == "pose":
        pose_actions[:, :6] = actions[:, :6]
    elif control_mode == "joint":
        for i, row in enumerate(actions):
            pose_actions[i, :6] = get_pose(row[:6]).astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported control_mode={control_mode!r}")

    for i, row in enumerate(actions):
        pose_actions[i, 6] = read_gripper(row, threshold)
    return pose_actions


def encode_processed_actions(pose_actions, threshold):
    continuous = np.stack(
        [encode_pose_to_continuous(row[:6]) for row in pose_actions],
        axis=0,
    ).astype(np.float32)
    gripper = np.asarray(
        [gripper_to_label(row[6], threshold) for row in pose_actions],
        dtype=np.float32,
    )
    return continuous, gripper


def copy_file(src_path, dst_path, pose_actions, continuous, gripper, source_mode, index):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src.keys():
            if key in {"action", "action_continuous", "gripper_label"}:
                continue
            src.copy(key, dst)

        dst.create_dataset("action", data=pose_actions, dtype=np.float32)
        dst.create_dataset("action_continuous", data=continuous, dtype=np.float32)
        dst.create_dataset("gripper_label", data=gripper, dtype=np.float32)

        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["control_mode"] = "pose"
        dst.attrs["source_control_mode"] = source_mode
        dst.attrs["source_path"] = os.path.relpath(src_path, PROJECT_ROOT)
        dst.attrs["source_basename"] = os.path.basename(src_path)
        dst.attrs["unified_index"] = index
        dst.attrs["standardized_action"] = "pose_xyz_rpy_gripper_binary"
        dst.attrs["processed_action"] = "xyz_rot6d_plus_gripper_label"


def convert_one(src_path, dst_path, index, threshold):
    with h5py.File(src_path, "r") as f:
        actions = np.asarray(f["action"], dtype=np.float32)
        attrs = dict(f.attrs)

    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"{src_path}: action must be [T, >=6], got {actions.shape}")

    source_mode = infer_control_mode(src_path, attrs)
    pose_actions = to_pose_actions(src_path, actions, source_mode, threshold)
    continuous, gripper = encode_processed_actions(pose_actions, threshold)
    copy_file(src_path, dst_path, pose_actions, continuous, gripper, source_mode, index)
    return source_mode, actions.shape, pose_actions.shape


def main():
    parser = argparse.ArgumentParser(
        description="Build one uniformly named pose dataset from mixed Piper joint/pose HDF5 files."
    )
    parser.add_argument("inputs", nargs="+", help="Input HDF5 directories or globs.")
    parser.add_argument("--output-dir", default="datasets/unified_pose_all")
    parser.add_argument("--prefix", default="piper_pose_unified")
    parser.add_argument("--gripper-threshold", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    paths = expand_inputs(args.inputs, output_dir)
    if not paths:
        raise SystemExit("No input HDF5 files found.")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    rows = []
    counts = {"joint": 0, "pose": 0}
    for index, src_path in enumerate(paths):
        dst_name = f"{args.prefix}_{index:04d}.hdf5"
        dst_path = os.path.join(output_dir, dst_name)
        source_mode, old_shape, new_shape = convert_one(
            src_path,
            dst_path,
            index,
            args.gripper_threshold,
        )
        counts[source_mode] = counts.get(source_mode, 0) + 1
        rows.append(
            {
                "index": index,
                "file": dst_name,
                "source_path": os.path.relpath(src_path, PROJECT_ROOT),
                "source_control_mode": source_mode,
                "old_shape": str(tuple(old_shape)),
                "pose_shape": str(tuple(new_shape)),
            }
        )
        print(f"[OK] {rows[-1]['source_path']} ({source_mode}) -> {dst_name}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "file",
                "source_path",
                "source_control_mode",
                "old_shape",
                "pose_shape",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Done. files={len(rows)} pose={counts.get('pose', 0)} "
        f"joint_converted={counts.get('joint', 0)} output={output_dir}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
