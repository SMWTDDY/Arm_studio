#!/usr/bin/env python3
import argparse
import glob
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def expand_inputs(inputs, output_dir):
    output_dir = os.path.abspath(output_dir)
    paths = []
    for item in inputs:
        item = os.path.abspath(item)
        if os.path.isdir(item):
            matches = sorted(glob.glob(os.path.join(item, "*.hdf5")))
        else:
            matches = sorted(glob.glob(item))
        for path in matches:
            path = os.path.abspath(path)
            if path.startswith(output_dir + os.sep):
                continue
            paths.append(path)
    return list(dict.fromkeys(paths))


def resize_rgb_dataset(src_ds, dst_group, name, image_size, compression):
    n = src_ds.shape[0]
    dst = dst_group.create_dataset(
        name,
        shape=(n, 1, image_size, image_size, 3),
        dtype=np.uint8,
        compression=compression,
        compression_opts=4 if compression == "gzip" else None,
        chunks=(1, 1, image_size, image_size, 3),
    )

    for i in range(n):
        img = np.asarray(src_ds[i])
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        dst[i, 0] = img.astype(np.uint8, copy=False)


def copy_observation(src, dst, cameras, image_size, compression):
    obs_src = src["observation"]
    obs_dst = dst.create_group("observation")

    for key in obs_src.keys():
        if key == "sensor_data":
            continue
        obs_src.copy(key, obs_dst)

    sensor_dst = obs_dst.create_group("sensor_data")
    for camera in cameras:
        cam_src_path = f"observation/sensor_data/{camera}"
        rgb_src_path = f"{cam_src_path}/rgb"
        if rgb_src_path not in src:
            raise KeyError(f"missing {rgb_src_path}")
        cam_dst = sensor_dst.create_group(camera)
        resize_rgb_dataset(src[rgb_src_path], cam_dst, "rgb", image_size, compression)

        cam_src = src[cam_src_path]
        for key in cam_src.keys():
            if key == "rgb":
                continue
            cam_src.copy(key, cam_dst)


def copy_file(src_path, dst_path, cameras, image_size, compression):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src.keys():
            if key == "observation":
                continue
            src.copy(key, dst)

        copy_observation(src, dst, cameras, image_size, compression)

        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["image_cache_size"] = image_size
        dst.attrs["image_cache_cameras"] = ",".join(cameras)


def main():
    parser = argparse.ArgumentParser(
        description="Build resized-image HDF5 cache for training without changing source datasets."
    )
    parser.add_argument("inputs", nargs="+", help="Input HDF5 directories or globs.")
    parser.add_argument("--output-dir", default="datasets/unified_pose_all_128")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--cameras", nargs="+", default=["front_view", "hand_camera"])
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="lzf")
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    paths = expand_inputs(args.inputs, args.output_dir)
    if not paths:
        raise SystemExit("No input HDF5 files found.")

    os.makedirs(args.output_dir, exist_ok=True)
    for src_path in tqdm(paths, desc="Building image cache"):
        dst_path = os.path.join(args.output_dir, os.path.basename(src_path))
        copy_file(src_path, dst_path, args.cameras, args.image_size, compression)

    print(f"Done. files={len(paths)} output={os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
