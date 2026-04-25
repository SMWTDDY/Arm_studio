#!/usr/bin/env python3
import argparse
import glob
import os

import cv2
import h5py
import numpy as np


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return list(dict.fromkeys(paths))


def read_rgb(f, camera):
    data = np.asarray(f[f"observation/sensor_data/{camera}/rgb"])
    if data.ndim == 5:
        data = data[:, 0]
    return data.astype(np.uint8)


def resize_to_height(img, height):
    if img.shape[0] == height:
        return img
    scale = height / img.shape[0]
    width = int(round(img.shape[1] * scale))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def export_video(h5_path, output_path, cameras, fps, max_frames=None):
    with h5py.File(h5_path, "r") as f:
        streams = [read_rgb(f, camera) for camera in cameras]
        total = min(len(stream) for stream in streams)
        if max_frames:
            total = min(total, max_frames)
        if total <= 0:
            raise ValueError(f"{h5_path}: no frames")

        first = []
        target_h = min(streams[i][0].shape[0] for i in range(len(streams)))
        for stream in streams:
            first.append(resize_to_height(stream[0], target_h))
        width = sum(img.shape[1] for img in first)
        height = target_h

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        for frame_idx in range(total):
            frames = []
            for camera, stream in zip(cameras, streams):
                img = resize_to_height(stream[frame_idx], height)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    camera,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                frames.append(bgr)
            combined = np.hstack(frames)
            cv2.putText(
                combined,
                f"{os.path.basename(h5_path)} frame={frame_idx}/{total - 1}",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            writer.write(combined)
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Export side-by-side camera videos from HDF5 trajectories.")
    parser.add_argument("paths", nargs="+", help="HDF5 files or globs.")
    parser.add_argument("--output-dir", default="outputs/dataset_videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", default=["front_view", "hand_camera"])
    args = parser.parse_args()

    for h5_path in expand_paths(args.paths):
        if not os.path.exists(h5_path):
            print(f"[Skip] missing: {h5_path}")
            continue
        base = os.path.splitext(os.path.basename(h5_path))[0]
        output_path = os.path.join(args.output_dir, f"{base}.mp4")
        export_video(
            h5_path,
            output_path,
            args.cameras,
            args.fps,
            max_frames=args.max_frames or None,
        )
        print(f"[OK] {h5_path} -> {output_path}")


if __name__ == "__main__":
    main()
