#!/usr/bin/env python3
"""Estimate Piper wrist-camera optical pose from chessboard images.

Input:
  - a CSV with one row per image:
      image,x,y,z,roll,pitch,yaw
    where pose is base_T_link6 / base_T_gripper in meters and radians.
  - chessboard images that contain the printed board.

Output:
  - flange_T_camera_optical: OpenCV optical frame, x right, y down, z forward.
  - flange_T_sapien_camera: SAPIEN camera frame, x forward, y left, z up.
  - urdf_link6_T_sapien_camera when joint columns are present, ready to copy
    into the URDF fixed joint for ManiSkill/SAPIEN mounted cameras.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate Piper eye-in-hand camera pose with a chessboard."
    )
    parser.add_argument(
        "--samples-csv",
        required=True,
        help="CSV with columns: image,x,y,z,roll,pitch,yaw",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Base directory for relative image paths. Defaults to the CSV directory.",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="Internal chessboard corner count as COLSxROWS, e.g. 9x6.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Chessboard square size in meters, e.g. 0.025.",
    )
    parser.add_argument(
        "--output",
        default="outputs/camera_calibration/hand_eye_chessboard_result.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--urdf-path",
        default="robot/piper/piper_assets/urdf/piper_description_with_camera_right.urdf",
        help=(
            "URDF used by simulation. If CSV includes j1..j6, also convert "
            "flange_T_camera into URDF link6_T_camera."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="park",
        help="OpenCV hand-eye calibration method.",
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Optional directory to save detected-corner preview images.",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "classic", "sb"],
        default="auto",
        help="Chessboard detector. auto tries classic first, then OpenCV SB.",
    )
    return parser.parse_args()


def _parse_pattern(pattern: str) -> Tuple[int, int]:
    parts = pattern.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError("--pattern must look like COLSxROWS, e.g. 9x6")
    cols, rows = int(parts[0]), int(parts[1])
    if cols <= 1 or rows <= 1:
        raise ValueError("Chessboard internal corner counts must be > 1")
    return cols, rows


def _pose_xyz_rpy_to_matrix(values: List[float]) -> np.ndarray:
    if len(values) != 6:
        raise ValueError(f"Expected 6 pose values, got {len(values)}")
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = np.asarray(values[:3], dtype=np.float64)
    mat[:3, :3] = R.from_euler("xyz", values[3:6]).as_matrix()
    return mat


def _read_samples(csv_path: Path, image_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "x", "y", "z", "roll", "pitch", "yaw"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        for row in reader:
            image = Path(row["image"])
            if not image.is_absolute():
                image = image_dir / image
            pose = [
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["roll"]),
                float(row["pitch"]),
                float(row["yaw"]),
            ]
            sample = {"image": image, "base_T_gripper": _pose_xyz_rpy_to_matrix(pose)}
            joint_keys = [f"j{i}" for i in range(1, 7)]
            if all(k in row and row[k] != "" for k in joint_keys):
                sample["joints"] = [float(row[k]) for k in joint_keys]
            rows.append(sample)
    if len(rows) < 6:
        raise ValueError("Need at least 6 samples; 12-20 diverse poses are recommended.")
    return rows


def _make_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square_size)
    return points


def _find_chessboard_corners(
    gray: np.ndarray,
    cols: int,
    rows: int,
    detector: str,
    criteria,
):
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    methods = [detector] if detector != "auto" else ["classic", "sb"]

    for method in methods:
        if method == "classic":
            ok, corners = cv2.findChessboardCorners(gray, (cols, rows), classic_flags)
            if ok:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return True, corners, method
            continue

        if method == "sb" and hasattr(cv2, "findChessboardCornersSB"):
            sb_flags = (
                cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_EXHAUSTIVE
                + cv2.CALIB_CB_ACCURACY
            )
            ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), sb_flags)
            if ok:
                corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
                return True, corners, method

    return False, None, None


def _detect_chessboards(
    samples: List[Dict[str, object]],
    cols: int,
    rows: int,
    square_size: float,
    preview_dir: Path | None,
    detector: str,
):
    objp = _make_object_points(cols, rows, square_size)
    object_points = []
    image_points = []
    valid_samples = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    detector_counts: Dict[str, int] = {}

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        image_path = Path(sample["image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[Skip] Cannot read image: {image_path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        ok, corners, detector_used = _find_chessboard_corners(
            gray, cols, rows, detector, criteria
        )
        if not ok:
            print(f"[Skip] Chessboard not found: {image_path}")
            continue
        detector_counts[detector_used] = detector_counts.get(detector_used, 0) + 1
        object_points.append(objp.copy())
        image_points.append(corners)
        valid_samples.append(sample)

        if preview_dir is not None:
            vis = image.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, ok)
            out = preview_dir / f"{image_path.stem}_corners.jpg"
            cv2.imwrite(str(out), vis)

    if len(valid_samples) < 6:
        raise RuntimeError(f"Only {len(valid_samples)} valid chessboard samples; need at least 6.")

    print(
        "[OK] Chessboard detections:",
        ", ".join(f"{k}={v}" for k, v in sorted(detector_counts.items())),
    )
    return object_points, image_points, valid_samples, image_size


def _method_id(name: str) -> int:
    return {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }[name]


def _matrix_to_xyz_rpy(mat: np.ndarray) -> Tuple[List[float], List[float]]:
    xyz = mat[:3, 3].astype(float).tolist()
    rpy = R.from_matrix(mat[:3, :3]).as_euler("xyz").astype(float).tolist()
    return xyz, rpy


def _invert_matrix(mat: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = mat[:3, :3].T
    inv[:3, 3] = -(inv[:3, :3] @ mat[:3, 3])
    return inv


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = xyz
    mat[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    return mat


def _parse_xyz_rpy(origin) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    return xyz, rpy


def _urdf_joint_transform(joint, q: float) -> np.ndarray:
    xyz, rpy = _parse_xyz_rpy(joint.find("origin"))
    mat = _transform_from_xyz_rpy(xyz, rpy)
    if joint.attrib.get("type") == "revolute":
        axis_el = joint.find("axis")
        axis = (
            np.fromstring(axis_el.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
            if axis_el is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        motion = np.eye(4, dtype=np.float64)
        motion[:3, :3] = R.from_rotvec(axis * float(q)).as_matrix()
        mat = mat @ motion
    return mat


def _urdf_base_T_link6(joints: List[float], urdf_path: Path) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    urdf_joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    mat = np.eye(4, dtype=np.float64)
    for idx, q in enumerate(joints[:6], start=1):
        name = f"joint{idx}"
        if name not in urdf_joints:
            raise ValueError(f"URDF missing {name}; cannot compute link6 correction")
        mat = mat @ _urdf_joint_transform(urdf_joints[name], float(q))
    return mat


def _mean_transform(mats: List[np.ndarray]) -> np.ndarray:
    mean = np.eye(4, dtype=np.float64)
    mean[:3, 3] = np.mean([mat[:3, 3] for mat in mats], axis=0)
    mean[:3, :3] = R.from_matrix([mat[:3, :3] for mat in mats]).mean().as_matrix()
    return mean


def _transform_scatter(mats: List[np.ndarray], mean: np.ndarray) -> Dict[str, float]:
    mean_rot = R.from_matrix(mean[:3, :3])
    trans_err = [float(np.linalg.norm(mat[:3, 3] - mean[:3, 3])) for mat in mats]
    rot_err = [
        float(np.linalg.norm((mean_rot.inv() * R.from_matrix(mat[:3, :3])).as_rotvec()))
        for mat in mats
    ]
    return {
        "translation_mean_m": float(np.mean(trans_err)),
        "translation_max_m": float(np.max(trans_err)),
        "rotation_mean_rad": float(np.mean(rot_err)),
        "rotation_max_rad": float(np.max(rot_err)),
    }


def _opencv_optical_to_sapien_camera(optical_T: np.ndarray) -> np.ndarray:
    """Convert OpenCV optical camera pose to SAPIEN camera pose.

    OpenCV optical: x right, y down, z forward.
    SAPIEN camera: x forward, y left, z up.
    """
    optical_T_sapien = np.eye(4, dtype=np.float64)
    optical_T_sapien[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return optical_T @ optical_T_sapien


def _urdf_origin(xyz: List[float], rpy: List[float]) -> str:
    return (
        f'<origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" '
        f'rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}" />'
    )


def _origin_result(mat: np.ndarray) -> Dict[str, object]:
    xyz, rpy = _matrix_to_xyz_rpy(mat)
    return {
        "xyz": xyz,
        "rpy_xyz": rpy,
        "matrix": mat.tolist(),
        "urdf_origin": _urdf_origin(xyz, rpy),
    }


def main():
    args = _parse_args()
    samples_csv = Path(args.samples_csv).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else samples_csv.parent
    urdf_path = Path(args.urdf_path).resolve()
    cols, rows = _parse_pattern(args.pattern)
    preview_dir = Path(args.preview_dir).resolve() if args.preview_dir else None

    samples = _read_samples(samples_csv, image_dir)
    object_points, image_points, valid_samples, image_size = _detect_chessboards(
        samples, cols, rows, args.square_size, preview_dir, args.detector
    )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    R_gripper2base = []
    t_gripper2base = []
    for sample in valid_samples:
        base_T_gripper = sample["base_T_gripper"]
        R_gripper2base.append(base_T_gripper[:3, :3])
        t_gripper2base.append(base_T_gripper[:3, 3])

    R_target2cam = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    t_target2cam = [tvec.reshape(3) for tvec in tvecs]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=_method_id(args.method),
    )

    gripper_T_camera = np.eye(4, dtype=np.float64)
    gripper_T_camera[:3, :3] = R_cam2gripper
    gripper_T_camera[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    optical_xyz, optical_rpy = _matrix_to_xyz_rpy(gripper_T_camera)
    # OpenCV 标定得到的是 optical camera frame；SAPIEN 相机坐标轴定义不同，
    # 所以必须先转成 SAPIEN camera frame，不能把 optical 结果直接写进仿真。
    gripper_T_sapien_camera = _opencv_optical_to_sapien_camera(gripper_T_camera)
    sapien_xyz, sapien_rpy = _matrix_to_xyz_rpy(gripper_T_sapien_camera)
    urdf_link6_T_flange = None
    urdf_link6_T_sapien_camera = None
    urdf_scatter = None
    if urdf_path.exists() and all("joints" in sample for sample in valid_samples):
        corrections = []
        for sample in valid_samples:
            base_T_urdf_link6 = _urdf_base_T_link6(sample["joints"], urdf_path)
            base_T_flange = sample["base_T_gripper"]
            corrections.append(_invert_matrix(base_T_urdf_link6) @ base_T_flange)
        urdf_link6_T_flange = _mean_transform(corrections)
        urdf_scatter = _transform_scatter(corrections, urdf_link6_T_flange)
        # CSV 里的 flange pose 来自真实 SDK；URDF 的 link6 坐标系与 SDK 法兰不完全等价。
        # 这里先估计 urdf_link6_T_flange，再得到最终可写入 hand_camera_joint 的 link6_T_camera。
        urdf_link6_T_sapien_camera = urdf_link6_T_flange @ gripper_T_sapien_camera
    warnings = []
    if float(rms) > 3.0:
        warnings.append(
            "High camera calibration RMS; recapture sharper, more diverse chessboard images "
            "or provide trusted camera intrinsics."
        )
    if float(np.linalg.norm(gripper_T_camera[:3, 3])) > 0.5:
        warnings.append(
            "Estimated camera translation is larger than 0.5m from flange; "
            "this is unlikely for a wrist camera and the result should not be copied to URDF."
        )
    if urdf_link6_T_sapien_camera is None:
        warnings.append(
            "CSV has no complete j1..j6 columns or URDF path is unavailable; "
            "cannot convert flange_T_camera into URDF link6_T_camera."
        )

    result = {
        "note": (
            "OpenCV hand-eye returns flange_T_camera_optical because the CSV pose "
            "is the SDK/pose-mode flange frame. For ManiSkill/SAPIEN mounted "
            "cameras, copy urdf_link6_T_sapien_camera.urdf_origin into the "
            "hand_cam fixed joint when it is available."
        ),
        "samples_csv": str(samples_csv),
        "valid_sample_count": len(valid_samples),
        "chessboard": {
            "internal_corners": [cols, rows],
            "square_size_m": args.square_size,
        },
        "camera_intrinsics": {
            "rms_reprojection_error_px": float(rms),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        },
        "flange_T_camera_optical": {
            "frame": "OpenCV optical: x right, y down, z forward",
            "xyz": optical_xyz,
            "rpy_xyz": optical_rpy,
            "matrix": gripper_T_camera.tolist(),
            "urdf_origin_not_for_sapien_mount": _urdf_origin(optical_xyz, optical_rpy),
        },
        "flange_T_sapien_camera": {
            "frame": "SAPIEN camera: x forward, y left, z up",
            "xyz": sapien_xyz,
            "rpy_xyz": sapien_rpy,
            "matrix": gripper_T_sapien_camera.tolist(),
        },
        "method": args.method,
        "warnings": warnings,
        "valid_images": [str(sample["image"]) for sample in valid_samples],
    }
    if urdf_link6_T_flange is not None and urdf_link6_T_sapien_camera is not None:
        result["urdf_link6_T_flange"] = {
            **_origin_result(urdf_link6_T_flange),
            "scatter": urdf_scatter,
        }
        result["urdf_link6_T_sapien_camera"] = {
            **_origin_result(urdf_link6_T_sapien_camera),
            "frame": "SAPIEN camera: x forward, y left, z up",
            "source": "urdf_link6_T_flange @ flange_T_sapien_camera",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(f"[OK] Valid samples: {len(valid_samples)}")
    print(f"[OK] Camera calibration RMS: {rms:.4f}px")
    print("[OK] flange_T_camera_optical (OpenCV; do not copy directly to SAPIEN mount):")
    print("  xyz =", " ".join(f"{v:.9f}" for v in optical_xyz))
    print("  rpy =", " ".join(f"{v:.9f}" for v in optical_rpy))
    print("[OK] flange_T_sapien_camera:")
    print("  xyz =", " ".join(f"{v:.9f}" for v in sapien_xyz))
    print("  rpy =", " ".join(f"{v:.9f}" for v in sapien_rpy))
    if urdf_link6_T_sapien_camera is not None:
        urdf_xyz, urdf_rpy = _matrix_to_xyz_rpy(urdf_link6_T_sapien_camera)
        print("[OK] urdf_link6_T_sapien_camera (copy this URDF origin):")
        print("  xyz =", " ".join(f"{v:.9f}" for v in urdf_xyz))
        print("  rpy =", " ".join(f"{v:.9f}" for v in urdf_rpy))
        print("  URDF:", result["urdf_link6_T_sapien_camera"]["urdf_origin"])
    for warning in warnings:
        print(f"[WARN] {warning}")
    print(f"[OK] Wrote {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
