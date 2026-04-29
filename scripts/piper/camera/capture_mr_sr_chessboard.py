#!/usr/bin/env python3
"""Capture right wrist-camera chessboard samples while teleoperating SR with MR.

Default setup:
  - leader arm: can_mr
  - follower arm: can_sr
  - wrist camera on SR: CH2592100GJ

Press:
  s: save current camera frame and follower pose to samples.csv
  q / esc: quit
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_PYAGXARM_ROOT = REPO_ROOT / "pyAgxArm"
if (LOCAL_PYAGXARM_ROOT / "pyAgxArm" / "__init__.py").exists():
    sys.path.insert(0, str(LOCAL_PYAGXARM_ROOT))

from agent_infra.Piper_Env.Camera.orbbec_camera import OrbbecCamera
from agent_infra.Piper_Env.Env.utils.piper_arm import PiperArm


class _TerminalKeyReader:
    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdin.isatty()
        self._old_settings = None

    def __enter__(self):
        if self.enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def read_key(self, timeout: float = 0.03) -> int:
        if not self.enabled:
            time.sleep(timeout)
            return 255
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return 255
        return ord(sys.stdin.read(1))


def _merge_keys(*keys: int) -> int:
    for key in keys:
        if key != 255:
            return key
    return 255


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Use MR leader pose + SR wrist camera to capture chessboard calibration samples."
    )
    parser.add_argument(
        "--master-can",
        "--can-name",
        dest="master_can",
        default="can_mr",
        help="MR leader CAN interface.",
    )
    parser.add_argument("--follower-can", default="can_sr", help="SR follower CAN interface.")
    parser.add_argument("--camera-serial", default="CH2592100GJ", help="SR wrist camera serial.")
    parser.add_argument("--robot-model", default="piper")
    parser.add_argument("--firmware-version", default="v183")
    parser.add_argument(
        "--joint-stream-command",
        default="auto",
        choices=["auto", "move_js", "move_j"],
        help="Follower joint command used for continuous teleop.",
    )
    parser.add_argument(
        "--disable-teleop",
        action="store_true",
        help="Only capture camera and follower pose; assume another process is already moving SR.",
    )
    parser.add_argument("--backend", default="v4l2", choices=["v4l2", "sdk", "auto"])
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rotate-degrees", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument(
        "--output-dir",
        default="outputs/camera_calibration/mr_sr_hand_eye",
        help="Directory for images and samples.csv.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not open an OpenCV preview window; keyboard still reads from terminal.",
    )
    parser.add_argument(
        "--auto-save-interval",
        type=float,
        default=0.0,
        help="Automatically save one frame every N seconds. 0 disables auto save.",
    )
    parser.add_argument(
        "--max-auto-saves",
        type=int,
        default=0,
        help="Maximum automatic saves; 0 means unlimited when --auto-save-interval is enabled.",
    )
    return parser.parse_args()


def _rotate_rgb(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _read_master_state(arm: PiperArm) -> dict:
    state = arm.get_master_state()
    joints = np.asarray(state.get("joint_pos", []), dtype=np.float32).reshape(-1)
    if joints.shape[0] < 6 or not state.get("has_leader_joint", False):
        raise RuntimeError("MR leader joint read is not ready.")
    return state


def _read_follower_state(arm: PiperArm) -> dict:
    state = arm.get_state()
    joints = np.asarray(state.get("joint_pos", []), dtype=np.float32).reshape(-1)
    pose = np.asarray(state.get("ee_pose", []), dtype=np.float32).reshape(-1)
    if joints.shape[0] < 6 or pose.shape[0] < 6:
        raise RuntimeError("SR follower state read is not ready.")
    return state


def _read_states(arm: PiperArm, disable_teleop: bool) -> tuple[dict | None, dict | None, str | None]:
    """Read MR/SR states; MR failure should not block saving SR calibration pose."""
    master_state = None
    master_error = None

    if not disable_teleop:
        try:
            master_state = _read_master_state(arm)
            gripper = float(
                np.asarray(master_state["gripper_pos"], dtype=np.float32).reshape(-1)[0]
            )
            arm.apply_action(master_state["joint_pos"][:6], gripper, mode="joint")
        except Exception as exc:
            master_error = f"MR not ready: {exc}"

    follower_state = _read_follower_state(arm)
    return master_state, follower_state, master_error


def _append_sample(csv_path: Path, row: dict):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "j1",
                "j2",
                "j3",
                "j4",
                "j5",
                "j6",
                "master_j1",
                "master_j2",
                "master_j3",
                "master_j4",
                "master_j5",
                "master_j6",
                "master_x",
                "master_y",
                "master_z",
                "master_roll",
                "master_pitch",
                "master_yaw",
                "timestamp",
                "camera_serial",
                "master_can",
                "follower_can",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "samples.csv"

    arm = None
    camera = None
    sample_idx = 0
    auto_save_count = 0
    last_auto_save_time = 0.0
    key_reader_context = _TerminalKeyReader(True)
    if not sys.stdin.isatty():
        print(
            "[Warn] 当前 stdin 不是交互终端，终端按键可能不可用；"
            "建议用 conda run --no-capture-output 或直接激活 SL 后运行 python。"
        )

    try:
        print(f"[Arm] Connecting MR->SR: {args.master_can} -> {args.follower_can}")
        arm = PiperArm(
            master_can=args.master_can,
            follower_can=args.follower_can,
            name="mr_sr_capture",
            robot_model=args.robot_model,
            firmware_version=args.firmware_version,
            joint_stream_command=args.joint_stream_command,
        )
        arm.connect()
        if arm.master is not None:
            try:
                arm.master.enable()
            except Exception as exc:
                print(f"[Warn] MR enable failed, continuing: {exc}")
            for _ in range(3):
                try:
                    arm.master.set_leader_mode()
                except Exception as exc:
                    print(f"[Warn] MR set_leader_mode failed, continuing: {exc}")
                    break
                time.sleep(0.05)
        if arm.follower is not None:
            arm.follower.enable()

        print(f"[Camera] Starting SR wrist camera: {args.camera_serial}")
        camera = OrbbecCamera(
            serial_number=args.camera_serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend=args.backend,
            fourcc=args.fourcc,
            auto_exposure=True,
            enable_depth=False,
        )

        print("\n操作：拖动 MR 让 SR 手眼相机看到棋盘格；按 s 保存；按 q 或 ESC 退出。")
        print("按键可以在预览窗口里按，也可以直接在当前终端里按。")
        if args.auto_save_interval > 0.0:
            print(f"自动保存: 每 {args.auto_save_interval:.2f}s 保存一张。")
        print(f"输出目录: {output_dir.resolve()}")
        print(f"照片目录: {image_dir.resolve()}")
        print(f"CSV: {csv_path.resolve()}")

        with key_reader_context as key_reader:
            while True:
                color_rgb, _ = camera.get_frames()
                if color_rgb is None:
                    time.sleep(0.03)
                    continue

                color_rgb = _rotate_rgb(color_rgb, args.rotate_degrees)
                clean_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                preview_bgr = clean_bgr.copy()

                try:
                    master_state, follower_state, master_error = _read_states(
                        arm, disable_teleop=args.disable_teleop
                    )
                    joints = np.asarray(follower_state["joint_pos"], dtype=np.float32).reshape(-1)[:6]
                    pose = np.asarray(follower_state["ee_pose"], dtype=np.float32).reshape(-1)[:6]
                    status = (
                        f"SR x={pose[0]:.3f} y={pose[1]:.3f} z={pose[2]:.3f} "
                        f"r={pose[3]:.2f} p={pose[4]:.2f} y={pose[5]:.2f}"
                    )
                    if master_error:
                        status = f"{status} | {master_error}"
                except Exception as exc:
                    master_state = None
                    follower_state = None
                    joints = None
                    pose = None
                    status = f"pose read failed: {exc}"

                cv2.putText(
                    preview_bgr,
                    f"s:save q:quit | {status}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                if not args.no_preview:
                    cv2.imshow("MR->SR chessboard capture", preview_bgr)
                    preview_key = cv2.waitKey(10) & 0xFF
                    terminal_key = key_reader.read_key(timeout=0.0)
                    key = _merge_keys(preview_key, terminal_key)
                else:
                    key = key_reader.read_key()

                if key in (ord("q"), ord("Q"), 27):
                    break

                now = time.monotonic()
                auto_save_due = (
                    args.auto_save_interval > 0.0
                    and now - last_auto_save_time >= args.auto_save_interval
                    and (args.max_auto_saves <= 0 or auto_save_count < args.max_auto_saves)
                )
                manual_save = key in (ord("s"), ord("S"))
                if not manual_save and not auto_save_due:
                    continue
                if auto_save_due:
                    last_auto_save_time = now
                    auto_save_count += 1

                timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_name = f"sample_{sample_idx:04d}_{timestamp}.jpg"
                image_path = image_dir / image_name
                if not cv2.imwrite(str(image_path), clean_bgr):
                    print(f"[Error] 写入图片失败: {image_path}")
                    continue

                if joints is None or pose is None:
                    sample_idx += 1
                    print(
                        f"[SavedImageOnly] {image_path.resolve()} | "
                        f"未写 CSV，因为没有有效 SR 位姿: {status}"
                    )
                    continue

                rel_image = image_path.relative_to(output_dir)
                master_joints = np.zeros(6, dtype=np.float32)
                master_pose = np.zeros(6, dtype=np.float32)
                if master_state is not None:
                    master_joints = np.asarray(
                        master_state.get("joint_pos", master_joints), dtype=np.float32
                    ).reshape(-1)[:6]
                    master_pose = np.asarray(
                        master_state.get("ee_pose", master_pose), dtype=np.float32
                    ).reshape(-1)[:6]

                row = {
                    "image": rel_image.as_posix(),
                    "x": f"{pose[0]:.9f}",
                    "y": f"{pose[1]:.9f}",
                    "z": f"{pose[2]:.9f}",
                    "roll": f"{pose[3]:.9f}",
                    "pitch": f"{pose[4]:.9f}",
                    "yaw": f"{pose[5]:.9f}",
                    "j1": f"{joints[0]:.9f}",
                    "j2": f"{joints[1]:.9f}",
                    "j3": f"{joints[2]:.9f}",
                    "j4": f"{joints[3]:.9f}",
                    "j5": f"{joints[4]:.9f}",
                    "j6": f"{joints[5]:.9f}",
                    "master_j1": f"{master_joints[0]:.9f}",
                    "master_j2": f"{master_joints[1]:.9f}",
                    "master_j3": f"{master_joints[2]:.9f}",
                    "master_j4": f"{master_joints[3]:.9f}",
                    "master_j5": f"{master_joints[4]:.9f}",
                    "master_j6": f"{master_joints[5]:.9f}",
                    "master_x": f"{master_pose[0]:.9f}",
                    "master_y": f"{master_pose[1]:.9f}",
                    "master_z": f"{master_pose[2]:.9f}",
                    "master_roll": f"{master_pose[3]:.9f}",
                    "master_pitch": f"{master_pose[4]:.9f}",
                    "master_yaw": f"{master_pose[5]:.9f}",
                    "timestamp": timestamp,
                    "camera_serial": args.camera_serial,
                    "master_can": args.master_can,
                    "follower_can": args.follower_can,
                }
                _append_sample(csv_path, row)
                sample_idx += 1
                print(
                    f"[Saved] {image_path} | pose=("
                    f"{row['x']}, {row['y']}, {row['z']}, "
                    f"{row['roll']}, {row['pitch']}, {row['yaw']})"
                )

    finally:
        if camera is not None:
            camera.stop()
        if arm is not None:
            arm.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f"[Done] samples.csv: {csv_path}")


if __name__ == "__main__":
    main()
