#!/usr/bin/env python3
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import environments.conveyor_env
import robot.piper.agent
from robot.piper.pose_ik import BoundedPiperIK
from robot.piper.agent import PiperActionWrapper


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def unwrap_env(env):
    return env.unwrapped


def make_env(args):
    env = gym.make(
        args.env_name,
        obs_mode="state",
        control_mode="pd_ee_pose",
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human" if args.render else None,
    )
    return PiperActionWrapper(env, binary_gripper=args.binary_gripper)


def get_arm_controller(env):
    controller = unwrap_env(env).agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers["arm"]
    return controller


def reset_robot_to_qpos(env, qpos):
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    if qpos.shape[0] == 6:
        qpos = np.concatenate([qpos, [0.0, 0.0]]).astype(np.float32)
    unwrap_env(env).agent.robot.set_qpos(qpos)
    unwrap_env(env).agent.robot.set_qvel(np.zeros_like(qpos, dtype=np.float32))


def solve_pose_action(env, action7):
    action7 = np.asarray(action7, dtype=np.float32)
    full_action = env.action(action7)
    full_action = np.asarray(full_action, dtype=np.float32)
    if full_action.ndim == 1:
        full_action = full_action[None, :]
    action_tensor = torch.from_numpy(full_action).to(unwrap_env(env).device)
    unwrap_env(env).agent.set_action(action_tensor)
    target_qpos = to_numpy(get_arm_controller(env)._target_qpos).reshape(-1)[:6]
    return target_qpos, full_action.reshape(-1)


def joint_error(target_qpos, reference_qpos):
    target_qpos = np.asarray(target_qpos, dtype=np.float64)[:6]
    reference_qpos = np.asarray(reference_qpos, dtype=np.float64)[:6]
    return (target_qpos - reference_qpos + np.pi) % (2 * np.pi) - np.pi


def pose_error(a, b):
    err = np.asarray(a[:6], dtype=np.float64) - np.asarray(b[:6], dtype=np.float64)
    err[3:6] = (err[3:6] + np.pi) % (2 * np.pi) - np.pi
    return err


def sim_current_ee_pose(env):
    pose = get_arm_controller(env).ee_pose_at_base
    p = to_numpy(pose.p).reshape(-1, 3)[0]
    q_wxyz = to_numpy(pose.q).reshape(-1, 4)[0]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    rpy = R.from_quat(q_xyzw).as_euler("xyz")
    return np.concatenate([p, rpy]).astype(np.float64)


def print_frame(
    frame,
    pose_action,
    reference_qpos,
    target_qpos,
    full_action,
    sim_pose=None,
    bounded_ik=None,
    after_qpos=None,
):
    err = joint_error(target_qpos, reference_qpos)
    print(f"\nframe={frame}")
    print("  pose_action xyz/rpy:", np.array2string(np.asarray(pose_action[:6]), precision=5, suppress_small=True))
    if sim_pose is not None:
        perr = pose_error(sim_pose, pose_action)
        print("  sim_ee_pose xyz/rpy:", np.array2string(sim_pose, precision=5, suppress_small=True))
        print(
            "  sim_pose_minus_action:",
            np.array2string(perr, precision=5, suppress_small=True),
            f"| pos_norm={np.linalg.norm(perr[:3]):.6f}m rot_norm={np.linalg.norm(perr[3:6]):.6f}rad",
        )
    print("  full_env_action:", np.array2string(full_action, precision=5, suppress_small=True))
    print("  reference_qpos:", np.array2string(np.asarray(reference_qpos[:6]), precision=5, suppress_small=True))
    print("  ik_target_qpos:", np.array2string(np.asarray(target_qpos[:6]), precision=5, suppress_small=True))
    print("  error:", np.array2string(err, precision=5, suppress_small=True))
    print(
        "  abs_error:",
        " ".join(f"{name}={abs(v):.5f}" for name, v in zip(JOINT_NAMES, err)),
        f"| max={np.max(np.abs(err)):.5f} rad",
    )
    if bounded_ik is not None:
        bounded_qpos = bounded_ik["qpos"]
        bounded_err = joint_error(bounded_qpos, reference_qpos)
        print("  bounded_ik_qpos:", np.array2string(bounded_qpos, precision=5, suppress_small=True))
        print(
            "  bounded_ik_abs_error:",
            " ".join(f"{name}={abs(v):.5f}" for name, v in zip(JOINT_NAMES, bounded_err)),
            f"| max={np.max(np.abs(bounded_err)):.5f} rad "
            f"| pos={bounded_ik['pos_error']:.6f}m rot={bounded_ik['rot_error']:.6f}rad "
            f"| status={bounded_ik['status']}",
        )
    if after_qpos is not None:
        after_err = joint_error(after_qpos, reference_qpos)
        print("  qpos_after_step:", np.array2string(np.asarray(after_qpos[:6]), precision=5, suppress_small=True))
        print(
            "  after_step_abs_error:",
            " ".join(f"{name}={abs(v):.5f}" for name, v in zip(JOINT_NAMES, after_err)),
            f"| max={np.max(np.abs(after_err)):.5f} rad",
        )


def load_dataset(path):
    with h5py.File(path, "r") as f:
        actions = np.asarray(f["action"], dtype=np.float32)
        state = np.asarray(f["observation/state"], dtype=np.float32)
        if state.ndim == 3:
            state = state[:, 0, :]
        attrs = dict(f.attrs)
    return actions, state, attrs


def iter_indices(n, start, count, stride, frames):
    if frames:
        for frame in frames:
            if 0 <= frame < n:
                yield frame
        return
    end = n if count is None else min(n, start + count * stride)
    for frame in range(start, end, stride):
        yield frame


def compare_dataset(args):
    actions, state, attrs = load_dataset(args.dataset)
    print(f"dataset={args.dataset}")
    print(f"frames={len(actions)} control_mode={attrs.get('control_mode')} action_shape={actions.shape}")

    env = make_env(args)
    bounded_solver = BoundedPiperIK()
    try:
        env.reset()
        errors = []
        after_errors = []
        for frame in iter_indices(len(actions), args.start, args.count, args.stride, args.frames):
            reference_qpos = state[frame, :6]
            reset_robot_to_qpos(env, state[frame, :8])
            sim_pose = sim_current_ee_pose(env)
            target_qpos, full_action = solve_pose_action(env, actions[frame])
            bounded_ik = bounded_solver.solve(actions[frame], reference_qpos)
            errors.append(joint_error(target_qpos, reference_qpos))

            after_qpos = None
            if args.step:
                env.step(actions[frame])
                after_qpos = to_numpy(unwrap_env(env).agent.robot.get_qpos()).reshape(-1)[:6]
                after_errors.append(joint_error(after_qpos, reference_qpos))

            if args.verbose:
                print_frame(frame, actions[frame], reference_qpos, target_qpos, full_action, sim_pose, bounded_ik, after_qpos)

        errors = np.asarray(errors)
        abs_errors = np.abs(errors)
        print("\nIK target qpos error summary")
        for i, name in enumerate(JOINT_NAMES):
            print(
                f"  {name}: mean={abs_errors[:, i].mean():.6f} "
                f"p95={np.percentile(abs_errors[:, i], 95):.6f} "
                f"max={abs_errors[:, i].max():.6f} rad"
            )
        print(f"  overall_max={abs_errors.max():.6f} rad")

        bad = np.where(abs_errors.max(axis=1) > args.threshold)[0]
        if len(bad) > 0:
            print(f"\nWARN: {len(bad)}/{len(errors)} checked frames exceed {args.threshold} rad")

        if after_errors:
            after_abs = np.abs(np.asarray(after_errors))
            print("\nqpos after one env.step error summary")
            for i, name in enumerate(JOINT_NAMES):
                print(
                    f"  {name}: mean={after_abs[:, i].mean():.6f} "
                    f"p95={np.percentile(after_abs[:, i], 95):.6f} "
                    f"max={after_abs[:, i].max():.6f} rad"
                )
            print(f"  overall_max={after_abs.max():.6f} rad")
    finally:
        env.close()


def compare_real(args):
    from teleop.real_to_sim import RealToSimTeleop

    env = make_env(args)
    bounded_solver = BoundedPiperIK()
    teleop = RealToSimTeleop(can_name=args.can_name)
    try:
        env.reset()
        time.sleep(args.warmup)
        for frame in range(args.count):
            joint_action = teleop.get_action(mode="joint", use_binary_gripper=args.binary_gripper)
            pose_action = teleop.get_action(mode="pose", use_binary_gripper=args.binary_gripper)
            reset_robot_to_qpos(env, joint_action[:6])
            sim_pose = sim_current_ee_pose(env)
            target_qpos, full_action = solve_pose_action(env, pose_action)
            bounded_ik = bounded_solver.solve(pose_action, joint_action[:6])
            print_frame(frame, pose_action, joint_action[:6], target_qpos, full_action, sim_pose, bounded_ik)
            time.sleep(args.period)
    finally:
        if hasattr(teleop, "close"):
            teleop.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare ManiSkill pd_ee_pose IK target qpos against direct joint qpos."
    )
    parser.add_argument("--dataset", help="Pose HDF5 dataset to replay.")
    parser.add_argument("--real", action="store_true", help="Read live real arm instead of a dataset.")
    parser.add_argument("--env-name", default="PiperConveyor-v0")
    parser.add_argument("--binary-gripper", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--frames", type=int, nargs="*", help="Exact dataset frame indices to check.")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--step", action="store_true", help="Also run one env.step and compare resulting qpos.")
    parser.add_argument("--verbose", action="store_true", help="Print every checked frame.")
    parser.add_argument("--can-name", default="can_master")
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--period", type=float, default=0.2)
    args = parser.parse_args()

    if args.real == bool(args.dataset):
        parser.error("Choose exactly one source: --dataset PATH or --real")

    if args.real:
        compare_real(args)
    else:
        compare_dataset(args)


if __name__ == "__main__":
    main()
