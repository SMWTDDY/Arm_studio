#!/usr/bin/env python3
import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym
import environments.conveyor_env
import robot.piper.agent
from robot.piper.agent import PiperActionWrapper
from robot.piper.pose_ik import BoundedPiperIK, load_joint_limits, pose_error


DEFAULT_QPOS = np.array(
    [-1.6, 0.55, -0.75, -1.55, 0.1, 2.0, 0.035, 0.035],
    dtype=np.float32,
)


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def pose_to_xyz_rpy(pose):
    p = to_numpy(pose.p).reshape(-1, 3)[0]
    q_wxyz = to_numpy(pose.q).reshape(-1, 4)[0]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    rpy = R.from_quat(q_xyzw).as_euler("xyz")
    return np.concatenate([p, rpy]).astype(np.float64)


def ee_pose_at_base(agent):
    base_link = agent.robot.find_link_by_name("base_link")
    ee_link = agent.robot.find_link_by_name("link6")
    return pose_to_xyz_rpy(base_link.pose.inv() * ee_link.pose)


def robot_qpos(agent):
    return to_numpy(agent.robot.get_qpos()).reshape(-1)[:8].astype(np.float32)


def set_robot_qpos(agent, qpos):
    qpos = np.asarray(qpos, dtype=np.float32)
    agent.robot.set_qpos(qpos)
    agent.robot.set_qvel(np.zeros_like(qpos, dtype=np.float32))


def print_pose(label, pose):
    print(f"{label}: {np.array2string(np.asarray(pose[:6]), precision=6, suppress_small=True)}")


def print_joint_report(label, qpos):
    qpos = np.asarray(qpos[:6], dtype=np.float64)
    limits = load_joint_limits()
    lower_margin = qpos - limits[:, 0]
    upper_margin = limits[:, 1] - qpos
    print(f"{label}: {np.array2string(qpos, precision=6, suppress_small=True)}")
    print(f"{label}_lower_margin: {np.array2string(lower_margin, precision=6, suppress_small=True)}")
    print(f"{label}_upper_margin: {np.array2string(upper_margin, precision=6, suppress_small=True)}")
    near = np.where(np.minimum(lower_margin, upper_margin) < 0.02)[0]
    if len(near) > 0:
        names = ", ".join(f"joint{i + 1}" for i in near)
        print(f"{label}_near_limit: {names}")


def main():
    parser = argparse.ArgumentParser(
        description="Test whether BoundedPiperIK + pd_joint_pos reaches a desired end-effector pose."
    )
    parser.add_argument("--env-name", default="PiperConveyor-v0")
    parser.add_argument("--target", type=float, nargs=6, help="Absolute target pose in robot base frame: x y z roll pitch yaw")
    parser.add_argument(
        "--delta",
        type=float,
        nargs=6,
        default=[0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
        help="Target offset from the initial EE pose when --target is omitted.",
    )
    parser.add_argument("--start-qpos", type=float, nargs=8, default=DEFAULT_QPOS.tolist())
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--settle", type=float, default=0.0, help="Sleep after each env step, useful with rendering.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--pos-weight", type=float, default=100.0)
    parser.add_argument("--rot-weight", type=float, default=1.0)
    parser.add_argument("--pos-threshold", type=float, default=0.005)
    parser.add_argument("--rot-threshold", type=float, default=0.03)
    args = parser.parse_args()

    env = gym.make(
        args.env_name,
        obs_mode="state",
        control_mode="pd_joint_pos",
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human" if args.render else None,
    )
    env = PiperActionWrapper(env, binary_gripper=True)
    solver = BoundedPiperIK(pos_weight=args.pos_weight, rot_weight=args.rot_weight, max_nfev=300)

    try:
        env.reset()
        agent = env.unwrapped.agent
        set_robot_qpos(agent, np.asarray(args.start_qpos, dtype=np.float32))

        initial_pose = ee_pose_at_base(agent)
        target_pose = np.asarray(args.target if args.target is not None else initial_pose + np.asarray(args.delta), dtype=np.float64)

        ik = solver.solve(target_pose, robot_qpos(agent)[:6])
        sim_action = np.concatenate([ik["qpos"], [1.0]]).astype(np.float32)

        print_pose("initial_ee_pose_base", initial_pose)
        print_pose("target_ee_pose_base ", target_pose)
        print_joint_report("start_qpos", robot_qpos(agent)[:6])
        print_joint_report("ik_qpos", ik["qpos"])
        print(f"ik_error: pos={ik['pos_error']:.6f}m rot={ik['rot_error']:.6f}rad status={ik['status']}")

        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(sim_action)
            if args.render:
                env.render()
            if args.settle > 0:
                time.sleep(args.settle)

        final_pose = ee_pose_at_base(agent)
        final_qpos = robot_qpos(agent)[:6]
        err = pose_error(final_pose, target_pose)
        pos_norm = float(np.linalg.norm(err[:3]))
        rot_norm = float(np.linalg.norm(err[3:6]))

        print_pose("final_ee_pose_base  ", final_pose)
        print_joint_report("final_qpos", final_qpos)
        print_pose("final_minus_target ", err)
        print(f"tracking_error: pos={pos_norm:.6f}m rot={rot_norm:.6f}rad")

        if pos_norm <= args.pos_threshold and rot_norm <= args.rot_threshold:
            print("PASS: final EE pose is close to target.")
            return 0

        print("FAIL: final EE pose is not close enough to target.")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
