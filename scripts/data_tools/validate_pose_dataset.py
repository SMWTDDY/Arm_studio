#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from teleop.get_pose import get_pose


DEFAULT_URDF = os.path.join(
    PROJECT_ROOT, "robot/piper/piper_assets/urdf/piper_description.urdf"
)


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def pose_action_from_qpos(qpos):
    pose = get_pose(np.asarray(qpos, dtype=np.float32)[:6]).astype(np.float64)
    return pose


def pose_error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    err = a[:6] - b[:6]
    err[3:6] = wrap_angle(err[3:6])
    return err


def load_joint_limits(urdf_path):
    fallback = np.array(
        [
            [-2.618, 2.618],
            [0.0, 3.14],
            [-2.967, 0.0],
            [-1.745, 1.745],
            [-1.22, 1.22],
            [-2.0944, 2.0944],
        ],
        dtype=np.float64,
    )
    if not os.path.exists(urdf_path):
        return fallback

    root = ET.parse(urdf_path).getroot()
    limits = []
    for i in range(1, 7):
        joint = root.find(f"./joint[@name='joint{i}']")
        if joint is None:
            return fallback
        limit = joint.find("limit")
        if limit is None:
            return fallback
        limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
    return np.array(limits, dtype=np.float64)


def flatten_state(state):
    state = np.asarray(state)
    if state.ndim == 3:
        state = state[:, 0, :]
    elif state.ndim == 1:
        state = state[None, :]
    return state


def load_dataset(path):
    with h5py.File(path, "r") as f:
        actions = np.asarray(f["action"], dtype=np.float64)
        state = None
        if "observation/state" in f:
            state = flatten_state(f["observation/state"][:])
        attrs = dict(f.attrs)
    return actions, state, attrs


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def validate_fk_alignment(actions, state, max_lag):
    if state is None:
        return None

    n = min(len(actions), len(state))
    target = actions[:n, :6]
    qpos = state[:n, :6]
    fk = np.stack([pose_action_from_qpos(q) for q in qpos], axis=0)

    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            pred = fk[lag:n]
            ref = target[: n - lag]
        else:
            pred = fk[: n + lag]
            ref = target[-lag:n]
        if len(pred) == 0:
            continue
        err = np.stack([pose_error(p, r) for p, r in zip(pred, ref)], axis=0)
        pos = np.linalg.norm(err[:, :3], axis=1)
        rot = np.linalg.norm(err[:, 3:6], axis=1)
        score = np.median(pos) + 0.05 * np.median(rot)
        if best is None or score < best["score"]:
            best = {
                "lag": lag,
                "score": score,
                "pos": summarize(pos),
                "rot": summarize(rot),
            }
    return best


def make_residual(target_pose, pos_weight, rot_weight):
    target_pose = np.asarray(target_pose, dtype=np.float64)

    def residual(q):
        err = pose_error(pose_action_from_qpos(q), target_pose)
        return np.concatenate([err[:3] * pos_weight, err[3:6] * rot_weight])

    return residual


def validate_sequential_ik(
    actions,
    state,
    limits,
    stride,
    max_frames,
    pos_weight,
    rot_weight,
    pos_threshold,
    rot_threshold,
):
    n = len(actions)
    indices = np.arange(0, n, max(1, stride), dtype=int)
    if max_frames is not None:
        indices = indices[:max_frames]
    if len(indices) == 0:
        return None

    if state is not None and len(state) > 0:
        seed = np.clip(state[min(indices[0], len(state) - 1), :6], limits[:, 0], limits[:, 1])
    else:
        seed = np.mean(limits, axis=1)

    pos_errors = []
    rot_errors = []
    joint_steps = []
    failures = []
    prev_solution = seed.copy()

    for idx in indices:
        target = actions[idx, :6]
        res = least_squares(
            make_residual(target, pos_weight, rot_weight),
            prev_solution,
            bounds=(limits[:, 0], limits[:, 1]),
            max_nfev=80,
            diff_step=1e-4,
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
        )
        solved_pose = pose_action_from_qpos(res.x)
        err = pose_error(solved_pose, target)
        pos = float(np.linalg.norm(err[:3]))
        rot = float(np.linalg.norm(err[3:6]))
        step = float(np.linalg.norm(wrap_angle(res.x - prev_solution)))
        pos_errors.append(pos)
        rot_errors.append(rot)
        joint_steps.append(step)
        # least_squares may report status=0 when it reaches max_nfev even though
        # the final pose is already within tolerance. Treat replay validity as a
        # pose-accuracy check; keep solver status in the report for diagnosis.
        if pos > pos_threshold or rot > rot_threshold:
            failures.append(
                {
                    "frame": int(idx),
                    "pos_error": pos,
                    "rot_error": rot,
                    "joint_step": step,
                    "solver_status": int(res.status),
                }
            )
        prev_solution = res.x

    return {
        "frames_checked": int(len(indices)),
        "pos": summarize(pos_errors),
        "rot": summarize(rot_errors),
        "joint_step": summarize(joint_steps),
        "failures": failures,
    }


def print_stats(prefix, stats, unit):
    print(
        f"{prefix}: mean={stats['mean']:.6f}{unit}, "
        f"p95={stats['p95']:.6f}{unit}, max={stats['max']:.6f}{unit}"
    )


def validate_file(path, args, limits):
    actions, state, attrs = load_dataset(path)
    print(f"\n=== {path} ===")
    print(f"frames={len(actions)} action_shape={actions.shape} control_mode={attrs.get('control_mode')}")

    if actions.ndim != 2 or actions.shape[1] < 6:
        print("FAIL: action dataset must have shape [T, >=6]")
        return False

    if attrs.get("control_mode") != "pose":
        print("WARN: file attr control_mode is not 'pose'; FK/IK checks will treat action[:6] as pose anyway.")

    ok = True
    fk = validate_fk_alignment(actions, state, args.max_lag)
    if fk is None:
        print("FK alignment: skipped, observation/state not found")
    else:
        print(f"FK alignment best_lag={fk['lag']} frames")
        print_stats("  position", fk["pos"], " m")
        print_stats("  rotation", fk["rot"], " rad")
        if fk["pos"]["p95"] > args.fk_pos_threshold or fk["rot"]["p95"] > args.fk_rot_threshold:
            print("  FAIL: FK/action mismatch is above threshold")
            ok = False
        else:
            print("  PASS")

    ik = validate_sequential_ik(
        actions,
        state,
        limits,
        args.stride,
        args.max_frames,
        args.pos_weight,
        args.rot_weight,
        args.ik_pos_threshold,
        args.ik_rot_threshold,
    )
    if ik is None:
        print("IK replay: skipped, no frames")
    else:
        print(f"IK replay frames_checked={ik['frames_checked']}")
        print_stats("  position", ik["pos"], " m")
        print_stats("  rotation", ik["rot"], " rad")
        print_stats("  joint_step", ik["joint_step"], " rad")
        if ik["failures"]:
            print(f"  FAIL: {len(ik['failures'])} frames exceeded thresholds")
            for failure in ik["failures"][: args.show_failures]:
                print(
                    "    frame={frame} pos={pos_error:.6f}m rot={rot_error:.6f}rad "
                    "joint_step={joint_step:.6f}rad status={solver_status}".format(**failure)
                )
            ok = False
        else:
            print("  PASS")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Validate whether recorded pose actions match Piper FK and can be replayed by sequential IK."
    )
    parser.add_argument("paths", nargs="+", help="HDF5 file(s) or glob(s), e.g. datasets/*pose*.hdf5")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--max-lag", type=int, default=3, help="Frame lag search for action/state FK alignment.")
    parser.add_argument("--stride", type=int, default=5, help="Check every Nth frame for IK replay.")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum IK frames per file; use 0 for all.")
    parser.add_argument("--fk-pos-threshold", type=float, default=0.02)
    parser.add_argument("--fk-rot-threshold", type=float, default=0.20)
    parser.add_argument("--ik-pos-threshold", type=float, default=0.01)
    parser.add_argument("--ik-rot-threshold", type=float, default=0.10)
    parser.add_argument("--pos-weight", type=float, default=100.0)
    parser.add_argument("--rot-weight", type=float, default=1.0)
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    if args.max_frames == 0:
        args.max_frames = None

    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    paths = list(dict.fromkeys(paths))

    limits = load_joint_limits(args.urdf)
    print("joint_limits:")
    for i, (lo, hi) in enumerate(limits, start=1):
        print(f"  joint{i}: [{lo:.4f}, {hi:.4f}]")

    all_ok = True
    for path in paths:
        if not os.path.exists(path):
            print(f"\n=== {path} ===\nFAIL: file not found")
            all_ok = False
            continue
        all_ok = validate_file(path, args, limits) and all_ok

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
