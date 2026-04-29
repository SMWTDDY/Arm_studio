#!/usr/bin/env python3
"""Smoke probe for the unified Piper real/sim entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_infra.Piper_Env.Env.protocol import arm_names_for_mode, build_meta_keys
from agent_infra.Piper_Env.Env.unified_piper_env import load_unified_config, make_unified_piper_env


def _resolve_config_path(path: str | None, base_dir: Path | None = None) -> str | None:
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    bases = [Path.cwd()]
    if base_dir is not None:
        bases.append(base_dir)
    bases.append(REPO_ROOT)
    bases.append(REPO_ROOT.parent)
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((Path.cwd() / candidate).resolve())


def _camera_roles_from_real_config(config_path: str | None, base_dir: Path | None = None) -> list[str]:
    resolved = _resolve_config_path(config_path, base_dir)
    if not resolved or not Path(resolved).exists():
        return []
    cfg = load_unified_config(resolved)
    nodes = ((cfg.get("cameras") or {}).get("nodes") or [])
    return [node.get("role") for node in nodes if isinstance(node, dict) and node.get("role")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe unified Piper env creation and protocol keys.")
    parser.add_argument(
        "-cfg",
        "--config",
        default="agent_infra/Piper_Env/Config/unified_real_single.yaml",
        help="Unified config path.",
    )
    parser.add_argument("--backend", choices=["real", "sim"], default=None)
    parser.add_argument("--arm-mode", choices=["single", "dual"], default=None)
    parser.add_argument(
        "--control",
        choices=["joint", "pose", "delta_pose", "relative_pose_chunk"],
        default=None,
        help="Override control mode.",
    )
    parser.add_argument("--hz", type=int, default=None, help="Override control frequency.")
    parser.add_argument("--without-cameras", action="store_true", help="Real backend only.")
    parser.add_argument("--start-cameras", action="store_true", help="Real backend only.")
    parser.add_argument("--steps", type=int, default=0, help="Run N safe-action steps after reset.")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Instantiate env and print meta_keys, but skip reset.",
    )
    parser.add_argument(
        "--dry-config",
        action="store_true",
        help="Only parse unified config and print expected meta_keys; does not create hardware/sim env.",
    )
    return parser.parse_args()


def _summarize_meta(env):
    meta = env.unwrapped.meta_keys if hasattr(env, "unwrapped") else env["meta_keys"]
    print("[UnifiedTest] obs sections:", list(meta["obs"].keys()))
    print("[UnifiedTest] state keys:", list(meta["obs"].get("state", {}).keys()))
    print("[UnifiedTest] action keys:", list(meta["action"].keys()))
    print("[UnifiedTest] rgb roles:", list(meta["obs"].get("rgb", {}).keys()))
    print("[UnifiedTest] depth roles:", list(meta["obs"].get("depth", {}).keys()))


def _summarize_obs(obs):
    print("[UnifiedTest] obs keys:", list(obs.keys()))
    if "state" in obs:
        for key, value in obs["state"].items():
            arr = np.asarray(value)
            print(f"[UnifiedTest] state/{key}: shape={arr.shape} dtype={arr.dtype}")
    if "rgb" in obs:
        for key, value in obs["rgb"].items():
            arr = np.asarray(value)
            print(f"[UnifiedTest] rgb/{key}: shape={arr.shape} dtype={arr.dtype}")


def main():
    args = _parse_args()
    if args.dry_config:
        cfg = load_unified_config(args.config)
        backend = args.backend or cfg.get("backend", "real")
        arm_mode = args.arm_mode or cfg.get("arm_mode", "single")
        control_mode = args.control or cfg.get("control_mode", "joint")
        camera_roles = []
        if backend == "sim":
            camera_roles = list((cfg.get("sim") or {}).get("camera_roles", []))
        elif backend == "real" and cfg.get("with_cameras", True):
            base_dir = Path(cfg["_config_path"]).parent if "_config_path" in cfg else None
            real_cfg = cfg.get("real") or {}
            camera_roles = _camera_roles_from_real_config(real_cfg.get("config_path"), base_dir)
        meta = build_meta_keys(
            arm_names_for_mode(arm_mode),
            control_mode=control_mode,
            camera_roles=camera_roles,
        )
        print(
            "[UnifiedTest] dry config:",
            f"backend={backend}",
            f"arm_mode={arm_mode}",
            f"control={control_mode}",
        )
        _summarize_meta({"meta_keys": meta})
        return

    env = None
    try:
        env = make_unified_piper_env(
            config_path=args.config,
            backend=args.backend,
            arm_mode=args.arm_mode,
            control_mode=args.control,
            hz=args.hz,
            with_cameras=False if args.without_cameras else None,
            start_cameras=args.start_cameras,
        )
        print(
            "[UnifiedTest] ready:",
            f"backend={args.backend or 'config'}",
            f"arms={getattr(env.unwrapped, 'arm_names', None)}",
            f"control={getattr(env.unwrapped, 'control_mode', None)}",
            f"hz={getattr(env.unwrapped, 'hz', None)}",
        )
        _summarize_meta(env)

        if args.no_reset:
            return

        obs, info = env.reset()
        print("[UnifiedTest] reset info:", info)
        _summarize_obs(obs)

        for idx in range(max(args.steps, 0)):
            action = env.unwrapped.get_safe_action()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"[UnifiedTest] step={idx} reward={reward} "
                f"terminated={terminated} truncated={truncated} "
                f"intervened={info.get('intervened')}"
            )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
