"""Unified Piper environment factory for real and simulated backends."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import cv2

try:
    import yaml
except ImportError:
    yaml = None

try:
    import gymnasium as gym
except ImportError:
    class _MissingGymEnv:
        pass

    class _MissingGym:
        Env = _MissingGymEnv

    gym = _MissingGym()

from agent_infra.Piper_Env.Env.protocol import (
    PIPER_BACKENDS,
    PIPER_CONTROL_MODES,
    arm_names_for_mode,
    build_meta_keys,
    ensure_under_control,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _workspace_root() -> Path:
    return _repo_root().parent


def _as_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except Exception:
        return {}


def _resolve_path(path: Optional[str], base_dir: Optional[Path] = None) -> Optional[str]:
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    bases = [Path.cwd()]
    if base_dir is not None:
        bases.append(base_dir)
    bases.append(_repo_root())
    bases.append(_workspace_root())
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((Path.cwd() / candidate).resolve())


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: str) -> Dict[str, Any]:
    """Tiny YAML subset loader for unified configs when PyYAML is unavailable."""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML list placement in {path}: {raw_line}")
            parent.append(_parse_scalar(content[2:]))
            continue

        if ":" not in content:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line}")

        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"Unsupported YAML mapping placement in {path}: {raw_line}")

        if value:
            parent[key] = _parse_scalar(value)
            continue

        # Look ahead is intentionally avoided; unified configs only need dicts
        # and simple lists, so create a list for known plural list keys.
        child: Any = [] if key in {"camera_roles"} else {}
        parent[key] = child
        stack.append((indent, child))

    return root


def load_unified_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    resolved = _resolve_path(config_path)
    if yaml is None:
        cfg = _load_simple_yaml(resolved)
    else:
        with open(resolved, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = resolved
    return cfg


class SinglePiperSimEnv(gym.Env):
    """Adapter that makes Arm_studio's single-arm ManiSkill env follow Piper meta_keys."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_name: str = "PiperConveyor-v0",
        arm_studio_root: Optional[str] = None,
        obs_mode: str = "rgb+state",
        control_mode: str = "joint",
        render_mode: str = "rgb_array",
        robot_uids: str = "piper_arm",
        binary_gripper: bool = False,
        camera_roles: Optional[list[str]] = None,
        camera_role_sources: Optional[Dict[str, str]] = None,
        image_shape: tuple[int, int, int] = (3, 224, 224),
        **gym_kwargs,
    ):
        if control_mode not in PIPER_CONTROL_MODES:
            raise ValueError(f"Unsupported control_mode={control_mode!r}.")
        if control_mode not in {"joint", "pose"}:
            raise NotImplementedError(
                "SinglePiperSimEnv currently supports joint and pose control. "
                f"Got control_mode={control_mode!r}."
            )
        if not hasattr(gym, "make"):
            raise ImportError(
                "backend='sim' requires gymnasium and Arm_studio simulation dependencies."
            )

        self.arm_names = ["single"]
        self.control_mode = control_mode
        self.hz = int(gym_kwargs.pop("hz", 32))
        self.binary_gripper = bool(binary_gripper)
        self.camera_roles = camera_roles or ["base_camera", "wrist_camera"]
        default_camera_sources = {
            "base_camera": "front_view",
            "wrist_camera": "hand_camera",
            "front_view": "front_view",
            "hand_camera": "hand_camera",
        }
        self.camera_role_sources = camera_role_sources or {
            role: default_camera_sources.get(role, role) for role in self.camera_roles
        }
        from teleop.get_pose import get_pose

        self._get_pose = get_pose
        self.meta_keys = build_meta_keys(
            self.arm_names,
            control_mode=control_mode,
            camera_roles=self.camera_roles,
            image_shape=image_shape,
            include_under_control=True,
        )

        root = Path(arm_studio_root).expanduser().resolve() if arm_studio_root else _workspace_root() / "Arm_studio"
        if not root.exists():
            raise FileNotFoundError(f"Arm_studio root not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import environments.conveyor_env  # noqa: F401
        import robot.piper.agent  # noqa: F401
        from robot.piper.agent import PiperActionWrapper
        from robot.piper.pose_ik import BoundedPiperIK

        sim_control_mode = "pd_joint_pos" if control_mode == "pose" else "pd_joint_pos"
        env = gym.make(
            env_name,
            obs_mode=obs_mode,
            control_mode=sim_control_mode,
            robot_uids=robot_uids,
            render_mode=render_mode,
            **gym_kwargs,
        )
        self.env = PiperActionWrapper(env, binary_gripper=binary_gripper)
        self.pose_ik = BoundedPiperIK() if control_mode == "pose" else None
        self._last_obs: Optional[Dict[str, Any]] = None

    @property
    def unwrapped(self):
        return self

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _first_batch(self, value: Any) -> np.ndarray:
        arr = self._to_numpy(value)
        if arr.ndim >= 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _extract_qpos_qvel(self, raw_obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        agent = raw_obs.get("agent", {}) if isinstance(raw_obs, dict) else {}
        qpos = self._first_batch(agent.get("qpos", np.zeros(8, dtype=np.float32))).astype(np.float32)
        qvel = self._first_batch(agent.get("qvel", np.zeros_like(qpos))).astype(np.float32)
        if qpos.shape[0] < 8:
            qpos = np.pad(qpos, (0, 8 - qpos.shape[0]))
        if qvel.shape[0] < 8:
            qvel = np.pad(qvel, (0, 8 - qvel.shape[0]))
        return qpos, qvel

    def _extract_rgb(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        rgb: Dict[str, np.ndarray] = {}
        sensor_data = raw_obs.get("sensor_data", {}) if isinstance(raw_obs, dict) else {}
        for role in self.camera_roles:
            frame = None
            source_role = self.camera_role_sources.get(role, role)
            if source_role in sensor_data and "rgb" in sensor_data[source_role]:
                frame = self._first_batch(sensor_data[source_role]["rgb"])
            if frame is None:
                shape = self.meta_keys["obs"]["rgb"][role]
                rgb[role] = np.zeros(shape, dtype=np.uint8)
                continue
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                target_c, target_h, target_w = self.meta_keys["obs"]["rgb"][role]
                if frame.shape[:2] != (target_h, target_w):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                frame = frame.transpose(2, 0, 1)
            rgb[role] = frame.astype(np.uint8, copy=False)
        return rgb

    def _adapt_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        qpos, qvel = self._extract_qpos_qvel(raw_obs)
        joint_pos = qpos[:6].astype(np.float32)
        joint_vel = qvel[:6].astype(np.float32)
        from robot.piper.gripper import piper_gripper_qpos_to_width

        gripper = np.array([piper_gripper_qpos_to_width(qpos[6:8])], dtype=np.float32)
        obs: Dict[str, Any] = {
            "state": {
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "ee_pose": self._get_pose(joint_pos.copy()),
                "gripper_pos": gripper,
            },
            "rgb": self._extract_rgb(raw_obs),
        }
        ensure_under_control(obs, self.arm_names)
        return obs

    def _to_sim_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        arm_action = np.asarray(action["arm"], dtype=np.float32).reshape(-1)
        gripper = float(np.asarray(action["gripper"], dtype=np.float32).reshape(-1)[0])
        if self.control_mode == "pose":
            if self.pose_ik is None:
                raise RuntimeError("pose_ik is not initialized.")
            state = self._last_obs["state"] if self._last_obs is not None else self._adapt_obs(self.env.unwrapped._init_raw_obs)["state"]
            ik = self.pose_ik.solve(arm_action[:6], state["joint_pos"])
            arm_action = ik["qpos"]
        return np.concatenate([arm_action[:6], [gripper]]).astype(np.float32)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        obs = self._adapt_obs(raw_obs)
        self._last_obs = obs
        return obs, info

    def step(self, action: Dict[str, np.ndarray]):
        policy_action = {k: np.asarray(v).copy() for k, v in action.items()}
        sim_action = self._to_sim_action(policy_action)
        raw_obs, reward, terminated, truncated, info = self.env.step(sim_action)
        obs = self._adapt_obs(raw_obs)
        self._last_obs = obs
        info = dict(info or {})
        info["actual_action"] = policy_action
        info["policy_action"] = policy_action
        info["intervened"] = {"single": False}
        info["action_source"] = {"single": "policy"}
        return obs, reward, terminated, truncated, info

    def get_safe_action(self, state: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        if state is None:
            if self._last_obs is None:
                obs, _ = self.reset()
            else:
                obs = self._last_obs
            state = obs["state"]
        if self.control_mode == "pose":
            arm = state["ee_pose"]
        else:
            arm = state["joint_pos"]
        return {
            "arm": np.asarray(arm, dtype=np.float32).copy(),
            "gripper": np.asarray(state["gripper_pos"], dtype=np.float32).copy(),
        }

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class DualPiperSimEnv(gym.Env):
    """Dual-arm unified sim made from two Arm_studio single-arm sim instances.

    This is a protocol-complete dual sim adapter. The two arms currently run in
    separate ManiSkill scenes, which keeps data/control formats aligned while a
    true shared-scene dual-arm task is built underneath.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        control_mode: str = "joint",
        hz: Optional[int] = None,
        camera_roles: Optional[list[str]] = None,
        **kwargs,
    ):
        self.arm_names = ["left", "right"]
        self.control_mode = control_mode
        self.hz = int(hz or kwargs.get("hz", 32))
        # 兼容旧配置中的 robot_uids；双臂协议环境必须拆成左右两个单臂场景，
        # 否则 left_hand_camera/right_hand_camera 会错误地复用同一个 wrist-camera URDF。
        kwargs.pop("robot_uids", None)
        left_robot_uids = kwargs.pop("left_robot_uids", "piper_arm_left")
        right_robot_uids = kwargs.pop("right_robot_uids", "piper_arm_right")
        self.camera_roles = camera_roles or [
            "base_camera",
            "left_hand_camera",
            "right_hand_camera",
        ]

        left_kwargs = dict(kwargs)
        right_kwargs = dict(kwargs)
        # 当前双臂仿真是“协议层双臂”：左右臂各运行一个 ArmStudio 单臂场景。
        # 因此相机 role 要在合并观测时重新命名为真实双臂风格的 left/right hand camera。
        left_kwargs["camera_roles"] = [
            role for role in ["base_camera", "left_hand_camera"] if role in self.camera_roles
        ]
        left_kwargs["camera_role_sources"] = {
            "base_camera": "front_view",
            "left_hand_camera": "hand_camera",
        }
        left_kwargs["robot_uids"] = left_robot_uids
        right_kwargs["camera_roles"] = [
            role for role in ["right_hand_camera"] if role in self.camera_roles
        ]
        right_kwargs["camera_role_sources"] = {
            "right_hand_camera": "hand_camera",
        }
        right_kwargs["robot_uids"] = right_robot_uids

        self.envs = {
            "left": SinglePiperSimEnv(
                control_mode=control_mode,
                hz=self.hz,
                **left_kwargs,
            ),
            "right": SinglePiperSimEnv(
                control_mode=control_mode,
                hz=self.hz,
                **right_kwargs,
            ),
        }
        self.meta_keys = build_meta_keys(
            self.arm_names,
            control_mode=control_mode,
            camera_roles=self.camera_roles,
            image_shape=(3, 224, 224),
            include_under_control=True,
        )

    @property
    def unwrapped(self):
        return self

    @staticmethod
    def _prefix_state(obs: Dict[str, Any], arm_name: str) -> Dict[str, np.ndarray]:
        return {
            f"{arm_name}_{key}": np.asarray(value).copy()
            for key, value in obs.get("state", {}).items()
        }

    def _combine_obs(self, obs_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        combined: Dict[str, Any] = {
            "state": {},
            "under_control": {},
            "rgb": {},
        }
        for name, obs in obs_by_arm.items():
            combined["state"].update(self._prefix_state(obs, name))
            combined["under_control"][name] = np.array([False], dtype=np.bool_)
            combined["rgb"].update(obs.get("rgb", {}))
        return combined

    def _split_action(self, action: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "left": {
                "arm": np.asarray(action["left_arm"], dtype=np.float32),
                "gripper": np.asarray(action["left_gripper"], dtype=np.float32),
            },
            "right": {
                "arm": np.asarray(action["right_arm"], dtype=np.float32),
                "gripper": np.asarray(action["right_gripper"], dtype=np.float32),
            },
        }

    def reset(self, **kwargs):
        obs_by_arm = {}
        infos = {}
        for name, env in self.envs.items():
            obs_by_arm[name], infos[name] = env.reset(**kwargs)
        return self._combine_obs(obs_by_arm), {"per_arm": infos}

    def step(self, action: Dict[str, np.ndarray]):
        split = self._split_action(action)
        obs_by_arm = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}
        for name, env in self.envs.items():
            obs, reward, term, trunc, info = env.step(split[name])
            obs_by_arm[name] = obs
            rewards[name] = reward
            terminated[name] = bool(np.asarray(term).any())
            truncated[name] = bool(np.asarray(trunc).any())
            infos[name] = info

        actual_action = {key: np.asarray(value).copy() for key, value in action.items()}
        info = {
            "per_arm": infos,
            "actual_action": actual_action,
            "policy_action": actual_action,
            "intervened": {"left": False, "right": False},
            "action_source": {"left": "policy", "right": "policy"},
        }
        reward_value = float(np.mean([float(np.asarray(v).mean()) for v in rewards.values()]))
        return (
            self._combine_obs(obs_by_arm),
            reward_value,
            any(terminated.values()),
            any(truncated.values()),
            info,
        )

    def get_safe_action(self, state: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        if state is not None:
            action = {}
            for name in self.arm_names:
                prefix = f"{name}_"
                arm_key = f"{prefix}ee_pose" if self.control_mode == "pose" else f"{prefix}joint_pos"
                action[f"{prefix}arm"] = np.asarray(state[arm_key], dtype=np.float32).copy()
                action[f"{prefix}gripper"] = np.asarray(
                    state[f"{prefix}gripper_pos"],
                    dtype=np.float32,
                ).copy()
            return action

        left = self.envs["left"].get_safe_action()
        right = self.envs["right"].get_safe_action()
        return {
            "left_arm": left["arm"],
            "left_gripper": left["gripper"],
            "right_arm": right["arm"],
            "right_gripper": right["gripper"],
        }

    def render(self):
        return {name: env.render() for name, env in self.envs.items()}

    def close(self):
        for env in self.envs.values():
            env.close()


def make_real_piper_env(
    arm_mode: str,
    config_path: Optional[str],
    control_mode: Optional[str],
    hz: Optional[int],
    with_cameras: bool = True,
    start_cameras: bool = False,
    **kwargs,
):
    if with_cameras:
        if arm_mode == "single":
            from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv

            env = SinglePiperEnv(config_path=config_path, control_mode=control_mode, hz=hz, **kwargs)
        elif arm_mode == "dual":
            from agent_infra.Piper_Env.Env.dual_piper_env import DualPiperEnv

            env = DualPiperEnv(config_path=config_path, control_mode=control_mode, hz=hz, **kwargs)
        else:
            raise ValueError(f"Unsupported real arm_mode={arm_mode!r}.")
        if start_cameras and hasattr(env, "start_cameras"):
            env.start_cameras()
        return env

    from agent_infra.Piper_Env.Env.utils.piper_base_env import PiperEnv

    env = PiperEnv(config_path=config_path, control_mode=control_mode, hz=hz, **kwargs)
    expected = len(arm_names_for_mode(arm_mode))
    if len(env.arm_names) != expected:
        env.close()
        raise ValueError(f"Config created {len(env.arm_names)} arm(s), expected {expected} for {arm_mode}.")
    return env


def make_sim_piper_env(
    arm_mode: str,
    control_mode: Optional[str],
    hz: Optional[int],
    **kwargs,
):
    if hz is not None:
        kwargs.setdefault("hz", hz)
    if arm_mode == "single":
        return SinglePiperSimEnv(control_mode=control_mode or "joint", **kwargs)
    if arm_mode == "dual":
        return DualPiperSimEnv(control_mode=control_mode or "joint", **kwargs)
    raise ValueError(f"Unsupported sim arm_mode={arm_mode!r}.")


def make_unified_piper_env(
    config_path: Optional[str] = None,
    backend: Optional[str] = None,
    arm_mode: Optional[str] = None,
    control_mode: Optional[str] = None,
    hz: Optional[int] = None,
    with_cameras: Optional[bool] = None,
    start_cameras: Optional[bool] = None,
    **overrides,
):
    cfg = load_unified_config(config_path)
    base_dir = Path(cfg["_config_path"]).parent if "_config_path" in cfg else None

    backend = backend or cfg.get("backend", "real")
    arm_mode = arm_mode or cfg.get("arm_mode", "single")
    control_mode = control_mode or cfg.get("control_mode")
    hz = hz if hz is not None else cfg.get("hz")
    if backend not in PIPER_BACKENDS:
        raise ValueError(f"Unsupported backend={backend!r}; expected one of {PIPER_BACKENDS}.")

    if backend == "real":
        real_cfg = _as_plain_dict(cfg.get("real"))
        real_cfg.update(overrides)
        real_config = _resolve_path(real_cfg.pop("config_path", None), base_dir)
        actual_with_cameras = bool(cfg.get("with_cameras", True) if with_cameras is None else with_cameras)
        actual_start_cameras = bool(
            real_cfg.pop("start_cameras", cfg.get("start_cameras", False))
            if start_cameras is None
            else start_cameras
        )
        return make_real_piper_env(
            arm_mode=arm_mode,
            config_path=real_config,
            control_mode=control_mode,
            hz=hz,
            with_cameras=actual_with_cameras,
            start_cameras=actual_start_cameras,
            **real_cfg,
        )

    sim_cfg = _as_plain_dict(cfg.get("sim"))
    sim_cfg.update({key: value for key, value in overrides.items() if value is not None})
    sim_root = sim_cfg.get("arm_studio_root")
    if sim_root:
        sim_cfg["arm_studio_root"] = _resolve_path(sim_root, base_dir)
    return make_sim_piper_env(
        arm_mode=arm_mode,
        control_mode=control_mode,
        hz=hz,
        **sim_cfg,
    )
