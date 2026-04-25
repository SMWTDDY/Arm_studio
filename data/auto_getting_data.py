import sys
import os
import argparse
import numpy as np
import time
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
import sapien

# 确保可以导入项目中的模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义环境和机器人代理以进行注册
import environments.conveyor_env
import robot.piper.agent
from robot.piper.agent import PiperActionWrapper
from robot.piper.pose_ik import BoundedPiperIK, load_joint_limits
from teleop.get_pose import get_pose
from data.recorder import HDF5Recorder


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def get_link_xyz_rpy(link):
    pose = link.pose
    p = to_numpy(pose.p).reshape(-1, 3)[0]
    q_wxyz = to_numpy(pose.q).reshape(-1, 4)[0]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    rpy = R.from_quat(q_xyzw).as_euler("xyz")
    return p.astype(np.float32), rpy.astype(np.float32)


def get_ee_pose_base(agent, ee_link):
    base_link = agent.robot.find_link_by_name("base_link")
    pose = base_link.pose.inv() * ee_link.pose
    p = to_numpy(pose.p).reshape(-1, 3)[0]
    q_wxyz = to_numpy(pose.q).reshape(-1, 4)[0]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    rpy = R.from_quat(q_xyzw).as_euler("xyz")
    return p.astype(np.float32), rpy.astype(np.float32)


def world_point_to_base(agent, point_world):
    base_link = agent.robot.find_link_by_name("base_link")
    point_pose_base = base_link.pose.inv() * sapien.Pose(p=np.asarray(point_world, dtype=np.float32))
    return to_numpy(point_pose_base.p).reshape(-1, 3)[0].astype(np.float32)


def base_point_to_world(agent, point_base):
    base_link = agent.robot.find_link_by_name("base_link")
    point_pose_world = base_link.pose * sapien.Pose(p=np.asarray(point_base, dtype=np.float32))
    return to_numpy(point_pose_world.p).reshape(-1, 3)[0].astype(np.float32)


def actor_position(actor):
    return to_numpy(actor.pose.p).reshape(-1, 3)[0].astype(np.float32)


def robot_qpos(agent):
    return to_numpy(agent.robot.get_qpos()).reshape(-1)[:8].astype(np.float32)


def set_robot_qpos(agent, qpos):
    agent.robot.set_qpos(np.asarray(qpos, dtype=np.float32))
    agent.robot.set_qvel(np.zeros_like(qpos, dtype=np.float32))


def move_towards(current, target, max_step):
    delta = np.asarray(target, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-8:
        return np.asarray(target, dtype=np.float32)
    return np.asarray(current, dtype=np.float32) + delta / dist * max_step


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def move_rpy_towards(current, target, max_step):
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    delta = wrap_angle(target - current)
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-8:
        return target
    return wrap_angle(current + delta / dist * max_step).astype(np.float32)


def downward_rpy(yaw):
    # With XYZ Euler, roll=pi makes the end-effector local +Z axis point downward.
    return np.array([np.pi, 0.0, yaw], dtype=np.float32)


def local_z_axis(rpy):
    return R.from_euler("xyz", rpy).as_matrix() @ np.array([0.0, 0.0, 1.0])


def choose_downward_rpy(target_p, seed_qpos, solver, yaw_samples=37):
    best = None
    for yaw in np.linspace(-np.pi, np.pi, yaw_samples):
        candidate_pose = np.concatenate([target_p, downward_rpy(yaw)])
        result = solver.solve(candidate_pose, seed_qpos)
        score = result["pos_error"] * 100.0 + result["rot_error"]
        if best is None or score < best[0]:
            best = (score, yaw, result)
    return downward_rpy(best[1]), best[2]


def point_in_range(point, x_range, y_range, z_range):
    return (
        x_range[0] <= point[0] <= x_range[1]
        and y_range[0] <= point[1] <= y_range[1]
        and z_range[0] <= point[2] <= z_range[1]
    )


def select_track_item(items, x_range, y_range, z_range, lock_x_range, skip_item_ids):
    candidates = []
    for item in items:
        if id(item) in skip_item_ids:
            continue
        p = actor_position(item)
        if point_in_range(p, x_range, y_range, z_range) and lock_x_range[0] <= p[0] <= lock_x_range[1]:
            candidates.append((abs(p[0]), item, p))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    _, item, p = candidates[0]
    return item, p


READY_QPOS_PRESETS = np.array(
    [
        [-1.60, 0.55, -0.75, -1.20, 0.10, 2.00],
        [-1.35, 0.42, -0.62, -1.55, -0.25, 1.75],
        [-1.85, 0.72, -0.98, -0.90, 0.38, 2.15],
        [-1.52, 0.95, -1.18, -1.35, -0.45, 1.55],
        [-1.95, 0.35, -0.45, -1.70, 0.55, 2.05],
        [-1.25, 0.80, -1.05, -0.75, -0.15, 1.35],
    ],
    dtype=np.float32,
)


def sample_ready_qpos(rng, noise, joint_limits):
    preset = READY_QPOS_PRESETS[int(rng.integers(len(READY_QPOS_PRESETS)))].copy()
    per_joint_scale = np.array([1.0, 1.0, 1.0, 1.8, 1.4, 1.8], dtype=np.float32)
    jitter = rng.uniform(-noise, noise, 6).astype(np.float32) * per_joint_scale
    qpos = preset + jitter
    # Keep sampled starts away from hard limits so the first IK solve has room to move.
    lower = joint_limits[:, 0].astype(np.float32) + 0.08
    upper = joint_limits[:, 1].astype(np.float32) - 0.08
    qpos = np.clip(qpos, lower, upper)
    qpos = np.concatenate([qpos, np.array([0.035, 0.035], dtype=np.float32)])
    qpos[6:] = 0.035
    return qpos


def estimate_lead_time(current_p, target_p, max_step, dt, min_lead, max_lead):
    ee_speed = max(max_step / dt, 1e-4)
    travel_time = np.linalg.norm(np.asarray(target_p) - np.asarray(current_p)) / ee_speed
    return float(np.clip(travel_time, min_lead, max_lead))


def sample_grasp_bias(rng, xy_std, z_std, yaw_std):
    pos_bias = np.array(
        [
            rng.normal(0.0, xy_std),
            rng.normal(0.0, xy_std),
            rng.normal(0.0, z_std),
        ],
        dtype=np.float32,
    )
    pos_clip = np.array([xy_std, xy_std, z_std], dtype=np.float32) * 3.0
    if np.any(pos_clip > 0.0):
        pos_bias = np.clip(pos_bias, -pos_clip, pos_clip)
    yaw_bias = float(rng.normal(0.0, yaw_std))
    if yaw_std > 0.0:
        yaw_bias = float(np.clip(yaw_bias, -3.0 * yaw_std, 3.0 * yaw_std))
    return pos_bias, yaw_bias


def sample_control_jitter(rng, pos_std, rot_std):
    pos = np.asarray(rng.normal(0.0, pos_std, size=3), dtype=np.float32)
    rot = np.asarray(rng.normal(0.0, rot_std, size=3), dtype=np.float32)
    if pos_std > 0.0:
        pos = np.clip(pos, -3.0 * pos_std, 3.0 * pos_std)
    if rot_std > 0.0:
        rot = np.clip(rot, -3.0 * rot_std, 3.0 * rot_std)
    return pos, rot


def main():
    parser = argparse.ArgumentParser(description="自动采集 Piper conveyor pose 数据")
    parser.add_argument("--episodes", type=int, default=100, help="成功采集的 episode 数量")
    parser.add_argument("--attempts", type=int, default=300, help="最多尝试 episode 数，避免无限循环")
    parser.add_argument("--save_dir", type=str, default="datasets/auto_collected")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_step", type=float, default=0.008, help="每个控制步最大末端位移（base frame, m）")
    parser.add_argument("--max_rot_step", type=float, default=0.08, help="每个控制步最大末端姿态变化（XYZ Euler norm, rad）")
    parser.add_argument("--gripper_offset", type=float, default=0.115, help="link6 到夹爪抓取中心沿局部 z 轴的距离")
    parser.add_argument("--approach_lift", type=float, default=0.08, help="抓取前 link6 在目标上方额外抬高距离")
    parser.add_argument("--ready_joint_noise", type=float, default=0.18, help="每次等待新方块前的关节随机扰动；会叠加到多个初始姿态模板上")
    parser.add_argument("--track_x", type=float, nargs=2, default=[-0.35, 0.35], metavar=("MIN", "MAX"), help="只追踪 world x 在该范围内的方块；默认以机械臂 x=0 左右对称")
    parser.add_argument("--track_y", type=float, nargs=2, default=[-0.20, 0.20], metavar=("MIN", "MAX"), help="只追踪 world y 在该范围内的方块；默认覆盖整个传送带宽度")
    parser.add_argument("--track_z", type=float, nargs=2, default=[0.035, 0.08], metavar=("MIN", "MAX"), help="只追踪 world z 在该范围内的方块；默认过滤传送带上的正常方块高度")
    parser.add_argument("--lock_x", type=float, nargs=2, default=[-0.18, 0.18], metavar=("MIN", "MAX"), help="方块进入该 world x 子窗口后才锁定；避免在追踪窗口边缘就开始抓")
    parser.add_argument("--min_lead", type=float, default=0.05, help="传送带运动预测的最小提前时间")
    parser.add_argument("--max_lead", type=float, default=0.85, help="传送带运动预测的最大提前时间")
    parser.add_argument("--close_lead", type=float, default=0.12, help="夹爪闭合时继续跟随方块的提前时间")
    parser.add_argument("--grasp_xy_noise_std", type=float, default=0.0, help="每条轨迹对抓取目标加入固定 xy 高斯偏移 (m)")
    parser.add_argument("--grasp_z_noise_std", type=float, default=0.0, help="每条轨迹对抓取目标加入固定 z 高斯偏移 (m)")
    parser.add_argument("--grasp_yaw_noise_std", type=float, default=0.0, help="每条轨迹对抓取姿态 yaw 加入固定高斯偏移 (rad)")
    parser.add_argument("--control_pos_jitter_std", type=float, default=0.0, help="控制过程中每隔若干步加入微小 xyz 抖动 (m)")
    parser.add_argument("--control_rot_jitter_std", type=float, default=0.0, help="控制过程中每隔若干步加入微小 rpy 抖动 (rad)")
    parser.add_argument("--control_jitter_hold_steps", type=int, default=4, help="控制抖动保持多少个 control step 后重新采样")
    parser.add_argument("--spawn_interval_steps", type=int, default=None, help="覆盖环境自动刷方块间隔；不传则使用环境默认值")
    parser.add_argument(
        "--attach-object",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="调试用：闭合夹爪后用脚本方式让物体跟随末端。默认关闭，使用真实物理抓取。",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # 上层仍规划/记录 pose，底层用 BoundedPiperIK 转 joint 后交给 pd_joint_pos。
    env_id = "PiperConveyor-v0"
    env = gym.make(
        env_id,
        obs_mode="rgb+state",
        control_mode="pd_joint_pos",
        robot_uids="piper_arm",
        num_envs=1,
        render_mode="human" if args.render else None,
    )
    env = PiperActionWrapper(env, binary_gripper=True)
    obs, _ = env.reset()
    
    # 获取底层对象
    agent = env.unwrapped.agent
    recorder = HDF5Recorder(robot="piper", mode="pose", save_dir=args.save_dir)
    # 姿态需要让局部 z 轴朝下；yaw 会按目标点自动搜索可达解。
    pose_ik = BoundedPiperIK(pos_weight=100.0, rot_weight=5.0, joint_weight=0.02, max_nfev=160)
    
    # 查找末端执行器 link6 的索引
    ee_link_name = "link6"
    links = agent.robot.get_links()
    ee_link_idx = -1
    for i, link in enumerate(links):
        if link.name == ee_link_name:
            ee_link_idx = i
            break

    if args.spawn_interval_steps is not None:
        env.unwrapped.item_spawn_interval = args.spawn_interval_steps

    joint_limits = load_joint_limits()
    belt_vel_world = env.unwrapped.belt_velocity
    dt = 1.0 / env.unwrapped.control_freq
    place_p_world = np.array([0.25, -0.3, 0.12], dtype=np.float32)
    place_p_base = world_point_to_base(agent, place_p_world)

    set_robot_qpos(agent, sample_ready_qpos(rng, args.ready_joint_noise, joint_limits))
    obs = env.unwrapped.get_obs()

    print(
        "[AutoData] 准备就绪。只追踪进入窗口的方块："
        f"x={args.track_x}, y={args.track_y}, z={args.track_z}；"
        f"锁定x={args.lock_x}；"
        f"刷方块间隔={getattr(env.unwrapped, 'item_spawn_interval', 400)} steps；"
        f"grasp_noise(xy/z/yaw)=({args.grasp_xy_noise_std}, {args.grasp_z_noise_std}, {args.grasp_yaw_noise_std})；"
        f"control_jitter(pos/rot)=({args.control_pos_jitter_std}, {args.control_rot_jitter_std})."
    )

    episode_count = 0
    attempt_count = 0
    idle_steps = 0
    skip_item_ids = set()
    while episode_count < args.episodes and attempt_count < args.attempts:
        # 被丢弃过的方块离开追踪窗口后允许未来重新选择同一个 actor。
        for item in env.unwrapped.items:
            if id(item) in skip_item_ids and not point_in_range(actor_position(item), args.track_x, args.track_y, args.track_z):
                skip_item_ids.remove(id(item))

        target_item, target_start_world = select_track_item(
            env.unwrapped.items,
            args.track_x,
            args.track_y,
            args.track_z,
            args.lock_x,
            skip_item_ids,
        )

        if target_item is None:
            idle_action = np.concatenate([robot_qpos(agent)[:6], [1.0]]).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(idle_action)
            if args.render:
                env.render()
            idle_steps += 1
            if idle_steps % int(max(1.0 / dt, 1)) == 0:
                print(
                    f"[AutoData] 等待方块进入追踪窗口... "
                    f"success={episode_count}/{args.episodes} attempts={attempt_count}/{args.attempts}"
                )
            continue

        idle_steps = 0
        attempt_count += 1
        set_robot_qpos(agent, sample_ready_qpos(rng, args.ready_joint_noise, joint_limits))
        obs = env.unwrapped.get_obs()
        print(f"[AutoData] 选中 {target_item.name} at world={np.round(target_start_world, 3)}")

        recorder.start_episode()
        step = 0
        close_steps = 0
        phase = "approach"
        phase_start_step = 0
        attached = False
        grasp_p_base = None
        grasp_rpy = None
        place_rpy = None
        last_warning_step = -999
        failed = False
        grasp_pos_bias, grasp_yaw_bias = sample_grasp_bias(
            rng,
            args.grasp_xy_noise_std,
            args.grasp_z_noise_std,
            args.grasp_yaw_noise_std,
        )
        control_pos_jitter = np.zeros(3, dtype=np.float32)
        control_rpy_jitter = np.zeros(3, dtype=np.float32)

        while True:
            current_time = step * dt
            current_p, current_rpy = get_ee_pose_base(agent, links[ee_link_idx])
            
            item_p_world = actor_position(target_item)
            item_p_base = world_point_to_base(agent, item_p_world)
            belt_vel_base = world_point_to_base(agent, item_p_world + belt_vel_world) - item_p_base

            if phase in ("approach", "descend") and not point_in_range(item_p_world, args.track_x, args.track_y, args.track_z):
                print(f"[Failed] 方块离开追踪窗口 (phase={phase})，丢弃数据。pos={np.round(item_p_world, 3)}")
                recorder.is_recording = False
                recorder.reset_buffers()
                skip_item_ids.add(id(target_item))
                failed = True
                break

            if phase in ("approach", "descend", "close"):
                rough_grasp_p_base = item_p_base.copy()
                rough_grasp_p_base[2] = item_p_base[2] + args.gripper_offset
                rough_target_p = rough_grasp_p_base.copy()
                if phase == "approach":
                    rough_target_p[2] += args.approach_lift
                if phase == "close":
                    predict_t = args.close_lead
                else:
                    predict_t = estimate_lead_time(
                        current_p,
                        rough_target_p,
                        args.max_step,
                        dt,
                        args.min_lead,
                        args.max_lead,
                    )
                cube_center_base = item_p_base + belt_vel_base * predict_t
                # link6 is above the object; its local +Z points downward through the cube.
                grasp_p_base = cube_center_base.copy()
                grasp_p_base[2] = cube_center_base[2] + args.gripper_offset
                grasp_p_base += grasp_pos_bias
                approach_p_base = grasp_p_base.copy()
                approach_p_base[2] = grasp_p_base[2] + args.approach_lift
                if grasp_rpy is None:
                    grasp_rpy, grasp_preview = choose_downward_rpy(grasp_p_base, robot_qpos(agent)[:6], pose_ik)
                    grasp_rpy[2] = wrap_angle(grasp_rpy[2] + grasp_yaw_bias)
                    place_link6_p = place_p_base.copy()
                    place_link6_p[2] = place_p_base[2] + args.gripper_offset
                    place_rpy, _ = choose_downward_rpy(place_link6_p, grasp_preview["qpos"], pose_ik)
                    print(
                        f"[AutoData] grasp_bias_pos={np.round(grasp_pos_bias, 4)} grasp_bias_yaw={grasp_yaw_bias:.4f} "
                        f"grasp_rpy={np.round(grasp_rpy, 3)} "
                        f"local_z={np.round(local_z_axis(grasp_rpy), 3)} "
                        f"preview_pos={grasp_preview['pos_error']:.4f} preview_rot={grasp_preview['rot_error']:.4f}"
                    )

            if phase == "approach":
                desired_p = approach_p_base
                desired_rpy = grasp_rpy
                gripper_action = 1.0
                if np.linalg.norm(current_p - desired_p) < 0.015:
                    phase = "descend"
                    phase_start_step = step
            elif phase == "descend":
                desired_p = grasp_p_base
                desired_rpy = grasp_rpy
                gripper_action = 1.0
                if np.linalg.norm(current_p - desired_p) < 0.012:
                    phase = "close"
                    phase_start_step = step
            elif phase == "close":
                desired_p = grasp_p_base
                desired_rpy = grasp_rpy
                gripper_action = -1.0
                close_steps += 1
                if args.attach_object and close_steps >= int(0.15 / dt):
                    attached = True
                if close_steps >= int(0.35 / dt):
                    phase = "lift"
                    phase_start_step = step
            elif phase == "lift":
                desired_p = grasp_p_base.copy()
                desired_p[2] = grasp_p_base[2] + args.approach_lift
                desired_rpy = grasp_rpy
                gripper_action = -1.0
                if np.linalg.norm(current_p - desired_p) < 0.015:
                    phase = "place"
                    phase_start_step = step
            elif phase == "place":
                desired_p = place_p_base.copy()
                desired_p[2] = place_p_base[2] + args.gripper_offset
                desired_rpy = place_rpy
                gripper_action = -1.0
                if np.linalg.norm(current_p - desired_p) < 0.015:
                    phase = "open"
                    phase_start_step = step
            else:
                desired_p = place_p_base.copy()
                desired_p[2] = place_p_base[2] + args.gripper_offset
                desired_rpy = place_rpy
                gripper_action = 1.0

            if args.control_jitter_hold_steps > 0 and step % args.control_jitter_hold_steps == 0:
                control_pos_jitter, control_rpy_jitter = sample_control_jitter(
                    rng,
                    args.control_pos_jitter_std,
                    args.control_rot_jitter_std,
                )

            if phase in ("approach", "descend", "close", "lift", "place"):
                desired_p = desired_p + control_pos_jitter
                desired_rpy = wrap_angle(desired_rpy + control_rpy_jitter).astype(np.float32)
            
            action_pos = move_towards(current_p, desired_p, args.max_step)
            action_rpy = move_rpy_towards(current_rpy, desired_rpy, args.max_rot_step)

            # 构建完整的目标位姿动作向量 [x, y, z, rx, ry, rz, gripper_action]
            desired_pose = np.array(
                [action_pos[0], action_pos[1], action_pos[2], action_rpy[0], action_rpy[1], action_rpy[2], gripper_action],
                dtype=np.float32,
            )

            current_qpos = robot_qpos(agent)
            ik_result = pose_ik.solve(desired_pose, current_qpos[:6])
            sim_action = np.concatenate([ik_result["qpos"], [gripper_action]]).astype(np.float32)
            achievable_pose = get_pose(ik_result["qpos"])
            full_action = np.concatenate([achievable_pose, [gripper_action]]).astype(np.float32)
            if ik_result["pos_error"] > 0.015 and step - last_warning_step > 30:
                last_warning_step = step
                print(
                    f"[PoseIK Warning] phase={phase} pos={ik_result['pos_error']:.4f}m "
                    f"rot={ik_result['rot_error']:.4f}rad status={ik_result['status']} "
                    f"target={np.round(desired_pose[:3], 3)}"
                )

            next_obs, reward, terminated, truncated, info = env.step(sim_action)
            if attached:
                if phase == "open":
                    target_item.set_pose(sapien.Pose(p=place_p_world))
                    target_item.set_linear_velocity([0, 0, 0])
                    attached = False
                else:
                    ee_world_p = base_point_to_world(agent, get_ee_pose_base(agent, links[ee_link_idx])[0])
                    item_follow_p = ee_world_p + np.array([0.0, 0.0, -0.005], dtype=np.float32)
                    target_item.set_pose(sapien.Pose(p=item_follow_p))
                    target_item.set_linear_velocity([0, 0, 0])
            if args.render:
                env.render()

            # 录制
            recorder.add_step(obs, full_action, reward)
            obs = next_obs
            step += 1

            # --- 判定与退出逻辑 ---
            if phase == "open" and step - phase_start_step > int(0.5 / dt):
                # 检查物体是否在堆叠区内
                final_item_p = actor_position(target_item)
                dist = np.linalg.norm(final_item_p[:2] - place_p_world[:2])
                if dist < 0.1:
                    print(f"[Success] 轨迹录制成功，距离中心: {dist:.4f}")
                    recorder.save()
                    success = True
                    episode_count += 1
                    break
                else:
                    print(f"[Failed] 未放入目标区域，距离中心: {dist:.4f}，丢弃数据。")
                    recorder.is_recording = False
                    recorder.reset_buffers()
                    break

            if current_time > 12.0:
                print(f"[Failed] 任务超时 (phase={phase})，丢弃数据。")
                recorder.is_recording = False
                recorder.reset_buffers()
                skip_item_ids.add(id(target_item))
                failed = True
                break

        if failed or episode_count < args.episodes:
            set_robot_qpos(agent, sample_ready_qpos(rng, args.ready_joint_noise, joint_limits))
            obs = env.unwrapped.get_obs()

    env.close()

if __name__ == "__main__":
    main()
