import argparse
import socket
import threading
import time
from collections import Counter


def _guess_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(
        description="Minimal Realman teleop-state probe via UDP realtime callback."
    )
    parser.add_argument("--robot-ip", type=str, default="192.168.2.18", help="Robot controller IP")
    parser.add_argument("--port", type=int, default=8080, help="Robot controller port")
    parser.add_argument("--duration", type=float, default=120.0, help="Probe duration in seconds")
    parser.add_argument("--print-hz", type=float, default=5.0, help="Console print rate")
    parser.add_argument(
        "--configure-push",
        action="store_true",
        help="Actively configure UDP realtime push (optional; keep off for non-intrusive test).",
    )
    parser.add_argument("--local-ip", type=str, default="", help="Local NIC IP for UDP push target")
    parser.add_argument("--udp-port", type=int, default=8089, help="UDP target port when configuring push")
    parser.add_argument("--udp-cycle-ms", type=int, default=20, help="UDP push cycle in ms")
    args = parser.parse_args()

    try:
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm,
            rm_thread_mode_e,
            rm_realtime_push_config_t,
            rm_udp_custom_config_t,
            rm_realtime_arm_state_callback_ptr,
            rm_udp_arm_current_status_e,
        )
    except Exception as e:
        print(f"[Error] Failed to import Realman SDK: {e}")
        return

    status_name_by_value = {int(e.value): e.name for e in rm_udp_arm_current_status_e}
    shared = {
        "last_code": None,
        "last_t": 0.0,
        "cb_count": 0,
        "ip": "",
        "port": 0,
        "errCode": 0,
    }
    lock = threading.Lock()
    transition_counter = Counter()

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.robot_ip, args.port)
    if getattr(handle, "id", -1) < 0:
        print(f"[Error] Failed to connect robot: ip={args.robot_ip}, port={args.port}")
        return
    print(f"[Info] Connected. handle_id={handle.id}")

    try:
        ret_cfg, cfg = arm.rm_get_realtime_push()
        print(f"[Info] rm_get_realtime_push ret={ret_cfg}, cfg={cfg}")
    except Exception as e:
        print(f"[Warn] rm_get_realtime_push failed: {e}")

    if args.configure_push:
        local_ip = args.local_ip if args.local_ip else _guess_local_ip()
        cfg_obj = rm_realtime_push_config_t()
        cfg_obj.cycle = int(args.udp_cycle_ms)
        cfg_obj.enable = True
        cfg_obj.port = int(args.udp_port)
        cfg_obj.force_coordinate = -1
        cfg_obj.ip = local_ip.encode("utf-8")

        custom = rm_udp_custom_config_t()
        custom.joint_speed = 0
        custom.lift_state = 0
        custom.expand_state = 0
        custom.hand_state = 0
        custom.arm_current_status = 1
        custom.aloha_state = 0
        custom.plus_base = 0
        custom.plus_state = 0
        cfg_obj.custom_config = custom

        ret_set = arm.rm_set_realtime_push(cfg_obj)
        print(
            f"[Info] rm_set_realtime_push ret={ret_set}, local_ip={local_ip}, "
            f"udp_port={args.udp_port}, cycle_ms={args.udp_cycle_ms}"
        )

    @rm_realtime_arm_state_callback_ptr
    def _state_cb(state):
        code = int(state.arm_current_status)
        now = time.time()
        ip_raw = bytes(state.arm_ip).split(b"\x00", 1)[0]
        ip_txt = ip_raw.decode("utf-8", errors="ignore")
        with lock:
            prev = shared["last_code"]
            shared["last_code"] = code
            shared["last_t"] = now
            shared["cb_count"] += 1
            shared["ip"] = ip_txt
            shared["port"] = int(state.arm_port)
            shared["errCode"] = int(state.errCode)
            if prev is not None and prev != code:
                transition_counter[(prev, code)] += 1

    arm.rm_realtime_arm_state_call_back(_state_cb)
    print("[Info] Callback registered. Start probing arm_current_status ...")
    print("[Hint] If using teach pendant / drag mode, toggle it now and watch status changes.")

    t0 = time.time()
    period = 1.0 / max(0.1, float(args.print_hz))
    try:
        while time.time() - t0 < args.duration:
            with lock:
                code = shared["last_code"]
                cb_count = shared["cb_count"]
                last_t = shared["last_t"]
                src_ip = shared["ip"]
                src_port = shared["port"]
                err_code = shared["errCode"]

            if code is None:
                print("[Probe] no callback yet ...")
            else:
                name = status_name_by_value.get(code, f"UNKNOWN_{code}")
                age_ms = (time.time() - last_t) * 1000.0
                print(
                    f"[Probe] status={name}({code}) cb_count={cb_count} "
                    f"age_ms={age_ms:.1f} src={src_ip}:{src_port} errCode={err_code}"
                )
            time.sleep(period)
    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")
    finally:
        if transition_counter:
            print("[Summary] status transitions:")
            for (a, b), cnt in transition_counter.items():
                an = status_name_by_value.get(a, str(a))
                bn = status_name_by_value.get(b, str(b))
                print(f"  {an}({a}) -> {bn}({b}): {cnt}")
        else:
            print("[Summary] no status transitions recorded.")
        arm.rm_delete_robot_arm()
        print("[Info] Disconnected.")


if __name__ == "__main__":
    main()
