#!/usr/bin/env python3
"""Probe Orbbec devices through pyorbbecsdk and V4L2/sysfs fallback."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_infra.Piper_Env.Camera.orbbec_camera import (  # noqa: E402
    get_orbbec_usb_devices,
    get_orbbec_video_devices,
)


def _print_fallback() -> int:
    print("\n[V4L2/sysfs fallback]")

    usb_devices = get_orbbec_usb_devices()
    if usb_devices:
        print("USB devices:")
        for item in usb_devices:
            print(f"  - {item}")
    else:
        print("USB devices: none detected")

    video_devices = get_orbbec_video_devices()
    if video_devices:
        print("V4L2 RGB devices:")
        for item in video_devices:
            print(f"  - {item}")
    else:
        print("V4L2 RGB devices: none detected")
    return len(video_devices)


def main() -> int:
    try:
        from pyorbbecsdk import Context
    except Exception as exc:
        print(f"[SDK] import failed: {exc!r}")
        video_count = _print_fallback()
        return 0 if video_count > 0 else 1

    sdk_count = 0
    try:
        ctx = Context()
        device_list = ctx.query_devices()
        sdk_count = int(device_list.get_count())
        print("[SDK] device_count =", sdk_count)

        for i in range(sdk_count):
            dev = device_list.get_device_by_index(i)
            info = dev.get_device_info()
            print("name =", info.get_name())
            print("serial =", info.get_serial_number())
    except Exception as exc:
        print(f"[SDK] query failed: {exc!r}")

    video_count = _print_fallback()

    if sdk_count == 0:
        print(
            "\n[Hint] SDK saw no depth devices. RGB calibration can still use "
            "--backend v4l2 with the V4L2 serials above. If depth/SDK is needed, "
            "reload udev rules, reconnect cameras, test one camera at a time, "
            "and prefer direct USB3 ports/powered hubs."
        )
    return 0 if sdk_count > 0 or video_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
