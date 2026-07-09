#!/usr/bin/env python3
"""
NI cDAQ-9189 accelerometer quick test

What this script does:
1) Finds DAQ devices visible to NI-DAQmx (filters for cDAQ-9189 by default)
2) Picks an AI channel (or uses the channel you specify)
3) Acquires finite samples
4) Prints basic signal statistics and a simple signal-present verdict

Example:
  python cdaq_accel_test.py --mode accel --sample-rate 2048 --samples 4096
  python cdaq_accel_test.py --channel cDAQ9189Mod1/ai0 --mode voltage
"""

from __future__ import annotations

import argparse
import socket
import sys
from typing import Iterable, Optional

import numpy as np

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    from nidaqmx.errors import DaqError
    from nidaqmx.errors import DaqNotFoundError
except Exception as exc:
    print("[ERROR] nidaqmx import failed.")
    print("Install with: pip install nidaqmx numpy")
    print(f"Detail: {exc}")
    sys.exit(2)


def _safe_attr(obj: object, names: Iterable[str]) -> Optional[str]:
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def find_target_devices(filter_text: str):
    system = nidaqmx.system.System.local()
    devices = list(system.devices)

    if not devices:
        return []

    f = filter_text.lower().strip()
    if not f:
        return devices

    matched = []
    for dev in devices:
        prod = str(getattr(dev, "product_type", "")).lower()
        name = str(getattr(dev, "name", "")).lower()
        if f in prod or f in name:
            matched.append(dev)
    return matched


def get_local_ipv4_candidates() -> list[str]:
    addrs: set[str] = set()
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                addrs.add(ip)
    except Exception:
        return []
    return sorted(addrs)


def find_ni_network_systems(timeout_sec: float = 6.0) -> list[str]:
    try:
        import nisyscfg as ni  # Optional helper for IP-unknown network devices.
    except Exception:
        return []

    systems: list[str] = []
    try:
        with ni.Session() as session:
            for item in session.find_systems(detect_online_systems=True, timeout=timeout_sec):
                systems.append(str(item))
    except Exception:
        return []
    return systems


def print_device_info(dev) -> None:
    name = _safe_attr(dev, ["name"]) or "(unknown-name)"
    prod = _safe_attr(dev, ["product_type", "product_num"]) or "(unknown-product)"
    serial = _safe_attr(dev, ["serial_num", "serial_number", "dev_serial_num"]) or "(n/a)"
    host = _safe_attr(dev, ["tcpip_hostname", "hostname"]) or "(n/a)"
    ip = _safe_attr(dev, ["tcpip_address", "ip_address", "network_address"]) or "(n/a)"

    print(f"[DEVICE] name={name}, product={prod}, serial={serial}, host={host}, ip={ip}")


def pick_ai_channel(dev, user_channel: Optional[str]) -> Optional[str]:
    if user_channel:
        return user_channel

    try:
        ai_channels = list(dev.ai_physical_chans)
    except Exception:
        return None

    if not ai_channels:
        return None

    return ai_channels[0].name


def add_input_channel(task, channel: str, mode: str, sensitivity_mv_per_g: float, min_val: float, max_val: float):
    if mode == "voltage":
        task.ai_channels.add_ai_voltage_chan(channel, min_val=min_val, max_val=max_val)
        print(f"[INFO] Voltage channel configured: {channel} ({min_val} to {max_val} V)")
        return

    # Keep accel setup minimal for broad driver compatibility.
    task.ai_channels.add_ai_accel_chan(
        physical_channel=channel,
        min_val=min_val,
        max_val=max_val,
        sensitivity=sensitivity_mv_per_g,
    )
    print(
        f"[INFO] Accel channel configured: {channel} "
        f"({min_val} to {max_val} g, sensitivity={sensitivity_mv_per_g} mV/g)"
    )


def analyze_signal(samples: np.ndarray):
    mean = float(np.mean(samples))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    std = float(np.std(samples))
    vmin = float(np.min(samples))
    vmax = float(np.max(samples))
    ptp = vmax - vmin
    return {
        "mean": mean,
        "rms": rms,
        "std": std,
        "min": vmin,
        "max": vmax,
        "ptp": ptp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="cDAQ-9189 accelerometer/AI input test")
    parser.add_argument("--device-filter", default="cdaq-9189", help="Device name/product match text")
    parser.add_argument("--channel", default=None, help="Physical channel, e.g. cDAQ9189Mod1/ai0")
    parser.add_argument("--mode", choices=["accel", "voltage"], default="accel")
    parser.add_argument("--sample-rate", type=float, default=2048.0, help="Sampling rate (Hz)")
    parser.add_argument("--samples", type=int, default=4096, help="Number of finite samples")
    parser.add_argument("--sensitivity-mv-g", type=float, default=100.0, help="Accelerometer sensitivity")
    parser.add_argument("--min", dest="min_val", type=float, default=-5.0, help="Input min (g or V)")
    parser.add_argument("--max", dest="max_val", type=float, default=5.0, help="Input max (g or V)")
    parser.add_argument(
        "--signal-threshold-rms",
        type=float,
        default=0.005,
        help="Signal-present RMS threshold (g or V)",
    )
    args = parser.parse_args()

    try:
        devices = find_target_devices(args.device_filter)
    except DaqNotFoundError:
        print("[ERROR] NI-DAQmx runtime is not installed or not accessible.")
        print("        Missing library: nicaiu/nicai_utf8")
        print("        Install NI-DAQmx from NI Package Manager, then reboot.")
        print("        After install, verify NI MAX can see cDAQ-9189 and retry.")
        return 2
    except DaqError as exc:
        if getattr(exc, "error_code", None) == -200090:
            print("[ERROR] DAQmx initialization failed (status -200090: LIBRARY_NOT_PRESENT).")
            print("        NI services may not be started yet or install/repair is incomplete.")
            print("        1) Reboot Windows once")
            print("        2) Open PowerShell as Administrator")
            print("        3) Start NI services: niSvcLoc, mxssvr, nidevldu")
            print("        4) Run NI MAX once, then retry this script")
            return 2
        raise

    if not devices:
        print("[ERROR] No NI-DAQmx devices found matching the filter.")
        print("        Check NI MAX: device must be visible and reserved locally.")
        all_system_devices = find_target_devices("")
        print(f"[INFO] DAQmx total visible devices: {len(all_system_devices)}")

        local_ips = get_local_ipv4_candidates()
        if local_ips:
            print(f"[INFO] Local IPv4 candidates: {', '.join(local_ips)}")

        ni_systems = find_ni_network_systems(timeout_sec=8.0)
        if ni_systems:
            print("[INFO] NI network-discovered systems:")
            for item in ni_systems:
                print(f"       - {item}")
        else:
            print("[INFO] NI network discovery returned 0 systems.")
            print("       Verify cDAQ power/link LEDs, same subnet, and NI MAX Network Devices scan.")

        print("[TIP] Try running with --device-filter '' to avoid strict model-name matching.")
        return 1

    print(f"[INFO] Matched devices: {len(devices)}")
    for dev in devices:
        print_device_info(dev)

    dev = devices[0]
    channel = pick_ai_channel(dev, args.channel)
    if not channel:
        print("[ERROR] No AI channel found. Specify --channel manually.")
        return 1

    print(f"[INFO] Using channel: {channel}")

    timeout = max(10.0, args.samples / args.sample_rate + 5.0)

    with nidaqmx.Task() as task:
        add_input_channel(
            task,
            channel=channel,
            mode=args.mode,
            sensitivity_mv_per_g=args.sensitivity_mv_g,
            min_val=args.min_val,
            max_val=args.max_val,
        )

        task.timing.cfg_samp_clk_timing(
            rate=args.sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=args.samples,
        )

        print(
            f"[INFO] Acquiring {args.samples} samples @ {args.sample_rate} Hz "
            f"(timeout={timeout:.1f}s)..."
        )
        data = task.read(number_of_samples_per_channel=args.samples, timeout=timeout)

    arr = np.asarray(data, dtype=float)
    if arr.ndim > 1:
        arr = arr[0]

    stats = analyze_signal(arr)

    unit = "g" if args.mode == "accel" else "V"
    print("\n=== SIGNAL STATS ===")
    print(f"Samples : {arr.size}")
    print(f"Mean    : {stats['mean']:.6f} {unit}")
    print(f"RMS     : {stats['rms']:.6f} {unit}")
    print(f"Std     : {stats['std']:.6f} {unit}")
    print(f"Min/Max : {stats['min']:.6f} / {stats['max']:.6f} {unit}")
    print(f"Pk-Pk   : {stats['ptp']:.6f} {unit}")

    signal_present = stats["rms"] >= args.signal_threshold_rms and stats["ptp"] >= (2.0 * args.signal_threshold_rms)
    print("\n=== VERDICT ===")
    if signal_present:
        print("[PASS] Signal appears to be present.")
    else:
        print("[WARN] Signal level is very low or flat.")
        print("       Check sensor power/IEPE setting/channel wiring/sensitivity.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
