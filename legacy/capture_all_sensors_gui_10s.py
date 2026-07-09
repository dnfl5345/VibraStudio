#!/usr/bin/env python3
"""
Acquire all available AI channels from NI cDAQ for 10 seconds with live GUI plots.

What this script covers:
1) Practical high-rate re-measurement for vibration signals.
2) CSV logging of all channels and summary stats.
3) Multi-channel live monitoring GUI (Matplotlib interactive window).

Default settings are tuned for accelerometer-like signals with bandwidth up to ~5 kHz:
- sample rate: 12800 Hz (Nyquist 6400 Hz)
- duration: 10 s

Note:
- If channel sensitivity (mV/g) is unknown, values are kept in Volts.
- This script uses voltage channels for broad compatibility across mixed AI modules.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from nidaqmx.constants import AcquisitionType


@dataclass
class RunConfig:
    sample_rate: float
    duration_sec: float
    chunk_size: int
    v_min: float
    v_max: float
    out_dir: Path


class AcquisitionState:
    def __init__(self, channels: list[str], cfg: RunConfig) -> None:
        self.channels = channels
        self.cfg = cfg
        self.target_samples = int(cfg.sample_rate * cfg.duration_sec)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.done_event = threading.Event()

        self.total_samples = 0
        self.latest_chunk = np.zeros((len(channels), 0), dtype=float)
        self.data_parts: list[np.ndarray] = []
        self.error: Exception | None = None
        self.start_t = 0.0
        self.end_t = 0.0

    def append_chunk(self, arr: np.ndarray) -> None:
        with self.lock:
            self.data_parts.append(arr)
            self.latest_chunk = arr
            self.total_samples += arr.shape[1]

    def get_latest(self) -> np.ndarray:
        with self.lock:
            return self.latest_chunk.copy()

    def get_progress(self) -> float:
        with self.lock:
            if self.target_samples <= 0:
                return 0.0
            return min(1.0, self.total_samples / self.target_samples)

    def build_matrix(self) -> np.ndarray:
        with self.lock:
            if not self.data_parts:
                return np.zeros((len(self.channels), 0), dtype=float)
            return np.concatenate(self.data_parts, axis=1)


def discover_ai_channels() -> list[tuple[str, str, str]]:
    """Return list of (device_name, product_type, channel_name)."""
    result: list[tuple[str, str, str]] = []
    system = nidaqmx.system.System.local()
    for dev in system.devices:
        dev_name = str(getattr(dev, "name", "?"))
        product = str(getattr(dev, "product_type", "?"))
        try:
            for ch in dev.ai_physical_chans:
                result.append((dev_name, product, ch.name))
        except Exception:
            continue
    return result


def acquisition_worker(state: AcquisitionState) -> None:
    state.start_t = time.perf_counter()
    try:
        with nidaqmx.Task() as task:
            for ch in state.channels:
                task.ai_channels.add_ai_voltage_chan(ch, min_val=state.cfg.v_min, max_val=state.cfg.v_max)

            task.timing.cfg_samp_clk_timing(
                rate=state.cfg.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(4 * state.cfg.chunk_size, 4096),
            )

            while not state.stop_event.is_set():
                remaining = state.target_samples - state.total_samples
                if remaining <= 0:
                    break

                n = min(state.cfg.chunk_size, remaining)
                data = task.read(number_of_samples_per_channel=n, timeout=5.0)
                arr = np.asarray(data, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                state.append_chunk(arr)

    except Exception as exc:
        state.error = exc
    finally:
        state.end_t = time.perf_counter()
        state.done_event.set()


def save_outputs(state: AcquisitionState, cfg: RunConfig) -> tuple[Path, Path, Path]:
    matrix = state.build_matrix()
    n_channels, n_samples = matrix.shape

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = cfg.out_dir / f"capture_{ts}_raw.csv"
    summary_csv = cfg.out_dir / f"capture_{ts}_summary.csv"
    plot_png = cfg.out_dir / f"capture_{ts}_overview.png"

    if n_samples == 0:
        pd.DataFrame({"message": ["No samples captured"]}).to_csv(summary_csv, index=False)
        return raw_csv, summary_csv, plot_png

    t = np.arange(n_samples, dtype=float) / cfg.sample_rate
    raw_dict = {"time_s": t}
    for i, ch in enumerate(state.channels):
        raw_dict[ch] = matrix[i]
    pd.DataFrame(raw_dict).to_csv(raw_csv, index=False)

    rows = []
    for i, ch in enumerate(state.channels):
        y = matrix[i]
        rows.append(
            {
                "channel": ch,
                "samples": y.size,
                "mean_V": float(np.mean(y)),
                "rms_V": float(np.sqrt(np.mean(np.square(y)))),
                "std_V": float(np.std(y)),
                "min_V": float(np.min(y)),
                "max_V": float(np.max(y)),
                "pkpk_V": float(np.ptp(y)),
            }
        )
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    # Save static overview for quick review.
    fig, axes = plt.subplots(min(n_channels, 8), 1, figsize=(12, 2.2 * min(n_channels, 8)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, ax in enumerate(axes):
        y = matrix[i]
        ax.plot(t, y, lw=0.8)
        ax.set_ylabel("V")
        ax.set_title(state.channels[i])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(plot_png, dpi=140)
    plt.close(fig)

    return raw_csv, summary_csv, plot_png


def run_gui_capture(cfg: RunConfig, auto_start: bool = True) -> int:
    channel_info = discover_ai_channels()
    if not channel_info:
        print("[ERROR] No AI channels discovered.")
        return 1

    channels = [ch for _, _, ch in channel_info]
    print(f"[INFO] Discovered {len(channels)} AI channels")
    for dev, prod, ch in channel_info:
        print(f"  - {ch} ({dev}, {prod})")

    state = AcquisitionState(channels, cfg)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title("NI cDAQ Multi-Channel Live Capture (10s)")
    ax.set_title("Live Multi-Channel (latest chunk)")
    ax.set_xlabel("Sample Index (chunk-local)")
    ax.set_ylabel("Voltage (offset stacked)")
    ax.grid(alpha=0.25)

    lines = []
    offsets = np.arange(len(channels), dtype=float) * 0.05
    for i, ch in enumerate(channels):
        (line,) = ax.plot([], [], lw=0.9, label=ch)
        lines.append(line)

    status_text = ax.text(0.01, 0.98, "Initializing...", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=7, ncol=1)

    worker = threading.Thread(target=acquisition_worker, args=(state,), daemon=True)

    def on_close(_event) -> None:
        state.stop_event.set()

    fig.canvas.mpl_connect("close_event", on_close)

    def animate(_frame):
        latest = state.get_latest()
        if latest.size > 0:
            x = np.arange(latest.shape[1])
            y_min = float("inf")
            y_max = float("-inf")
            for i, line in enumerate(lines):
                y = latest[i] + offsets[i]
                line.set_data(x, y)
                y_min = min(y_min, float(np.min(y)))
                y_max = max(y_max, float(np.max(y)))
            ax.set_xlim(0, max(1, latest.shape[1]))
            if y_max <= y_min:
                y_min -= 1e-3
                y_max += 1e-3
            margin = max(1e-3, 0.1 * (y_max - y_min))
            ax.set_ylim(y_min - margin, y_max + margin)

        elapsed = max(0.0, time.perf_counter() - state.start_t) if state.start_t > 0 else 0.0
        status_text.set_text(
            f"Progress: {state.get_progress()*100:.1f}% | elapsed: {elapsed:.2f}s | "
            f"samples/ch: {state.total_samples}/{state.target_samples}"
        )

        if state.done_event.is_set():
            plt.close(fig)
        return lines

    if auto_start:
        worker.start()

    ani = FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)
    _ = ani
    plt.tight_layout()
    plt.show()

    # Ensure worker has stopped.
    state.stop_event.set()
    worker.join(timeout=2.0)

    if state.error is not None:
        print(f"[ERROR] Acquisition failed: {type(state.error).__name__}: {state.error}")
        return 2

    raw_csv, summary_csv, plot_png = save_outputs(state, cfg)
    matrix = state.build_matrix()
    print("\n=== CAPTURE RESULT ===")
    print(f"Captured channels : {len(channels)}")
    print(f"Captured samples  : {matrix.shape[1]} per channel")
    print(f"Requested duration: {cfg.duration_sec:.2f}s")
    print(f"Effective runtime : {max(0.0, state.end_t - state.start_t):.2f}s")
    print(f"Raw CSV           : {raw_csv}")
    print(f"Summary CSV       : {summary_csv}")
    print(f"Overview PNG      : {plot_png}")

    if matrix.shape[1] > 0:
        print("\n=== CHANNEL SUMMARY ===")
        for i, ch in enumerate(channels):
            y = matrix[i]
            rms = float(np.sqrt(np.mean(np.square(y))))
            ptp = float(np.ptp(y))
            mean = float(np.mean(y))
            print(f"[OK] {ch}: mean={mean:.6f} V, rms={rms:.6f} V, pkpk={ptp:.6f} V")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-channel 10s cDAQ GUI capture")
    parser.add_argument("--sample-rate", type=float, default=12800.0, help="Sampling rate in Hz")
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds")
    parser.add_argument("--chunk", type=int, default=1024, help="Chunk size per read")
    parser.add_argument("--min", dest="v_min", type=float, default=-10.0, help="Min voltage")
    parser.add_argument("--max", dest="v_max", type=float, default=10.0, help="Max voltage")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(
        sample_rate=args.sample_rate,
        duration_sec=args.duration,
        chunk_size=args.chunk,
        v_min=args.v_min,
        v_max=args.v_max,
        out_dir=Path(args.out_dir),
    )
    return run_gui_capture(cfg, auto_start=True)


if __name__ == "__main__":
    raise SystemExit(main())
