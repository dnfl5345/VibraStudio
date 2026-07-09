#!/usr/bin/env python3
"""
Integrated modal-analysis GUI for cDAQ AI channels.

Features:
- 10-second multi-channel acquisition (configurable)
- Sensor mapping for X/Z axes across Mod3~Mod5 layout
- FFT report export per channel and per sensor-axis
- Simple mode-shape animation for cantilever beam (equal spacing)

Default mapping follows user-described order:
Mod3/ai0=S1_X, Mod3/ai1=S1_Z, Mod3/ai2=S2_X,
Mod4/ai0=S2_Z, Mod4/ai1=S3_X, Mod4/ai2=S3_Z,
Mod5/ai0=S4_X, Mod5/ai1=S4_Z, Mod5/ai2=S5_X
"""

from __future__ import annotations

import csv
import argparse
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from nidaqmx.constants import AcquisitionType
from scipy.signal import find_peaks, windows


@dataclass
class CaptureConfig:
    sample_rate: float = 12800.0
    duration_sec: float = 10.0
    chunk_size: int = 1024
    min_v: float = -10.0
    max_v: float = 10.0
    fft_peaks: int = 6
    auto_modes: int = 3
    sensitivity_mv_g: float = 100.0
    line_ref_hz: float = 0.0
    out_dir: Path = Path("outputs")


@dataclass
class ChannelMap:
    physical_channel: str
    sensor_id: int
    axis: str  # X or Z
    y_pos: float  # normalized [0..1], bottom->top


DEFAULT_CHANNEL_MAP = [
    ChannelMap("cDAQ9189-233EBC6Mod3/ai0", 1, "X", 0.2),
    ChannelMap("cDAQ9189-233EBC6Mod3/ai1", 1, "Z", 0.2),
    ChannelMap("cDAQ9189-233EBC6Mod3/ai2", 2, "X", 0.4),
    ChannelMap("cDAQ9189-233EBC6Mod4/ai0", 2, "Z", 0.4),
    ChannelMap("cDAQ9189-233EBC6Mod4/ai1", 3, "X", 0.6),
    ChannelMap("cDAQ9189-233EBC6Mod4/ai2", 3, "Z", 0.6),
    ChannelMap("cDAQ9189-233EBC6Mod5/ai0", 4, "X", 0.8),
    ChannelMap("cDAQ9189-233EBC6Mod5/ai1", 4, "Z", 0.8),
    ChannelMap("cDAQ9189-233EBC6Mod5/ai2", 5, "X", 1.0),
]

# Mounted/connected channels allowlist for NI-9230 modal test setup.
# Rationale: NI-9230 does not expose reliable open-channel detection for this setup,
# so acquisition should only include channels known to have sensors mounted.
MOUNTED_CHANNELS = {
    "cDAQ9189-233EBC6Mod3/ai0",
    "cDAQ9189-233EBC6Mod3/ai1",
    "cDAQ9189-233EBC6Mod3/ai2",
    "cDAQ9189-233EBC6Mod4/ai0",
    "cDAQ9189-233EBC6Mod4/ai1",
    "cDAQ9189-233EBC6Mod4/ai2",
    "cDAQ9189-233EBC6Mod5/ai0",
    "cDAQ9189-233EBC6Mod5/ai1",
}


def discover_available_ai_channels() -> list[str]:
    channels: list[str] = []
    system = nidaqmx.system.System.local()
    for dev in system.devices:
        try:
            channels.extend(ch.name for ch in dev.ai_physical_chans)
        except Exception:
            continue
    return sorted(set(channels))


def resolve_channel_map(channel_map: list[ChannelMap]) -> tuple[list[ChannelMap], list[str]]:
    available = discover_available_ai_channels()
    by_suffix = {ch.split("/")[-2] + "/" + ch.split("/")[-1]: ch for ch in available if "/" in ch}

    resolved: list[ChannelMap] = []
    missing: list[str] = []
    for m in channel_map:
        if m.physical_channel not in MOUNTED_CHANNELS:
            continue

        if m.physical_channel in available:
            resolved.append(m)
            continue

        suffix = m.physical_channel.split("/")
        if len(suffix) == 2:
            key = suffix[0] + "/" + suffix[1]
        else:
            key = "/".join(suffix[-2:]) if len(suffix) >= 2 else m.physical_channel

        ch_match = by_suffix.get(key)
        if ch_match is None:
            missing.append(m.physical_channel)
            continue

        resolved.append(ChannelMap(ch_match, m.sensor_id, m.axis, m.y_pos))

    return resolved, missing


class ModalAnalysisApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("cDAQ Modal Analysis GUI (FFT + Mode Animation)")
        self.root.geometry("1380x900")

        self.sample_rate_var = tk.StringVar(value="12800")
        self.duration_var = tk.StringVar(value="10")
        self.chunk_var = tk.StringVar(value="1024")
        self.min_var = tk.StringVar(value="-10")
        self.max_var = tk.StringVar(value="10")
        self.fft_peaks_var = tk.StringVar(value="6")
        self.auto_modes_var = tk.StringVar(value="3")
        self.sensitivity_var = tk.StringVar(value="100")
        self.line_ref_var = tk.StringVar(value="0")
        self.axis_for_mode_var = tk.StringVar(value="Z")
        self.mode_freq_var = tk.StringVar(value="0")

        self.status_var = tk.StringVar(value="Ready")

        self.running = False
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.channel_map = list(DEFAULT_CHANNEL_MAP)
        self.channels = [m.physical_channel for m in self.channel_map]

        self.data_matrix = np.zeros((len(self.channels), 0), dtype=float)
        self.latest_chunk = np.zeros((len(self.channels), 0), dtype=float)
        self.capture_done = False

        self.freq = np.array([])
        self.fft_mag = np.array([])  # shape: (n_ch, n_freq)
        self.fft_top_table: list[tuple[str, float, float]] = []
        self.mode_candidates: list[float] = []

        self.anim_phase = 0.0
        self.anim: FuncAnimation | None = None

        self._build_ui()
        self._draw_mapping_table()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Fs (Hz)").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctrl, textvariable=self.sample_rate_var, width=10).grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Duration (s)").grid(row=0, column=2, sticky="w")
        ttk.Entry(ctrl, textvariable=self.duration_var, width=8).grid(row=0, column=3, padx=4)

        ttk.Label(ctrl, text="Chunk").grid(row=0, column=4, sticky="w")
        ttk.Entry(ctrl, textvariable=self.chunk_var, width=8).grid(row=0, column=5, padx=4)

        ttk.Label(ctrl, text="Min V").grid(row=0, column=6, sticky="w")
        ttk.Entry(ctrl, textvariable=self.min_var, width=8).grid(row=0, column=7, padx=4)

        ttk.Label(ctrl, text="Max V").grid(row=0, column=8, sticky="w")
        ttk.Entry(ctrl, textvariable=self.max_var, width=8).grid(row=0, column=9, padx=4)

        ttk.Label(ctrl, text="FFT Peaks").grid(row=0, column=10, sticky="w")
        ttk.Entry(ctrl, textvariable=self.fft_peaks_var, width=6).grid(row=0, column=11, padx=4)

        ttk.Label(ctrl, text="Auto Modes").grid(row=0, column=12, sticky="w")
        ttk.Entry(ctrl, textvariable=self.auto_modes_var, width=6).grid(row=0, column=13, padx=4)

        ttk.Label(ctrl, text="Sensitivity (mV/g)").grid(row=0, column=14, sticky="w")
        ttk.Entry(ctrl, textvariable=self.sensitivity_var, width=8).grid(row=0, column=15, padx=4)

        ttk.Label(ctrl, text="Line Ref (Hz)").grid(row=0, column=16, sticky="w")
        ttk.Entry(ctrl, textvariable=self.line_ref_var, width=6).grid(row=0, column=17, padx=4)

        ttk.Label(ctrl, text="Mode Axis").grid(row=0, column=18, sticky="w")
        ttk.Combobox(
            ctrl,
            textvariable=self.axis_for_mode_var,
            values=["X", "Z"],
            state="readonly",
            width=4,
        ).grid(row=0, column=19, padx=4)

        ttk.Label(ctrl, text="Mode Freq (Hz)").grid(row=0, column=20, sticky="w")
        ttk.Entry(ctrl, textvariable=self.mode_freq_var, width=8).grid(row=0, column=21, padx=4)

        self.start_btn = ttk.Button(ctrl, text="Start 10s Capture", command=self.start_capture)
        self.start_btn.grid(row=0, column=22, padx=(10, 4))

        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=23, padx=4)

        self.mode_btn = ttk.Button(ctrl, text="Apply Mode Freq", command=self.apply_mode_frequency)
        self.mode_btn.grid(row=0, column=24, padx=4)

        self.auto_extract_btn = ttk.Button(
            ctrl,
            text="Auto Peak + Extract",
            command=self.auto_pick_and_extract,
        )
        self.auto_extract_btn.grid(row=0, column=25, padx=4)

        ttk.Label(ctrl, textvariable=self.status_var).grid(row=1, column=0, columnspan=26, sticky="w", pady=(6, 0))

        body = ttk.Frame(self.root, padding=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Sensor Mapping (X/Z)").pack(anchor="w")
        self.map_text = tk.Text(left, width=52, height=28)
        self.map_text.pack(fill=tk.Y, expand=False)
        self.map_text.configure(state=tk.DISABLED)

        self.report_text = tk.Text(left, width=52, height=18)
        self.report_text.pack(fill=tk.Y, expand=False, pady=(8, 0))

        fig = Figure(figsize=(11.5, 8.2), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.25)

        self.ax_time = fig.add_subplot(gs[0, 0])
        self.ax_time.set_title("Time Signal (latest chunk, stacked)")
        self.ax_time.set_xlabel("Sample")
        self.ax_time.set_ylabel("Voltage + offset")
        self.ax_time.grid(alpha=0.25)

        self.ax_fft = fig.add_subplot(gs[0, 1])
        self.ax_fft.set_title("FFT Magnitude (selected axis channels)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.ax_fft.grid(alpha=0.25)

        self.ax_mode = fig.add_subplot(gs[1, :])
        self.ax_mode.set_title("Mode Shape Animation")
        self.ax_mode.set_xlabel("Lateral displacement (normalized)")
        self.ax_mode.set_ylabel("Beam height (normalized)")
        self.ax_mode.grid(alpha=0.25)

        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.time_lines = []
        offsets = np.arange(len(self.channels), dtype=float) * 0.03
        self.time_offsets = offsets
        for _ in self.channels:
            (line,) = self.ax_time.plot([], [], lw=0.9)
            self.time_lines.append(line)

        self.fft_lines = []
        for _ in range(6):
            (line,) = self.ax_fft.plot([], [], lw=1.0)
            self.fft_lines.append(line)

        (self.mode_line,) = self.ax_mode.plot([], [], "o-", lw=2.0, ms=6)
        self.mode_base_y = np.array([0.0])

    def _draw_mapping_table(self) -> None:
        self.map_text.configure(state=tk.NORMAL)
        self.map_text.delete("1.0", tk.END)
        self.map_text.insert(tk.END, "physical_channel,sensor_id,axis,y_pos\n")
        for m in self.channel_map:
            self.map_text.insert(
                tk.END,
                f"{m.physical_channel},{m.sensor_id},{m.axis},{m.y_pos:.3f}\n",
            )
        self.map_text.configure(state=tk.DISABLED)

    def _append_report(self, text: str) -> None:
        self.report_text.insert(tk.END, text + "\n")
        self.report_text.see(tk.END)

    def _read_cfg(self) -> CaptureConfig:
        return CaptureConfig(
            sample_rate=float(self.sample_rate_var.get()),
            duration_sec=float(self.duration_var.get()),
            chunk_size=int(self.chunk_var.get()),
            min_v=float(self.min_var.get()),
            max_v=float(self.max_var.get()),
            fft_peaks=max(1, int(self.fft_peaks_var.get())),
            auto_modes=max(1, int(self.auto_modes_var.get())),
            sensitivity_mv_g=float(self.sensitivity_var.get()),
            line_ref_hz=float(self.line_ref_var.get()),
            out_dir=Path("outputs"),
        )

    def start_capture(self) -> None:
        if self.running:
            return

        try:
            cfg = self._read_cfg()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        resolved_map, missing = resolve_channel_map(self.channel_map)
        if not resolved_map:
            msg = "No mapped AI channels are currently available. Check module power/recognition in NI MAX."
            if missing:
                msg += "\nMissing examples: " + ", ".join(missing[:4])
            messagebox.showerror("Channel Error", msg)
            return

        self.channel_map = resolved_map
        self.channels = [m.physical_channel for m in self.channel_map]
        self._draw_mapping_table()

        self.running = True
        self.capture_done = False
        self.stop_event.clear()
        self.data_matrix = np.zeros((len(self.channels), 0), dtype=float)
        self.latest_chunk = np.zeros((len(self.channels), 0), dtype=float)

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Capturing...")
        self._append_report("=== New Capture ===")

        self.worker = threading.Thread(target=self._capture_worker, args=(cfg,), daemon=True)
        self.worker.start()
        self._schedule_ui_refresh()

    def stop_capture(self) -> None:
        self.stop_event.set()

    def _capture_worker(self, cfg: CaptureConfig) -> None:
        target_samples = int(cfg.sample_rate * cfg.duration_sec)
        pieces: list[np.ndarray] = []
        total = 0
        t_start = time.perf_counter()
        err: Exception | None = None

        try:
            with nidaqmx.Task() as task:
                for ch in self.channels:
                    task.ai_channels.add_ai_voltage_chan(ch, min_val=cfg.min_v, max_val=cfg.max_v)
                task.timing.cfg_samp_clk_timing(
                    rate=cfg.sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=max(4 * cfg.chunk_size, 4096),
                )

                while not self.stop_event.is_set() and total < target_samples:
                    n = min(cfg.chunk_size, target_samples - total)
                    data = task.read(number_of_samples_per_channel=n, timeout=5.0)
                    arr = np.asarray(data, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    pieces.append(arr)
                    total += arr.shape[1]
                    self.latest_chunk = arr

        except Exception as exc:
            err = exc

        elapsed = time.perf_counter() - t_start
        self.running = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

        if err is not None:
            self.status_var.set("Capture failed")
            self._append_report(f"[ERROR] {type(err).__name__}: {err}")
            return

        if pieces:
            self.data_matrix = np.concatenate(pieces, axis=1)
        else:
            self.data_matrix = np.zeros((len(self.channels), 0), dtype=float)

        self.capture_done = True
        self.status_var.set(f"Capture done ({self.data_matrix.shape[1]} samples/ch, {elapsed:.2f}s)")

        try:
            self._compute_fft_and_report(cfg)
        except Exception as exc:
            self._append_report(f"[ERROR] FFT/report failed: {exc}")

    def _schedule_ui_refresh(self) -> None:
        self._update_plots()
        if self.running:
            self.root.after(120, self._schedule_ui_refresh)

    def _update_plots(self) -> None:
        latest = self.latest_chunk
        if latest.size > 0:
            x = np.arange(latest.shape[1])
            y_min = float("inf")
            y_max = float("-inf")
            for i, line in enumerate(self.time_lines):
                y = latest[i] + self.time_offsets[i]
                line.set_data(x, y)
                y_min = min(y_min, float(np.min(y)))
                y_max = max(y_max, float(np.max(y)))
            self.ax_time.set_xlim(0, max(1, latest.shape[1]))
            if y_max <= y_min:
                y_min -= 1e-6
                y_max += 1e-6
            margin = max(1e-6, 0.1 * (y_max - y_min))
            self.ax_time.set_ylim(y_min - margin, y_max + margin)

        self.canvas.draw_idle()

    def _compute_fft_and_report(self, cfg: CaptureConfig) -> None:
        data = self.data_matrix
        if data.size == 0:
            self._append_report("[WARN] No captured data")
            return

        n_ch, n = data.shape
        win = windows.hann(n, sym=False)
        freq = np.fft.rfftfreq(n, d=1.0 / cfg.sample_rate)

        mag_all = []
        peak_rows = []

        for i, ch in enumerate(self.channels):
            g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)
            y = (data[i] - np.mean(data[i])) * g_per_v
            spec = np.fft.rfft(y * win)
            mag = np.abs(spec) / max(1, n)
            mag_all.append(mag)

            valid = (freq >= 0.5) & (freq <= min(5000.0, cfg.sample_rate * 0.5))
            f_v = freq[valid]
            m_v = mag[valid]

            if m_v.size > 8:
                peaks, _ = find_peaks(m_v)
                if peaks.size > 0:
                    top = peaks[np.argsort(m_v[peaks])[-cfg.fft_peaks :]]
                    top = top[np.argsort(m_v[top])[::-1]]
                    for idx in top:
                        peak_rows.append((ch, float(f_v[idx]), float(m_v[idx])))

        self.freq = freq
        self.fft_mag = np.asarray(mag_all, dtype=float)
        self.fft_top_table = peak_rows

        self._draw_fft_axis()
        self._build_mode_candidates(cfg)
        self._export_reports(cfg)

    def _draw_fft_axis(self) -> None:
        self.ax_fft.cla()
        self.ax_fft.set_title("FFT Magnitude (selected axis channels)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.ax_fft.grid(alpha=0.25)

        if self.freq.size == 0 or self.fft_mag.size == 0:
            self.canvas.draw_idle()
            return

        axis_sel = self.axis_for_mode_var.get().strip().upper()
        labels = []
        for i, m in enumerate(self.channel_map):
            if m.axis == axis_sel:
                labels.append((i, f"S{m.sensor_id}_{m.axis}"))

        max_lines = min(len(labels), 8)
        for i in range(max_lines):
            ch_idx, lbl = labels[i]
            self.ax_fft.plot(self.freq, self.fft_mag[ch_idx], lw=1.0, label=lbl)

        self.ax_fft.set_xlim(0.0, min(5000.0, self.freq[-1]))
        self.ax_fft.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def _build_mode_candidates(self, cfg: CaptureConfig) -> None:
        axis_sel = self.axis_for_mode_var.get().strip().upper()
        idxs = [i for i, m in enumerate(self.channel_map) if m.axis == axis_sel]
        if not idxs or self.freq.size == 0:
            self.mode_candidates = []
            return

        avg = np.mean(self.fft_mag[idxs], axis=0)
        valid = (self.freq >= 0.5) & (self.freq <= min(5000.0, cfg.sample_rate * 0.5))
        f = self.freq[valid]
        a = avg[valid]

        peaks, _ = find_peaks(a)
        if peaks.size == 0:
            self.mode_candidates = []
            return

        top = peaks[np.argsort(a[peaks])[-cfg.fft_peaks :]]
        top = top[np.argsort(a[top])[::-1]]
        self.mode_candidates = [float(f[i]) for i in top]

        if self.mode_candidates:
            self.mode_freq_var.set(f"{self.mode_candidates[0]:.2f}")
            self._append_report("Mode candidates (Hz): " + ", ".join(f"{x:.2f}" for x in self.mode_candidates))

    def _export_reports(self, cfg: CaptureConfig) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = cfg.out_dir / f"modal_capture_{ts}_raw.csv"
        fft_path = cfg.out_dir / f"modal_capture_{ts}_fft_report.csv"
        summary_path = cfg.out_dir / f"modal_capture_{ts}_summary.txt"
        overview_path = cfg.out_dir / f"modal_capture_{ts}_overview.png"

        n_samples = self.data_matrix.shape[1]
        t = np.arange(n_samples, dtype=float) / cfg.sample_rate

        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", *self.channels])
            for k in range(n_samples):
                row = [f"{t[k]:.9f}"] + [f"{self.data_matrix[i, k]:.9e}" for i in range(len(self.channels))]
                writer.writerow(row)

        with fft_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["channel", "peak_freq_hz", "peak_mag"])
            for row in self.fft_top_table:
                writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.9e}"])

        with summary_path.open("w", encoding="utf-8") as f:
            f.write("Modal Analysis Summary\n")
            f.write(f"sample_rate_hz={cfg.sample_rate}\n")
            f.write(f"duration_sec={cfg.duration_sec}\n")
            f.write(f"samples_per_channel={n_samples}\n")
            f.write(f"sensitivity_mv_g={cfg.sensitivity_mv_g}\n")
            f.write("\nChannel Stats (Voltage):\n")
            for i, ch in enumerate(self.channels):
                y = self.data_matrix[i]
                mean = float(np.mean(y))
                rms = float(np.sqrt(np.mean(np.square(y))))
                ptp = float(np.ptp(y))
                g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)
                mean_g = mean * g_per_v
                rms_g = rms * g_per_v
                ptp_g = ptp * g_per_v
                f.write(
                    f"{ch}: mean={mean:.9e} V, rms={rms:.9e} V, pkpk={ptp:.9e} V, "
                    f"mean={mean_g:.9e} g, rms={rms_g:.9e} g, pkpk={ptp_g:.9e} g\n"
                )

            f.write("\nDC Offset Check (0-center diagnostics):\n")
            f.write("criteria=abs(mean)/ac_rms, notice_if_ratio>0.20\n")
            for i, ch in enumerate(self.channels):
                y = self.data_matrix[i]
                mean = float(np.mean(y))
                ac_rms = float(np.sqrt(np.mean(np.square(y - mean))))
                ratio = abs(mean) / max(1e-12, ac_rms)
                status = "OFFSET_NOTICE" if ratio > 0.20 else "OK"
                f.write(
                    f"{ch}: mean={mean:.9e} V, ac_rms={ac_rms:.9e} V, "
                    f"ratio={ratio:.3f}, status={status}\n"
                )

            f.write("\nMode candidates (Hz):\n")
            if self.mode_candidates:
                f.write(", ".join(f"{x:.4f}" for x in self.mode_candidates) + "\n")
                if cfg.line_ref_hz > 0:
                    nearest = min(self.mode_candidates, key=lambda x: abs(x - cfg.line_ref_hz))
                    f.write(
                        f"Nearest_to_{cfg.line_ref_hz:.2f}Hz={nearest:.4f}Hz, "
                        f"delta={abs(nearest-cfg.line_ref_hz):.4f}Hz\n"
                    )
            else:
                f.write("None\n")

        self._save_overview_png(overview_path, cfg)

        self._append_report(f"Saved raw: {raw_path}")
        self._append_report(f"Saved fft report: {fft_path}")
        self._append_report(f"Saved summary: {summary_path}")
        self._append_report(f"Saved overview: {overview_path}")

    def _save_overview_png(self, overview_path: Path, cfg: CaptureConfig) -> None:
        if self.data_matrix.size == 0 or self.freq.size == 0 or self.fft_mag.size == 0:
            return

        n_ch, n = self.data_matrix.shape
        t = np.arange(n, dtype=float) / cfg.sample_rate
        valid = (self.freq >= 0.5) & (self.freq <= min(5000.0, cfg.sample_rate * 0.5))
        f = self.freq[valid]

        n_rows = n_ch + 1
        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(15, max(9, 1.8 * n_rows)),
            dpi=130,
            sharex="col",
        )
        axes = np.atleast_2d(axes)

        peak_map: dict[str, list[tuple[float, float]]] = {}
        for ch, f_peak, m_peak in self.fft_top_table:
            peak_map.setdefault(ch, []).append((f_peak, m_peak))
        for ch in peak_map:
            peak_map[ch].sort(key=lambda x: x[1], reverse=True)

        y_time_min = float(np.min(self.data_matrix))
        y_time_max = float(np.max(self.data_matrix))
        if y_time_max <= y_time_min:
            y_time_min -= 1e-6
            y_time_max += 1e-6

        y_fft_min = 0.0
        y_fft_max = float(np.max(self.fft_mag[:, valid])) if np.any(valid) else 1.0
        y_fft_max = max(y_fft_max, 1e-12)

        for i in range(n_ch):
            ax_t = axes[i, 0]
            ax_f = axes[i, 1]
            ch_name = self.channels[i]

            ax_t.plot(t, self.data_matrix[i], lw=0.8, color="tab:blue")
            ax_t.axhline(0.0, color="tab:gray", ls=":", lw=0.8)
            ax_t.set_ylim(y_time_min, y_time_max)
            ax_t.set_ylabel("V")
            ax_t.grid(alpha=0.25)
            ax_t.set_title(f"{ch_name} Time")

            ax_f.plot(f, self.fft_mag[i][valid], lw=0.9, color="tab:purple")
            if cfg.line_ref_hz > 0:
                ax_f.axvline(cfg.line_ref_hz, color="tab:green", ls=":", lw=1.0)
            for fm in self.mode_candidates[: cfg.auto_modes]:
                ax_f.axvline(fm, color="tab:red", ls="--", lw=0.8)
            ch_peaks = peak_map.get(ch_name, [])[: cfg.fft_peaks]
            if ch_peaks:
                pfx = [x[0] for x in ch_peaks]
                pmy = [x[1] for x in ch_peaks]
                ax_f.scatter(pfx, pmy, s=14, color="tab:orange", zorder=5)
            ax_f.set_ylim(y_fft_min, y_fft_max)
            ax_f.grid(alpha=0.25)
            ax_f.set_title(f"{ch_name} FFT")

        ax_t_all = axes[-1, 0]
        ax_f_all = axes[-1, 1]

        for i, ch_name in enumerate(self.channels):
            ax_t_all.plot(t, self.data_matrix[i], lw=0.7, alpha=0.18, label=ch_name)
        mean_trace = np.mean(self.data_matrix, axis=0)
        ax_t_all.plot(t, mean_trace, lw=1.5, color="black", alpha=0.9, label="mean")
        ax_t_all.axhline(0.0, color="tab:gray", ls=":", lw=0.8)
        ax_t_all.set_ylim(y_time_min, y_time_max)
        ax_t_all.set_ylabel("V")
        ax_t_all.set_title("All Channels Time Overlay (alpha)")
        ax_t_all.grid(alpha=0.25)

        for i, ch_name in enumerate(self.channels):
            ax_f_all.plot(f, self.fft_mag[i][valid], lw=0.7, alpha=0.18, label=ch_name)
        mean_fft = np.mean(self.fft_mag[:, valid], axis=0)
        ax_f_all.plot(f, mean_fft, lw=1.5, color="black", alpha=0.9, label="mean")
        if cfg.line_ref_hz > 0:
            ax_f_all.axvline(cfg.line_ref_hz, color="tab:green", ls=":", lw=1.0)
        for fm in self.mode_candidates[: cfg.auto_modes]:
            ax_f_all.axvline(fm, color="tab:red", ls="--", lw=0.8)

        peaks_all, _ = find_peaks(mean_fft)
        if peaks_all.size > 0:
            top_all = peaks_all[np.argsort(mean_fft[peaks_all])[-cfg.fft_peaks :]]
            top_all = top_all[np.argsort(mean_fft[top_all])[::-1]]
            ax_f_all.scatter(f[top_all], mean_fft[top_all], s=16, color="tab:orange", zorder=6)

        ax_f_all.set_ylim(y_fft_min, y_fft_max)
        ax_f_all.set_title("All Channels FFT Overlay (alpha)")
        ax_f_all.grid(alpha=0.25)

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude (g)")

        if self.mode_candidates and cfg.line_ref_hz > 0:
            nearest = min(self.mode_candidates, key=lambda x: abs(x - cfg.line_ref_hz))
            fig.suptitle(
                f"Per-Channel Overview | Nearest mode to {cfg.line_ref_hz:.2f} Hz: "
                f"{nearest:.2f} Hz (delta {abs(nearest-cfg.line_ref_hz):.2f} Hz)",
                fontsize=12,
            )
        elif self.mode_candidates:
            shown = ", ".join(f"{x:.2f}" for x in self.mode_candidates[: cfg.auto_modes])
            fig.suptitle(f"Per-Channel Overview | Auto mode peaks: {shown} Hz", fontsize=12)
        else:
            fig.suptitle("Per-Channel Overview | No auto mode candidates", fontsize=12)

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(overview_path)
        plt.close(fig)

    def apply_mode_frequency(self) -> None:
        if self.freq.size == 0 or self.fft_mag.size == 0 or self.data_matrix.size == 0:
            messagebox.showwarning("No FFT", "Run a capture first.")
            return

        try:
            f_mode = float(self.mode_freq_var.get())
        except Exception:
            messagebox.showerror("Input Error", "Mode frequency must be numeric")
            return

        self._start_mode_animation(f_mode)
        try:
            cfg = self._read_cfg()
            paths = self._extract_frequency_component(f_mode, cfg, prefix="gui_manual")
            self._append_report(f"Extracted component table: {paths[0]}")
            self._append_report(f"Extracted component wave: {paths[1]}")
        except Exception as exc:
            self._append_report(f"[WARN] component extraction skipped: {exc}")

    def auto_pick_and_extract(self) -> None:
        if self.freq.size == 0 or self.fft_mag.size == 0 or self.data_matrix.size == 0:
            messagebox.showwarning("No FFT", "Run a capture first.")
            return

        cfg = self._read_cfg()
        if not self.mode_candidates:
            self._build_mode_candidates(cfg)

        if not self.mode_candidates:
            messagebox.showwarning("No Peaks", "No mode candidate peaks were found.")
            return

        selected = self.mode_candidates[: cfg.auto_modes]
        self._append_report("Auto-selected peak frequencies: " + ", ".join(f"{x:.4f}Hz" for x in selected))

        # Keep animation tied to the strongest mode for interactive inspection.
        f_mode = float(selected[0])
        self.mode_freq_var.set(f"{f_mode:.2f}")
        self._start_mode_animation(f_mode)

        for i, fm in enumerate(selected, start=1):
            try:
                paths = self._extract_frequency_component(float(fm), cfg, prefix=f"gui_auto_mode{i}")
                self._append_report(f"Mode {i} component table: {paths[0]}")
                self._append_report(f"Mode {i} component wave: {paths[1]}")
            except Exception as exc:
                self._append_report(f"[WARN] auto extraction failed for mode {i}: {exc}")

    def _extract_frequency_component(self, f_mode: float, cfg: CaptureConfig, prefix: str) -> tuple[Path, Path]:
        if self.data_matrix.size == 0:
            raise RuntimeError("No captured data")

        n = self.data_matrix.shape[1]
        if n <= 0:
            raise RuntimeError("Empty capture")

        fs = float(cfg.sample_rate)
        t = np.arange(n, dtype=float) / fs
        win = windows.hann(n, sym=False)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        idx = int(np.argmin(np.abs(freq - f_mode)))
        f_sel = float(freq[idx])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        table_path = cfg.out_dir / f"{prefix}_{ts}_component_{f_sel:.2f}Hz.csv"
        wave_path = cfg.out_dir / f"{prefix}_{ts}_wave_{f_sel:.2f}Hz.csv"

        g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)

        table_rows = []
        wave_dict: dict[str, np.ndarray] = {"time_s": t}
        for i, ch in enumerate(self.channels):
            yv = self.data_matrix[i] - np.mean(self.data_matrix[i])
            yg = yv * g_per_v
            spec = np.fft.rfft(yg * win)
            c = spec[idx]

            # Single-frequency reconstruction using estimated amplitude/phase.
            amp_pk = float(2.0 * np.abs(c) / max(1e-12, np.sum(win)))
            phase = float(np.angle(c))
            recon = amp_pk * np.cos(2.0 * np.pi * f_sel * t + phase)

            m = self.channel_map[i]
            table_rows.append(
                {
                    "channel": ch,
                    "sensor_id": m.sensor_id,
                    "axis": m.axis,
                    "selected_freq_hz": f"{f_sel:.6f}",
                    "amplitude_pk_g": f"{amp_pk:.9e}",
                    "amplitude_rms_g": f"{(amp_pk / np.sqrt(2.0)):.9e}",
                    "phase_deg": f"{np.degrees(phase):.6f}",
                }
            )
            wave_dict[ch] = recon

        with table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "channel",
                    "sensor_id",
                    "axis",
                    "selected_freq_hz",
                    "amplitude_pk_g",
                    "amplitude_rms_g",
                    "phase_deg",
                ],
            )
            writer.writeheader()
            writer.writerows(table_rows)

        with wave_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time_s", *self.channels]
            writer.writerow(header)
            for k in range(n):
                writer.writerow([f"{t[k]:.9f}"] + [f"{wave_dict[ch][k]:.9e}" for ch in self.channels])

        return table_path, wave_path

    def _start_mode_animation(self, f_mode: float) -> None:
        axis_sel = self.axis_for_mode_var.get().strip().upper()
        channel_entries = [m for m in self.channel_map if m.axis == axis_sel]
        if not channel_entries:
            messagebox.showwarning("No channels", f"No {axis_sel}-axis mapping found")
            return

        # Build sensor response at chosen frequency from complex FFT bins.
        n = self.data_matrix.shape[1]
        win = windows.hann(n, sym=False)
        freq = np.fft.rfftfreq(n, d=1.0 / float(self.sample_rate_var.get()))
        idx = int(np.argmin(np.abs(freq - f_mode)))

        y_pos = []
        amp = []
        phs = []

        for m in channel_entries:
            ch_i = self.channels.index(m.physical_channel)
            y = self.data_matrix[ch_i] - np.mean(self.data_matrix[ch_i])
            spec = np.fft.rfft(y * win)
            c = spec[idx]
            y_pos.append(m.y_pos)
            amp.append(float(np.abs(c)))
            phs.append(float(np.angle(c)))

        y_pos_arr = np.asarray(y_pos, dtype=float)
        amp_arr = np.asarray(amp, dtype=float)
        phs_arr = np.asarray(phs, dtype=float)

        if np.max(amp_arr) <= 0:
            amp_arr = np.ones_like(amp_arr)
        amp_arr = amp_arr / np.max(amp_arr)

        # Include fixed base point for cantilever visualization.
        y_nodes = np.concatenate(([0.0], y_pos_arr))

        self.ax_mode.cla()
        self.ax_mode.set_title(f"Mode Shape Animation @ {freq[idx]:.2f} Hz ({axis_sel}-axis)")
        self.ax_mode.set_xlabel("Lateral displacement (normalized)")
        self.ax_mode.set_ylabel("Beam height (normalized)")
        self.ax_mode.grid(alpha=0.25)
        self.ax_mode.set_ylim(-0.02, 1.05)
        self.ax_mode.set_xlim(-1.2, 1.2)

        (self.mode_line,) = self.ax_mode.plot([], [], "o-", lw=2.0, ms=6)

        self.anim_phase = 0.0

        def update(_frame):
            self.anim_phase += 0.20
            disp_nodes = [0.0]
            for a, p in zip(amp_arr, phs_arr):
                disp_nodes.append(float(a * np.cos(self.anim_phase + p)))
            self.mode_line.set_data(disp_nodes, y_nodes)
            return (self.mode_line,)

        if self.anim is not None:
            self.anim.event_source.stop()

        self.anim = FuncAnimation(self.ax_mode.figure, update, interval=50, blit=True)
        self.canvas.draw_idle()

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrated modal analysis GUI")
    parser.add_argument("--batch-run", action="store_true", help="Run one capture+FFT report without GUI")
    parser.add_argument("--sample-rate", type=float, default=12800.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--min", dest="min_v", type=float, default=-10.0)
    parser.add_argument("--max", dest="max_v", type=float, default=10.0)
    parser.add_argument("--fft-peaks", type=int, default=6)
    parser.add_argument("--auto-modes", type=int, default=3)
    parser.add_argument("--sensitivity-mv-g", type=float, default=100.0)
    parser.add_argument("--line-ref-hz", type=float, default=0.0)
    args = parser.parse_args()

    if args.batch_run:
        cfg = CaptureConfig(
            sample_rate=args.sample_rate,
            duration_sec=args.duration,
            chunk_size=args.chunk,
            min_v=args.min_v,
            max_v=args.max_v,
            fft_peaks=max(1, args.fft_peaks),
            auto_modes=max(1, args.auto_modes),
            sensitivity_mv_g=args.sensitivity_mv_g,
            line_ref_hz=args.line_ref_hz,
            out_dir=Path("outputs"),
        )
        return run_batch_capture_and_report(cfg)

    root = tk.Tk()
    ModalAnalysisApp(root)
    root.mainloop()
    return 0


def run_batch_capture_and_report(cfg: CaptureConfig) -> int:
    active_map, missing = resolve_channel_map(list(DEFAULT_CHANNEL_MAP))
    channels = [m.physical_channel for m in active_map]
    if not channels:
        print("[BATCH][ERROR] No mapped AI channels are currently available.")
        if missing:
            print("[BATCH][ERROR] Missing mapped channels (sample): " + ", ".join(missing[:5]))
        return 3
    pieces: list[np.ndarray] = []
    total = 0
    target_samples = int(cfg.sample_rate * cfg.duration_sec)

    print(f"[BATCH] capturing {len(channels)} channels for {cfg.duration_sec:.2f}s @ {cfg.sample_rate:.1f} Hz")

    with nidaqmx.Task() as task:
        for ch in channels:
            task.ai_channels.add_ai_voltage_chan(ch, min_val=cfg.min_v, max_val=cfg.max_v)

        task.timing.cfg_samp_clk_timing(
            rate=cfg.sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=max(4 * cfg.chunk_size, 4096),
        )

        while total < target_samples:
            n = min(cfg.chunk_size, target_samples - total)
            data = task.read(number_of_samples_per_channel=n, timeout=5.0)
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            pieces.append(arr)
            total += arr.shape[1]

    data_matrix = np.concatenate(pieces, axis=1) if pieces else np.zeros((len(channels), 0), dtype=float)
    if data_matrix.shape[1] == 0:
        print("[BATCH][ERROR] no data captured")
        return 2

    n = data_matrix.shape[1]
    win = windows.hann(n, sym=False)
    freq = np.fft.rfftfreq(n, d=1.0 / cfg.sample_rate)

    peak_rows: list[tuple[str, float, float]] = []
    for i, ch in enumerate(channels):
        g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)
        y = (data_matrix[i] - np.mean(data_matrix[i])) * g_per_v
        spec = np.fft.rfft(y * win)
        mag = np.abs(spec) / max(1, n)
        valid = (freq >= 0.5) & (freq <= min(5000.0, cfg.sample_rate * 0.5))
        f_v = freq[valid]
        m_v = mag[valid]
        if m_v.size > 8:
            peaks, _ = find_peaks(m_v)
            if peaks.size > 0:
                top = peaks[np.argsort(m_v[peaks])[-cfg.fft_peaks :]]
                top = top[np.argsort(m_v[top])[::-1]]
                for idx in top:
                    peak_rows.append((ch, float(f_v[idx]), float(m_v[idx])))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cfg.out_dir / f"modal_capture_{ts}_raw.csv"
    fft_path = cfg.out_dir / f"modal_capture_{ts}_fft_report.csv"
    summary_path = cfg.out_dir / f"modal_capture_{ts}_summary.txt"
    overview_path = cfg.out_dir / f"modal_capture_{ts}_overview.png"

    t = np.arange(n, dtype=float) / cfg.sample_rate
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", *channels])
        for k in range(n):
            writer.writerow([f"{t[k]:.9f}"] + [f"{data_matrix[i, k]:.9e}" for i in range(len(channels))])

    with fft_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "peak_freq_hz", "peak_mag"])
        for row in peak_rows:
            writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.9e}"])

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Modal Analysis Summary\n")
        f.write(f"sample_rate_hz={cfg.sample_rate}\n")
        f.write(f"duration_sec={cfg.duration_sec}\n")
        f.write(f"samples_per_channel={n}\n")
        f.write(f"sensitivity_mv_g={cfg.sensitivity_mv_g}\n")
        f.write("\nChannel Stats (Voltage):\n")
        for i, ch in enumerate(channels):
            y = data_matrix[i]
            mean = float(np.mean(y))
            rms = float(np.sqrt(np.mean(np.square(y))))
            ptp = float(np.ptp(y))
            g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)
            mean_g = mean * g_per_v
            rms_g = rms * g_per_v
            ptp_g = ptp * g_per_v
            f.write(
                f"{ch}: mean={mean:.9e} V, rms={rms:.9e} V, pkpk={ptp:.9e} V, "
                f"mean={mean_g:.9e} g, rms={rms_g:.9e} g, pkpk={ptp_g:.9e} g\n"
            )

        f.write("\nDC Offset Check (0-center diagnostics):\n")
        f.write("criteria=abs(mean)/ac_rms, notice_if_ratio>0.20\n")
        for i, ch in enumerate(channels):
            y = data_matrix[i]
            mean = float(np.mean(y))
            ac_rms = float(np.sqrt(np.mean(np.square(y - mean))))
            ratio = abs(mean) / max(1e-12, ac_rms)
            status = "OFFSET_NOTICE" if ratio > 0.20 else "OK"
            f.write(
                f"{ch}: mean={mean:.9e} V, ac_rms={ac_rms:.9e} V, "
                f"ratio={ratio:.3f}, status={status}\n"
            )

    # Auto-pick dominant peaks from selected axis (default Z) and export extracted components.
    axis_sel = "Z"
    axis_idxs = [i for i, m in enumerate(active_map) if m.axis == axis_sel]
    selected_freqs: list[float] = []
    if axis_idxs:
        win = windows.hann(n, sym=False)
        avg_mag = np.zeros_like(freq)
        g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)
        for i in axis_idxs:
            y = (data_matrix[i] - np.mean(data_matrix[i])) * g_per_v
            c = np.fft.rfft(y * win)
            avg_mag += np.abs(c) / max(1, n)
        avg_mag /= max(1, len(axis_idxs))

        valid = (freq >= 0.5) & (freq <= min(5000.0, cfg.sample_rate * 0.5))
        f_v = freq[valid]
        m_v = avg_mag[valid]
        peaks, _ = find_peaks(m_v)
        if peaks.size > 0:
            top = peaks[np.argsort(m_v[peaks])[-cfg.auto_modes :]]
            top = top[np.argsort(m_v[top])[::-1]]
            selected_freqs = [float(f_v[i]) for i in top]

    if selected_freqs:
        win = windows.hann(n, sym=False)
        f_bins = np.fft.rfftfreq(n, d=1.0 / cfg.sample_rate)
        t = np.arange(n, dtype=float) / cfg.sample_rate
        g_per_v = 1000.0 / max(1e-9, cfg.sensitivity_mv_g)

        for mode_idx, selected_freq in enumerate(selected_freqs, start=1):
            idx = int(np.argmin(np.abs(f_bins - selected_freq)))
            f_sel = float(f_bins[idx])

            extract_table = cfg.out_dir / f"modal_capture_{ts}_auto_mode{mode_idx}_component_{f_sel:.2f}Hz.csv"
            extract_wave = cfg.out_dir / f"modal_capture_{ts}_auto_mode{mode_idx}_wave_{f_sel:.2f}Hz.csv"

            rows = []
            wave_cols: dict[str, np.ndarray] = {"time_s": t}
            for i, ch in enumerate(channels):
                y = (data_matrix[i] - np.mean(data_matrix[i])) * g_per_v
                c = np.fft.rfft(y * win)[idx]
                amp_pk = float(2.0 * np.abs(c) / max(1e-12, np.sum(win)))
                phase = float(np.angle(c))
                wave_cols[ch] = amp_pk * np.cos(2.0 * np.pi * f_sel * t + phase)
                m = active_map[i]
                rows.append(
                    {
                        "channel": ch,
                        "sensor_id": m.sensor_id,
                        "axis": m.axis,
                        "selected_freq_hz": f"{f_sel:.6f}",
                        "amplitude_pk_g": f"{amp_pk:.9e}",
                        "amplitude_rms_g": f"{(amp_pk / np.sqrt(2.0)):.9e}",
                        "phase_deg": f"{np.degrees(phase):.6f}",
                    }
                )

            with extract_table.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "channel",
                        "sensor_id",
                        "axis",
                        "selected_freq_hz",
                        "amplitude_pk_g",
                        "amplitude_rms_g",
                        "phase_deg",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            with extract_wave.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", *channels])
                for k in range(n):
                    writer.writerow([f"{t[k]:.9f}"] + [f"{wave_cols[ch][k]:.9e}" for ch in channels])

            print(f"[BATCH] auto mode {mode_idx} frequency: {f_sel:.4f} Hz")
            print(f"[BATCH] saved auto mode {mode_idx} extract table: {extract_table}")
            print(f"[BATCH] saved auto mode {mode_idx} extract wave: {extract_wave}")

    with summary_path.open("a", encoding="utf-8") as f:
        f.write("\nAuto mode frequencies (Hz):\n")
        if selected_freqs:
            f.write(", ".join(f"{x:.4f}" for x in selected_freqs) + "\n")
            if cfg.line_ref_hz > 0:
                nearest = min(selected_freqs, key=lambda x: abs(x - cfg.line_ref_hz))
                f.write(
                    f"Nearest_to_{cfg.line_ref_hz:.2f}Hz={nearest:.4f}Hz, "
                    f"delta={abs(nearest-cfg.line_ref_hz):.4f}Hz\n"
                )
        else:
            f.write("None\n")

    # Save per-channel overview image: left=time, right=FFT (same scale in each column).
    valid = (freq >= 0.5) & (freq <= min(5000.0, cfg.sample_rate * 0.5))
    fft_matrix = []
    for i in range(len(channels)):
        y = (data_matrix[i] - np.mean(data_matrix[i])) * (1000.0 / max(1e-9, cfg.sensitivity_mv_g))
        c = np.fft.rfft(y * windows.hann(n, sym=False))
        fft_matrix.append(np.abs(c) / max(1, n))
    fft_matrix = np.asarray(fft_matrix, dtype=float)

    peak_map: dict[str, list[tuple[float, float]]] = {}
    for ch, f_peak, m_peak in peak_rows:
        peak_map.setdefault(ch, []).append((f_peak, m_peak))
    for ch in peak_map:
        peak_map[ch].sort(key=lambda x: x[1], reverse=True)

    n_rows = len(channels) + 1
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(15, max(9, 1.8 * n_rows)),
        dpi=130,
        sharex="col",
    )
    axes = np.atleast_2d(axes)

    y_time_min = float(np.min(data_matrix))
    y_time_max = float(np.max(data_matrix))
    if y_time_max <= y_time_min:
        y_time_min -= 1e-6
        y_time_max += 1e-6

    y_fft_min = 0.0
    y_fft_max = float(np.max(fft_matrix[:, valid])) if np.any(valid) else 1.0
    y_fft_max = max(y_fft_max, 1e-12)

    t = np.arange(n, dtype=float) / cfg.sample_rate
    for i, ch_name in enumerate(channels):
        ax_t = axes[i, 0]
        ax_f = axes[i, 1]

        ax_t.plot(t, data_matrix[i], lw=0.8, color="tab:blue")
        ax_t.axhline(0.0, color="tab:gray", ls=":", lw=0.8)
        ax_t.set_ylim(y_time_min, y_time_max)
        ax_t.set_ylabel("V")
        ax_t.grid(alpha=0.25)
        ax_t.set_title(f"{ch_name} Time")

        ax_f.plot(freq[valid], fft_matrix[i][valid], lw=0.9, color="tab:purple")
        if cfg.line_ref_hz > 0:
            ax_f.axvline(cfg.line_ref_hz, color="tab:green", ls=":", lw=1.0)
        for fm in selected_freqs:
            ax_f.axvline(fm, color="tab:red", ls="--", lw=0.8)
        ch_peaks = peak_map.get(ch_name, [])[: cfg.fft_peaks]
        if ch_peaks:
            pfx = [x[0] for x in ch_peaks]
            pmy = [x[1] for x in ch_peaks]
            ax_f.scatter(pfx, pmy, s=14, color="tab:orange", zorder=5)
        ax_f.set_ylim(y_fft_min, y_fft_max)
        ax_f.grid(alpha=0.25)
        ax_f.set_title(f"{ch_name} FFT")

    ax_t_all = axes[-1, 0]
    ax_f_all = axes[-1, 1]

    for i, ch_name in enumerate(channels):
        ax_t_all.plot(t, data_matrix[i], lw=0.7, alpha=0.18, label=ch_name)
    mean_trace = np.mean(data_matrix, axis=0)
    ax_t_all.plot(t, mean_trace, lw=1.5, color="black", alpha=0.9, label="mean")
    ax_t_all.axhline(0.0, color="tab:gray", ls=":", lw=0.8)
    ax_t_all.set_ylim(y_time_min, y_time_max)
    ax_t_all.set_ylabel("V")
    ax_t_all.set_title("All Channels Time Overlay (alpha)")
    ax_t_all.grid(alpha=0.25)

    for i, ch_name in enumerate(channels):
        ax_f_all.plot(freq[valid], fft_matrix[i][valid], lw=0.7, alpha=0.18, label=ch_name)
    mean_fft = np.mean(fft_matrix[:, valid], axis=0)
    ax_f_all.plot(freq[valid], mean_fft, lw=1.5, color="black", alpha=0.9, label="mean")
    if cfg.line_ref_hz > 0:
        ax_f_all.axvline(cfg.line_ref_hz, color="tab:green", ls=":", lw=1.0)
    for fm in selected_freqs:
        ax_f_all.axvline(fm, color="tab:red", ls="--", lw=0.8)

    peaks_all, _ = find_peaks(mean_fft)
    if peaks_all.size > 0:
        top_all = peaks_all[np.argsort(mean_fft[peaks_all])[-cfg.fft_peaks :]]
        top_all = top_all[np.argsort(mean_fft[top_all])[::-1]]
        ax_f_all.scatter(freq[valid][top_all], mean_fft[top_all], s=16, color="tab:orange", zorder=6)

    ax_f_all.set_ylim(y_fft_min, y_fft_max)
    ax_f_all.set_title("All Channels FFT Overlay (alpha)")
    ax_f_all.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude (g)")

    if selected_freqs and cfg.line_ref_hz > 0:
        nearest = min(selected_freqs, key=lambda x: abs(x - cfg.line_ref_hz))
        fig.suptitle(
            f"Per-Channel Overview | Nearest mode to {cfg.line_ref_hz:.2f} Hz: "
            f"{nearest:.2f} Hz (delta {abs(nearest-cfg.line_ref_hz):.2f} Hz)",
            fontsize=12,
        )
    elif selected_freqs:
        shown = ", ".join(f"{x:.2f}" for x in selected_freqs[: cfg.auto_modes])
        fig.suptitle(f"Per-Channel Overview | Auto mode peaks: {shown} Hz", fontsize=12)
    else:
        fig.suptitle("Per-Channel Overview | No auto mode candidates", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(overview_path)
    plt.close(fig)

    print(f"[BATCH] saved raw: {raw_path}")
    print(f"[BATCH] saved fft: {fft_path}")
    print(f"[BATCH] saved summary: {summary_path}")
    print(f"[BATCH] saved overview: {overview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
