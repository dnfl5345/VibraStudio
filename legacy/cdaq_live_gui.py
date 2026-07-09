#!/usr/bin/env python3
"""
Live NI-DAQ viewer for quick signal-presence checks.

- Scans available AI channels
- Streams data from one selected channel
- Displays rolling time-series plot
- Supports voltage mode (default) and accelerometer mode
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import nidaqmx
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from nidaqmx.constants import AcquisitionType
from scipy import signal as spsig
from scipy.stats import kurtosis


class LiveDAQApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NI cDAQ Live Signal Viewer")
        self.root.geometry("1150x820")

        self.running = False
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.sample_rate_var = tk.StringVar(value="12800")
        self.chunk_size_var = tk.StringVar(value="256")
        self.window_sec_var = tk.StringVar(value="3")
        self.mode_var = tk.StringVar(value="voltage")
        self.sensitivity_var = tk.StringVar(value="100")
        self.min_var = tk.StringVar(value="-30")
        self.max_var = tk.StringVar(value="30")
        self.channel_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.detrend_var = tk.BooleanVar(value=True)
        self.bandpass_var = tk.BooleanVar(value=False)
        self.bp_low_var = tk.StringVar(value="10")
        self.bp_high_var = tk.StringVar(value="2000")
        self.envelope_var = tk.BooleanVar(value=False)
        self.auto_peak_var = tk.BooleanVar(value=True)
        self.peak_count_var = tk.StringVar(value="3")

        self.time_buf: deque[float] = deque(maxlen=40000)
        self.data_buf: deque[float] = deque(maxlen=40000)
        self.start_time = 0.0
        self.current_sample_rate = 12800.0
        self.fft_peak_markers = []
        self.fft_peak_texts = []
        self.tf_tick = 0

        self._build_ui()
        self.refresh_channels()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Channel").grid(row=0, column=0, sticky="w")
        self.channel_box = ttk.Combobox(top, textvariable=self.channel_var, width=42, state="readonly")
        self.channel_box.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(top, text="Refresh", command=self.refresh_channels).grid(row=0, column=2, padx=5)

        ttk.Label(top, text="Mode").grid(row=0, column=3, sticky="w", padx=(10, 0))
        ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["voltage", "accel"],
            width=10,
            state="readonly",
        ).grid(row=0, column=4, padx=5)

        ttk.Label(top, text="Fs(Hz)").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.sample_rate_var, width=12).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(top, text="Chunk").grid(row=1, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.chunk_size_var, width=10).grid(row=1, column=3, sticky="w", padx=5)

        ttk.Label(top, text="Window(s)").grid(row=1, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.window_sec_var, width=10).grid(row=1, column=5, sticky="w", padx=5)

        ttk.Label(top, text="Sensitivity(mV/g)").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.sensitivity_var, width=12).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(top, text="Min").grid(row=2, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.min_var, width=10).grid(row=2, column=3, sticky="w", padx=5)

        ttk.Label(top, text="Max").grid(row=2, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.max_var, width=10).grid(row=2, column=5, sticky="w", padx=5)

        ttk.Checkbutton(top, text="Detrend", variable=self.detrend_var).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Checkbutton(top, text="Bandpass", variable=self.bandpass_var).grid(row=3, column=1, sticky="w")
        ttk.Label(top, text="BP Low(Hz)").grid(row=3, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.bp_low_var, width=10).grid(row=3, column=3, sticky="w", padx=5)
        ttk.Label(top, text="BP High(Hz)").grid(row=3, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.bp_high_var, width=10).grid(row=3, column=5, sticky="w", padx=5)

        ttk.Checkbutton(top, text="Envelope(Hilbert)", variable=self.envelope_var).grid(row=4, column=0, sticky="w", pady=2)
        ttk.Checkbutton(top, text="Auto Peaks", variable=self.auto_peak_var).grid(row=4, column=1, sticky="w")
        ttk.Label(top, text="Peak Count").grid(row=4, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.peak_count_var, width=10).grid(row=4, column=3, sticky="w", padx=5)

        btns = ttk.Frame(self.root, padding=(10, 0))
        btns.pack(side=tk.TOP, fill=tk.X)
        self.scan_btn = ttk.Button(btns, text="Auto Scan", command=self.auto_scan_channels)
        self.scan_btn.pack(side=tk.LEFT)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_stream, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)
        ttk.Label(btns, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        fig = Figure(figsize=(10, 8), dpi=100)
        self.ax_time = fig.add_subplot(311)
        self.ax_time.set_title("Live Time Series")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Value")
        self.line_time, = self.ax_time.plot([], [], lw=1.0)
        self.ax_time.grid(True, alpha=0.3)

        self.ax_fft = fig.add_subplot(312)
        self.ax_fft.set_title("Live Spectrum (FFT)")
        self.ax_fft.set_xlabel("Frequency (Hz)")
        self.ax_fft.set_ylabel("Magnitude")
        self.line_fft, = self.ax_fft.plot([], [], lw=1.0)
        self.ax_fft.grid(True, alpha=0.3)

        self.ax_tf = fig.add_subplot(313)
        self.ax_tf.set_title("Hilbert Envelope Spectrogram")
        self.ax_tf.set_xlabel("Time Offset (s)")
        self.ax_tf.set_ylabel("Frequency (Hz)")

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = canvas

        self.stats_label = ttk.Label(
            self.root,
            text="RMS: -, PkPk: -, Mean: -, Fdom: -, Crest: -, Kurtosis: -",
            padding=10,
        )
        self.stats_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _apply_preprocessing(self, y: np.ndarray, fs: float) -> np.ndarray:
        out = y.astype(float, copy=True)
        if out.size < 8:
            return out

        if self.detrend_var.get():
            out = spsig.detrend(out, type="linear")

        if self.bandpass_var.get() and fs > 0:
            try:
                low = float(self.bp_low_var.get())
                high = float(self.bp_high_var.get())
                nyq = 0.5 * fs
                low = max(0.1, min(low, nyq * 0.95))
                high = max(low + 0.1, min(high, nyq * 0.98))
                b, a = spsig.butter(4, [low / nyq, high / nyq], btype="band")
                out = spsig.filtfilt(b, a, out)
            except Exception:
                pass

        return out

    def _clear_fft_peak_labels(self) -> None:
        for marker in self.fft_peak_markers:
            try:
                marker.remove()
            except Exception:
                pass
        self.fft_peak_markers = []

        for txt in self.fft_peak_texts:
            try:
                txt.remove()
            except Exception:
                pass
        self.fft_peak_texts = []

    def _add_fft_peak_labels(self, freq: np.ndarray, mag: np.ndarray) -> None:
        self._clear_fft_peak_labels()
        if not self.auto_peak_var.get() or freq.size < 8:
            return

        try:
            peak_count = max(1, int(self.peak_count_var.get()))
        except Exception:
            peak_count = 3

        mag_work = mag.copy()
        if mag_work.size > 0:
            mag_work[0] = 0.0

        peaks, _ = spsig.find_peaks(mag_work)
        if peaks.size == 0:
            return

        best_idx = peaks[np.argsort(mag_work[peaks])[-peak_count:]]
        best_idx = best_idx[np.argsort(freq[best_idx])]

        for idx in best_idx:
            f = float(freq[idx])
            m = float(mag_work[idx])
            marker, = self.ax_fft.plot(f, m, "ro", ms=4)
            self.fft_peak_markers.append(marker)
            txt = self.ax_fft.text(f, m, f" {f:.1f} Hz", fontsize=8, va="bottom", ha="left")
            self.fft_peak_texts.append(txt)

    def _get_available_channels(self) -> list[str]:
        channels: list[str] = []
        for dev in nidaqmx.system.System.local().devices:
            try:
                channels.extend(ch.name for ch in dev.ai_physical_chans)
            except Exception:
                continue
        return sorted(set(channels))

    def refresh_channels(self) -> None:
        try:
            channels = self._get_available_channels()
            self.channel_box["values"] = channels
            if channels and not self.channel_var.get():
                self.channel_var.set(channels[0])
            self.status_var.set(f"Found {len(channels)} AI channel(s)")
        except Exception as exc:
            self.status_var.set("Channel scan failed")
            messagebox.showerror("DAQ Error", f"Failed to scan channels:\n{exc}")

    def auto_scan_channels(self) -> None:
        if self.running:
            messagebox.showwarning("Busy", "Stop streaming before auto scan.")
            return

        try:
            sample_rate = float(self.sample_rate_var.get())
            chunk_size = int(self.chunk_size_var.get())
            min_val = float(self.min_var.get())
            max_val = float(self.max_var.get())
            sensitivity = float(self.sensitivity_var.get())
            mode = self.mode_var.get()
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        channels = self._get_available_channels()
        if not channels:
            messagebox.showwarning("No Channels", "No AI channels were found.")
            return

        self.scan_btn.configure(state=tk.DISABLED)
        self.status_var.set(f"Auto scanning {len(channels)} channel(s)...")

        threading.Thread(
            target=self._scan_worker,
            args=(channels, sample_rate, chunk_size, min_val, max_val, sensitivity, mode),
            daemon=True,
        ).start()

    def _scan_worker(
        self,
        channels: list[str],
        sample_rate: float,
        chunk_size: int,
        min_val: float,
        max_val: float,
        sensitivity_mv_g: float,
        mode: str,
    ) -> None:
        results: list[tuple[str, float, float]] = []

        for channel in channels:
            try:
                with nidaqmx.Task() as task:
                    if mode == "voltage":
                        task.ai_channels.add_ai_voltage_chan(channel, min_val=min_val, max_val=max_val)
                    else:
                        task.ai_channels.add_ai_accel_chan(
                            physical_channel=channel,
                            min_val=min_val,
                            max_val=max_val,
                            sensitivity=sensitivity_mv_g,
                        )
                    task.timing.cfg_samp_clk_timing(
                        rate=sample_rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=chunk_size,
                    )
                    data = task.read(number_of_samples_per_channel=chunk_size, timeout=2.0)
                    arr = np.asarray(data, dtype=float)
                    if arr.ndim > 1:
                        arr = arr[0]
                    rms = float(np.sqrt(np.mean(np.square(arr))))
                    ptp = float(np.ptp(arr))
                    results.append((channel, rms, ptp))
            except Exception:
                continue

        def finish_scan() -> None:
            self.scan_btn.configure(state=tk.NORMAL)
            if not results:
                self.status_var.set("Auto scan failed")
                messagebox.showwarning("Auto Scan", "No readable channels found.")
                return

            results.sort(key=lambda x: x[1], reverse=True)
            best_channel, best_rms, best_ptp = results[0]
            self.channel_var.set(best_channel)
            self.status_var.set(f"Auto scan complete. Best: {best_channel}")

            top_lines = [f"{ch}: RMS={rms:.6f}, PkPk={ptp:.6f}" for ch, rms, ptp in results[:8]]
            messagebox.showinfo(
                "Auto Scan Result",
                "Best channel selected:\n"
                f"{best_channel} (RMS={best_rms:.6f}, PkPk={best_ptp:.6f})\n\n"
                "Top channels:\n"
                + "\n".join(top_lines),
            )

        self.root.after(0, finish_scan)

    def start_stream(self) -> None:
        if self.running:
            return

        try:
            sample_rate = float(self.sample_rate_var.get())
            chunk_size = int(self.chunk_size_var.get())
            min_val = float(self.min_var.get())
            max_val = float(self.max_var.get())
            sensitivity = float(self.sensitivity_var.get())
            channel = self.channel_var.get().strip()
            if not channel:
                raise ValueError("Select a channel first")
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self.time_buf.clear()
        self.data_buf.clear()
        self.start_time = time.perf_counter()
        self.stop_event.clear()
        self.running = True
        self.current_sample_rate = sample_rate
        self.tf_tick = 0

        self.scan_btn.configure(state=tk.DISABLED)
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Streaming {channel}")

        self.worker = threading.Thread(
            target=self._stream_worker,
            args=(channel, sample_rate, chunk_size, min_val, max_val, sensitivity, self.mode_var.get()),
            daemon=True,
        )
        self.worker.start()
        self._update_plot()

    def stop_stream(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.running = False
        self.scan_btn.configure(state=tk.NORMAL)
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._clear_fft_peak_labels()
        self.status_var.set("Stopped")

    def _stream_worker(
        self,
        channel: str,
        sample_rate: float,
        chunk_size: int,
        min_val: float,
        max_val: float,
        sensitivity_mv_g: float,
        mode: str,
    ) -> None:
        try:
            with nidaqmx.Task() as task:
                if mode == "voltage":
                    task.ai_channels.add_ai_voltage_chan(channel, min_val=min_val, max_val=max_val)
                    y_unit = "V"
                else:
                    task.ai_channels.add_ai_accel_chan(
                        physical_channel=channel,
                        min_val=min_val,
                        max_val=max_val,
                        sensitivity=sensitivity_mv_g,
                    )
                    y_unit = "g"

                task.timing.cfg_samp_clk_timing(
                    rate=sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=max(chunk_size * 4, 1024),
                )

                self.root.after(0, lambda: self.ax_time.set_ylabel(y_unit))

                while not self.stop_event.is_set():
                    data = task.read(number_of_samples_per_channel=chunk_size, timeout=2.0)
                    arr = np.asarray(data, dtype=float)
                    if arr.ndim > 1:
                        arr = arr[0]

                    now = time.perf_counter()
                    t0 = now - (arr.size / sample_rate)
                    ts = np.linspace(t0, now, arr.size, endpoint=False) - self.start_time

                    self.time_buf.extend(ts.tolist())
                    self.data_buf.extend(arr.tolist())

        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("DAQ Stream Error", str(exc)))
        finally:
            self.root.after(0, self.stop_stream)

    def _update_plot(self) -> None:
        if not self.running:
            self.canvas.draw_idle()
            return

        if self.time_buf and self.data_buf:
            t = np.asarray(self.time_buf, dtype=float)
            y = np.asarray(self.data_buf, dtype=float)

            try:
                window_sec = max(0.5, float(self.window_sec_var.get()))
            except Exception:
                window_sec = 3.0

            t_end = t[-1]
            mask = t >= (t_end - window_sec)
            t_view = t[mask]
            y_view = y[mask]

            if t_view.size > 1:
                y_proc = self._apply_preprocessing(y_view, self.current_sample_rate)

                y_plot = y_proc
                if self.envelope_var.get() and y_proc.size > 8:
                    y_plot = np.abs(spsig.hilbert(y_proc))

                self.line_time.set_data(t_view, y_plot)
                self.ax_time.set_xlim(t_view[0], t_view[-1])

                y_min = float(np.min(y_plot))
                y_max = float(np.max(y_plot))
                if y_max <= y_min:
                    y_min -= 1e-6
                    y_max += 1e-6
                margin = max(1e-6, 0.1 * (y_max - y_min))
                self.ax_time.set_ylim(y_min - margin, y_max + margin)

                mean = float(np.mean(y_plot))
                rms = float(np.sqrt(np.mean(np.square(y_plot))))
                ptp = float(np.ptp(y_plot))
                peak_abs = float(np.max(np.abs(y_plot))) if y_plot.size else 0.0
                crest = peak_abs / max(rms, 1e-12)
                krt = float(kurtosis(y_plot, fisher=False, bias=False)) if y_plot.size > 8 else float("nan")

                y_fft = y_plot - mean
                n = y_fft.size
                if n >= 16 and self.current_sample_rate > 0:
                    win = np.hanning(n)
                    spec = np.fft.rfft(y_fft * win)
                    freq = np.fft.rfftfreq(n, d=1.0 / self.current_sample_rate)
                    mag = np.abs(spec) / max(1, n)
                    self.line_fft.set_data(freq, mag)
                    self.ax_fft.set_xlim(0.0, self.current_sample_rate / 2.0)
                    mag_max = float(np.max(mag)) if mag.size else 1.0
                    self.ax_fft.set_ylim(0.0, max(1e-12, mag_max * 1.15))
                    self._add_fft_peak_labels(freq, mag)

                    dominant_freq = 0.0
                    if mag.size > 1:
                        idx = int(np.argmax(mag[1:]) + 1)
                        dominant_freq = float(freq[idx])
                    self.stats_label.configure(
                        text=(
                            f"RMS: {rms:.6f}, PkPk: {ptp:.6f}, Mean: {mean:.6f}, "
                            f"Fdom: {dominant_freq:.2f} Hz, Crest: {crest:.3f}, Kurtosis: {krt:.3f}"
                        )
                    )

                    self.tf_tick += 1
                    if self.tf_tick % 4 == 0:
                        self.ax_tf.clear()
                        self.ax_tf.set_title("Hilbert Envelope Spectrogram")
                        self.ax_tf.set_xlabel("Time Offset (s)")
                        self.ax_tf.set_ylabel("Frequency (Hz)")
                        env = np.abs(spsig.hilbert(y_proc))
                        nperseg = min(256, env.size)
                        if nperseg >= 32:
                            noverlap = nperseg // 2
                            f_sp, t_sp, sxx = spsig.spectrogram(
                                env,
                                fs=self.current_sample_rate,
                                window="hann",
                                nperseg=nperseg,
                                noverlap=noverlap,
                                detrend=False,
                                scaling="spectrum",
                                mode="magnitude",
                            )
                            sxx_db = 20.0 * np.log10(np.maximum(sxx, 1e-12))
                            self.ax_tf.pcolormesh(t_sp - t_sp.max(), f_sp, sxx_db, shading="auto")
                            self.ax_tf.set_ylim(0.0, self.current_sample_rate / 2.0)

        self.canvas.draw_idle()
        self.root.after(80, self._update_plot)

    def on_close(self) -> None:
        self.stop_stream()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    LiveDAQApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
