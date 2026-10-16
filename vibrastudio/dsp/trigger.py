"""Band-limited level trigger: record automatically when excitation starts.

Built for shaker tests driven by a separate controller: the app watches the
response (or the shaker's drive-monitor signal on any input), and when the
level inside a frequency band crosses a threshold it saves a recording that
includes a pre-trigger segment. Everything runs on absolute sample indices
of the engine's ring buffer, so trigger times and segment edges are exact.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal as spsig

DISARMED, SETTLING, ARMED, CAPTURING = "DISARMED", "SETTLING", "ARMED", "CAPTURING"


@dataclass
class TriggerSettings:
    source: int = -1             # channel index; -1 = any enabled channel
    f_lo: float = 0.0            # band edges in Hz; 0 = open edge
    f_hi: float = 0.0
    metric: str = "rms"          # "rms" | "peak" of the band-passed signal
    window_s: float = 0.1        # level averaging window
    threshold: float = 0.05      # g
    pre_s: float = 1.0           # kept before the trigger point
    stop_mode: str = "duration"  # "duration" | "quiet"
    duration_s: float = 10.0     # duration mode: length after the trigger
    quiet_s: float = 2.0         # quiet mode: stop after this long below threshold
    max_s: float = 120.0         # hard cap after the trigger
    rearm: bool = True
    hysteresis: float = 0.8      # re-arm / "quiet" means level < threshold * this
    write_csv: bool = False


class BandLevelMeter:
    """Streaming band-pass + windowed level for many channels.

    Levels are produced once per sub-block (``sub`` samples, ~10 ms) so a
    trigger point is located to that resolution.
    """

    def __init__(self, fs: float, n_ch: int, f_lo: float, f_hi: float,
                 metric: str, window_s: float, sub: int | None = None) -> None:
        self.fs = float(fs)
        self.n_ch = n_ch
        self.metric = metric
        self.sub = sub or max(16, int(round(0.01 * fs)))
        nyq = 0.5 * self.fs
        lo = max(float(f_lo), 1.0)                  # always drop DC / drift
        hi = float(f_hi) if 0 < f_hi < 0.98 * nyq else 0.0
        if hi and hi <= lo:
            raise ValueError(f"band upper edge {hi:g} Hz must exceed lower edge {lo:g} Hz")
        if hi:
            self.sos = spsig.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        else:
            self.sos = spsig.butter(4, lo / nyq, btype="highpass", output="sos")
        self.band = (lo, hi or nyq)
        self._zi: np.ndarray | None = None
        self._rest = np.zeros((n_ch, 0))
        self._rest_start = 0
        k = max(1, int(round(window_s * self.fs / self.sub)))
        self._win: deque = deque(maxlen=k)

    def push(self, block: np.ndarray, start_abs: int) -> tuple[np.ndarray, np.ndarray]:
        """Feed samples starting at absolute index ``start_abs``.

        Returns (ends, levels): absolute end index of each completed
        sub-block and the level per channel at that point, shape (m, n_ch).
        """
        if block.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros((0, self.n_ch))
        if self._zi is None:
            # Start in steady state for the first sample (no startup ring).
            zi0 = spsig.sosfilt_zi(self.sos)                       # (sections, 2)
            self._zi = zi0[:, None, :] * block[None, :, :1]        # (sections, ch, 2)
            self._rest_start = start_abs
        y, self._zi = spsig.sosfilt(self.sos, block, axis=1, zi=self._zi)
        if self._rest.shape[1] == 0:
            self._rest_start = start_abs
        buf = np.concatenate((self._rest, y), axis=1)
        m = buf.shape[1] // self.sub
        ends = np.zeros(m, dtype=np.int64)
        levels = np.zeros((m, self.n_ch))
        for j in range(m):
            seg = buf[:, j * self.sub:(j + 1) * self.sub]
            stat = np.mean(seg ** 2, axis=1) if self.metric == "rms" else np.max(np.abs(seg), axis=1)
            self._win.append(stat)
            w = np.array(self._win)
            levels[j] = np.sqrt(w.mean(axis=0)) if self.metric == "rms" else w.max(axis=0)
            ends[j] = self._rest_start + (j + 1) * self.sub
        self._rest = buf[:, m * self.sub:]
        self._rest_start += m * self.sub
        return ends, levels


class TriggerController:
    """State machine: ARMED -> CAPTURING -> ARMED (or DISARMED).

    Levels are evaluated one ~10 ms sub-block at a time, so a late GUI tick
    or a long batch after a stall gives the same trigger and stop points as
    real-time processing. Files are written by a background worker: saving
    a 48-channel segment never blinds the detector to the next event.
    """

    HISTORY_S = 30.0

    def __init__(self, out_dir: str | Path = "recordings") -> None:
        self.out_dir = Path(out_dir)
        self.settings = TriggerSettings()
        self.labels: list[str] = []
        self.state = DISARMED
        self.events: list[dict] = []
        self.message = ""
        self._engine = None
        self._meter: BandLevelMeter | None = None
        self._mark = 0
        self._need_quiet = False
        self._blank_until = 0
        self._cap: dict | None = None
        self._blocks: list[tuple[int, np.ndarray]] = []
        self._blk_lock = threading.Lock()
        self._listener = None
        self._n_captures = 0
        self._save_queue: queue.Queue = queue.Queue()
        self._pending = 0
        self._worker = threading.Thread(target=self._save_worker, daemon=True,
                                        name="trigger-save")
        self._worker.start()
        self.history: deque = deque()          # (abs_end, level, channel_index)
        self.last_level = 0.0
        self.last_channel = -1
        self.fs = 0.0

    @property
    def saving(self) -> int:
        """Captures waiting to be written to disk."""
        return self._pending

    # -- control -------------------------------------------------------------
    def configure(self, settings: TriggerSettings, labels: list[str]) -> None:
        self.settings = settings
        self.labels = list(labels)
        self._meter = None                     # rebuilt on the next process()

    def arm(self, engine) -> None:
        self._drop_listener()
        self._cap = None
        self._engine = engine
        self.fs = engine.fs
        self._mark = engine.ring.total
        self._meter = None
        self._need_quiet = False
        self.history.clear()
        self.state = SETTLING if engine.iepe_settling() else ARMED
        self.message = ""

    def disarm(self) -> None:
        """Stop watching. A capture in progress is saved up to now."""
        if self.state == CAPTURING and self._cap is not None and self._engine is not None:
            self._cap["reason"] += ", stopped early"
            self._finish(self._engine, self._engine.ring.total)
        self._drop_listener()
        self._cap = None
        self.state = DISARMED

    def force(self) -> None:
        """Trigger now, regardless of level (manual start)."""
        if self._engine is not None and self.state in (ARMED, SETTLING):
            self._start_capture(self._engine.ring.total, self.last_level, self.last_channel,
                                reason="manual")

    # -- per GUI tick --------------------------------------------------------
    def process(self, engine) -> None:
        if self.state == DISARMED or engine is None:
            return
        s = self.settings
        ring = engine.ring
        block, new_mark = ring.read_since(self._mark, max_samples=ring.capacity)
        start_abs = new_mark - block.shape[1]
        self._mark = new_mark
        if block.size == 0:
            return

        rows = list(range(block.shape[0])) if s.source < 0 else [s.source]
        rows = [r for r in rows if r < block.shape[0]]
        if not rows:
            return
        if self._meter is None or self._meter.n_ch != len(rows):
            try:
                self._meter = BandLevelMeter(engine.fs, len(rows), s.f_lo, s.f_hi,
                                             s.metric, s.window_s)
            except ValueError as exc:
                self.message = str(exc)
                self.state = DISARMED
                return
            # A fresh filter rings for a moment; never trigger on that.
            self._blank_until = start_abs + int(0.3 * engine.fs)
        ends, levels = self._meter.push(block[rows], start_abs)
        if ends.size == 0:
            return
        best = np.argmax(levels, axis=1)
        lvl = levels[np.arange(levels.shape[0]), best]
        for e, l, b in zip(ends, lvl, best):
            self.history.append((int(e), float(l), rows[int(b)]))
        while self.history and self.history[0][0] < ends[-1] - self.HISTORY_S * engine.fs:
            self.history.popleft()
        self.last_level, self.last_channel = float(lvl[-1]), rows[int(best[-1])]

        if self.state == SETTLING:
            if engine.iepe_settling():
                return
            self.state = ARMED
            self._need_quiet = True            # settle tail must decay first

        fs = engine.fs
        loud = s.threshold * s.hysteresis
        for e, l, b in zip(ends, lvl, best):
            e, l = int(e), float(l)
            if self.state == ARMED:
                if e < self._blank_until:
                    continue
                if self._need_quiet:
                    if l < loud:
                        self._need_quiet = False
                    continue
                if l >= s.threshold:
                    # Trigger point: start of the first sub-block over threshold.
                    self._start_capture(e - self._meter.sub, l, rows[int(b)])
                continue
            if self.state != CAPTURING or self._cap is None:
                break
            cap = self._cap
            if e <= cap["trigger_abs"]:
                continue
            cap["peak"] = max(cap["peak"], l)
            if l >= loud:
                cap["last_loud"] = e
            trig = cap["trigger_abs"]
            hard_end = trig + int(round(s.max_s * fs))
            if s.stop_mode == "duration":
                end_abs = min(trig + int(round(s.duration_s * fs)), hard_end)
            else:
                end_abs = min(cap["last_loud"] + int(round(s.quiet_s * fs)), hard_end)
            if e >= end_abs:
                self._finish(engine, end_abs)
                if s.rearm:
                    self.state = ARMED
                    # Continuous excitation must drop first, or every
                    # recording would be followed by an instant re-trigger.
                    self._need_quiet = l >= loud
                else:
                    self.state = DISARMED

    # -- capture -------------------------------------------------------------
    def _start_capture(self, trigger_abs: int, level: float, channel: int,
                       reason: str = "level") -> None:
        engine = self._engine
        s = self.settings
        fs = engine.fs
        with self._blk_lock:
            self._blocks = []

        def feed(block: np.ndarray, ring=engine.ring) -> None:
            # Runs on the DAQ thread right after the ring append, so
            # ring.total is this block's end index.
            with self._blk_lock:
                self._blocks.append((ring.total, block.astype(np.float32)))

        self._listener = feed
        engine.add_listener(feed)              # listen first, then read the ring
        start_abs = trigger_abs - int(round(s.pre_s * fs))
        pre, got_start, got_end = engine.ring.read_range(start_abs)
        self._cap = {
            "trigger_abs": trigger_abs, "start_abs": got_start,
            "pre": pre.astype(np.float32), "pre_end": got_end,
            "level": level, "peak": level, "channel": channel,
            "last_loud": trigger_abs, "t_wall": datetime.now(), "reason": reason,
        }
        self.state = CAPTURING

    def _finish(self, engine, end_abs: int) -> None:
        cap = self._cap
        self._drop_listener()
        data = self._assemble(engine, cap, end_abs)
        self._cap = None
        self._n_captures += 1
        self._pending += 1
        self._save_queue.put((self._n_captures, data, cap, engine.fs, self.settings))

    def _assemble(self, engine, cap: dict, end_abs: int) -> np.ndarray:
        start = cap["start_abs"]
        data, got_start, got_end = engine.ring.read_range(start, end_abs)
        if got_start == start and got_end == end_abs:
            return data.astype(np.float32)          # whole segment still in the ring
        # Longer than the ring: pre-trigger part + blocks seen by the listener.
        n = end_abs - start
        out = np.zeros((cap["pre"].shape[0], n), dtype=np.float32)
        filled = min(cap["pre_end"], end_abs) - start
        out[:, :filled] = cap["pre"][:, :filled]
        with self._blk_lock:
            blocks = list(self._blocks)
            self._blocks = []
        for blk_end, blk in blocks:
            b0 = blk_end - blk.shape[1]
            lo, hi = max(b0, start + filled), min(blk_end, end_abs)
            if hi > lo:
                out[:, lo - start:hi - start] = blk[:, lo - b0:hi - b0]
                filled = max(filled, hi - start)
        return out[:, :filled]

    def _drop_listener(self) -> None:
        if self._listener is not None and self._engine is not None:
            self._engine.remove_listener(self._listener)
        self._listener = None

    def _save_worker(self) -> None:
        while True:
            n_ev, data, cap, fs, settings = self._save_queue.get()
            try:
                self._save(n_ev, data, cap, fs, settings)
            except Exception as exc:
                self.message = f"save failed: {exc}"
            finally:
                self._pending -= 1

    def wait_saved(self, timeout: float = 30.0) -> bool:
        t_end = time.time() + timeout
        while self._pending and time.time() < t_end:
            time.sleep(0.02)
        return self._pending == 0

    def _save(self, n_ev: int, data: np.ndarray, cap: dict, fs: float,
              s: TriggerSettings) -> None:
        stamp = cap["t_wall"].strftime("%Y%m%d_%H%M%S")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"trig_{stamp}_{n_ev:03d}"
        labels = self.labels[: data.shape[0]]
        np.savez_compressed(base.with_suffix(".npz"), data=data, fs=fs,
                            channels=np.array(labels))
        pre_s = (cap["trigger_abs"] - cap["start_abs"]) / fs
        ac = data - data.mean(axis=1, keepdims=True)
        ch_rms = np.sqrt(np.mean(ac.astype(np.float64) ** 2, axis=1))
        ch_peak = np.max(np.abs(ac), axis=1).astype(np.float64)
        ch_label = self.labels[cap["channel"]] if 0 <= cap["channel"] < len(self.labels) else "?"
        meta = {
            "started_at": stamp, "sample_rate_hz": fs, "channels": labels,
            "samples_per_channel": int(data.shape[1]),
            "duration_sec": data.shape[1] / fs, "units": "g",
            "trigger": {
                "pre_trigger_s": pre_s, "trigger_sample": int(round(pre_s * fs)),
                "channel": ch_label, "level_at_trigger": cap["level"],
                "peak_level": cap["peak"], "reason": cap["reason"],
                "settings": asdict(s),
            },
            "per_channel": [
                {"label": lbl, "rms_g": float(r), "peak_g": float(pk)}
                for lbl, r, pk in zip(labels, ch_rms, ch_peak)
            ],
        }
        base.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if s.write_csv:
            t = np.arange(data.shape[1]) / fs - pre_s
            header = "time_s," + ",".join(labels)
            np.savetxt(base.with_suffix(".csv"), np.column_stack((t, data.T)),
                       delimiter=",", header=header, comments="", fmt="%.6e")
        self.events.append({
            "n": n_ev, "time": cap["t_wall"].strftime("%H:%M:%S"), "channel": ch_label,
            "level": cap["level"], "peak": cap["peak"], "pre_s": pre_s,
            "duration_s": data.shape[1] / fs, "file": str(base.with_suffix(".npz")),
            "trigger_abs": cap["trigger_abs"], "reason": cap["reason"],
            "n_channels": int(data.shape[0]), "ch_rms": ch_rms.tolist(),
        })
