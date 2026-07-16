"""Stream recorder: tees acquisition chunks to disk (NPZ + optional CSV)."""

from __future__ import annotations

import csv
import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np


class StreamRecorder:
    """Collects chunks from the acquisition engine on a writer thread."""

    MAX_SECONDS = 600.0

    def __init__(self, out_dir: str | Path = "recordings") -> None:
        self.out_dir = Path(out_dir)
        self._queue: queue.Queue = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._parts: list[np.ndarray] = []
        self.samples = 0
        self.recording = False
        self.fs = 0.0
        self.channel_names: list[str] = []
        self.started_at = ""

    def start(self, fs: float, channel_names: list[str]) -> None:
        if self.recording:
            return
        self.fs = fs
        self.channel_names = list(channel_names)
        self._parts = []
        self.samples = 0
        self.started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stop.clear()
        self.recording = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="recorder")
        self._thread.start()

    def feed(self, block: np.ndarray) -> None:
        """Called from the acquisition thread."""
        if not self.recording:
            return
        try:
            self._queue.put_nowait(block.copy())
        except queue.Full:
            pass

    def _run(self) -> None:
        max_samples = int(self.MAX_SECONDS * max(self.fs, 1.0))
        while not self._stop.is_set() or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.samples < max_samples:
                self._parts.append(block)
                self.samples += block.shape[1]

    def stop(self, write_csv: bool = False) -> Path | None:
        """Stop recording and write files. Returns the NPZ path."""
        if not self.recording:
            return None
        self.recording = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if not self._parts:
            return None

        data = np.concatenate(self._parts, axis=1)
        self._parts = []
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"record_{self.started_at}"

        npz_path = base.with_suffix(".npz")
        np.savez_compressed(
            npz_path, data=data.astype(np.float32), fs=self.fs,
            channels=np.array(self.channel_names),
        )
        meta = {
            "started_at": self.started_at,
            "sample_rate_hz": self.fs,
            "channels": self.channel_names,
            "samples_per_channel": int(data.shape[1]),
            "duration_sec": data.shape[1] / max(self.fs, 1.0),
            "units": "g",
        }
        base.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if write_csv:
            t = np.arange(data.shape[1]) / self.fs
            with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["time_s", *self.channel_names])
                for k in range(data.shape[1]):
                    # 9 decimals: 7 was too coarse to recover fs from the file.
                    w.writerow([f"{t[k]:.9f}", *(f"{v:.6e}" for v in data[:, k])])
        return npz_path
