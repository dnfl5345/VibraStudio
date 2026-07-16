"""File playback backend: replay recorded data through the live pipeline.

Loads VibraStudio recordings (.npz written by StreamRecorder, or .csv with a
``time_s`` column) and streams them like a live DAQ source, so every analysis
view — FFT averaging, octave, FRF, ODS — works identically on stored data.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RecordingData:
    data: np.ndarray            # (n_channels, n_samples), float64, unit g
    fs: float
    channels: list[str]
    path: str

    @property
    def duration(self) -> float:
        return self.data.shape[1] / max(self.fs, 1e-9)


def load_recording(path: str | Path) -> RecordingData:
    """Load a .npz (VibraStudio) or .csv (time_s + channel columns) recording."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            data = np.asarray(z["data"], dtype=np.float64)
            fs = float(z["fs"])
            channels = [str(c) for c in z["channels"]]
        if data.ndim != 2 or data.shape[0] != len(channels):
            raise ValueError(f"Unexpected NPZ layout in {path.name}")
        return RecordingData(data, fs, channels, str(path))

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header or header[0].strip().lower() not in ("time_s", "time", "t"):
                raise ValueError("CSV must start with a time_s column")
            channels = [h.strip() for h in header[1:]]
            rows = [r for r in reader if len(r) == len(header)]
        if len(rows) < 16:
            raise ValueError("CSV contains too few samples")
        arr = np.asarray(rows, dtype=np.float64)
        t = arr[:, 0]
        dt = np.median(np.diff(t))
        if dt <= 0:
            raise ValueError("CSV time column is not increasing")
        return RecordingData(arr[:, 1:].T.copy(), 1.0 / dt, channels, str(path))

    raise ValueError(f"Unsupported file type: {path.suffix} (use .npz or .csv)")


class PlaybackBackend:
    """DAQ-backend-compatible source that replays a RecordingData buffer.

    Loops seamlessly at the end of file so spectral averaging keeps running;
    ``position``/``loops`` expose progress for the UI.
    """

    def __init__(
        self,
        rec: RecordingData,
        chunk_size: int,
        channel_rows: list[int] | None = None,
        realtime: bool = True,
        loop: bool = True,
    ) -> None:
        rows = channel_rows if channel_rows is not None else list(range(rec.data.shape[0]))
        self.data = rec.data[rows]
        self.fs = rec.fs
        self.chunk = max(16, int(chunk_size))
        self.realtime = realtime
        self.loop = loop
        self.n = self.data.shape[1]
        self.position = 0
        self.loops = 0
        self._served = 0
        self._t0 = 0.0

    def start(self) -> None:
        self.position = 0
        self.loops = 0
        self._served = 0
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
        if self.realtime:
            target = (self._served + self.chunk) / self.fs
            wait = self._t0 + target - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

        end = self.position + self.chunk
        if end <= self.n:
            out = self.data[:, self.position:end].copy()
            self.position = end % self.n
            if self.position == 0:
                self.loops += 1
        elif self.loop:
            head = self.data[:, self.position:]
            tail = self.data[:, : end - self.n]
            out = np.concatenate((head, tail), axis=1)
            self.position = end - self.n
            self.loops += 1
        else:
            out = self.data[:, self.position:].copy()
            self.position = self.n
            if out.shape[1] == 0:
                raise EOFError("End of recording")
        self._served += out.shape[1]
        return out
