"""Acquisition engine: NI-DAQmx backend, ring buffer and streaming thread."""

from __future__ import annotations

import threading
import traceback

import numpy as np

from ..config import AcquisitionConfig, ChannelConfig
from .simulator import SimBackend


# ----------------------------------------------------------------------
def discover_ai_channels() -> list[str]:
    """List all NI-DAQmx AI physical channels visible on this system."""
    try:
        import nidaqmx
    except Exception:
        return []
    channels: list[str] = []
    try:
        for dev in nidaqmx.system.System.local().devices:
            try:
                channels.extend(ch.name for ch in dev.ai_physical_chans)
            except Exception:
                continue
    except Exception:
        return []
    return sorted(set(channels))


def hardware_available() -> bool:
    return len(discover_ai_channels()) > 0


# ----------------------------------------------------------------------
class NidaqBackend:
    """Continuous multi-channel IEPE accelerometer (or voltage) acquisition."""

    def __init__(self, channels: list[ChannelConfig], acq: AcquisitionConfig) -> None:
        self.channels = channels
        self.acq = acq
        self._task = None
        self._reader = None
        self._buf: np.ndarray | None = None

    def start(self) -> None:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, ExcitationSource, AccelUnits
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        task = nidaqmx.Task()
        try:
            for ch in self.channels:
                if self.acq.coupling == "iepe":
                    task.ai_channels.add_ai_accel_chan(
                        physical_channel=ch.physical_channel,
                        min_val=self.acq.min_val,
                        max_val=self.acq.max_val,
                        units=AccelUnits.G,
                        sensitivity=ch.sensitivity_mv_g,
                        current_excit_source=ExcitationSource.INTERNAL,
                        current_excit_val=0.002,
                    )
                else:
                    task.ai_channels.add_ai_voltage_chan(
                        ch.physical_channel,
                        min_val=self.acq.min_val,
                        max_val=self.acq.max_val,
                    )
            task.timing.cfg_samp_clk_timing(
                rate=self.acq.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(8 * self.acq.chunk_size, 8192),
            )
            self._reader = AnalogMultiChannelReader(task.in_stream)
            self._buf = np.zeros((len(self.channels), self.acq.chunk_size), dtype=np.float64)
            task.start()
        except Exception:
            task.close()
            raise
        self._task = task

    def read(self) -> np.ndarray:
        assert self._reader is not None and self._buf is not None
        self._reader.read_many_sample(
            self._buf, number_of_samples_per_channel=self.acq.chunk_size, timeout=5.0
        )
        data = self._buf.copy()
        if self.acq.coupling == "voltage":
            # Convert volts to g using per-channel sensitivity.
            sens = np.array([1000.0 / max(1e-9, c.sensitivity_mv_g) for c in self.channels])
            data *= sens[:, None]
        return data

    def stop(self) -> None:
        if self._task is not None:
            try:
                self._task.stop()
            finally:
                self._task.close()
                self._task = None


# ----------------------------------------------------------------------
class RingBuffer:
    """Thread-safe multi-channel sample ring buffer."""

    def __init__(self, n_channels: int, capacity: int) -> None:
        self.n_channels = n_channels
        self.capacity = int(capacity)
        self._data = np.zeros((n_channels, self.capacity), dtype=np.float64)
        self._write = 0            # next write position
        self.total = 0             # total samples ever written
        self._lock = threading.Lock()

    def append(self, block: np.ndarray) -> None:
        n = block.shape[1]
        with self._lock:
            if n >= self.capacity:
                self._data[:] = block[:, -self.capacity:]
                self._write = 0
                self.total += n
                return
            end = self._write + n
            if end <= self.capacity:
                self._data[:, self._write:end] = block
            else:
                k = self.capacity - self._write
                self._data[:, self._write:] = block[:, :k]
                self._data[:, : end - self.capacity] = block[:, k:]
            self._write = end % self.capacity
            self.total += n

    def latest(self, n: int) -> np.ndarray:
        """Return the most recent n samples (may be fewer early on)."""
        with self._lock:
            n = min(n, self.total, self.capacity)
            if n <= 0:
                return np.zeros((self.n_channels, 0))
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return self._data[:, start : start + n].copy()
            k = self.capacity - start
            return np.concatenate((self._data[:, start:], self._data[:, : n - k]), axis=1)

    def read_since(self, total_mark: int, max_samples: int) -> tuple[np.ndarray, int]:
        """Return samples written after ``total_mark`` (up to max), and new mark."""
        with self._lock:
            available = self.total - total_mark
            if available <= 0:
                return np.zeros((self.n_channels, 0)), total_mark
            n = int(min(available, max_samples, self.capacity))
        block = self.latest(n)
        return block, self.total


# ----------------------------------------------------------------------
class AcquisitionEngine:
    """Background streaming thread feeding a ring buffer and listeners."""

    def __init__(
        self,
        channels: list[ChannelConfig],
        acq: AcquisitionConfig,
        buffer_seconds: float = 30.0,
    ) -> None:
        self.channels = channels
        self.acq = acq
        self.fs = acq.sample_rate
        capacity = int(acq.sample_rate * buffer_seconds)
        self.ring = RingBuffer(len(channels), capacity)
        self.simulated = acq.simulate
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list = []   # callables(block) e.g. recorder
        self._listener_lock = threading.Lock()

        if acq.simulate:
            self.backend = SimBackend(channels, acq.sample_rate, acq.chunk_size)
        else:
            self.backend = NidaqBackend(channels, acq)

    # -- listeners (recorder tee) --------------------------------------
    def add_listener(self, fn) -> None:
        with self._listener_lock:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        with self._listener_lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.backend.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="daq-stream")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        try:
            self.backend.stop()
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                block = self.backend.read()
                self.ring.append(block)
                with self._listener_lock:
                    listeners = list(self._listeners)
                for fn in listeners:
                    try:
                        fn(block)
                    except Exception:
                        pass
        except Exception as exc:
            if not self._stop.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
