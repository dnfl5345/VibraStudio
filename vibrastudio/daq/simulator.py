"""Simulation backend: cantilever-beam modal response generator.

Produces realistic 18-channel accelerometer signals so every analysis view
(FFT, PSD, octave, FRF, ODS animation) can be exercised without hardware.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import ChannelConfig

# Cantilever beam eigenvalue solutions (beta*L) for the first modes.
_BETA_L = np.array([1.8751, 4.6941, 7.8548, 10.9955, 14.1372])


def _cantilever_shape(beta_l: float, xi: np.ndarray) -> np.ndarray:
    """Analytic cantilever mode shape, xi normalized [0..1], clamped at 0.

    Normalized to unit tip deflection (|phi(1)| = 1) so different sensor
    positions map onto the true spatial shape.
    """
    sigma = (np.sinh(beta_l) - np.sin(beta_l)) / (np.cosh(beta_l) + np.cos(beta_l))

    def raw(x: np.ndarray) -> np.ndarray:
        bx = beta_l * x
        return (np.cosh(bx) - np.cos(bx)) - sigma * (np.sinh(bx) - np.sin(bx))

    tip = abs(float(raw(np.array([1.0]))[0])) or 1.0
    return raw(xi) / tip


class SimBackend:
    """Drop-in replacement for the NI-DAQmx backend (same read interface)."""

    # First natural frequency per direction; higher modes follow beam theory ratios.
    F1 = {"X": 14.2, "Y": 17.8, "Z": 11.6}
    MODE_AMPS = np.array([0.60, 0.28, 0.12, 0.05, 0.02])  # g, per mode

    def __init__(self, channels: list[ChannelConfig], sample_rate: float, chunk_size: int) -> None:
        self.channels = channels
        self.fs = float(sample_rate)
        self.chunk = int(chunk_size)
        self._n = 0
        self._t0 = 0.0
        self._rng = np.random.default_rng(20260709)
        self._mode_phases = self._rng.uniform(0, 2 * np.pi, size=(3, len(_BETA_L)))

        ratios = (_BETA_L / _BETA_L[0]) ** 2
        self._freqs = {d: self.F1[d] * ratios for d in ("X", "Y", "Z")}
        self._shapes = {}
        for i, ch in enumerate(channels):
            xi = np.clip(ch.position, 0.0, 1.0)
            self._shapes[i] = np.array([_cantilever_shape(b, np.array([xi]))[0] for b in _BETA_L])

    def start(self) -> None:
        self._n = 0
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
        """Return one chunk of shape (n_channels, chunk), paced to real time."""
        target = (self._n + self.chunk) / self.fs
        wait = self._t0 + target - time.perf_counter()
        if wait > 0:
            time.sleep(wait)

        t = (self._n + np.arange(self.chunk)) / self.fs
        self._n += self.chunk

        # Slowly varying excitation level to mimic operational (random) forcing.
        env = 1.0 + 0.35 * np.sin(2 * np.pi * 0.13 * t) + 0.2 * np.sin(2 * np.pi * 0.031 * t + 1.0)

        out = np.empty((len(self.channels), self.chunk), dtype=np.float64)
        for i, ch in enumerate(self.channels):
            d = ch.direction if ch.direction in self._freqs else "Z"
            di = "XYZ".index(d)
            freqs = self._freqs[d]
            shapes = self._shapes[i]
            y = np.zeros(self.chunk)
            for m in range(len(_BETA_L)):
                y += (
                    self.MODE_AMPS[m]
                    * shapes[m]
                    * np.sin(2 * np.pi * freqs[m] * t + self._mode_phases[di, m])
                )
            # Rotating-machine tone + harmonic, bearing-like modulated HF burst.
            y += 0.08 * np.sin(2 * np.pi * 29.5 * t + 0.3 * di)
            y += 0.03 * np.sin(2 * np.pi * 59.0 * t)
            carrier = np.sin(2 * np.pi * 3200.0 * t)
            y += 0.02 * (1 + np.sign(np.sin(2 * np.pi * 87.0 * t))) * carrier
            y = y * env + self._rng.normal(0.0, 0.02, self.chunk)
            out[i] = y
        return out
