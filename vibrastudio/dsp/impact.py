"""Impact (modal hammer) testing: trigger capture, FRF estimation, modal ID.

Implements the classic LabVIEW SVMS impact-test workflow: level trigger with
pre-trigger, force/exponential windows, H1 FRF + coherence accumulated over
hits, then natural frequency / damping (half-power) / mode-shape extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .analysis import WINDOWS  # noqa: F401  (kept for future window options)


# -- impact location guide (cantilever theory) ---------------------------
_BETA_L = np.array([1.8751, 4.6941, 7.8548, 10.9955])


def _phi(beta_l: float, xi: np.ndarray) -> np.ndarray:
    """Cantilever mode shape, xi from clamped base [0..1], tip-normalized."""
    sigma = (np.sinh(beta_l) - np.sin(beta_l)) / (np.cosh(beta_l) + np.cos(beta_l))
    bx = beta_l * xi
    raw = (np.cosh(bx) - np.cos(bx)) - sigma * (np.sinh(bx) - np.sin(bx))
    tip = (np.cosh(beta_l) - np.cos(beta_l)) - sigma * (np.sinh(beta_l) - np.sin(beta_l))
    return raw / (abs(tip) or 1.0)


def impact_guide(free_len_cm: float, zone_from_tip_cm: float, n_modes: int = 4,
                 n_grid: int = 400) -> dict:
    """Score hitting positions inside the sensor-free zone near the tip.

    score(x) = min over modes |phi_m(x)| — a hit exciting every mode of
    interest. Returns positions (cm from tip), score, best point and the
    node positions (score dips) to avoid.
    """
    d = np.linspace(0.0, zone_from_tip_cm, n_grid)     # distance from free tip
    xi = (free_len_cm - d) / free_len_cm               # from clamped base
    shapes = np.abs(np.array([_phi(b, xi) for b in _BETA_L[:n_modes]]))
    score = shapes.min(axis=0)
    best_i = int(np.argmax(score))
    # nodes: local minima with low modal participation
    nodes = []
    for m in range(shapes.shape[0]):
        idx, _ = find_peaks(-shapes[m])
        nodes.extend(float(d[i]) for i in idx if shapes[m][i] < 0.15)
    return {
        "d_cm": d, "score": score,
        "best_cm": float(d[best_i]), "best_score": float(score[best_i]),
        "avoid_cm": sorted(nodes),
    }


# -- windows -------------------------------------------------------------
def force_window(n: int, fs: float, flat_ms: float = 5.0) -> np.ndarray:
    """Unity over the pulse, half-cosine rolloff, zero elsewhere."""
    w = np.zeros(n)
    n_flat = max(2, int(flat_ms * 1e-3 * fs))
    n_roll = max(2, n_flat // 2)
    w[:n_flat] = 1.0
    end = min(n, n_flat + n_roll)
    w[n_flat:end] = 0.5 * (1 + np.cos(np.pi * np.arange(end - n_flat) / n_roll))
    return w


def exp_window(n: int, final_fraction: float = 0.05) -> np.ndarray:
    """exp decay reaching final_fraction at the end of the block."""
    tau = (n - 1) / max(np.log(1.0 / max(final_fraction, 1e-6)), 1e-9)
    return np.exp(-np.arange(n) / tau)


# -- accumulation --------------------------------------------------------
@dataclass
class ImpactAccumulator:
    """Averaged H1 FRF and coherence over accepted hammer hits."""

    fs: float
    n_resp: int
    nfft: int
    gff: np.ndarray = field(init=False)
    gxf: np.ndarray = field(init=False)
    gxx: np.ndarray = field(init=False)
    navg: int = 0

    def __post_init__(self) -> None:
        nb = self.nfft // 2 + 1
        self.freq = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)
        self.gff = np.zeros(nb)
        self.gxf = np.zeros((self.n_resp, nb), dtype=np.complex128)
        self.gxx = np.zeros((self.n_resp, nb))

    @property
    def exp_tau_sec(self) -> float:
        """Time constant of the exponential window (for damping correction)."""
        return (self.nfft - 1) / (np.log(1.0 / 0.05) * self.fs)

    def add_hit(self, force: np.ndarray, responses: np.ndarray,
                flat_ms: float = 5.0, exp_final: float = 0.05) -> None:
        n = self.nfft
        f = np.zeros(n)
        f[: force.size] = force[:n]
        r = np.zeros((self.n_resp, n))
        r[:, : responses.shape[1]] = responses[:, :n]

        we = exp_window(n, exp_final)
        fw = np.fft.rfft(f * force_window(n, self.fs, flat_ms) * we)
        rw = np.fft.rfft(r * we[None, :], axis=1)

        self.gff += np.abs(fw) ** 2
        self.gxf += rw * np.conj(fw)[None, :]
        self.gxx += np.abs(rw) ** 2
        self.navg += 1

    def frf(self) -> tuple[np.ndarray, np.ndarray]:
        """(H1, coherence) — shapes (n_resp, n_bins)."""
        gff = np.maximum(self.gff, 1e-30)
        h1 = self.gxf / gff[None, :]
        coh = np.abs(self.gxf) ** 2 / (gff[None, :] * np.maximum(self.gxx, 1e-30))
        return h1, np.clip(coh, 0.0, 1.0)

    def reset(self) -> None:
        self.gff[:] = 0
        self.gxf[:] = 0
        self.gxx[:] = 0
        self.navg = 0


# -- modal identification ------------------------------------------------
def identify_modes(freq: np.ndarray, h1: np.ndarray, f_min: float = 5.0,
                   f_max: float | None = None, max_modes: int = 8,
                   exp_tau_sec: float = 0.0) -> list[dict]:
    """Peak-pick natural frequencies, half-power damping and mode shapes.

    Shapes come from Im(H) at resonance (valid for light, well-separated
    modes with a force reference).
    """
    if h1.size == 0 or freq.size < 8:
        return []
    if f_max is None:
        f_max = freq[-1]
    mag = np.mean(np.abs(h1), axis=0)
    mask = (freq >= f_min) & (freq <= f_max)
    if not np.any(mask):
        return []
    lo = int(np.argmax(mask))
    m = mag[mask]
    pk, props = find_peaks(m, prominence=np.max(m) * 0.02)
    if pk.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:max_modes]
    modes = []
    df = freq[1] - freq[0]
    for p in sorted(pk[order]):
        i = lo + p
        fn = float(freq[i])
        # half-power bandwidth on the averaged magnitude
        half = mag[i] / np.sqrt(2.0)
        i1, i2 = i, i
        while i1 > 1 and mag[i1] > half:
            i1 -= 1
        while i2 < mag.size - 2 and mag[i2] > half:
            i2 += 1
        bw = max(freq[i2] - freq[i1], df)
        zeta = float(bw / (2.0 * fn)) if fn > 0 else 0.0
        if exp_tau_sec > 0 and fn > 0:
            # Remove the artificial damping added by the exponential window.
            zeta = max(0.0, zeta - 1.0 / (2.0 * np.pi * fn * exp_tau_sec))
        shape = np.imag(h1[:, i])
        peak_ref = shape[np.argmax(np.abs(shape))] if shape.size else 1.0
        if abs(peak_ref) > 0:
            shape = shape / peak_ref
        modes.append({
            "fn": fn, "zeta": min(zeta, 0.5), "amp": float(mag[i]),
            "shape": shape,
        })
    return modes
