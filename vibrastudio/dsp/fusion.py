"""One shape model for accelerometers and video: fit, model order, fusion.

At one frequency the horizontal motion of the beam along its height z is

    u(z) = B(z) q,   B(z) = [1, z/L, phi_1(z), ..., phi_m(z)]

rigid translation, rigid rotation and the first m clamped-free bending shapes
(tip value 1), with complex coefficients q (amplitude and phase).

    sensors   y_s = B(z_s) q + e_s
    video     y_v = g B(z_v) q + C(z_v) r + e_v,   C = [1, z/L]

g is the complex video/sensor gain: |g| is the scale mismatch (px/mm, sensor
sensitivity), arg g the time offset between the two clocks at this frequency.
r is rigid motion of the camera itself; it has the same columns as the rigid
part of B, so only the sensors can tell the two apart.

Everything is solved by weighted least squares with weights 1/sigma.

The beam shapes are only the default. ``basis`` (a function z_cm -> matrix, one
column per shape, most important first) replaces B with any other set, e.g. the
mode shapes of a finite-element model of the structure with its sensors on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

BETA = (1.8751, 4.6941, 7.8548, 10.9955, 14.1372)
MODEL_FLOOR = 0.01            # the shape model itself is not better than ~1 % of the largest amplitude
# Which rigid motions of the base the structure may have. A shaker moves the clamp
# ("both"); in a hammer test the clamp stands still, or at most gives a little
# in rotation ("rotation"), and saying so lets the model reach below the camera's view.
RIGID = {"both": (0, 1), "rotation": (1,), "none": ()}


def _columns(rigid: str, max_modes: int) -> tuple[list[int], int]:
    r = RIGID[rigid]
    return list(r) + list(range(2, 2 + max_modes)), len(r)


def _full(q: np.ndarray, idx: list[int], n_rigid: int) -> tuple[np.ndarray, int]:
    """Coefficients on the full [1, z/L, phi_1..] basis, and the number of bending shapes used."""
    n_modes = max(0, len(q) - n_rigid)
    out = np.zeros(2 + n_modes, complex)
    out[idx[:len(q)]] = q
    return out, n_modes


def alias(f: float, fps: float) -> tuple[float, bool]:
    """(frequency seen by the camera, conjugated?) for a tone at f sampled at fps."""
    r = f % fps
    return (fps - r, True) if r > fps / 2 else (r, False)


def mac(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real + 1e-30))


def beam_basis(z_cm, length_cm: float, n_modes: int) -> np.ndarray:
    """Columns [1, z/L, phi_1..phi_n] at heights z (clipped to the free length)."""
    x = np.clip(np.asarray(z_cm, float) / length_cm, 0.0, 1.0)
    cols = [np.ones_like(x), x]
    for b in BETA[:n_modes]:
        s = (np.sinh(b) - np.sin(b)) / (np.cosh(b) + np.cos(b))

        def phi(u, b=b, s=s):
            return (np.cosh(b * u) - np.cos(b * u)) - s * (np.sinh(b * u) - np.sin(b * u))
        cols.append(phi(x) / abs(phi(1.0)))
    return np.array(cols).T


def wls(A: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    Aw, yw = A * w[:, None], y * w
    M = Aw.conj().T @ Aw
    return np.linalg.solve(M + 1e-12 * np.trace(M).real / len(M) * np.eye(len(M)), Aw.conj().T @ yw)


def _weights(y: np.ndarray, sigma) -> np.ndarray:
    floor = MODEL_FLOOR * float(np.abs(y).max()) + 1e-12
    s = np.full(len(y), floor) if sigma is None else np.hypot(np.asarray(sigma, float), floor)
    return 1.0 / s


@dataclass
class ShapeFit:
    q: np.ndarray                 # (2 + m,) complex, in the units of the sensor data
    n_modes: int
    length_cm: float
    source: str                   # "sensors" | "fusion"
    cv_error: float               # relative held-out error that picked the model order (nan: no choice possible)
    gain: complex | None = None   # video / structure, fusion only
    camera: np.ndarray | None = None
    n_sensors: int = 0
    n_video: int = 0
    basis: Callable | None = None  # custom shapes (then q holds one coefficient per column used)

    def __call__(self, z_cm) -> np.ndarray:
        if self.basis is not None:
            return np.asarray(self.basis(z_cm))[:, :len(self.q)] @ self.q
        return beam_basis(z_cm, self.length_cm, self.n_modes) @ self.q


def _basis(z, length_cm, max_modes, rigid, basis):
    """(matrix at z, column bookkeeping) for the beam shapes or a custom set."""
    if basis is not None:
        B = np.asarray(basis(z), float)
        return B, list(range(B.shape[1])), 0
    idx, nr = _columns(rigid, max_modes)
    return beam_basis(z, length_cm, max_modes)[:, idx], idx, nr


def _result(q, idx, nr, basis):
    return (np.asarray(q, complex), len(q)) if basis is not None else _full(q, idx, nr)


def fit_sensors(z_s, y_s, length_cm: float, sigma_s=None, max_modes: int = 4, rigid: str = "both",
                basis: Callable | None = None) -> ShapeFit:
    """Sensors only. The number of bending shapes is the one that predicts a left-out sensor best."""
    z_s, y_s = np.asarray(z_s, float), np.asarray(y_s, complex)
    n = len(y_s)
    w = _weights(y_s, sigma_s)
    B, idx, nr = _basis(z_s, length_cm, max_modes, rigid, basis)
    pmin = max(nr, 1)
    if n <= pmin:                                        # nothing to leave out: the first columns only
        p, err = min(n, pmin), float("nan")
    else:
        errs = {}
        for p in range(pmin, min(n - 1, len(idx)) + 1):
            res = []
            for i in range(n):
                k = np.arange(n) != i
                res.append(B[i, :p] @ wls(B[k][:, :p], y_s[k], w[k]) - y_s[i])
            errs[p] = float(np.linalg.norm(res) / (np.linalg.norm(y_s) + 1e-30))
        p = min(errs, key=errs.get)
        err = errs[p]
    q, n_modes = _result(wls(B[:, :p], y_s, w), idx, nr, basis)
    return ShapeFit(q, n_modes, length_cm, "sensors", err, n_sensors=n, basis=basis)


def _video_order(B: np.ndarray, y: np.ndarray, w: np.ndarray, pmin: int, folds: int = 5) -> tuple[int, float]:
    """Number of columns that predicts left-out video points best (interleaved folds)."""
    n = len(y)
    errs = {}
    for p in range(pmin, min(n - n // folds - 1, B.shape[1]) + 1):
        num = 0.0
        for k in range(folds):
            out = np.arange(n) % folds == k
            c = wls(B[~out][:, :p], y[~out], w[~out])
            num += float(np.sum(np.abs(w[out] * (B[out][:, :p] @ c - y[out])) ** 2))
        errs[p] = float(np.sqrt(num / (np.sum(np.abs(w * y) ** 2) + 1e-30)))
    best = min(errs.values())
    p = min(k for k, v in errs.items() if v <= 1.1 * best)     # the smallest model that is as good
    return p, errs[p]


def fit_fusion(z_s, y_s, z_v, y_v, length_cm: float, sigma_s=None, sigma_v=None, *,
               gain: complex | None = None, gain_abs: float | None = None, camera: bool = False,
               max_modes: int = 4, iters: int = 20, rigid: str = "both",
               basis: Callable | None = None) -> ShapeFit:
    """Sensors + video on one shape.

    gain      known complex video/sensor gain (a harmonic of the line used for alignment)
    gain_abs  only |g| known, the phase is found (a line unrelated to the alignment line)
    neither   g is found from the data (the alignment line itself)
    camera    also solve rigid motion of the camera; needs 2 sensors with a known gain, 3 otherwise
    rigid     base motion the structure may have: "both", "rotation" or "none" (see RIGID)
    """
    z_s, y_s = np.asarray(z_s, float), np.asarray(y_s, complex)
    z_v, y_v = np.asarray(z_v, float), np.asarray(y_v, complex)
    ns, nv = len(y_s), len(y_v)
    ws, wv = _weights(y_s, sigma_s), _weights(y_v, sigma_v)
    Bs, idx, nr = _basis(z_s, length_cm, max_modes, rigid, basis)
    Bv = _basis(z_v, length_cm, max_modes, rigid, basis)[0]
    p, err = _video_order(Bv, y_v, wv, max(nr, 1))
    Bs, Bv = Bs[:, :p], Bv[:, :p]
    camera = bool(camera and ns >= (2 if gain is not None else 3))
    Cv = beam_basis(z_v, length_cm, 0) if camera else np.zeros((nv, 0))
    Zs = np.zeros((ns, Cv.shape[1]))

    def solve(g: complex) -> np.ndarray:
        return wls(np.r_[np.c_[g * Bv, Cv], np.c_[Bs, Zs]], np.r_[y_v, y_s], np.r_[wv, ws])

    if gain is not None:
        g, x = complex(gain), solve(complex(gain))
    else:
        c = wls(Bv, y_v, wv)                              # video alone: the shape up to the gain
        ref = Bs @ c
        g = np.vdot(y_s * ws, ref * ws) / (np.vdot(y_s * ws, y_s * ws) + 1e-30)
        if gain_abs is not None:
            g = gain_abs * np.exp(1j * np.angle(g))
        x = solve(g)
        for _ in range(iters):
            m = Bv @ x[:p]
            resid = y_v - Cv @ x[p:]
            g_new = np.vdot(m * wv, resid * wv) / (np.vdot(m * wv, m * wv) + 1e-30)
            if gain_abs is not None:
                g_new = gain_abs * np.exp(1j * np.angle(g_new))
            if abs(g_new - g) < 1e-9 * abs(g):
                break
            g, x = g_new, solve(g_new)
    q, n_modes = _result(x[:p], idx, nr, basis)
    return ShapeFit(q, n_modes, length_cm, "fusion", err, complex(g),
                    x[p:].copy() if camera else None, ns, nv, basis)


def screen_patches(z_v, y_v, seed: np.ndarray, length_cm: float, tol: float = 0.05, max_modes: int = 4) -> np.ndarray:
    """Video points that follow a smooth beam shape, grown from trusted ``seed`` points.

    Cables, weak contrast and patches straddling a block give biased amplitudes
    that a noise estimate does not show; they are dropped where the strong line
    (large motion everywhere) says they do not belong to the beam.
    """
    z_v, y_v = np.asarray(z_v, float), np.asarray(y_v, complex)
    on = (z_v >= 0) & (z_v <= length_cm)
    keep = seed.copy()
    B = beam_basis(z_v, length_cm, max_modes)
    for _ in range(4):
        p = min(B.shape[1], max(1, int(keep.sum()) - 1))
        fit = B[:, :p] @ wls(B[keep][:, :p], y_v[keep], np.ones(int(keep.sum())))
        keep = seed | (on & (np.abs(y_v - fit) < tol * np.abs(fit).max()))
    return keep
