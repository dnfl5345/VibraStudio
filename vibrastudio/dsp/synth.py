"""Sensor-guided synthesis, one frequency line at a time.

For every line the amplification factor comes from the sensors,
alpha = target_px / (largest sensor motion in px), so each line is drawn the
same size whatever its real amplitude. Two pictures are made at that alpha:

* video magnification: the camera's own phase motion at the line, times alpha;
* synthesis: the still reference frame warped by alpha x the shape that the
  sensors give (dsp.fusion), plus the video where the video can be trusted.

Whether the video magnification can be trusted is measured, not assumed: the
magnified motion at the sensor blocks is compared with alpha x sensors, and
the same alpha is applied a little off the line, where there is only noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import fusion as fu

HARM_FLOOR = 0.005            # harmonics of the drive in the video: ~0.5 % of the drive line is not structure
NOISE_MAX = 0.10              # noise alone may draw at most this fraction of the target motion
MIN_SENSORS_ALONE = 4         # fewer sensors than this cannot show bending on their own
ALPHA_MAX = 5.0e4


@dataclass
class SynthOptions:
    freqs: list | None = None          # Hz; None: the drive and the strongest other sensor lines
    max_lines: int = 5
    target_px: float = 10.0
    source: str = "auto"               # "sensors" | "fusion" | "auto" (fusion when fewer than 4 sensors are used)
    use_sensors: list | None = None    # labels of the sensors the shape may use; None: all
    length_cm: float | None = None     # free length of the beam; None: 2 % above the top sensor
    phases: int = 16
    control_hz: float = 1.5


@dataclass
class Line:
    f: float                           # Hz on the DAQ clock
    f_video: float                     # Hz where the camera sees it (aliased above fps/2)
    folded: bool
    harmonic: int                      # k if f = k x drive, else 0
    f_control: float = 0.0
    sensor_um: float = 0.0
    alpha: float = 0.0
    snr_video: float = 0.0
    measured_ratio: float = float("nan")
    mac: float = float("nan")
    noise_only: float = float("nan")
    valid: bool = False                # video magnification agrees with the sensors
    source: str = "sensors"            # what the synthesis was drawn from
    n_modes: int = 0
    sensors_used: list = field(default_factory=list)
    labels: list = field(default_factory=list)      # every sensor, in the order of z_s / y_s
    loo_error: float = float("nan")    # all sensors, one left out at a time (sensor-only model)
    held_out_error: float = float("nan")   # at the sensors the shape was not allowed to use
    synth_px: float = float("nan")     # largest block motion measured in the synthesized frames
    note: str = ""                     # why the synthesis should not be trusted, if so
    gain: complex = 1.0                # sensor um -> the video's phase reference (|g|, time offset)
    fit: fu.ShapeFit | None = None
    z_s: np.ndarray | None = None
    y_s: np.ndarray | None = None      # (n_s,) um, image-right, DAQ phase
    z_v: np.ndarray | None = None
    y_v: np.ndarray | None = None      # (n_v,) um / gain, brought to the DAQ phase (kept patches)
    frames_video: list = field(default_factory=list)
    frames_synth: list = field(default_factory=list)

    @property
    def name(self) -> str:
        s = f"{self.f:.1f} Hz"
        if self.harmonic > 1:
            s += f" (harmonic {self.harmonic})"
        if self.folded or abs(self.f_video - self.f) > 0.02 * self.f:
            s += f", seen at {self.f_video:.1f} Hz"
        return s


def line_and_floor(x: np.ndarray, fs: float, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Complex amplitude at f and the median amplitude a few resolution cells away."""
    from .magnify import phasor
    cell = fs / x.shape[-1]
    off = [o * cell for o in (-6, -4, 4, 6) if f + o * cell > 0]
    return phasor(x, fs, f), np.median([np.abs(phasor(x, fs, f + o)) for o in off], axis=0)


def _refine(x: np.ndarray, fs: float, f: float, span: float) -> float:
    from .magnify import phasor
    for _ in range(2):
        grid = np.linspace(f - span, f + span, 21)
        f = float(grid[np.argmax([np.abs(phasor(x, fs, v)).sum() for v in grid])])
        span /= 8
    return f


def plan(opts: SynthOptions, ods, a_y: np.ndarray, fs: float, f_drive: float, fps: float, f_vid: float,
         n_frames: int, mains: float = 60.0) -> list[Line]:
    """Which lines, and where the camera sees them. ``a_y``: (n_s, N) acceleration along the image axis (g)."""
    from scipy.signal import find_peaks

    from .ods_video import G
    f1 = _refine(a_y, fs, f_drive, 0.003 * f_drive)
    rho = f_vid / f1                                   # camera clock / DAQ clock
    cands = [f1]
    if opts.freqs:
        cands += [float(f) for f in opts.freqs]
    else:
        n_nodes = ods.auto.shape[0] // 3
        amp = ods.amplitude_g().reshape(n_nodes, 3, -1)[:, 1].max(axis=0)
        band = np.nonzero((ods.freq >= 3.0) & (ods.freq <= min(0.4 * fs, 1000.0)))[0]
        d = amp[band] * G * 1e6 / (2 * np.pi * ods.freq[band]) ** 2
        pk, _ = find_peaks(np.log(d + 1e-30), prominence=np.log(6.0), distance=max(1, int(1.0 / ods.df)))
        for k in pk[np.argsort(d[pk])[::-1]][:opts.max_lines + 2]:
            cands.append(float(ods.freq[band[k]]))
    lines: list[Line] = []
    for f in cands:
        if lines:
            f = _refine(a_y, fs, f, 0.3)
        k = int(round(f / f1))
        harm = k if k >= 1 and abs(f - k * f1) < 0.03 + 0.002 * k * f1 else 0
        if harm:
            f = harm * f1
        if any(abs(f - ln.f) < 0.3 for ln in lines):
            continue
        fv, folded = fu.alias(f * rho, fps)
        lines.append(Line(f, fv, folded, harm))
        if len(lines) >= opts.max_lines and not opts.freqs:
            break
    busy = [ln.f_video for ln in lines] + [fu.alias(m * 2 * mains, fps)[0] for m in (1, 2, 3)]
    for ln in lines:
        for off in (1, -1, 1.5, -1.5, 2.1, -2.1, 2.7, -2.7):
            c = ln.f_video + off * opts.control_hz
            if 1.0 < c < fps / 2 - 1.0 and min(abs(c - b) for b in busy) > 0.6:
                break
        ln.f_control = c
    return sorted(lines, key=lambda ln: ln.f)


def build(lines: list[Line], opts: SynthOptions, *, a: np.ndarray, fs: float, s_h: np.ndarray, labels: list,
          tracks: np.ndarray, roi: list, blocks, idx: list, fps: float, px_mm: float, gain: complex,
          mag, mag_gain: float, base: np.ndarray, f_vid: float,
          log: Callable[[str], None] = print,
          progress: Callable[[str, int, int], bool] | None = None) -> list[Line]:
    """Fill every line: alpha, check of the video magnification, shape fit, both frame sets."""
    import cv2

    from .magnify import flow_points, phasor
    from .ods_video import acc_to_disp_mm

    um = 1e3 / px_mm
    L = float(opts.length_cm or 1.02 * np.max(s_h))
    hv = np.array([q["h"] for q in roi])
    is_blk = np.array([q["kind"] == "block" for q in roi])
    blk = np.nonzero(is_blk)[0]
    tx = tracks[:, :, 0]
    p1 = phasor(tx, fps, f_vid) * um
    keep = fu.screen_patches(hv, p1, is_blk, L)
    use = [i for i, lab in enumerate(labels) if opts.use_sensors is None or lab in opts.use_sensors]
    if not use:
        raise ValueError(f"none of the sensors {opts.use_sensors} is in the recording ({', '.join(labels)})")
    rest = [i for i in range(len(labels)) if i not in use]
    src = opts.source if opts.source != "auto" else ("fusion" if len(use) < 4 else "sensors")
    log(f"synthesis: beam length {L:.1f} cm; {int(keep.sum())} of {len(hv)} video patches follow the beam; "
        f"shape from {'sensors + video where it passes the check' if src == 'fusion' else 'sensors'} "
        f"({', '.join(labels[i] for i in use)})")

    thetas = 2 * np.pi * np.arange(opts.phases) / opts.phases
    centers = np.c_[blocks.cols, blocks.rows].astype(float)
    win = max(15, 2 * blocks.half + 1)
    Hh, Ww = base.shape
    rows_h = (blocks.row0 - np.arange(Hh)) / blocks.px_per_cm
    xx, yy = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32))
    edge = np.minimum(np.arange(Ww), Ww - 1 - np.arange(Ww)) / 30.0
    taper = (0.5 - 0.5 * np.cos(np.pi * np.clip(edge, 0, 1))).astype(np.float32)   # the frame border stays put
    base32 = base.astype(np.float32)

    def measured(frames) -> np.ndarray:
        d = np.array([flow_points(base, f, centers, win) for f in frames])[..., 0] * np.exp(-1j * thetas)[:, None]
        ok = np.isfinite(d).sum(axis=0) > 0                        # a block the flow lost in every frame stays nan
        out = np.full(d.shape[1], np.nan, complex)
        out[ok] = 2 * np.nanmean(d[:, ok], axis=0)
        return out

    chk = np.array([i in use for i in idx])                        # blocks whose sensor the user allows

    def u8(f) -> np.ndarray:
        return np.clip(f, 0, 255).astype(np.uint8)

    for n, ln in enumerate(lines):
        ps, ns = line_and_floor(a[:, 1], fs, ln.f)
        ys = -acc_to_disp_mm(ps, ln.f) * 1e3                       # um, image right = -Y
        sig_s = np.abs(acc_to_disp_mm(ns, ln.f)) * 1e3
        pv, nv = line_and_floor(tx, fps, ln.f_video)
        yv = (np.conj(pv) if ln.folded else pv) * um               # um, on the true-frequency phase
        sig_v = nv * um
        if ln.harmonic > 1:
            sig_v = np.hypot(sig_v, HARM_FLOOR * np.abs(p1))
        if ln.harmonic == 1:
            gl = gain
        elif ln.harmonic > 1:
            gl = abs(gain) * np.exp(1j * ln.harmonic * np.angle(gain))
        else:                                                      # unrelated line: only |g| carries over
            gl = abs(gain) * np.exp(1j * np.angle(np.vdot(ys[idx], yv[blk])))
        pred = gl * ys[idx] / um                                   # px at the blocks
        if ln.folded:
            pred = np.conj(pred)                                   # as the camera plays it back
        ln.sensor_um = float(np.abs(ys).max())
        ln.alpha = float(min(opts.target_px / max(float(np.abs(pred).max()), 1e-9), ALPHA_MAX))
        ln.snr_video = float(np.sqrt(np.mean(np.abs(yv[blk]) ** 2)) / (np.sqrt(np.mean(sig_v[blk] ** 2)) + 1e-12))

        # video magnification at this alpha, and the same alpha where there is only noise
        a_mag = ln.alpha / max(mag_gain, 1e-6)
        fr = [mag.frame(a_mag, th, ln.f_video) for th in thetas]
        dm = measured(fr)
        dc = measured([mag.frame(a_mag, th, ln.f_control) for th in thetas])
        target = ln.alpha * pred
        big = chk & (np.abs(pred) > 0.25 * np.abs(pred[chk]).max())
        ln.measured_ratio = float(np.nanmedian(np.abs(dm[big]) / np.abs(target[big])))
        ln.mac = fu.mac(target[chk], np.nan_to_num(dm[chk])) if chk.sum() >= 3 else float("nan")
        ln.noise_only = float(np.sqrt(np.nanmean(np.abs(dc) ** 2)) / np.sqrt(np.mean(np.abs(target[chk]) ** 2)))
        ln.valid = bool(ln.noise_only < NOISE_MAX and 0.8 < ln.measured_ratio < 1.25
                        and not ln.mac <= 0.9)                     # with fewer than 3 sensors there is no shape to match
        ln.frames_video = [u8(f) for f in fr]

        # shape
        fit_all = fu.fit_sensors(s_h, ys, L, sig_s)
        ln.loo_error = fit_all.cv_error
        if src == "fusion" and ln.valid:
            kw = {} if ln.harmonic == 1 else dict(gain=gl) if ln.harmonic > 1 else dict(gain_abs=abs(gain))
            fit = fu.fit_fusion(s_h[use], ys[use], hv[keep], yv[keep], L, sig_s[use], sig_v[keep],
                                camera=ln.harmonic != 1, **kw)
        else:
            fit = fit_all if not rest else fu.fit_sensors(s_h[use], ys[use], L, sig_s[use])
        if rest:
            ln.held_out_error = float(np.linalg.norm(fit(s_h[rest]) - ys[rest]) / (np.linalg.norm(ys[rest]) + 1e-30))
        ln.fit, ln.source, ln.n_modes, ln.gain = fit, fit.source, fit.n_modes, complex(gl)
        if fit.source == "sensors" and len(use) < MIN_SENSORS_ALONE:
            ln.note = (f"{len(use)} sensor{'s' if len(use) > 1 else ''} alone and no usable video at this line: "
                       "a straight line, bending is not shown")
        ln.sensors_used, ln.labels = [labels[i] for i in use], list(labels)
        ln.z_s, ln.y_s, ln.z_v = s_h, ys, hv[keep]
        ln.y_v = yv[keep] / (fit.gain if fit.gain is not None else gl)

        # synthesis: the still frame moved by alpha x shape
        shape_px = gl * fit(rows_h) / um
        if ln.folded:
            shape_px = np.conj(shape_px)
        fs_ = []
        for th in thetas:
            dx = (ln.alpha * np.real(shape_px * np.exp(1j * th))).astype(np.float32)
            fs_.append(cv2.remap(base32, xx - taper[None, :] * dx[:, None], yy, cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE))
        ln.synth_px = float(np.nanmax(np.abs(measured(fs_))))
        ln.frames_synth = [u8(f) for f in fs_]
        log(f"  {ln.name}: sensors {ln.sensor_um:.2f} um, alpha {ln.alpha:,.0f}; video magnification "
            f"measured/target {ln.measured_ratio:.2f}, shape match {ln.mac:.2f}, noise only {ln.noise_only:.2f} "
            f"-> {'valid' if ln.valid else 'NOT valid'}; synthesis from {ln.source} ({ln.n_modes} bending shapes"
            + (f", left-out sensors {100 * ln.held_out_error:.0f} %" if rest else
               f", leave-one-out {100 * ln.loo_error:.0f} %") + f"), drawn {ln.synth_px:.1f} px"
            + (f"  [{ln.note}]" if ln.note else ""))
        if progress and not progress("synth", n + 1, len(lines)):
            raise InterruptedError("cancelled")
    return lines
