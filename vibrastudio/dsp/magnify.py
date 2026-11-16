"""Video motion magnification checked against the accelerometers.

For a structure driven at one frequency (shaker sine) and filmed while a
recording was taken:

  1. frame rate: given, or from mains light flicker (lights blink at
     2 x mains; the alias that shows the sensors' drive tone wins),
  2. regions of interest, chosen automatically: the lit beam strip, the
     sensor blocks (template matching, then matched to the rig heights,
     which also gives px/mm) and a motion map (signal/noise at the drive),
  3. true sub-pixel motion of each region (gradient least squares) as a
     phasor at the drive frequency,
  4. narrow-band phase-based magnification: complex steerable pyramid,
     per-pixel phase phasor accumulated frame by frame (memory does not
     grow with video length), one steady cycle synthesized for any alpha,
  5. magnified motion measured back with pyramidal optical flow ->
     linearity per alpha; recommended alpha = largest that stays linear,
  6. video vs sensors per sensor (image plane: right = -Y, up = +Z; one
     complex gain absorbs the unknown time offset between camera and DAQ).

Needs OpenCV (``cv2``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Callable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _need_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed (pip install opencv-python-headless)")


# ----------------------------------------------------------------- basics
def read_gray(path: str, x0: int = 0, x1: int | None = None, max_frames: int | None = None,
              progress: Callable[[int], bool] | None = None) -> np.ndarray:
    """(T, H, W) uint8 luma, optionally cropped to columns x0:x1."""
    _need_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    out = []
    while max_frames is None or len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[:, x0:x1].copy())
        if progress and len(out) % 100 == 0 and not progress(len(out)):
            raise InterruptedError("cancelled")
    if not out:
        raise RuntimeError(f"no frames in {path}")
    return np.array(out)


def phasor(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """Complex amplitude at f0 along the last axis (Hann window)."""
    x = np.asarray(x, float)
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    w = np.hanning(n)
    e = np.exp(-2j * np.pi * f0 * np.arange(n) / fs)
    return 2 * (x * w * e).sum(axis=-1) / w.sum()


# --------------------------------------------------------------- pyramid
def csp_filters(h: int, w: int, n_scales: int = 5, n_orient: int = 4):
    """Self-inverting complex steerable pyramid filters (FFT domain).

    hi^2 + lo^2 + sum(|B|^2 + |B_mirror|^2) = 1 at every frequency, so a frame
    is rebuilt exactly as IFFT(F(hi^2+lo^2)) + sum 2 Re[IFFT(FFT(B) conj(H))].
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.clip(np.sqrt(fx ** 2 + fy ** 2) / 0.5, 1e-9, None)
    theta = np.arctan2(fy, fx)
    log_r = np.log2(r)

    def step(cut_log):
        x = np.clip(log_r - cut_log + 1.0, 0.0, 1.0)
        return np.sin(0.5 * np.pi * x)

    us = [step(-float(s)) for s in range(n_scales + 1)]
    hi = us[0].copy()
    lows = np.sqrt(np.clip(1.0 - us[0] ** 2, 0, 1))
    rads = []
    for s in range(1, n_scales + 1):
        rads.append(lows * us[s])
        lows = lows * np.sqrt(np.clip(1.0 - us[s] ** 2, 0, 1))
    lo = lows
    bands = []
    for rad in rads:
        for k in range(n_orient):
            ang = np.angle(np.exp(1j * (theta - np.pi * k / n_orient)))
            A = np.where(np.abs(ang) < np.pi / 2, np.cos(ang) ** (n_orient - 1), 0.0)
            bands.append((rad * A).astype(np.float32))
    total = hi ** 2 + lo ** 2
    for B in bands:
        Bm = np.roll(np.abs(B)[::-1, ::-1], (1, 1), axis=(0, 1))
        total = total + B ** 2 + Bm ** 2
    norm = np.sqrt(np.clip(total, 1e-12, None)).astype(np.float32)
    return [(b / norm).astype(np.float32) for b in bands], (lo / norm).astype(np.float32), \
        (hi / norm).astype(np.float32)


class NarrowbandMagnifier:
    """Per-pixel phase phasor at f0 in every pyramid band, accumulated frame by frame.

    5 scales keep large magnified motions in the coarse bands: on the 11-02
    side video the result stays linear to about 20 px of drawn motion.
    """

    def __init__(self, ref: np.ndarray, n_scales: int = 5, n_orient: int = 4, smooth_px: float = 1.0) -> None:
        h, w = ref.shape
        self.bands, self.lo, self.hi = csp_filters(h, w, n_scales, n_orient)
        self.ref = ref.astype(np.float32)
        self.Fref = np.fft.fft2(self.ref)
        self.Bref = [np.fft.ifft2(self.Fref * H) for H in self.bands]
        self.acc = [np.zeros((h, w), np.complex128) for _ in self.bands]
        self.noise = [np.zeros((h, w), np.complex128) for _ in self.bands]
        self.smooth = smooth_px
        self.phi: list[np.ndarray] = []

    def run(self, V: np.ndarray, fps: float, f0: float,
            progress: Callable[[int, int], bool] | None = None) -> None:
        T = len(V)
        win = np.hanning(T)
        t = np.arange(T) / fps
        e = np.exp(-2j * np.pi * f0 * t) * win
        en = np.exp(-2j * np.pi * f0 * 1.07 * t) * win          # off-frequency: noise floor
        for i in range(T):
            F = np.fft.fft2(V[i].astype(np.float32))
            for b, H in enumerate(self.bands):
                d = np.angle(np.fft.ifft2(F * H) * np.conj(self.Bref[b]))
                self.acc[b] += d * e[i]
                self.noise[b] += d * en[i]
            if progress and i % 20 == 0 and not progress(i, T):
                raise InterruptedError("cancelled")
        wsum = win.sum()
        self.amp = [np.abs(B) for B in self.Bref]
        self.phi = []
        for b in range(len(self.bands)):                         # amplitude-weighted smoothing
            wgt = self.amp[b] ** 2
            a = self.acc[b] * wgt
            s = self.smooth
            num = cv2.GaussianBlur(a.real, (0, 0), s) + 1j * cv2.GaussianBlur(a.imag, (0, 0), s)
            self.phi.append(2 * num / (cv2.GaussianBlur(wgt, (0, 0), s) + 1e-9) / wsum)

    def motion_map(self) -> np.ndarray:
        """Per-pixel signal/noise of the phase motion at f0."""
        s = sum(np.abs(a) * m ** 2 for a, m in zip(self.acc, self.amp))
        n = sum(np.abs(a) * m ** 2 for a, m in zip(self.noise, self.amp))
        return s / (n + 1e-12)

    def frame(self, alpha: float, theta: float) -> np.ndarray:
        out = np.fft.ifft2(self.Fref * (self.hi ** 2 + self.lo ** 2)).real
        for H, B, ph in zip(self.bands, self.Bref, self.phi):
            Bm = B * np.exp(1j * alpha * np.real(ph * np.exp(1j * theta)))
            out += 2 * np.fft.ifft2(np.fft.fft2(Bm) * np.conj(H)).real
        return out


# ------------------------------------------------------------ auto ROIs
def find_beam(ref: np.ndarray) -> int:
    """Column of the brightest long vertical strip (the lit beam face)."""
    _need_cv2()
    prof = ref[: ref.shape[0] // 2].astype(np.float32).mean(axis=0)
    return int(np.argmax(prof - cv2.blur(prof[None], (41, 1))[0]))


@dataclass
class Blocks:
    rows: np.ndarray            # (k,) block centre rows, bottom -> top
    cols: np.ndarray
    heights_cm: np.ndarray      # matched rig heights
    half: int                   # half box size, px
    px_per_cm: float
    row0: float                 # row of height 0
    fit_rms: float
    side: int                   # +1 right of the beam, -1 left
    missing: list = field(default_factory=list)


def find_blocks(ref: np.ndarray, beam_x: int, heights_cm, search: int = 110) -> Blocks:
    """Sensor blocks beside the beam, matched to the rig heights.

    A well-lit compact blob (top-hat) gives a template; normalized template
    matching along its column finds the others even where the light is poor;
    the subset whose rows fit row = row0 - px_per_cm * height best is kept.
    """
    _need_cv2()
    from scipy.signal import find_peaks
    ref = ref.astype(np.float32)
    H_all = np.sort(np.asarray(heights_cm, float))
    best_tpl = None
    for side in (+1, -1):
        lo, hi = (beam_x + 6, beam_x + search) if side > 0 else (beam_x - search, beam_x - 6)
        lo, hi = max(lo, 0), min(hi, ref.shape[1])
        if hi - lo < 20:
            continue
        band = ref[:, lo:hi]
        th = band - cv2.GaussianBlur(band, (0, 0), 15)
        mask = (th > max(12.0, 2.5 * th.std())).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if 250 <= a <= 5000 and 0.5 <= w / max(h, 1) <= 2.0:
                score = a * float(th[y:y + h, x:x + w].mean())
                if best_tpl is None or score > best_tpl[0]:
                    best_tpl = (score, side, int(round(cen[i][1])), int(round(cen[i][0])) + lo, max(w, h) // 2)
    if best_tpl is None:
        raise RuntimeError("no sensor block found beside the beam")
    _, side, cy, cx, hw = best_tpl
    hw = int(np.clip(hw, 10, 30))
    tpl = ref[cy - hw:cy + hw, cx - hw:cx + hw]
    x0, x1 = max(cx - 40 - hw, 0), min(cx + 40 + hw, ref.shape[1])
    res = cv2.matchTemplate(ref[:, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
    score, col = res.max(axis=1), res.argmax(axis=1)
    pk, _ = find_peaks(score, height=0.25, distance=max(20, int(1.5 * hw)))
    cand_r = pk + hw
    cand_c = x0 + col[pk] + hw
    cand_s = score[pk]
    if len(cand_r) < 3:
        raise RuntimeError("too few sensor blocks found")
    # RANSAC: a line row = row0 - k * height through two strong matches, scored
    # by how many heights find a matching candidate near the line (weighted by
    # match score); then a least-squares refit on the inliers. Heights with no
    # candidate near the line are reported missing instead of being forced.
    strong = np.nonzero(cand_s >= 0.5)[0]
    if len(strong) < 2:
        strong = np.argsort(cand_s)[::-1][:3]
    spacing = float(np.median(np.diff(np.sort(H_all)))) if len(H_all) > 1 else 1.0

    def assign(r0, k, tol):
        used, out = set(), []
        for h in H_all:
            pred = r0 - k * h
            d = np.abs(cand_r - pred)
            for j in np.argsort(d):
                if d[j] > tol:
                    break
                if j not in used:
                    used.add(j); out.append((h, j)); break
        return out

    best = None
    for i, j in permutations(strong, 2):
        for a, b in combinations(range(len(H_all)), 2):
            ha, hb = H_all[a], H_all[b]
            ri, rj = cand_r[i], cand_r[j]
            if ri == rj or ha == hb:
                continue
            k = (ri - rj) / (hb - ha)                            # rows grow downward, heights up
            if k <= 0:
                continue
            r0 = ri + k * ha
            tol = 0.3 * k * spacing
            m = assign(r0, k, tol)
            sc = sum(cand_s[jj] for _, jj in m) - 0.002 * sum(abs(cand_r[jj] - (r0 - k * hh)) for hh, jj in m)
            if best is None or sc > best[0]:
                best = (sc, r0, k)
    if best is None:
        raise RuntimeError("sensor blocks do not line up with the rig heights")
    _, r0, k = best
    m = assign(r0, k, 0.3 * k * spacing)
    hh = np.array([h for h, _ in m]); jj = np.array([j for _, j in m])
    A = np.c_[np.ones(len(hh)), -hh]
    coef = np.linalg.lstsq(A, cand_r[jj].astype(float), rcond=None)[0]
    m = assign(coef[0], coef[1], 0.3 * coef[1] * spacing)       # final assignment on the refit line
    hh = np.array([h for h, _ in m]); jj = np.array([j for _, j in m])
    rms = float(np.sqrt(np.mean((coef[0] - coef[1] * hh - cand_r[jj]) ** 2)))
    o = np.argsort(hh)
    missing = [float(h) for h in H_all if not np.any(np.isclose(hh, h))]
    return Blocks(cand_r[jj][o].astype(float), cand_c[jj][o].astype(float), hh[o], int(hw * 0.7),
                  float(coef[1]), float(coef[0]), rms, side, missing)


def lk_patches(V: np.ndarray, ref: np.ndarray, boxes) -> np.ndarray:
    """(n, T, 2) sub-pixel dx (right), dy (down) per box by gradient least squares."""
    _need_cv2()
    rb = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 1.0)
    Iy, Ix = np.gradient(rb)
    H, W = ref.shape
    out = []
    for (y0, y1, x0, x1) in boxes:
        y0, y1, x0, x1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
        A = np.c_[Ix[y0:y1, x0:x1].ravel(), Iy[y0:y1, x0:x1].ravel()]
        AtA = A.T @ A + 1e-6 * np.eye(2)
        py0, py1, px0, px1 = max(y0 - 4, 0), min(y1 + 4, H), max(x0 - 4, 0), min(x1 + 4, W)
        sub = np.stack([cv2.GaussianBlur(v[py0:py1, px0:px1].astype(np.float32), (0, 0), 1.0) for v in V])
        dI = (sub[:, y0 - py0:y1 - py0, x0 - px0:x1 - px0] - rb[y0:y1, x0:x1]).reshape(len(V), -1)
        out.append(-np.linalg.solve(AtA, A.T @ dI.T).T)
    return np.array(out)


def choose_fps(bright: np.ndarray, motion: np.ndarray, f_drive: float, mains: float,
               fps_hint: float | None = None) -> tuple[float, float]:
    """Frame rate from light flicker (2 x mains), the alias that shows the drive tone."""
    n = len(bright)
    X = np.abs(np.fft.rfft((bright - bright.mean()) * np.hanning(n)))
    k = 5 + int(np.argmax(X[5:]))
    a, b, c = np.log(X[k - 1:k + 2] + 1e-12)
    cpf = (k + 0.5 * (a - c) / (a - 2 * b + c)) / n
    best = None
    for m in range(0, 6):
        for s in (+1, -1):
            denom = m + s * cpf
            if denom <= 0:
                continue
            fps = 2 * mains / denom
            if (fps_hint and abs(fps / fps_hint - 1) > 0.5) or f_drive >= fps / 2:
                continue
            p = abs(phasor(motion, fps, f_drive))
            noise = np.median([abs(phasor(motion, fps, f_drive * r)) for r in (0.93, 0.95, 1.05, 1.07)])
            snr = float(p / max(noise, 1e-12))
            if best is None or snr > best[1]:
                best = (fps, snr)
    if best is None:
        raise RuntimeError("no frame rate found that shows the drive tone; give --fps")
    return best


def flow_points(a: np.ndarray, b: np.ndarray, pts: np.ndarray, win: int = 21) -> np.ndarray:
    """Pyramidal LK displacement (N, 2) of points from image a to b."""
    to8 = lambda im: np.clip(im, 0, 255).astype(np.uint8)
    win = int(win) | 1
    p1, st, _ = cv2.calcOpticalFlowPyrLK(to8(a), to8(b), pts.astype(np.float32).reshape(-1, 1, 2), None,
                                         winSize=(win, win), maxLevel=4,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
    d = p1.reshape(-1, 2) - pts
    d[st.ravel() == 0] = np.nan
    return d


# ---------------------------------------------------------------- driver
@dataclass
class MagnifyResult:
    fps: float
    fps_snr: float
    f_video: float
    f_sensor: float
    crop: tuple
    beam_x: int
    blocks: Blocks
    px_per_mm: float
    ref: np.ndarray
    snr_map: np.ndarray
    roi: list                        # dicts: kind, h, box
    P: np.ndarray                    # (n_roi, 2) px phasor (dx right, dy down)
    sensor_labels: list
    pred: np.ndarray                 # (k, 2) sensor-predicted px phasor at the blocks (time-aligned)
    gain: complex                    # video / sensor
    per_sensor: list
    alphas: list
    linearity: dict                  # alpha -> (k,) ratio, relative to the smallest alpha
    mag_gain: float                  # magnified/true at the smallest alpha
    alpha_recommended: float
    frames: dict                     # alpha -> list of K frames (one cycle)
    base: np.ndarray
    thetas: np.ndarray


def run(video: str, model, acc_nodes: np.ndarray, fs: float, *, fps: float | None = None,
        mains: float = 60.0, fps_hint: float | None = None, alphas=None, target_px: float = 12.0,
        cycle_frames: int = 24, log: Callable[[str], None] = print,
        progress: Callable[[str, int, int], bool] | None = None) -> MagnifyResult:
    """Whole pipeline. ``model``/``acc_nodes`` come from dsp.ods_video (rig-mapped recording)."""
    from . import ods_video as ov
    _need_cv2()
    prog = (lambda stage: (lambda i, n=0: progress(stage, i, n))) if progress else (lambda stage: None)

    ods = ov.SpectralODS(acc_nodes, fs, 0.5)
    v = ods.velocity_spectrum().sum(axis=0)
    band = (ods.freq > 2) & (ods.freq < 200)
    f_drive = float(ods.freq[band][np.argmax(v[band])])
    shape, f_drive, coh, _ = ods.shape(f_drive)
    sens = [i for i in range(model.n_nodes) if model.measured[i].any()]
    s_h = model.nodes[sens, 2] * 100.0
    log(f"sensors: drive {f_drive:.3f} Hz (coherence {coh:.2f})")

    first = read_gray(video, max_frames=60).astype(np.float32).mean(axis=0)
    beam_full = find_beam(first)
    x0, x1 = max(0, beam_full - 110), min(first.shape[1], beam_full + 150)
    V = read_gray(video, x0, x1, progress=prog("read"))
    T, Hh, Ww = V.shape
    ref = V[:50].astype(np.float32).mean(axis=0)
    beam_x = beam_full - x0
    bl = find_blocks(ref, beam_x, s_h)
    px_mm = bl.px_per_cm / 10.0
    log(f"video: {T} frames; beam at x={x0 + beam_x}; {len(bl.rows)} sensor blocks "
        f"({'right' if bl.side > 0 else 'left'} of the beam), {px_mm:.3f} px/mm, fit rms {bl.fit_rms:.1f} px"
        + (f"; not found: {', '.join(f'{h:g} cm' for h in bl.missing)}" if bl.missing else ""))

    roi = []
    for r, c, h in zip(bl.rows, bl.cols, bl.heights_cm):
        roi.append(dict(kind="block", h=float(h), box=(int(r - bl.half), int(r + bl.half),
                                                        int(c - bl.half), int(c + bl.half))))
    for y in range(20, Hh - 20, 30):
        roi.append(dict(kind="beam", h=float((bl.row0 - y) / bl.px_per_cm),
                        box=(y - 15, y + 15, beam_x - 12, beam_x + 12)))
    tracks = lk_patches(V, ref, [r["box"] for r in roi])
    iblk = [i for i, r in enumerate(roi) if r["kind"] == "block"]
    top = iblk[-1]
    if fps is None:
        bgx = slice(5, 40) if bl.side > 0 else slice(Ww - 40, Ww - 5)
        bright = V[:, 100:400, bgx].astype(np.float32).mean(axis=(1, 2))
        fps, fsnr = choose_fps(bright, tracks[top, :, 0], f_drive, mains, fps_hint)
    else:
        fsnr = float("nan")
    grid = np.linspace(f_drive * 0.995, f_drive * 1.005, 201)
    f_vid = float(grid[np.argmax([abs(phasor(tracks[top, :, 0], fps, f)) for f in grid])])
    log(f"frame rate {fps:.3f} fps; drive in video {f_vid:.3f} Hz (DAQ {f_drive:.3f} Hz)")
    P = phasor(tracks.transpose(0, 2, 1), fps, f_vid)

    mag = NarrowbandMagnifier(ref)
    log("magnifier: accumulating phase phasors ...")
    mag.run(V, fps, f_vid, prog("magnify"))
    del V

    # sensors -> image plane (right = -Y, down = -Z), aligned by one complex gain
    sh = shape[sens]
    idx = [int(np.argmin(np.abs(s_h - h))) for h in bl.heights_cm]
    pv = np.c_[-sh[idx, 1], -sh[idx, 2]] * px_mm
    vb = P[iblk]
    big = np.abs(pv[:, 0]) > 0.25 * np.abs(pv[:, 0]).max()
    g = np.vdot(pv[big, 0], vb[big, 0]) / np.vdot(pv[big, 0], pv[big, 0])
    pred = pv * g
    rows = []
    for k, i in enumerate(idx):
        rows.append(dict(sensor=model.labels[sens[i]], height_cm=round(float(bl.heights_cm[k]), 1),
                         sensor_um=round(float(abs(pv[k, 0]) / px_mm * 1e3), 1),
                         video_um=round(float(abs(vb[k, 0]) / px_mm * 1e3), 1),
                         ratio=round(float(abs(vb[k, 0]) / max(abs(pred[k, 0]), 1e-12)), 3),
                         phase_deg=round(float(np.degrees(np.angle(vb[k, 0] / pred[k, 0]))), 1),
                         small=bool(not big[k])))
    log(f"video/sensor amplitude {abs(g):.3f}, time offset phase {np.degrees(np.angle(g)):+.1f} deg")

    # alpha sweep
    a_auto = target_px / max(float(np.abs(vb[:, 0]).max()), 1e-6)
    if alphas is None:
        alphas = sorted({float(max(1, round(a_auto * m, -1) or round(a_auto * m))) for m in
                         (0.125, 0.25, 0.5, 1, 2, 4)})
    thetas = 2 * np.pi * np.arange(cycle_frames) / cycle_frames
    base = mag.frame(0.0, 0.0)
    centers = np.c_[bl.cols, bl.rows].astype(float)
    win = max(15, 2 * bl.half + 1)
    raw, frames = {}, {}
    for n_a, al in enumerate(alphas):
        fr = [mag.frame(al, th) for th in thetas]
        d = np.array([flow_points(base, f, centers, win) for f in fr])
        dm = 2 * np.nanmean(d[..., 0] * np.exp(-1j * thetas)[:, None], axis=0)
        raw[al] = np.abs(dm) / (al * np.abs(vb[:, 0]) + 1e-12)
        frames[al] = fr
        if progress and not progress("alpha", n_a + 1, len(alphas)):
            raise InterruptedError("cancelled")
    a0 = min(alphas)
    gain0 = float(np.nanmedian(raw[a0][big]))
    lin = {al: raw[al] / max(gain0, 1e-9) for al in alphas}
    for al in alphas:
        log(f"  alpha {al:7g}: top block {raw[al][-1] * al * abs(vb[-1, 0]):6.2f} px "
            f"(alpha x true {al * abs(vb[-1, 0]):6.2f}), linearity {np.nanmedian(lin[al][big]):.2f}")
    ok = [al for al in alphas if np.all(np.abs(lin[al][big] - 1) < 0.2)]
    a_best = max(ok) if ok else a0
    log(f"recommended alpha {a_best:g}; magnifier gain {gain0:.2f}")
    return MagnifyResult(fps, fsnr, f_vid, f_drive, (x0, x1), beam_x, bl, px_mm, ref, mag.motion_map(), roi, P,
                         [r["sensor"] for r in rows], pred, complex(g), rows, list(alphas), lin, gain0, a_best,
                         frames, base, thetas)
