"""3D ODS animation data from a recording: node motion, shapes, frames.

Pure numpy/scipy (no Qt, no plotting) so the dialog, the CLI and tests share it.

Pipeline
  1. ``NodeModel``: which recording row measures which node, along which
     global direction (``ChannelConfig.vector()``; a rig gives exact vectors).
  2. ``node_acceleration``: per node, the 3D acceleration in g from all of its
     channels (least squares, so 1, 2 or 3 channels per node all work).
  3. Shapes or time histories, in displacement (mm):
       * ``SpectralODS.shape(f)`` — operating shape at one frequency from
         averaged cross-spectra against the strongest input (the usual
         spectral ODS; for a sine drive it equals the drive phasor);
       * ``integrate_displacement`` — time history by FFT integration in a
         band, with an optional low cut per global axis.
  4. ``tone_frames`` / ``sweep_frames`` / ``time_frames``: one ``Frame`` per
     video frame, ready for the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

G = 9.80665


# --------------------------------------------------------------------- model
@dataclass
class NodeModel:
    nodes: np.ndarray                   # (n, 3) geometry units
    edges: list[tuple[int, int]]
    fixed: list[int]
    labels: list[str]
    rows: list[int]                     # recording rows used
    node_of: list[int]                  # node index for each used row
    vecs: np.ndarray                    # (k, 3) global unit vector of each used row
    units: str = ""                     # "m" for rig geometry

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def measured(self) -> np.ndarray:
        """(n, 3) bool: which node directions any channel sees."""
        m = np.zeros((self.n_nodes, 3), bool)
        for node, v in zip(self.node_of, self.vecs):
            m[node] |= np.abs(v) > 1e-6
        return m


def model_from(geometry, channels, n_rows: int) -> NodeModel:
    """Map channels onto geometry nodes.

    ``channels`` is aligned with the recording rows: a channel whose
    physical_channel is 'file/ch<i>' reads row i, otherwise channel k reads row k.
    Disabled channels and channels with no node (point 0, AUX) are skipped.
    """
    rows, node_of, vecs = [], [], []
    for k, c in enumerate(channels):
        row = k
        if c.physical_channel.startswith("file/ch"):
            try:
                row = int(c.physical_channel[7:])
            except ValueError:
                pass
        node = c.point - 1
        if not c.enabled or row >= n_rows or not (0 <= node < geometry.n_nodes):
            continue
        v = c.vector()
        if np.linalg.norm(v) < 1e-9:
            continue
        rows.append(row); node_of.append(node); vecs.append(v / np.linalg.norm(v))
    labels = list(geometry.node_labels) or [str(i + 1) for i in range(geometry.n_nodes)]
    units = "m" if geometry.params.get("shape") == "rig" else ""
    return NodeModel(np.asarray(geometry.nodes, float), list(geometry.edges), list(geometry.fixed),
                     labels, rows, node_of, np.array(vecs).reshape(-1, 3), units)


def node_acceleration(data: np.ndarray, model: NodeModel, base_follows: bool = True) -> np.ndarray:
    """(n_nodes, 3, N) global acceleration (g). Unmeasured directions are 0.

    Fixed nodes stay at rest, except with ``base_follows``: a fixed node with
    a sensor within 5 % of the structure span (e.g. 0.7 cm above a clamp on
    a shaker) is the moving base, so it takes that sensor's motion.
    """
    n = data.shape[1]
    out = np.zeros((model.n_nodes, 3, n))
    by_node: dict[int, list[int]] = {}
    for j, node in enumerate(model.node_of):
        by_node.setdefault(node, []).append(j)
    for node, js in by_node.items():
        V = model.vecs[js]                                   # (k, 3)
        r = data[[model.rows[j] for j in js]].astype(float)
        r = r - r.mean(axis=1, keepdims=True)
        out[node] = np.linalg.pinv(V) @ r                    # min-norm least squares
    span = float(np.max(np.ptp(model.nodes, axis=0))) if model.n_nodes > 1 else 1.0
    measured = sorted(by_node)
    for i in model.fixed:
        if not 0 <= i < model.n_nodes:
            continue
        out[i] = 0.0
        if base_follows and measured:
            dist = [np.linalg.norm(model.nodes[j] - model.nodes[i]) for j in measured]
            k = int(np.argmin(dist))
            if dist[k] <= 0.05 * (span or 1.0):
                out[i] = out[measured[k]]
    return out


def acc_to_disp_mm(a_g, f_hz):
    return np.asarray(a_g) * G / -(2 * np.pi * np.asarray(f_hz)) ** 2 * 1e3


# ------------------------------------------------------------------ spectral
class SpectralODS:
    """Segment FFTs of every node direction, kept for shapes at any frequency."""

    def __init__(self, acc: np.ndarray, fs: float, resolution_hz: float = 0.5,
                 t0: float = 0.0, t1: float | None = None) -> None:
        n_all = acc.shape[-1]
        i0 = int(max(0.0, t0) * fs)
        i1 = n_all if t1 is None else min(n_all, int(t1 * fs))
        x = acc[..., i0:i1].reshape(-1, i1 - i0)              # (dof, N)
        self.dof_ok = np.abs(x).max(axis=1) > 0
        n = x.shape[1]
        nper = int(2 ** np.ceil(np.log2(max(64.0, fs / max(resolution_hz, 1e-3)))))
        nper = min(nper, int(2 ** np.floor(np.log2(n))))
        hop = nper // 2
        w = np.hanning(nper)
        starts = range(0, n - nper + 1, hop)
        segs = []
        for s in starts:
            seg = x[:, s:s + nper]
            seg = seg - seg.mean(axis=1, keepdims=True)
            segs.append(np.fft.rfft(seg * w, axis=1))
        self.X = np.array(segs)                               # (nseg, dof, nf)
        self._x, self._starts, self._w = x, list(starts), w
        self.freq = np.fft.rfftfreq(nper, 1.0 / fs)
        self.fs, self.nper, self.wsum = fs, nper, w.sum()
        self.n_nodes = acc.shape[0]
        self.auto = (np.abs(self.X) ** 2).mean(axis=0)        # (dof, nf), unscaled

    @property
    def df(self) -> float:
        return float(self.freq[1])

    def bin(self, f: float) -> int:
        return int(np.clip(round(f / self.df), 1, len(self.freq) - 1))

    def amplitude_g(self) -> np.ndarray:
        """(dof, nf) peak amplitude spectrum in g (tone-calibrated)."""
        return 2.0 * np.sqrt(self.auto) / self.wsum

    def velocity_spectrum(self) -> np.ndarray:
        """(3, nf) per global axis, summed over nodes: acceleration / f (mm/s)."""
        a = self.amplitude_g().reshape(self.n_nodes, 3, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = a * G * 1e3 / (2 * np.pi * np.maximum(self.freq, 1e-9))
        v[..., 0] = 0
        return np.sqrt((v ** 2).sum(axis=0))

    def peaks(self, fmin: float, fmax: float, n: int = 8) -> list[float]:
        """Strongest resonance-like peaks (velocity, all directions)."""
        v = self.velocity_spectrum().sum(axis=0)
        band = np.nonzero((self.freq >= fmin) & (self.freq <= fmax))[0]
        if band.size < 3:
            return []
        lv = np.log(v[band] + 1e-30)
        pk, prop = find_peaks(lv, prominence=0.5, distance=max(1, int(1.0 / self.df)))
        if pk.size == 0:
            return []
        order = np.argsort(prop["prominences"] + 0.3 * (lv[pk] - lv.max()))[::-1][:n]
        return sorted(float(self.freq[band[k]]) for k in pk[order])

    def refine(self, b: int) -> float:
        """Peak frequency between bins (Hann: parabola through log power)."""
        if not 1 <= b < len(self.freq) - 1:
            return float(self.freq[b])
        p = self.auto[:, b - 1:b + 2].sum(axis=0)
        if not (p[1] >= p[0] and p[1] >= p[2]) or np.any(p <= 0):
            return float(self.freq[b])
        a, c, e = np.log(p)
        den = a - 2 * c + e
        delta = 0.5 * (a - e) / den if den < 0 else 0.0
        return float(self.freq[b] + np.clip(delta, -0.5, 0.5) * self.df)

    def _dft(self, f: float) -> np.ndarray:
        """(nseg, dof) windowed DFT of every segment at exactly ``f``."""
        e = np.exp(-2j * np.pi * f * np.arange(self.nper) / self.fs) * self._w
        out = []
        for s in self._starts:
            seg = self._x[:, s:s + self.nper]
            out.append((seg - seg.mean(axis=1, keepdims=True)) @ e)
        return np.array(out)

    def shape(self, f: float, ref: int | None = None,
              exact: bool = True) -> tuple[np.ndarray, float, float, int]:
        """Operating shape near ``f``: (n_nodes, 3) complex displacement in mm.

        The strongest bin within +-1 bin is taken and, with ``exact``, its
        peak frequency is refined between bins and the shape is computed by a
        DFT at that frequency, so a sine drive between two bins keeps its
        true amplitude and frequency. Phase is relative to ``ref`` (a dof;
        default the strongest). Returns (shape, frequency, coherence, ref).
        """
        b = self.bin(f)
        lo, hi = max(1, b - 1), min(len(self.freq) - 1, b + 1)
        b = lo + int(np.argmax(self.auto[:, lo:hi + 1].sum(axis=0)))
        if exact:
            fx = self.refine(b)
            Xb = self._dft(fx)
        else:
            fx, Xb = float(self.freq[b]), self.X[:, :, b]      # (nseg, dof)
        auto = (np.abs(Xb) ** 2).mean(axis=0)
        if ref is None:
            ref = int(np.argmax(auto))
        pxy = (np.conj(Xb[:, ref])[:, None] * Xb).mean(axis=0)
        prr = auto[ref]
        amp = 2.0 / self.wsum * pxy / np.sqrt(max(prr, 1e-300))   # g, phase vs ref
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = np.abs(pxy) ** 2 / (prr * auto)
        # weighted by response power: directions with only noise barely count
        wgt = auto * self.dof_ok
        coh = float(np.nansum(coh * wgt) / wgt.sum()) if wgt.sum() > 0 else 0.0
        disp = acc_to_disp_mm(amp, fx).reshape(self.n_nodes, 3)
        return disp, fx, coh, ref


def integrate_displacement(acc: np.ndarray, fs: float, lo_hz=(3.0, 3.0, 3.0),
                           hi_hz: float = 400.0) -> np.ndarray:
    """(n_nodes, 3, N) displacement in mm by FFT double integration.

    A smooth high-pass at ``lo_hz`` (per global axis X, Y, Z) and low-pass at
    ``hi_hz``. Zero-phase, so a sensor that was hit directly (baseline jump)
    shows a symmetric low-frequency wobble around the hit: raise the low cut
    of that axis above it if the structure has no mode there.
    """
    n = acc.shape[-1]
    A = np.fft.rfft(acc, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    hi = np.clip((hi_hz * 1.15 - f) / (hi_hz * 0.15), 0, 1)
    taper = np.array([np.clip((f - lo) / max(0.5, 0.25 * lo), 0, 1) * hi for lo in lo_hz])
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(f > 0, G * 1e3 / -(2 * np.pi * f) ** 2, 0.0)
    return np.fft.irfft(A * (k * taper)[None], n=n, axis=-1)


# -------------------------------------------------------------------- frames
@dataclass
class Frame:
    disp: np.ndarray                    # (n, 3) mm, real, this instant
    gain: float                         # geometry units per mm (exaggeration)
    path: np.ndarray | None             # (n, 2, K) mm: orbit (tone) or trail (time), X-Y
    cursor: float                       # Hz or s, for the strip
    title: str
    subtitle: str
    extra: dict = field(default_factory=dict)


def _vis_gain(model: NodeModel, peak_mm: float, vis_fraction: float) -> float:
    span = float(np.max(np.ptp(model.nodes, axis=0))) if model.n_nodes > 1 else 1.0
    return vis_fraction * (span or 1.0) / max(peak_mm, 1e-12)


def _fmt_mm(v: float) -> str:
    return f"{v * 1e3:.1f} µm" if v < 1.0 else f"{v:.3f} mm"


def _drawn(model: NodeModel, gain: float) -> str:
    if model.units == "m":                 # geometry in metres: true exaggeration factor
        return f"motion drawn ×{gain * 1e3:,.0f}"
    return "motion scaled to fit"


def _peak_node(shape: np.ndarray) -> int:
    return int(np.argmax(np.linalg.norm(np.abs(shape), axis=1)))


def tone_frames(model: NodeModel, shapes: list[dict], fps: float = 30.0,
                seconds_each: float = 3.0, cycles_per_second: float = 1.0,
                vis_fraction: float = 0.12) -> list[Frame]:
    """Animate each shape {f, shape, label, note} for ``seconds_each``, own scale."""
    out: list[Frame] = []
    theta = 0.0
    dth = 2 * np.pi * cycles_per_second / fps
    ring = np.exp(1j * np.linspace(0, 2 * np.pi, 49))
    for s in shapes:
        phi = s["shape"]
        peak = float(np.abs(phi).max())
        gain = _vis_gain(model, peak, vis_fraction)
        orbit = np.real(phi[:, :2, None] * ring[None, None, :])
        k = _peak_node(phi)
        amp = np.abs(phi[k])
        sub = (f"{s.get('note', '')}  ·  largest motion at node {model.labels[k]}: "
               f"X {_fmt_mm(amp[0])}, Y {_fmt_mm(amp[1])}, Z {_fmt_mm(amp[2])}  ·  "
               + _drawn(model, gain))
        for _ in range(max(1, int(seconds_each * fps))):
            theta += dth
            out.append(Frame(np.real(phi * np.exp(1j * theta)), gain, orbit, s["f"],
                             f"{s.get('label', 'ODS')}   f = {s['f']:.2f} Hz", sub))
    return out


def sweep_frames(model: NodeModel, ods: SpectralODS, f1: float, f2: float,
                 seconds: float = 20.0, fps: float = 30.0, cycles_per_second: float = 1.0,
                 vis_fraction: float = 0.12, label: str = "ODS sweep") -> list[Frame]:
    """Frequency slides f1 -> f2; one scale for the whole sweep, so resonances grow."""
    b1, b2 = ods.bin(min(f1, f2)), ods.bin(max(f1, f2))
    bins = np.arange(b1, b2 + 1)
    ref = int(np.argmax(ods.auto[:, bins].sum(axis=1)))       # one reference for the sweep
    shapes = []
    for b in bins:
        s, _, _, _ = ods.shape(ods.freq[b], ref=ref, exact=False)
        shapes.append(s)
    shapes = np.array(shapes)                                  # (nb, n, 3)
    peak = float(np.abs(shapes).max())
    gain = _vis_gain(model, peak, vis_fraction)
    n_frames = max(2, int(seconds * fps))
    fq = np.linspace(ods.freq[b1], ods.freq[b2], n_frames)
    out: list[Frame] = []
    theta = 0.0
    dth = 2 * np.pi * cycles_per_second / fps
    ring = np.exp(1j * np.linspace(0, 2 * np.pi, 49))
    for f in fq:
        x = (f - ods.freq[b1]) / ods.df
        i = int(np.clip(np.floor(x), 0, len(bins) - 1)); j = min(i + 1, len(bins) - 1); w = x - i
        phi = (1 - w) * shapes[i] + w * shapes[j]
        theta += dth
        k = _peak_node(phi); amp = np.abs(phi[k])
        sub = (f"amplitude to scale across the sweep  ·  node {model.labels[k]}: "
               f"X {_fmt_mm(amp[0])}, Y {_fmt_mm(amp[1])}, Z {_fmt_mm(amp[2])}  ·  "
               + _drawn(model, gain))
        out.append(Frame(np.real(phi * np.exp(1j * theta)), gain,
                         np.real(phi[:, :2, None] * ring[None, None, :]), float(f),
                         f"{label}   f = {f:.2f} Hz", sub))
    return out


def time_frames(model: NodeModel, disp: np.ndarray, t: np.ndarray, slow: float = 20.0,
                fps: float = 30.0, trail: int = 24, vis_fraction: float = 0.12,
                label: str = "Time response", note: str = "") -> list[Frame]:
    """Displacement history ``disp`` (n,3,N) at times ``t``, played ``slow``x slower."""
    fs = 1.0 / (t[1] - t[0])
    n_out = max(2, int((t[-1] - t[0]) * slow * fps))
    idx = np.clip(np.round(np.arange(n_out) / (slow * fps) * fs).astype(int), 0, disp.shape[-1] - 1)
    gain = _vis_gain(model, float(np.abs(disp).max()), vis_fraction)
    k = _peak_node(np.abs(disp).max(axis=-1))
    out: list[Frame] = []
    for m, i in enumerate(idx):
        d = disp[:, :, i]
        tr = disp[:, :2, idx[max(0, m - trail):m + 1]]
        sub = (f"{note}  ·  node {model.labels[k]}: X {d[k, 0] * 1e3:+.1f}, Y {d[k, 1] * 1e3:+.1f}, "
               f"Z {d[k, 2] * 1e3:+.1f} µm  ·  " + _drawn(model, gain))
        out.append(Frame(d, gain, tr, float(t[i]), f"{label} ({slow:g}× slow)   t = {t[i]:.4f} s", sub))
    return out
