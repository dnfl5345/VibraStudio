"""3D ODS restoration from motion-amplification motion vectors.

Rebuilds an Operating Deflection Shape on the cylinder node-link geometry
(vibrastudio.geometry.make_cylinder) from 2D motion-vector time series of
the kind produced by motion-amplification systems (RDI Iris M, gDAC, ...)
on rotating machinery, and renders the restored shape as an animation.

Pipeline
--------
1. Motion vectors  — one of:
     --video  path   dense optical flow (Farneback) sampled on a grid
     --npz    path   pre-exported vectors: points (P,2), u (T,P), v (T,P), fps
     --demo          synthetic vectors representative of a rotating machine
                     (1x running-speed rocking/bending + ring ovalization)
2. Spectral ODS    — Hann-windowed FFT per point, dominant-peak pick on the
   averaged spectrum, complex amplitude/phase of u and v at that frequency.
3. 3D restoration  — measurement points are binned into axial stations
   (rings). Per ring, a linear least-squares fit separates what a single
   side-view camera can observe:
       u(theta) = dX                          (axial translation)
       v(theta) = dZ + [a2 cos2t + b2 sin2t] * cos(theta)
                  (vertical translation + n=2 ovalization, projected)
   The out-of-plane translation dY is unobservable from one view and is
   set to zero. Ring motion is expanded onto the cylinder nodes as
   dX*x^ + dZ*z^ + d_r(theta)*r^, giving a full 3D deflection shape.
4. Animation       — node-link wireframe (undeformed ghost + deformed,
   colored by amplitude) exported as GIF (and MP4 when ffmpeg is present).

The cylinder is the same parametric template used by the ODS view
(n nodes/ring x n layers), rotated so its axis lies along X to match a
horizontal machine train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibrastudio.geometry import Geometry, make_cylinder, make_grid  # noqa: E402

# Repo dark-theme palette (vibrastudio.ui.theme)
BG = "#1f1e1c"
ACCENT = "#d97757"
TEXT = "#a8a49e"
FAINT = "#7d7a74"


# ---------------------------------------------------------------------------
# 1. Motion-vector acquisition
# ---------------------------------------------------------------------------
class MotionVectors:
    """Grid of tracked points with per-frame displacement components.

    points : (P, 2) image coordinates (x right, y up) in arbitrary units
    u, v   : (T, P) horizontal / vertical displacement time series
    fps    : frame rate of the series
    """

    def __init__(self, points: np.ndarray, u: np.ndarray, v: np.ndarray, fps: float):
        self.points = np.asarray(points, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.fps = float(fps)
        if self.u.shape != self.v.shape or self.u.shape[1] != self.points.shape[0]:
            raise ValueError("u/v must be (T, P) matching points (P, 2)")

    @classmethod
    def from_npz(cls, path: str) -> "MotionVectors":
        d = np.load(path)
        return cls(d["points"], d["u"], d["v"], float(d["fps"]))

    @classmethod
    def from_video(cls, path: str, grid: tuple[int, int] = (24, 12),
                   max_frames: int = 1024,
                   roi: tuple[int, int, int, int] | None = None,
                   ref_roi: tuple[int, int, int, int] | None = None
                   ) -> "MotionVectors":
        """Dense Farneback optical flow sampled on a regular grid.

        Flow is computed frame-to-reference (first frame) so u/v are
        displacements, not velocities — the same convention vendors use
        when exporting motion vectors.

        roi     (x0, y0, x1, y1) image region to place the grid on
                (default: central 80% of the frame).
        ref_roi optional stationary region (e.g. the mounting plate or
                a rigid background). Its mean per-frame motion is
                subtracted from every tracked point — common-mode
                rejection of handheld camera shake.
        """
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ok, ref = cap.read()
        if not ok:
            raise IOError("empty video")
        ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        h, w = ref_g.shape

        x0, y0, x1, y1 = roi if roi else (int(w * .1), int(h * .1),
                                          int(w * .9), int(h * .9))
        gx = np.linspace(x0, x1 - 1, grid[0]).astype(int)
        gy = np.linspace(y0, y1 - 1, grid[1]).astype(int)
        pts = np.array([(x, y) for y in gy for x in gx])
        if ref_roi:
            rx0, ry0, rx1, ry1 = ref_roi
            rmask = np.zeros((h, w), bool)
            rmask[ry0:ry1, rx0:rx1] = True

        us, vs, ru, rv = [], [], [], []
        prev = ref_g
        acc = np.zeros((h, w, 2), np.float32)
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, g, None, pyr_scale=0.5, levels=3, winsize=21,
                iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
            acc += flow          # integrate frame-to-frame flow -> displacement
            us.append(acc[pts[:, 1], pts[:, 0], 0])
            vs.append(-acc[pts[:, 1], pts[:, 0], 1])   # image y is down
            if ref_roi:
                ru.append(acc[..., 0][rmask].mean())
                rv.append(-acc[..., 1][rmask].mean())
            prev = g
        cap.release()
        u, v = np.array(us), np.array(vs)
        if ref_roi:
            u -= np.array(ru)[:, None]
            v -= np.array(rv)[:, None]
        pts_up = np.column_stack([pts[:, 0], h - pts[:, 1]])   # y up
        return cls(pts_up, u, v, fps)

    def rotated(self, axis_angle_deg: float) -> "MotionVectors":
        """Rotate points and vector components so a machine axis that sits
        at `axis_angle_deg` (CCW from image x, y-up frame) maps to +x."""
        a = np.radians(-axis_angle_deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        pts = self.points @ R.T
        u = c * self.u - s * self.v
        v = s * self.u + c * self.v
        return MotionVectors(pts, u, v, self.fps)

    @classmethod
    def demo(cls, fps: float = 240.0, n_frames: int = 1024,
             f1: float = 29.7, seed: int = 7) -> "MotionVectors":
        """Synthetic vectors representative of the target video's machine.

        A horizontal rotating machine (motor-driven train, ~1780 RPM) whose
        casing rocks/bends at 1x running speed: vertical deflection grows
        toward the free end with a phase lag along the axis, a smaller
        in-phase axial component, 2x harmonic, and measurement noise. A
        mild n=2 ring ovalization is added near the drive end so the 3D
        restoration has genuine cross-section deformation to recover.
        """
        rng = np.random.default_rng(seed)
        nx_, ny_ = 24, 12
        # casing occupies x in [0,1], the visible face spans y in [-1,1]*R
        xs = np.linspace(0.0, 1.0, nx_)
        ys = np.linspace(-0.92, 0.92, ny_)          # normalized z/R on the face
        pts = np.array([(x, y) for y in ys for x in xs])
        t = np.arange(n_frames) / fps
        w1 = 2 * np.pi * f1

        x, y = pts[:, 0], pts[:, 1]
        theta = np.arcsin(np.clip(y, -1, 1))         # surface angle from view
        # 1x rocking/bending: amplitude ramps along the axis, 60 deg phase lag
        amp_z = 0.25 + 1.0 * x ** 1.5
        ph_z = np.radians(60) * x
        # axial (u) component, weaker, uniform phase
        amp_x = 0.18 * (1 - 0.4 * x)
        # n=2 ovalization near drive end, projected on the camera axis
        oval = 0.35 * np.exp(-((x - 0.15) / 0.25) ** 2) * np.cos(2 * theta)
        u = (amp_x[None, :] * np.sin(w1 * t[:, None])
             + 0.05 * np.sin(2 * w1 * t[:, None] + 0.8))
        v = (amp_z[None, :] * np.sin(w1 * t[:, None] - ph_z[None, :])
             + oval[None, :] * np.cos(theta)[None, :] * np.sin(w1 * t[:, None] - 1.1)
             + 0.08 * np.sin(2 * w1 * t[:, None]))
        u += 0.04 * rng.standard_normal(u.shape)
        v += 0.04 * rng.standard_normal(v.shape)
        return cls(pts, u, v, fps)

    @classmethod
    def demo_bridge(cls, fps: float = 60.0, n_frames: int = 2048,
                    f1: float = 2.2, seed: int = 11) -> "MotionVectors":
        """Synthetic vectors representative of a bridge span, side view.

        Traffic/wind-driven first vertical bending of the deck: half-sine
        shape between the abutments at ~2 Hz, a weaker second bending
        harmonic, and the section rotation that Euler beam theory couples
        to bending (theta = dw/dx), visible as longitudinal motion that
        varies linearly over the girder depth. Ends are supported.
        """
        rng = np.random.default_rng(seed)
        nx_, ny_ = 30, 6
        xs = np.linspace(0.0, 1.0, nx_)              # along the span
        ys = np.linspace(-0.5, 0.5, ny_)             # over the girder depth
        pts = np.array([(x, y) for y in ys for x in xs])
        t = np.arange(n_frames) / fps
        w1 = 2 * np.pi * f1
        x, y = pts[:, 0], pts[:, 1]

        # mode 1 (half-sine) + mode 2, standing waves
        w_shape = np.sin(np.pi * x)
        w2_shape = 0.22 * np.sin(2 * np.pi * x)
        # section rotation theta = dw/dx (per unit span), scaled by depth
        th_shape = np.pi * np.cos(np.pi * x)
        depth = 0.12                                  # girder depth / span
        v = (w_shape[None, :] * np.sin(w1 * t[:, None])
             + w2_shape[None, :] * np.sin(1.9 * w1 * t[:, None] + 0.7))
        u = (depth * y[None, :] * th_shape[None, :] * np.sin(w1 * t[:, None])
             + 0.05 * np.sin(w1 * t[:, None] + 0.3))
        u += 0.03 * rng.standard_normal(u.shape)
        v += 0.03 * rng.standard_normal(v.shape)
        return cls(pts, u, v, fps)


# ---------------------------------------------------------------------------
# 2. Spectral ODS at the dominant frequency
# ---------------------------------------------------------------------------
def spectral_ods(mv: MotionVectors, freq: float | None = None,
                 min_freq: float = 2.0):
    """Complex u/v amplitudes at the dominant (or requested) frequency.

    Returns (f0, U, V): U/V are complex, phase-referenced so the strongest
    point has zero phase — exactly the cross-spectrum convention the live
    ODS view uses, collapsed to a single record.
    """
    T = mv.u.shape[0]
    win = np.hanning(T)
    scale = 2.0 / win.sum()
    U = np.fft.rfft((mv.u - mv.u.mean(0)) * win[:, None], axis=0) * scale
    V = np.fft.rfft((mv.v - mv.v.mean(0)) * win[:, None], axis=0) * scale
    f = np.fft.rfftfreq(T, 1.0 / mv.fps)

    if freq is None:
        power = (np.abs(U) ** 2 + np.abs(V) ** 2).mean(axis=1)
        power[f < min_freq] = 0.0
        k = int(np.argmax(power))
    else:
        k = int(np.argmin(np.abs(f - freq)))
    Uk, Vk = U[k], V[k]

    ref = np.argmax(np.abs(Uk) ** 2 + np.abs(Vk) ** 2)
    rot = np.exp(-1j * np.angle(Vk[ref] if abs(Vk[ref]) > abs(Uk[ref]) else Uk[ref]))
    spectrum = np.sqrt((np.abs(U) ** 2 + np.abs(V) ** 2).mean(axis=1))
    return f[k], Uk * rot, Vk * rot, (f, spectrum)


# ---------------------------------------------------------------------------
# 3. 3D restoration on the cylinder node-link structure
# ---------------------------------------------------------------------------
def restore_cylinder_ods(mv: MotionVectors, Uk: np.ndarray, Vk: np.ndarray,
                         n_per_ring: int = 12, n_layers: int = 6,
                         radius: float = 0.35, length: float = 1.6):
    """Fit per-ring motion from image-plane vectors, expand to 3D nodes.

    Returns (geometry, disp) with disp (n_nodes, 3) complex — the full 3D
    deflection shape: axial + vertical rigid translation per ring plus the
    n=2 radial ovalization the projection equations allow us to observe.
    """
    geom = make_cylinder(n_per_ring=n_per_ring, n_layers=n_layers,
                         radius=radius, height=length)
    # rotate the tested cylinder template so its axis lies along X
    geom = Geometry(geom.name, geom.nodes[:, [2, 1, 0]].copy(), geom.edges,
                    dict(geom.params, orientation="axis-X"), [])

    px = mv.points[:, 0]
    py = mv.points[:, 1]
    x01 = (px - px.min()) / max(np.ptp(px), 1e-12)
    yr = py - py.mean()
    yr = yr / max(np.abs(yr).max(), 1e-12) * 0.98      # normalized z/R
    theta_p = np.arcsin(np.clip(yr, -1, 1))

    edges_x = np.linspace(0, 1, n_layers + 1)
    station = np.clip(np.digitize(x01, edges_x[1:-1]), 0, n_layers - 1)

    dX = np.zeros(n_layers, complex)
    dZ = np.zeros(n_layers, complex)
    a2 = np.zeros(n_layers, complex)
    b2 = np.zeros(n_layers, complex)
    filled = np.zeros(n_layers, bool)
    for j in range(n_layers):
        m = station == j
        if m.sum() < 4:
            continue
        th = theta_p[m]
        dX[j] = Uk[m].mean()
        # v = dZ + (a2 cos2t + b2 sin2t) cos(t)  — complex linear lstsq
        A = np.column_stack([np.ones(m.sum()),
                             np.cos(2 * th) * np.cos(th),
                             np.sin(2 * th) * np.cos(th)])
        coef, *_ = np.linalg.lstsq(A, Vk[m], rcond=None)
        dZ[j], a2[j], b2[j] = coef
        filled[j] = True
    if not filled.all():                       # interpolate empty stations
        idx = np.arange(n_layers)
        for arr in (dX, dZ, a2, b2):
            arr[~filled] = np.interp(idx[~filled], idx[filled],
                                     arr[filled].real) + 1j * np.interp(
                idx[~filled], idx[filled], arr[filled].imag)

    disp = np.zeros((geom.n_nodes, 3), complex)
    for j in range(n_layers):
        for i in range(n_per_ring):
            k = j * n_per_ring + i
            _, ny_, nz_ = geom.nodes[k]
            th = np.arctan2(ny_, nz_)          # angle from +Z toward +Y
            d_r = a2[j] * np.cos(2 * th) + b2[j] * np.sin(2 * th)
            disp[k, 0] = dX[j]
            disp[k, 1] = d_r * np.sin(th)      # dY: ovalization only
            disp[k, 2] = dZ[j] + d_r * np.cos(th)
    rings = {"dX": dX, "dZ": dZ, "a2": a2, "b2": b2}
    return geom, disp, rings


def restore_deck_ods(mv: MotionVectors, Uk: np.ndarray, Vk: np.ndarray,
                     n_stations: int = 16, n_width: int = 5,
                     span: float = 3.4, width: float = 0.7):
    """Restore a bridge-deck ODS on the grid (plate) node-link template.

    Side-view projection equations per axial station:
        v(y) = dZ                    (vertical bending translation)
        u(y) = dX + theta * y        (longitudinal + section rotation)
    theta is fit independently from the depth-wise gradient of u; Euler
    beam theory predicts theta ~ d(dZ)/dx, so agreement between the two
    is an internal consistency check of the restoration idea. The deck
    roll (torsion across the width) is unobservable from a side view and
    left at zero. End stations are marked as supports and clamped.

    Returns (geometry, disp, stations) with stations holding dX/dZ/theta
    plus the beam-theory cross-check series.
    """
    geom = make_grid(nx=n_stations, ny=n_width, width=span, depth=width)
    # deck spans X: supports at both abutments (first/last column of nodes)
    geom.fixed = [j * n_stations + i for j in range(n_width)
                  for i in (0, n_stations - 1)]

    px, py = mv.points[:, 0], mv.points[:, 1]
    x01 = (px - px.min()) / max(np.ptp(px), 1e-12)
    yc = py - py.mean()
    yc = yc / max(np.abs(yc).max(), 1e-12)           # depth coordinate [-1,1]

    edges_x = np.linspace(0, 1, n_stations + 1)
    station = np.clip(np.digitize(x01, edges_x[1:-1]), 0, n_stations - 1)

    dX = np.zeros(n_stations, complex)
    dZ = np.zeros(n_stations, complex)
    th = np.zeros(n_stations, complex)
    filled = np.zeros(n_stations, bool)
    for j in range(n_stations):
        m = station == j
        if m.sum() < 3:
            continue
        dZ[j] = Vk[m].mean()
        A = np.column_stack([np.ones(m.sum()), yc[m]])
        coef, *_ = np.linalg.lstsq(A, Uk[m], rcond=None)
        dX[j], th[j] = coef
        filled[j] = True
    if not filled.all():
        idx = np.arange(n_stations)
        for arr in (dX, dZ, th):
            arr[~filled] = np.interp(idx[~filled], idx[filled],
                                     arr[filled].real) + 1j * np.interp(
                idx[~filled], idx[filled], arr[filled].imag)

    disp = np.zeros((geom.n_nodes, 3), complex)
    for j in range(n_width):
        for i in range(n_stations):
            k = j * n_stations + i
            disp[k, 0] = dX[i]
            disp[k, 2] = dZ[i]
    for k in geom.fixed:
        disp[k] = 0.0

    # beam-theory cross-check: theta should track d(dZ)/dx along the span
    dz_dx = np.gradient(dZ, 1.0 / max(n_stations - 1, 1))
    stations = {"dX": dX, "dZ": dZ, "theta": th, "dz_dx": dz_dx}
    return geom, disp, stations


# ---------------------------------------------------------------------------
# 4. Rendering
# ---------------------------------------------------------------------------
def render_animation(geom: Geometry, disp: np.ndarray, f0: float,
                     out_gif: Path, out_mp4: Path | None = None,
                     n_frames: int = 48, cycles: int = 2, fps_out: int = 24,
                     gain_frac: float = 0.16, title: str | None = None,
                     view: tuple[float, float] = (18, -58),
                     aspect: tuple[float, float, float] = (1.35, 1, 1)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import imageio.v2 as imageio

    nodes, edges = geom.nodes, np.asarray(geom.edges)
    amp = np.abs(disp).sum(axis=1)
    gain = gain_frac * geom.bbox_size() / max(np.abs(disp).max(), 1e-12)
    edge_amp = amp[edges].mean(axis=1) / max(amp.max(), 1e-12)
    cmap = colormaps["plasma"]
    edge_colors = cmap(0.15 + 0.8 * edge_amp)

    fig = plt.figure(figsize=(7.2, 5.4), facecolor=BG)
    ax = fig.add_subplot(projection="3d", facecolor=BG)
    frames = []
    for n in range(n_frames):
        ph = 2 * np.pi * cycles * n / n_frames
        d = (disp * np.exp(1j * ph)).real
        p = nodes + gain * d

        ax.cla()
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.add_collection3d(Line3DCollection(
            nodes[edges], colors=FAINT, linewidths=0.7, alpha=0.35))
        ax.add_collection3d(Line3DCollection(
            p[edges], colors=edge_colors, linewidths=2.2))
        node_c = cmap(0.15 + 0.8 * amp / max(amp.max(), 1e-12))
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=node_c, s=18, depthshade=False)
        s = geom.bbox_size()
        c = nodes.mean(axis=0)
        ax.set_xlim(c[0] - 0.62 * s, c[0] + 0.62 * s)
        ax.set_ylim(c[1] - 0.45 * s, c[1] + 0.45 * s)
        ax.set_zlim(c[2] - 0.45 * s, c[2] + 0.45 * s)
        ax.set_box_aspect(aspect)
        ax.view_init(elev=view[0], azim=view[1])
        if geom.fixed:
            fx = nodes[geom.fixed]
            ax.scatter(fx[:, 0], fx[:, 1], fx[:, 2], c="#e05a5a", s=26,
                       depthshade=False, marker="s")
        ax.set_title(title or f"3D ODS restoration — {geom.name} @ {f0:.2f} Hz",
                     color=TEXT, fontsize=10)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(img.copy())
    plt.close(fig)

    imageio.mimsave(out_gif, frames, fps=fps_out, loop=0)
    if out_mp4 is not None:
        try:
            imageio.mimsave(out_mp4, frames, fps=fps_out, quality=8)
        except Exception as e:                       # ffmpeg missing is fine
            print(f"[warn] mp4 skipped: {e}")
            out_mp4 = None
    return out_gif, out_mp4


def render_summary(mv: MotionVectors, spec, f0, Uk, Vk, rings, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, s = spec
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=BG)
    for a in axes:
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color(FAINT)
        a.tick_params(colors=TEXT, labelsize=8)

    ax = axes[0]
    amp = np.hypot(np.abs(Uk), np.abs(Vk))
    q = ax.quiver(mv.points[:, 0], mv.points[:, 1], Uk.real, Vk.real,
                  amp, cmap="plasma", scale=25, width=3e-3)
    ax.set_title(f"motion vectors @ {f0:.2f} Hz (Re)", color=TEXT, fontsize=9)
    ax.set_aspect("auto")
    fig.colorbar(q, ax=ax).ax.tick_params(colors=TEXT, labelsize=7)

    ax = axes[1]
    ax.semilogy(f, s + 1e-12, color=ACCENT, lw=1.2)
    ax.axvline(f0, color="#5aa9d6", ls="--", lw=1)
    ax.set_xlim(0, min(f.max(), 5 * f0))
    ax.set_title(f"avg motion spectrum — peak {f0:.2f} Hz", color=TEXT, fontsize=9)
    ax.set_xlabel("Hz", color=TEXT, fontsize=8)

    ax = axes[2]
    st = np.arange(len(rings["dZ"])) + 1
    ax.plot(st, np.abs(rings["dZ"]), "o-", color=ACCENT, label="|dZ| vertical")
    ax.plot(st, np.abs(rings["dX"]), "s-", color="#5aa9d6", label="|dX| axial")
    ax.plot(st, np.abs(rings["a2"]) + np.abs(rings["b2"]), "^-",
            color="#a3c95a", label="|n=2| ovalization")
    ax2 = ax.twinx()
    ax2.plot(st, np.degrees(np.angle(rings["dZ"])), ":", color=TEXT, lw=1)
    ax2.set_ylabel("dZ phase (deg)", color=TEXT, fontsize=8)
    ax2.tick_params(colors=TEXT, labelsize=7)
    ax.set_title("per-ring restored motion", color=TEXT, fontsize=9)
    ax.set_xlabel("ring (drive end -> free end)", color=TEXT, fontsize=8)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, edgecolor=FAINT)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor=BG)
    plt.close(fig)


def render_summary_deck(mv: MotionVectors, spec, f0, Uk, Vk, st, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, s = spec
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=BG)
    for a in axes:
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color(FAINT)
        a.tick_params(colors=TEXT, labelsize=8)

    ax = axes[0]
    amp = np.hypot(np.abs(Uk), np.abs(Vk))
    q = ax.quiver(mv.points[:, 0], mv.points[:, 1], Uk.real, Vk.real,
                  amp, cmap="plasma", scale=25, width=3e-3)
    ax.set_title(f"motion vectors @ {f0:.2f} Hz (Re)", color=TEXT, fontsize=9)
    ax.set_aspect("auto")
    fig.colorbar(q, ax=ax).ax.tick_params(colors=TEXT, labelsize=7)

    ax = axes[1]
    ax.semilogy(f, s + 1e-12, color=ACCENT, lw=1.2)
    ax.axvline(f0, color="#5aa9d6", ls="--", lw=1)
    ax.set_xlim(0, min(f.max(), 5 * f0))
    ax.set_title(f"avg motion spectrum — peak {f0:.2f} Hz", color=TEXT, fontsize=9)
    ax.set_xlabel("Hz", color=TEXT, fontsize=8)

    ax = axes[2]
    n = len(st["dZ"])
    xs = np.arange(n) + 1
    ax.plot(xs, np.abs(st["dZ"]), "o-", color=ACCENT, label="|dZ| bending")
    ax.plot(xs, np.abs(st["dX"]), "s-", color="#5aa9d6", label="|dX| longitudinal")
    # beam-theory consistency: fitted rotation vs derivative of bending
    th, dz = st["theta"], st["dz_dx"]
    k = (np.vdot(th, dz) / max(np.vdot(th, th).real, 1e-12)).real
    r = np.corrcoef(np.abs(th), np.abs(dz))[0, 1]
    ax.plot(xs, np.abs(th) * abs(k), "^-", color="#a3c95a",
            label=f"|theta|·k (fit rotation)")
    ax.plot(xs, np.abs(dz), ":", color=TEXT, lw=1.4,
            label="|d(dZ)/dx| (beam theory)")
    ax.set_title(f"per-station restore — theta vs d(dZ)/dx: r={r:.3f}",
                 color=TEXT, fontsize=9)
    ax.set_xlabel("station (abutment -> abutment)", color=TEXT, fontsize=8)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, edgecolor=FAINT)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor=BG)
    plt.close(fig)
    return r


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file to extract motion vectors from")
    src.add_argument("--npz", help="exported vectors: points,u,v,fps")
    src.add_argument("--demo", action="store_true",
                     help="synthetic motion vectors for the chosen structure")
    ap.add_argument("--structure", choices=["cylinder", "deck"],
                    default="cylinder",
                    help="restoration target: rotating-machine cylinder or "
                         "bridge deck (grid template)")
    ap.add_argument("--freq", type=float, default=None,
                    help="analysis frequency in Hz (default: dominant peak)")
    ap.add_argument("--nodes-per-ring", type=int, default=12)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--stations", type=int, default=16,
                    help="deck: stations along the span")
    ap.add_argument("--width-nodes", type=int, default=5,
                    help="deck: nodes across the width")
    ap.add_argument("--roi", default=None,
                    help="video: x0,y0,x1,y1 region to track (image px)")
    ap.add_argument("--ref-roi", default=None,
                    help="video: x0,y0,x1,y1 stationary region for "
                         "camera-shake common-mode rejection")
    ap.add_argument("--angle", type=float, default=0.0,
                    help="machine-axis angle in deg (CCW from image x, "
                         "y-up); points/vectors are rotated to align it +x")
    ap.add_argument("--out", default="docs/ods_restoration",
                    help="output directory")
    args = ap.parse_args(argv)

    def _parse_roi(s):
        return tuple(int(x) for x in s.split(",")) if s else None

    if args.demo:
        mv = (MotionVectors.demo_bridge() if args.structure == "deck"
              else MotionVectors.demo())
    elif args.npz:
        mv = MotionVectors.from_npz(args.npz)
    else:
        mv = MotionVectors.from_video(args.video, roi=_parse_roi(args.roi),
                                      ref_roi=_parse_roi(args.ref_roi))
    if args.angle:
        mv = mv.rotated(args.angle)
    print(f"[1] motion vectors: {mv.points.shape[0]} points x "
          f"{mv.u.shape[0]} frames @ {mv.fps:g} fps")

    f0, Uk, Vk, spec = spectral_ods(mv, freq=args.freq)
    print(f"[2] spectral ODS: dominant frequency {f0:.3f} Hz, "
          f"max |v| {np.abs(Vk).max():.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.structure == "deck":
        geom, disp, st = restore_deck_ods(
            mv, Uk, Vk, n_stations=args.stations, n_width=args.width_nodes)
        print(f"[3] restored on deck grid: {geom.n_nodes} nodes, "
              f"{len(geom.edges)} edges ({args.stations} stations x "
              f"{args.width_nodes} width, {len(geom.fixed)} supported)")
        stem = "ods_deck_restoration"
        gif, mp4 = render_animation(
            geom, disp, f0, out / f"{stem}.gif", out / f"{stem}.mp4",
            title=f"3D ODS restoration — bridge deck "
                  f"{args.stations}x{args.width_nodes} @ {f0:.2f} Hz",
            view=(24, -66), aspect=(2.3, 1, 0.9), gain_frac=0.22)
        r = render_summary_deck(mv, spec, f0, Uk, Vk, st,
                                out / f"{stem}_summary.png")
        print(f"[3b] beam-theory consistency (theta vs d(dZ)/dx): r={r:.3f}")
    else:
        geom, disp, rings = restore_cylinder_ods(
            mv, Uk, Vk, n_per_ring=args.nodes_per_ring, n_layers=args.layers)
        print(f"[3] restored on cylinder: {geom.n_nodes} nodes, "
              f"{len(geom.edges)} edges "
              f"({args.nodes_per_ring}/ring x {args.layers} rings)")
        stem = "ods_cylinder_restoration"
        gif, mp4 = render_animation(
            geom, disp, f0, out / f"{stem}.gif", out / f"{stem}.mp4",
            title=f"3D ODS restoration — cylinder {args.nodes_per_ring}"
                  f"x{args.layers} @ {f0:.2f} Hz")
        render_summary(mv, spec, f0, Uk, Vk, rings,
                       out / f"{stem}_summary.png")
    print(f"[4] wrote {gif}" + (f", {mp4}" if mp4 else "")
          + f", {out / (stem + '_summary.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
