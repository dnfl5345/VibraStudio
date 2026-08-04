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

from vibrastudio.geometry import Geometry, make_cylinder  # noqa: E402

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
                   max_frames: int = 1024) -> "MotionVectors":
        """Dense Farneback optical flow sampled on a regular grid.

        Flow is computed frame-to-reference (first frame) so u/v are
        displacements, not velocities — the same convention vendors use
        when exporting motion vectors.
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

        gx = np.linspace(w * 0.1, w * 0.9, grid[0]).astype(int)
        gy = np.linspace(h * 0.1, h * 0.9, grid[1]).astype(int)
        pts = np.array([(x, y) for y in gy for x in gx])

        us, vs = [], []
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
            prev = g
        cap.release()
        pts_up = np.column_stack([pts[:, 0], h - pts[:, 1]])   # y up
        return cls(pts_up, np.array(us), np.array(vs), fps)

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


# ---------------------------------------------------------------------------
# 4. Rendering
# ---------------------------------------------------------------------------
def render_animation(geom: Geometry, disp: np.ndarray, f0: float,
                     out_gif: Path, out_mp4: Path | None = None,
                     n_frames: int = 48, cycles: int = 2, fps_out: int = 24,
                     gain_frac: float = 0.16):
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
        ax.set_box_aspect((1.35, 1, 1))
        ax.view_init(elev=18, azim=-58)
        ax.set_title(f"3D ODS restoration — cylinder {geom.params.get('n_per_ring')}"
                     f"x{geom.params.get('n_layers')} @ {f0:.2f} Hz",
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


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file to extract motion vectors from")
    src.add_argument("--npz", help="exported vectors: points,u,v,fps")
    src.add_argument("--demo", action="store_true",
                     help="synthetic rotating-machine motion vectors")
    ap.add_argument("--freq", type=float, default=None,
                    help="analysis frequency in Hz (default: dominant peak)")
    ap.add_argument("--nodes-per-ring", type=int, default=12)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out", default="docs/ods_restoration",
                    help="output directory")
    args = ap.parse_args(argv)

    if args.demo:
        mv = MotionVectors.demo()
    elif args.npz:
        mv = MotionVectors.from_npz(args.npz)
    else:
        mv = MotionVectors.from_video(args.video)
    print(f"[1] motion vectors: {mv.points.shape[0]} points x "
          f"{mv.u.shape[0]} frames @ {mv.fps:g} fps")

    f0, Uk, Vk, spec = spectral_ods(mv, freq=args.freq)
    print(f"[2] spectral ODS: dominant frequency {f0:.3f} Hz, "
          f"max |v| {np.abs(Vk).max():.3f}")

    geom, disp, rings = restore_cylinder_ods(
        mv, Uk, Vk, n_per_ring=args.nodes_per_ring, n_layers=args.layers)
    print(f"[3] restored on cylinder: {geom.n_nodes} nodes, "
          f"{len(geom.edges)} edges "
          f"({args.nodes_per_ring}/ring x {args.layers} rings)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gif, mp4 = render_animation(geom, disp, f0,
                                out / "ods_cylinder_restoration.gif",
                                out / "ods_cylinder_restoration.mp4")
    render_summary(mv, spec, f0, Uk, Vk, rings, out / "ods_restoration_summary.png")
    print(f"[4] wrote {gif}" + (f", {mp4}" if mp4 else "")
          + f", {out / 'ods_restoration_summary.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
