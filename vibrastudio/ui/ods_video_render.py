"""Render ODS frames to an MP4 (or GIF): 3D, front, side and top views.

Uses matplotlib's Agg canvas directly (no pyplot), so it is safe in a worker
thread. Video encoding goes through the ffmpeg binary that ships with
``imageio-ffmpeg`` (H.264, plays on phones and in PowerPoint); without it the
frames are written as an animated GIF via Pillow.

Layout (16:9): [3D] [front X-Z] [side Y-Z] [top X-Y] over a strip with the
spectrum (frequency videos) or the time trace (time videos) and a cursor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..dsp import ods_video as ov
from ..dsp.ods_video import Frame, NodeModel

AXIS_COLORS = {"X": "#d62728", "Y": "#2ca02c", "Z": "#1f77b4"}
BEAM = "#1d4e89"
SENSOR = "#f28e2b"


@dataclass
class Strip:
    kind: str                      # "spectrum" or "time"
    x: np.ndarray
    y: np.ndarray                  # (3, n): per global axis
    title: str = ""
    marks: tuple = ()
    log_x: bool = True


class Renderer:
    def __init__(self, model: NodeModel, strip: Strip, size: tuple[int, int] = (1920, 1080),
                 vis_fraction: float = 0.12) -> None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        self.model, self.strip = model, strip
        w, h = size
        scale = h / 1080.0
        self.fig = Figure(figsize=(w / 100, h / 100), dpi=100, facecolor="white")
        self.canvas = FigureCanvasAgg(self.fig)
        fs = lambda v: v * scale                                   # font scale
        gs = self.fig.add_gridspec(2, 4, height_ratios=[3.3, 1], width_ratios=[1.25, 1, 1, 1.1],
                                   left=0.045, right=0.985, top=0.86, bottom=0.07,
                                   wspace=0.3, hspace=0.32)
        self.a3 = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.af = self.fig.add_subplot(gs[0, 1])
        self.asd = self.fig.add_subplot(gs[0, 2])
        self.at = self.fig.add_subplot(gs[0, 3])
        self.ast = self.fig.add_subplot(gs[1, :])
        self.ttl = self.fig.text(0.045, 0.955, "", fontsize=fs(20), weight="bold", va="center")
        self.sub = self.fig.text(0.045, 0.905, "", fontsize=fs(12), va="center", color="#333")

        P = model.nodes
        span = float(np.max(np.ptp(P, axis=0))) if model.n_nodes > 1 else 1.0
        span = span or 1.0
        margin = (vis_fraction * 1.5 + 0.06) * span
        lo, hi = P.min(axis=0) - margin, P.max(axis=0) + margin
        u = f" ({model.units})" if model.units else ""
        E = np.asarray(model.edges, int).reshape(-1, 2)
        self.E = E
        meas = model.measured.any(axis=1)
        fixed = np.zeros(model.n_nodes, bool)
        fixed[[i for i in model.fixed if 0 <= i < model.n_nodes]] = True
        self.meas, self.fixed = meas, fixed

        # 3D
        a3 = self.a3
        a3.set_xlim(lo[0], hi[0]); a3.set_ylim(lo[1], hi[1]); a3.set_zlim(lo[2], hi[2])
        a3.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9)), zoom=1.1)
        a3.view_init(elev=18, azim=-58)
        a3.set_xlabel("X" + u, fontsize=fs(9)); a3.set_ylabel("Y" + u, fontsize=fs(9))
        a3.set_zlabel("Z" + u, fontsize=fs(9))
        a3.tick_params(labelsize=fs(8))
        from matplotlib.ticker import MaxNLocator
        for axis in (a3.xaxis, a3.yaxis, a3.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        a3.set_title("3D", fontsize=fs(14))
        a3.add_collection3d(Line3DCollection(P[E], colors="#aaaaaa", linestyles="--", linewidths=1))
        self.l3 = Line3DCollection(P[E], colors=BEAM, linewidths=3)
        a3.add_collection3d(self.l3)
        self.n3, = a3.plot(P[meas, 0], P[meas, 1], P[meas, 2], "o", color=SENSOR, mec="k", ms=fs(6))
        if fixed.any():
            a3.plot(P[fixed, 0], P[fixed, 1], P[fixed, 2], "s", color="#c0392b", ms=fs(8))
        L = 0.18 * span
        o = lo + 0.05 * (hi - lo)
        for k, (name, c) in enumerate(AXIS_COLORS.items()):
            v = np.zeros(3); v[k] = L
            a3.quiver(*o, *v, color=c, lw=2, arrow_length_ratio=0.25)
            a3.text(*(o + 1.3 * v), name, color=c, fontsize=fs(11), weight="bold")

        # 2D views: (axes, horizontal index, vertical index, title)
        self.views = []
        for ax, i, j, title in ((self.af, 0, 2, "Front (X-Z), seen from -Y"),
                                (self.asd, 1, 2, "Side (Y-Z), seen from +X"),
                                (self.at, 0, 1, "Top (X-Y), seen from +Z")):
            ax.set_title(title, fontsize=fs(14))
            ax.set_xlabel("XYZ"[i] + u, fontsize=fs(10)); ax.set_ylabel("XYZ"[j] + u, fontsize=fs(10))
            ax.tick_params(labelsize=fs(8)); ax.grid(alpha=0.3)
            self._equal_limits(ax, lo[i], hi[i], lo[j], hi[j], w, h)
            ax.add_collection(LineCollection(P[E][:, :, [i, j]], colors="#aaaaaa", linestyles="--",
                                             linewidths=1))
            lc = LineCollection(P[E][:, :, [i, j]], colors=BEAM, linewidths=3)
            ax.add_collection(lc)
            nodes, = ax.plot(P[meas, i], P[meas, j], "o", color=SENSOR, mec="k", ms=fs(7), zorder=5)
            if fixed.any():
                ax.plot(P[fixed, i], P[fixed, j], "s", color="#c0392b", ms=fs(8), zorder=4)
            self.views.append((ax, i, j, lc, nodes))
        from matplotlib import colormaps
        cmap = colormaps["viridis"]
        z = P[:, 2]; zn = (z - z.min()) / (np.ptp(z) or 1.0)
        self.paths = [self.at.plot([], [], color=cmap(zn[n]), lw=1, alpha=0.7)[0]
                      for n in range(model.n_nodes)]

        # strip
        ax = self.ast
        for k, name in enumerate("XYZ"):
            if strip.kind == "spectrum":
                ax.semilogy(strip.x, strip.y[k] + 1e-30, color=AXIS_COLORS[name], lw=1, label=f"{name}")
            else:
                ax.plot(strip.x, strip.y[k], color=AXIS_COLORS[name], lw=1, label=f"{name}")
        if strip.kind == "spectrum":
            top = float(np.max(strip.y)) or 1.0
            ax.set_ylim(top * 1e-4, top * 2)
            if strip.log_x and strip.x[0] > 0:
                ax.set_xscale("log")
            ax.set_xlim(strip.x[0], strip.x[-1]); ax.set_xlabel("Hz", fontsize=fs(10))
            from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
            if ax.get_xscale() == "log":
                ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            for f in strip.marks:
                ax.axvline(f, color="#bbbbbb", lw=0.8, ls=":")
        else:
            ax.set_xlim(strip.x[0], strip.x[-1]); ax.set_xlabel("time (s)", fontsize=fs(10))
        ax.set_title(strip.title, fontsize=fs(11)); ax.tick_params(labelsize=fs(8))
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=fs(9), ncol=3)
        self.cur = ax.axvline(strip.x[0], color="r", lw=2)

    def _equal_limits(self, ax, x0, x1, y0, y1, w_px, h_px) -> None:
        """Equal data scale on both axes, filling the panel (no aspect warnings)."""
        pos = ax.get_position()
        bw, bh = pos.width * w_px, pos.height * h_px
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        k = max((x1 - x0) / bw, (y1 - y0) / bh)
        ax.set_xlim(cx - k * bw / 2, cx + k * bw / 2)
        ax.set_ylim(cy - k * bh / 2, cy + k * bh / 2)

    def draw(self, fr: Frame) -> np.ndarray:
        P = self.model.nodes + fr.gain * fr.disp
        seg = P[self.E]
        self.l3.set_segments(seg)
        m = self.meas
        self.n3.set_data_3d(P[m, 0], P[m, 1], P[m, 2])
        for ax, i, j, lc, nodes in self.views:
            lc.set_segments(seg[:, :, [i, j]])
            nodes.set_data(P[m, i], P[m, j])
        if fr.path is not None:
            base = self.model.nodes[:, :2]
            for n, ln in enumerate(self.paths):
                if self.meas[n]:
                    ln.set_data(base[n, 0] + fr.gain * fr.path[n, 0], base[n, 1] + fr.gain * fr.path[n, 1])
        self.cur.set_xdata([fr.cursor, fr.cursor])
        self.ttl.set_text(fr.title)
        self.sub.set_text(fr.subtitle)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3]


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_video(frames: list[Frame], model: NodeModel, strip: Strip, out: str | Path,
                size: tuple[int, int] = (1920, 1080), fps: float = 30.0,
                progress: Callable[[int, int], bool] | None = None,
                stills: tuple[int, ...] = ()) -> Path:
    """Encode ``frames``. ``progress(i, n)`` returning False cancels.

    Writes MP4 (H.264) when ffmpeg is available, else GIF. ``stills`` are frame
    indices also saved as PNG next to the video. Returns the written path.
    """
    out = Path(out)
    r = Renderer(model, strip, size)
    exe = ffmpeg_exe()
    if out.suffix.lower() != ".gif" and exe is None:
        out = out.with_suffix(".gif")
    n = len(frames)
    if out.suffix.lower() == ".gif":
        from PIL import Image
        imgs = []
        for i, fr in enumerate(frames):
            if progress and not progress(i, n):
                raise InterruptedError("cancelled")
            im = Image.fromarray(r.draw(fr))
            imgs.append(im.resize((im.width // 2, im.height // 2)))
            if i in stills:
                im.save(out.with_name(f"{out.stem}_still{stills.index(i)}.png"))
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
        return out

    import os
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", exe)
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(str(out), size, fps=fps, codec="libx264", quality=None,
                                    output_params=["-crf", "20", "-preset", "medium",
                                                   "-movflags", "+faststart"],
                                    pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    try:
        for i, fr in enumerate(frames):
            if progress and not progress(i, n):
                raise InterruptedError("cancelled")
            img = np.ascontiguousarray(r.draw(fr))
            w.send(img)
            if i in stills:
                from PIL import Image
                Image.fromarray(img).save(out.with_name(f"{out.stem}_still{stills.index(i)}.png"))
    finally:
        w.close()
    return out


# ------------------------------------------------------------------- jobs
@dataclass
class VideoOptions:
    mode: str = "peaks"            # "peaks" | "list" | "sweep" | "time"
    freqs: tuple = ()              # for "list"
    n_peaks: int = 6
    fmin: float = 3.0
    fmax: float = 500.0
    seconds_each: float = 3.0      # per shape (peaks/list)
    sweep_seconds: float = 20.0
    t0: float = 0.0                # analysis window (s); time mode: shown span
    t1: float | None = None
    slow: float = 20.0             # time mode slow-motion factor
    lo_xyz: tuple = (3.0, 3.0, 3.0)
    hi_hz: float = 400.0
    resolution_hz: float = 0.5
    fps: float = 30.0
    size: tuple = (1920, 1080)


def build_frames(data: np.ndarray, fs: float, model: NodeModel, o: VideoOptions,
                 name: str = "") -> tuple[list[Frame], Strip]:
    """Frames and strip for one recording (``data``: rows x samples, g)."""
    acc = ov.node_acceleration(data, model)
    if not model.rows:
        raise ValueError("No channel is placed on the structure (all AUX or disabled).")
    if o.mode == "time":
        disp = ov.integrate_displacement(acc, fs, o.lo_xyz, o.hi_hz)
        i0 = int(max(0.0, o.t0) * fs)
        i1 = disp.shape[-1] if o.t1 is None else min(disp.shape[-1], int(o.t1 * fs))
        if i1 - i0 < 4:
            raise ValueError("Time window is empty.")
        d, t = disp[..., i0:i1], np.arange(i0, i1) / fs
        k = int(np.argmax(np.abs(d).max(axis=(1, 2))))
        lo = ",".join(f"{v:g}" for v in o.lo_xyz)
        strip = Strip("time", t, d[k], f"{name}  ·  displacement at node {model.labels[k]} "
                                       f"(mm; low cut X,Y,Z = {lo} Hz, high cut {o.hi_hz:g} Hz)")
        frames = ov.time_frames(model, d, t, o.slow, o.fps, label="Time response", note=name)
        return frames, strip

    ods = ov.SpectralODS(acc, fs, o.resolution_hz, o.t0, o.t1)
    band = (ods.freq >= max(o.fmin, ods.df)) & (ods.freq <= o.fmax)
    v = ods.velocity_spectrum()
    if o.mode == "sweep":
        frames = ov.sweep_frames(model, ods, o.fmin, o.fmax, o.sweep_seconds, o.fps, label="ODS sweep")
        marks = ()
    else:
        freqs = list(o.freqs) if o.mode == "list" else ods.peaks(o.fmin, o.fmax, o.n_peaks)
        if not freqs:
            raise ValueError("No peaks found in the band; enter frequencies instead.")
        shapes = []
        for f in freqs:
            sh, fb, coh, ref = ods.shape(f)
            share = (np.abs(sh) ** 2).sum(axis=0); share = share / (share.sum() or 1)
            shapes.append(dict(f=fb, shape=sh, label="ODS",
                               note=(f"{name}  ·  X:Y:Z energy {share[0]:.2f}:{share[1]:.2f}:{share[2]:.2f}"
                                     f"  ·  coherence {coh:.2f}")))
        frames = ov.tone_frames(model, shapes, o.fps, o.seconds_each)
        marks = tuple(s["f"] for s in shapes)
    strip = Strip("spectrum", ods.freq[band], v[:, band],
                  f"{name}  ·  velocity spectrum of all nodes per axis (mm/s); red = shown frequency",
                  marks)
    return frames, strip


def resolve_mapping(rec, rig=None, prefer: str = "auto"):
    """(channels, geometry, description) for a recording.

    ``prefer``: "rig" (match the rig by physical channel), "file" (the file's
    own stored map and structure) or "auto" (rig if it matches, else file).
    """
    from ..config import ChannelConfig
    from ..geometry import Geometry, make_line

    stored = rec.meta.get("channel_map") or []
    if prefer in ("auto", "rig") and rig is not None and len(stored) == len(rec.channels):
        mapped = rig.channels_for_recording(stored)
        if mapped is not None:
            n = sum(1 for c in mapped if c.point)
            return mapped, rig.to_geometry(), f"rig \"{rig.name}\" ({n} inputs matched by physical channel)"
    if len(stored) == len(rec.channels):
        chans = [ChannelConfig(**{**c, "physical_channel": f"file/ch{i}"}) for i, c in enumerate(stored)]
        g = rec.meta.get("geometry")
        geo = Geometry.from_dict(g) if g and g.get("nodes") else make_line(max(c.point for c in chans) or 1)
        return chans, geo, "the file's own channel map"
    import re
    chans = []
    for i, label in enumerate(rec.channels):
        m = re.match(r"^S(\d+)[_-]?([XYZ])$", label.strip(), re.IGNORECASE)
        chans.append(ChannelConfig(physical_channel=f"file/ch{i}", label=label,
                                   point=int(m.group(1)) if m else 0,
                                   direction=m.group(2).upper() if m else "Z"))
    n = max((c.point for c in chans), default=1) or 1
    return chans, make_line(n), "channel labels (S<node>_<axis>) on a straight line"
