"""Figures and video for dsp.magnify results (matplotlib Agg, thread-safe)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Callable

import numpy as np

from ..dsp.magnify import MagnifyResult, flow_points

ORANGE, CYAN, RED = "#f28e2b", "#4fc3f7", "#d62728"


def save_results(r: MagnifyResult, out: Path, video: str, recording: str) -> Path:
    d = dict(video=video, recording=recording, fps=r.fps, f_video=r.f_video, f_sensor=r.f_sensor,
             px_per_mm=r.px_per_mm, crop=list(r.crop), beam_x=r.beam_x,
             blocks_side="right" if r.blocks.side > 0 else "left", blocks_fit_rms_px=r.blocks.fit_rms,
             blocks_missing_cm=r.blocks.missing,
             video_over_sensor=abs(r.gain), time_offset_phase_deg=float(np.degrees(np.angle(r.gain))),
             per_sensor=r.per_sensor, alphas=r.alphas, alpha_recommended=r.alpha_recommended,
             magnifier_gain=r.mag_gain,
             linearity={f"{a:g}": [round(float(x), 3) for x in v] for a, v in r.linearity.items()})
    if r.synth:
        d["synthesis"] = [synth_row(ln) for ln in r.synth]
    p = out / "magnify_results.json"
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def _num(v: float, nd: int = 3):
    return None if not np.isfinite(v) else round(float(v), nd)


def synth_row(ln) -> dict:
    return dict(line=ln.name, f_hz=_num(ln.f, 4), f_video_hz=_num(ln.f_video, 4), folded=ln.folded,
                harmonic=ln.harmonic, sensor_um=_num(ln.sensor_um), alpha=_num(ln.alpha, 1),
                video_snr=_num(ln.snr_video, 1), measured_over_target=_num(ln.measured_ratio),
                shape_match=_num(ln.mac), noise_only=_num(ln.noise_only),
                video_magnification_valid=ln.valid, synthesis_from=ln.source, bending_shapes=ln.n_modes,
                sensors_used=ln.sensors_used, leave_one_out_error=_num(ln.loo_error),
                held_out_error=_num(ln.held_out_error), synthesis_drawn_px=_num(ln.synth_px, 2), note=ln.note)


def _profile(ln, z: np.ndarray):
    """Fitted shape, sensors and video points turned so the largest sensor is real (um)."""
    rot = np.exp(-1j * np.angle(ln.y_s[np.argmax(np.abs(ln.y_s))]))
    return (ln.fit(z) * rot).real, (ln.y_s * rot).real, (ln.y_v * rot).real


def synthesis_figure(r: MagnifyResult, out: Path, title: str = "") -> Path:
    """Per line: the shape the synthesis draws, the sensors and the video points it was fitted to."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(r.synth)
    fig = Figure(figsize=(3.6 * n + 0.6, 6.4), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    fig.suptitle(title or "Sensor-guided synthesis: shape per frequency", fontsize=14, weight="bold")
    for i, ln in enumerate(r.synth):
        ax = fig.add_subplot(1, n, i + 1)
        z = np.linspace(0, ln.fit.length_cm, 120)
        curve, ys, yv = _profile(ln, z)
        used = np.array([lab in ln.sensors_used for lab in ln.labels])
        ax.plot(yv, ln.z_v, ".", color=CYAN, alpha=0.9 if ln.valid else 0.35,
                label="video" + ("" if ln.valid else " (failed the check)"))
        ax.plot(curve, z, "-", color="k", lw=1.6, label=f"model ({ln.source}, {ln.n_modes} bending)")
        ax.plot(ys[used], ln.z_s[used], "o", color=RED, ms=7, label="sensors used")
        if (~used).any():
            ax.plot(ys[~used], ln.z_s[~used], "o", mfc="none", mec=RED, ms=7, label="sensors left out")
        span = 1.5 * max(np.abs(ys).max(), np.abs(curve).max())
        ax.set_xlim(-span, span); ax.axvline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
        err = (f"left-out sensors {100 * ln.held_out_error:.0f} %" if np.isfinite(ln.held_out_error) else
               f"leave-one-out {100 * ln.loo_error:.0f} %" if np.isfinite(ln.loo_error) else "")
        ax.set_title(f"{ln.name}\nα = {ln.alpha:,.0f} · video {'valid' if ln.valid else 'not valid'}\n"
                     + ("straight line only: not reliable" if ln.note else err), fontsize=10,
                     color="#b71c1c" if ln.note else "k")
        ax.set_xlabel("horizontal motion (µm)")
        if i == 0:
            ax.set_ylabel("height (cm)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    p = out / "synthesis_shapes.png"
    fig.savefig(p)
    return p


def synthesis_video(r: MagnifyResult, out: Path, fps: float = 12.0, cycles: int = 3,
                    progress: Callable[[int, int], bool] | None = None) -> Path:
    """Per line: original | video magnification | sensor-guided synthesis, same alpha, one cycle looped."""
    import imageio_ffmpeg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    W, H = 1600, 896
    fig = Figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 5, width_ratios=[0.55, 0.55, 0.55, 0.75, 1.0], left=0.02, right=0.98, top=0.84,
                          bottom=0.08, wspace=0.12)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    a_tx = fig.add_subplot(gs[0, 3]); a_tx.axis("off")
    a_pl = fig.add_subplot(gs[0, 4])
    for a, t in zip(axs, ("original", "video magnification", "sensor-guided synthesis")):
        a.set_title(t, fontsize=13); a.axis("off")
    ref8 = np.clip(r.base, 0, 255)
    axs[0].imshow(ref8, cmap="gray", vmin=0, vmax=255)
    im1 = axs[1].imshow(ref8, cmap="gray", vmin=0, vmax=255)
    im2 = axs[2].imshow(ref8, cmap="gray", vmin=0, vmax=255)
    ttl = fig.text(0.02, 0.93, "", fontsize=19, weight="bold")
    txt = a_tx.text(0.0, 0.5, "", fontsize=12, va="center", transform=a_tx.transAxes)
    curve, = a_pl.plot([], [], "-", color="k", lw=1.8, label="synthesis (model × α)")
    pts, = a_pl.plot([], [], "o", color=RED, ms=8, label="sensors used × α")
    pts_out, = a_pl.plot([], [], "o", mfc="none", mec=RED, ms=8, label="sensors left out × α")
    a_pl.axvline(0, color="k", lw=0.6); a_pl.grid(alpha=0.3); a_pl.legend(loc="lower right", fontsize=9)
    a_pl.set_xlabel("drawn horizontal motion (px)"); a_pl.set_ylabel("height (cm)")
    p = out / "synthesis_vs_video.mp4"
    w = imageio_ffmpeg.write_frames(str(p), (W, H), fps=fps, codec="libx264", quality=None,
                                    output_params=["-crf", "20", "-movflags", "+faststart"],
                                    pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    total = sum(cycles * len(ln.frames_synth) for ln in r.synth)
    n = 0
    try:
        for ln in r.synth:
            K = len(ln.frames_synth)
            thetas = 2 * np.pi * np.arange(K) / K
            z = np.linspace(0, ln.fit.length_cm, 100)
            sc = ln.alpha * ln.gain * r.px_per_mm / 1e3                  # um -> drawn px, in step with the frames
            cz, cs = ln.fit(z) * sc, ln.y_s * sc
            if ln.folded:
                cz, cs = np.conj(cz), np.conj(cs)
            used = np.array([lab in ln.sensors_used for lab in ln.labels])
            span = 1.5 * max(np.abs(cs).max(), np.abs(cz).max(), 1.0)
            a_pl.set_xlim(-span, span); a_pl.set_ylim(-2, ln.fit.length_cm + 2)
            err = (f"left-out sensors {100 * ln.held_out_error:.0f} %" if np.isfinite(ln.held_out_error) else
                   f"leave-one-out {100 * ln.loo_error:.0f} %" if np.isfinite(ln.loo_error) else "not checked")
            ttl.set_text(f"{ln.name}    α = {ln.alpha:,.0f}")
            mac = f"{ln.mac:.2f}" if np.isfinite(ln.mac) else "—  (needs 3 sensors)"
            note = "\n\n" + "\n".join(textwrap.wrap("! " + ln.note, 34)) if ln.note else ""
            txt.set_text(f"sensor motion {ln.sensor_um:.2f} µm\nvideo signal/noise {ln.snr_video:.1f}\n\n"
                         f"video magnification check\n  measured / target  {ln.measured_ratio:.2f}\n"
                         f"  shape match  {mac}\n  noise only  {ln.noise_only:.2f}\n"
                         f"  → {'valid' if ln.valid else 'NOT valid' if ln.note else 'NOT valid: use the synthesis'}\n\n"
                         f"synthesis\n  from {ln.source}: {', '.join(ln.sensors_used)}\n"
                         f"  {ln.n_modes} bending shapes + rigid\n  model error: {err}"
                         + note)
            for _ in range(cycles):
                for k, th in enumerate(thetas):
                    im1.set_data(ln.frames_video[k]); im2.set_data(ln.frames_synth[k])
                    e = np.exp(1j * th)
                    curve.set_data((cz * e).real, z)
                    pts.set_data((cs[used] * e).real, ln.z_s[used])
                    pts_out.set_data((cs[~used] * e).real, ln.z_s[~used])
                    fig.canvas.draw()
                    w.send(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
                    n += 1
                    if progress and not progress(n, total):
                        raise InterruptedError("cancelled")
    finally:
        w.close()
    return p


def summary_figure(r: MagnifyResult, out: Path, title: str = "") -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    fig = Figure(figsize=(19, 7.5), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 0.8, 1.3, 1.3], left=0.03, right=0.99, top=0.86,
                          bottom=0.09, wspace=0.25)
    ax = [fig.add_subplot(gs[0, i]) for i in range(4)]
    fig.suptitle(title or "Motion magnification vs accelerometers", fontsize=15, weight="bold")
    ax[0].imshow(r.ref, cmap="gray")
    ax[0].set_title("Auto-selected regions\n(orange: sensor blocks, cyan: beam face)")
    for q in r.roi:
        y0, y1, x0, x1 = q["box"]
        ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                  ec=ORANGE if q["kind"] == "block" else CYAN, lw=1))
    for row, y, x in zip(r.per_sensor, r.blocks.rows, r.blocks.cols):
        ax[0].text(x + r.blocks.half + 4, y, row["sensor"], color=ORANGE, fontsize=8, va="center")
    ax[0].axis("off")
    im = ax[1].imshow(np.log10(np.clip(r.snr_map, 1, None)), cmap="inferno", vmin=0, vmax=2)
    ax[1].set_title(f"Motion map at {r.f_video:.2f} Hz\n(log10 signal / noise)"); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.05)

    blk = [i for i, q in enumerate(r.roi) if q["kind"] == "block"]
    bt = [i for i, q in enumerate(r.roi) if q["kind"] == "beam"]
    rot = np.exp(-1j * np.angle(r.P[blk[-1], 0]))
    um = 1e3 / r.px_per_mm
    hb = np.array([r.roi[i]["h"] for i in bt])
    ax[2].plot((r.P[bt, 0] * rot).real * um, hb, ".-", color=CYAN, label="video: beam face (every 30 px)")
    ax[2].plot((r.P[blk, 0] * rot).real * um, r.blocks.heights_cm, "s", ms=10, mfc="none", mec=ORANGE, mew=2,
               label="video: sensor blocks")
    ax[2].plot((r.pred[:, 0] * rot).real * um, r.blocks.heights_cm, "o", color=RED,
               label=f"accelerometers (x{abs(r.gain):.2f}, time-aligned)")
    ax[2].axvline(0, color="k", lw=0.8); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=9)
    ax[2].set_xlabel("horizontal motion in the image (µm; + = in phase with the top block)")
    ax[2].set_ylabel("height (cm)")
    ax[2].set_title(f"Shape at {r.f_video:.2f} Hz: video vs sensors")
    small = np.array([row["small"] for row in r.per_sensor])
    for a in r.alphas:
        y = np.where(small, np.nan, r.linearity[a])          # near-node blocks: ratio is noise
        ax[3].plot(r.blocks.heights_cm, y, "o-", label=f"α = {a:g}")
    ax[3].axhline(1, color="k", lw=0.8); ax[3].axhspan(0.8, 1.2, color="g", alpha=0.1)
    ax[3].set_ylim(0, 1.6); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8, ncol=2)
    ax[3].set_xlabel("sensor block height (cm)  (blocks near a node left out)")
    ax[3].set_ylabel("magnified motion / (α × true), relative to smallest α")
    ax[3].set_title(f"Linearity of the magnification (recommended α = {r.alpha_recommended:g})")
    p = out / "magnify_summary.png"
    fig.savefig(p)
    return p


def overlay_video(r: MagnifyResult, out: Path, fps: float = 24.0, cycles: int = 2,
                  progress: Callable[[int, int], bool] | None = None) -> Path:
    """Per alpha: original | magnified + sensor-predicted positions (x alpha) | difference | profile."""
    import imageio_ffmpeg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    W, H = 1920, 1080
    fig = Figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.55, 0.55, 0.55, 1.6], left=0.03, right=0.98, top=0.86,
                          bottom=0.07, wspace=0.12)
    a_ref, a_mag, a_dif, a_pl = (fig.add_subplot(gs[0, i]) for i in range(4))
    for a, t in ((a_ref, "original"), (a_mag, "magnified + sensor prediction"), (a_dif, "magnified − original")):
        a.set_title(t, fontsize=13); a.axis("off")
    a_ref.imshow(r.ref, cmap="gray", vmin=0, vmax=255)
    i_mag = a_mag.imshow(r.ref, cmap="gray", vmin=0, vmax=255)
    i_dif = a_dif.imshow(np.zeros_like(r.ref), cmap="coolwarm", vmin=-60, vmax=60)
    cross, = a_mag.plot([], [], "+", color="#ff1744", ms=16, mew=2.5)
    ttl = fig.text(0.03, 0.95, "", fontsize=20, weight="bold")
    sub = fig.text(0.03, 0.91, "", fontsize=12, color="#333")
    lv, = a_pl.plot([], [], "s", ms=10, mfc="none", mec=ORANGE, mew=2, label="measured in magnified video")
    ls, = a_pl.plot([], [], "o-", color=RED, label="accelerometers × α")
    a_pl.axvline(0, color="k", lw=0.8); a_pl.grid(alpha=0.3); a_pl.legend(loc="lower right")
    a_pl.set_ylabel("height (cm)"); a_pl.set_xlabel("horizontal motion after magnification (px)")
    hh = r.blocks.heights_cm
    a_pl.set_ylim(min(hh) - 3, max(hh) + 3)
    centers = np.c_[r.blocks.cols, r.blocks.rows].astype(float)
    win = max(15, 2 * r.blocks.half + 1)
    p = out / "magnify_vs_sensors.mp4"
    w = imageio_ffmpeg.write_frames(str(p), (W, H), fps=fps, codec="libx264", quality=None,
                                    output_params=["-crf", "20", "-movflags", "+faststart"],
                                    pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    total = len(r.alphas) * cycles * len(r.thetas)
    n = 0
    try:
        for al in r.alphas:
            span = max(3.0, 1.4 * al * r.mag_gain * float(np.abs(r.pred[:, 0]).max()))
            a_pl.set_xlim(-span, span)
            lin = float(np.nanmedian(r.linearity[al]))
            for _ in range(cycles):
                for k, th in enumerate(r.thetas):
                    f = r.frames[al][k]
                    d = flow_points(r.base, f, centers, win)
                    pr = np.real(r.pred * al * r.mag_gain * np.exp(1j * th))
                    i_mag.set_data(f); i_dif.set_data(f - r.base)
                    cross.set_data(centers[:, 0] + pr[:, 0], centers[:, 1] + pr[:, 1])
                    lv.set_data(d[:, 0], hh); ls.set_data(pr[:, 0], hh)
                    ttl.set_text(f"Motion magnification α = {al:g}   ({r.f_video:.2f} Hz, one cycle looped)"
                                 + ("   ← recommended" if al == r.alpha_recommended else ""))
                    sub.set_text(f"red + = accelerometer shape × α projected into the image · linearity {lin:.2f} · "
                                 f"{r.px_per_mm:.3f} px/mm · {r.fps:.2f} fps")
                    fig.canvas.draw()
                    w.send(np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
                    n += 1
                    if progress and not progress(n, total):
                        raise InterruptedError("cancelled")
    finally:
        w.close()
    return p
