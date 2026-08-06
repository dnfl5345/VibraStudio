"""Graded-resolution (LOD) phase magnification — an FEM-style mesh for CSP.

A finite element mesh is dense where the field varies fast and coarse in
the far field. The same idea applies to phase-based magnification, but the
direction that pays is the *coarse* one: upsampling a region does not add
information (phase accuracy is set by the pyramid band, not by pixel
count), while downsampling a region cuts cost as 1/s^2.

So this module grades the spatial resolution the pyramid sees:

  zone 1 (s=1)  strong-motion hotspot        -> full resolution
  zone 2 (s=2)  the rest of the moving object -> half resolution
  zone 3 (s=4)  everything else               -> quarter resolution

Every pixel still gets magnified — unlike a hard ROI, which drops whatever
falls outside its box — so captured motion stays at 100% and the trade
becomes spatial fidelity per zone instead of all-or-nothing coverage.

Zones are composited through the *residual* (magnified - original), not
the image: the residual is the band-limited motion signal, so upsampling
it from a coarse zone is far more benign than upsampling a blurred frame,
and a feathered weight across zone borders plays the role of a transition
element. cost = sum over zones of (zone box area) / s^2.

Usage:
  python tools/lod_magnify.py --video clip.mp4 --band 5 7 --alpha 15 \
      --frames 120 --label motor --out docs/roi_segment/motor_lod
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_motion_magnify import BG, ACCENT, FAINT, TEXT, feather_mask, magnify  # noqa: E402
from roi_segment_benchmark import bbox_roi, prescan_power, segment_roi  # noqa: E402


def magnify_scaled(Y: np.ndarray, fps: float, band, alpha: float, scale: int,
                   box=None, scales=3, orients=4):
    """Magnify a (optionally cropped) region at 1/scale resolution.

    Returns the residual at FULL resolution over the crop, plus the pixel
    count actually pushed through the pyramid.
    """
    T, H, W = Y.shape
    x0, y0, x1, y1 = box if box else (0, 0, W, H)
    crop = Y[:, y0:y1, x0:x1]
    if scale > 1:
        small = np.stack([cv2.resize(f, (crop.shape[2] // scale,
                                         crop.shape[1] // scale),
                                     interpolation=cv2.INTER_AREA)
                          for f in crop])
    else:
        small = crop
    mag = magnify(small, fps, band, alpha, scales, orients)
    res = mag - small
    if scale > 1:
        res = np.stack([cv2.resize(r, (crop.shape[2], crop.shape[1]),
                                   interpolation=cv2.INTER_CUBIC)
                        for r in res])
    return res.astype(np.float32), small.shape[1] * small.shape[2] * T


def build_zones(power, scale_pre, shape, mask_obj, hot_box, ring=24):
    """fine = motion hotspot, mid = rest of the object, coarse = frame."""
    H, W = shape
    fine = np.zeros((H, W), np.float32)
    fine[hot_box[1]:hot_box[3], hot_box[0]:hot_box[2]] = 1.0
    mid = np.clip(mask_obj, 0, 1).astype(np.float32)
    mid = cv2.dilate(mid, np.ones((ring, ring), np.uint8))
    mid = np.maximum(mid, fine)
    ys, xs = np.nonzero(mid > 0.5)
    mid_box = ((int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                int(ys.max()) + 1) if xs.size else (0, 0, W, H))
    return fine, mid, mid_box


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--band", nargs=2, type=float, required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--coarse", type=int, default=4, help="far-field 1/s")
    ap.add_argument("--mid", type=int, default=2, help="object 1/s")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated coarse factors, e.g. 2,3,4,6 — "
                         "runs the mesh-convergence study against one baseline")
    ap.add_argument("--label", default="clip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    band = tuple(args.band)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < args.frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    bgr = np.stack(frames)
    yuv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2YUV) for f in frames])
    Y = yuv[..., 0].astype(np.float32)
    T, H, W = Y.shape
    print(f"[{args.label}] {T} frames {W}x{H} @ {fps:g} fps, "
          f"band {band[0]}-{band[1]} Hz, alpha {args.alpha}")

    t0 = time.perf_counter()
    power = prescan_power(Y, fps, band, scale=4)
    hot_box, hot_mask = bbox_roi(power, 4, (H, W))
    seg_boxes, seg_mask, _ = segment_roi(bgr[0], power, 4)
    fine_w, mid_w, mid_box = build_zones(power, 4, (H, W), seg_mask, hot_box)
    t_sel = time.perf_counter() - t0
    a_fine = (hot_box[2] - hot_box[0]) * (hot_box[3] - hot_box[1]) / (H * W)
    a_mid = (mid_box[2] - mid_box[0]) * (mid_box[3] - mid_box[1]) / (H * W)
    print(f"  zones: fine {100*a_fine:.1f}% (s=1), mid {100*a_mid:.1f}% "
          f"(s={args.mid}), coarse 100% (s={args.coarse}); select {t_sel:.2f}s")

    # -- baseline -------------------------------------------------------
    t0 = time.perf_counter()
    Y_full = magnify(Y, fps, band, args.alpha)
    t_full = time.perf_counter() - t0
    print(f"  full : {t_full:7.2f} s  {T*H*W/1e6:6.1f} Mpx")

    # -- graded pass(es) -------------------------------------------------
    def graded(coarse: int):
        t0 = time.perf_counter()
        res = np.zeros_like(Y)
        px = 0
        r_c, p_c = magnify_scaled(Y, fps, band, args.alpha, coarse)
        res += r_c
        px += p_c
        r_m, p_m = magnify_scaled(Y, fps, band, args.alpha, args.mid, mid_box)
        px += p_m
        x0, y0, x1, y1 = mid_box
        wm = (mid_w[y0:y1, x0:x1] *
              feather_mask(y1 - y0, x1 - x0, border=16))[None]
        res[:, y0:y1, x0:x1] = res[:, y0:y1, x0:x1] * (1 - wm) + r_m * wm
        r_f, p_f = magnify_scaled(Y, fps, band, args.alpha, 1, hot_box)
        px += p_f
        x0, y0, x1, y1 = hot_box
        wf = (fine_w[y0:y1, x0:x1] *
              feather_mask(y1 - y0, x1 - x0, border=24))[None]
        res[:, y0:y1, x0:x1] = res[:, y0:y1, x0:x1] * (1 - wf) + r_f * wf
        return Y + res, time.perf_counter() - t0 + t_sel, px

    sweep = ([int(s) for s in args.sweep.split(",")] if args.sweep
             else [args.coarse])
    sweep_rows = []
    Y_lod = None
    for c in sweep:
        Yc, tc, pxc = graded(c)
        d_f, d_c = Y_full - Y, Yc - Y
        rc = float(np.corrcoef(d_f.ravel(), d_c.ravel())[0, 1])
        sweep_rows.append(dict(coarse=c, t=tc, mpx=pxc / 1e6,
                               speedup=t_full / tc, r_all=rc))
        print(f"  lod s_coarse={c}: {tc:7.2f} s  {pxc/1e6:6.1f} Mpx  "
              f"speedup x{t_full/tc:.2f}  motion r={rc:.4f}")
        if c == args.coarse or Y_lod is None:
            Y_lod, t_lod, px = Yc, tc, pxc

    # -- zone-wise fidelity against the full-frame result ----------------
    d_full, d_lod = Y_full - Y, Y_lod - Y
    zones = {
        "fine (s=1)": fine_w > 0.5,
        "mid (s=%d)" % args.mid: (mid_w > 0.5) & (fine_w <= 0.5),
        "coarse (s=%d)" % args.coarse: (mid_w <= 0.5),
    }
    zone_stats = {}
    for name, m in zones.items():
        if m.sum() < 50:
            continue
        a, b = d_full[:, m].ravel(), d_lod[:, m].ravel()
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        share = float((a ** 2).sum() / (d_full ** 2).sum())
        zone_stats[name] = dict(area=float(m.mean()), r=r,
                                energy_share=share,
                                nrmse=float(np.sqrt(np.mean((a - b) ** 2)) /
                                            (a.max() - a.min() + 1e-9)))
        print(f"    {name:14s} area {100*m.mean():5.1f}%  "
              f"motion share {100*share:5.1f}%  r={r:.4f}")
    r_all = float(np.corrcoef(d_full.ravel(), d_lod.ravel())[0, 1])
    print(f"  whole-frame motion correlation r={r_all:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lod_metrics.json").write_text(json.dumps(dict(
        label=args.label, size=[W, H], frames=T, fps=fps, band=band,
        alpha=args.alpha, t_full=t_full, t_lod=t_lod, mpx_full=T * H * W / 1e6,
        mpx_lod=px / 1e6, speedup=t_full / t_lod, r_all=r_all,
        zones=zone_stats, sweep=sweep_rows), indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    zone_img = np.zeros((H, W, 3), np.uint8)
    zone_img[:] = (60, 60, 80)
    zone_img[mid_w > 0.5] = (90, 169, 214)
    zone_img[fine_w > 0.5] = (217, 119, 87)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=BG)
    for a_ in ax:
        a_.set_facecolor(BG)
        a_.tick_params(colors=TEXT, labelsize=7)
        for sp in a_.spines.values():
            sp.set_color(FAINT)
    blend = (0.55 * cv2.cvtColor(bgr[0], cv2.COLOR_BGR2RGB) +
             0.45 * zone_img).astype(np.uint8)
    ax[0].imshow(blend)
    ax[0].set_title(f"LOD zones — fine {100*a_fine:.0f}% / mid {100*a_mid:.0f}%"
                    f" / coarse 100%", color=TEXT, fontsize=9)
    ax[0].axis("off")
    err = np.abs(d_full - d_lod).mean(axis=0)
    im = ax[1].imshow(err, cmap="plasma")
    fig.colorbar(im, ax=ax[1]).ax.tick_params(colors=TEXT, labelsize=7)
    ax[1].set_title(f"|full - lod| motion error (r={r_all:.3f})",
                    color=TEXT, fontsize=9)
    ax[1].axis("off")
    if len(sweep_rows) > 1:   # mesh-convergence curve (FEM h-refinement)
        sp = [w["speedup"] for w in sweep_rows]
        rr = [w["r_all"] for w in sweep_rows]
        ax[2].plot(sp, rr, "o-", color=ACCENT)
        for w in sweep_rows:
            ax[2].annotate(f"1/{w['coarse']}", (w["speedup"], w["r_all"]),
                           textcoords="offset points", xytext=(6, -10),
                           color=TEXT, fontsize=8)
        ax[2].set_xlabel("speedup vs full frame", color=TEXT, fontsize=8)
        ax[2].set_ylabel("whole-frame motion correlation", color=TEXT, fontsize=8)
        ax[2].set_title(f"{args.label}: cost/fidelity as the far-field mesh "
                        f"coarsens", color=TEXT, fontsize=9)
        ax[2].grid(alpha=0.15, color=FAINT)
    else:
        names = list(zone_stats)
        xs_ = np.arange(len(names))
        ax[2].bar(xs_, [zone_stats[n]["r"] for n in names], 0.5, color=ACCENT)
        ax[2].set_ylim(0, 1.05)
        ax[2].set_xticks(xs_)
        ax[2].set_xticklabels(names, color=TEXT, fontsize=8)
        ax[2].set_ylabel("motion correlation vs full", color=TEXT, fontsize=8)
        ax[2].set_title(f"{args.label}: {t_full:.0f}s -> {t_lod:.0f}s "
                        f"(x{t_full/t_lod:.2f})", color=TEXT, fontsize=9)
        for i, n in enumerate(names):
            ax[2].text(i, zone_stats[n]["r"] + 0.02,
                       f"{zone_stats[n]['r']:.3f}", ha="center", color=TEXT,
                       fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.label}_lod.png", dpi=110, facecolor=BG)
    plt.close(fig)

    def to_bgr(Ys):
        z = yuv.copy()
        z[..., 0] = np.clip(Ys, 0, 255).astype(np.uint8)
        return np.stack([cv2.cvtColor(f, cv2.COLOR_YUV2BGR) for f in z])

    tiles = []
    for name, stack in (("original", bgr), ("full CSP", to_bgr(Y_full)),
                        ("LOD graded", to_bgr(Y_lod))):
        s = stack.copy()
        for f in s:
            cv2.putText(f, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (87, 119, 217), 2, cv2.LINE_AA)
        tiles.append(s)
    grid = np.concatenate(tiles, axis=2)
    imageio.mimsave(out / f"{args.label}_lod.mp4",
                    [f[..., ::-1] for f in grid], fps=int(round(fps)), quality=7)
    imageio.mimsave(out / f"{args.label}_lod.gif",
                    [f[..., ::-1] for f in grid[::3, ::3, ::3]][:40],
                    fps=max(1, int(round(fps / 3))), loop=0)
    print(f"  wrote {out}/{args.label}_lod.png/.mp4/.gif, lod_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
