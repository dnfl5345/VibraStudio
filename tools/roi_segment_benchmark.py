"""Segmentation-guided ROI vs box ROI vs full-frame phase magnification.

Extends the ROI cost study with a *rough object segmentation* selector and
benchmarks three modes against each other on the same clip:

  full     magnify every pixel (reference result)
  bbox     largest band-power hotspot -> padded rectangle (previous study)
  segment  SLIC superpixels over the reference frame, seeded by band power,
           grown into color-similar neighbours -> object mask (+ its bbox)

The segment mode exists because the bbox study left motion on the table:
a rectangle around the strongest hotspot misses low-texture parts of the
same object (a smooth flange reflects little texture, so its band power is
weak even though it moves). Region growing over superpixels absorbs those
neighbours by appearance, so the ROI follows the object rather than the
texture.

Reported per mode:
  t_scan / t_magnify   wall-clock (s)
  Mpx                  pixels actually pushed through the pyramid
  area                 ROI area as a fraction of the frame
  captured             fraction of the full-frame magnified-motion ENERGY
                       that falls inside the ROI (1 - captured = motion the
                       ROI silently drops -- the metric the bbox study
                       flagged as its weakness)
  r_motion / NRMSE     in-ROI fidelity against the full-frame result

Usage:
  python tools/roi_segment_benchmark.py --video clip.mp4 --band 5 7 \
      --alpha 15 --frames 120 --label motor --out docs/roi_segment/motor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_motion_magnify import BG, ACCENT, FAINT, TEXT, feather_mask, magnify  # noqa: E402


# ---------------------------------------------------------------------------
def prescan_power(Y: np.ndarray, fps: float, band, scale: int = 4):
    """Per-pixel band power on a 1/scale grid — the cheap first pass."""
    T = Y.shape[0]
    small = Y[:, ::scale, ::scale]
    S = np.fft.rfft(small - small.mean(0), axis=0)
    f = np.fft.rfftfreq(T, 1.0 / fps)
    m = (f >= band[0]) & (f <= band[1])
    return (np.abs(S[m]) ** 2).sum(axis=0)


def bbox_roi(power: np.ndarray, scale: int, shape, pad: int = 48):
    """Largest connected hotspot -> padded rectangle (previous study)."""
    h, w = shape
    thr = power.mean() + 1.5 * power.std()
    mask = cv2.morphologyEx((power > thr).astype(np.uint8),
                            cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return (0, 0, w, h), np.ones((h, w), np.float32)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[k, :4]
    box = (max(0, x * scale - pad), max(0, y * scale - pad),
           min(w, (x + bw) * scale + pad), min(h, (y + bh) * scale + pad))
    m = np.zeros((h, w), np.float32)
    m[box[1]:box[3], box[0]:box[2]] = 1.0
    return box, m


def segment_roi(ref_bgr: np.ndarray, power: np.ndarray, scale: int,
                pad: int = 24, n_segments: int = 400, seed_frac: float = 0.30,
                keep_frac: float = 0.08, color_tau: float = 14.0,
                max_area: float = 0.6):
    """Rough object segmentation, seeded by motion, grown by appearance.

    SLIC gives ~n_segments superpixels of the reference frame. Segments
    whose mean band power exceeds seed_frac of the strongest one become
    seeds; the region then grows into adjacent segments that either still
    carry motion (>= keep_frac) or match the region's mean Lab color within
    color_tau — the latter is what pulls smooth, weakly-textured parts of
    the same object into the ROI. Growth stops at max_area of the frame.
    """
    from skimage.segmentation import slic

    h, w = ref_bgr.shape[:2]
    rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    labels = slic(rgb, n_segments=n_segments, compactness=12.0,
                  start_label=0, channel_axis=-1)
    n = int(labels.max()) + 1

    P = cv2.resize(power.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    idx = labels.ravel()
    cnt = np.maximum(np.bincount(idx, minlength=n), 1)
    seg_pw = np.bincount(idx, weights=P.ravel(), minlength=n) / cnt
    seg_col = np.stack([np.bincount(idx, weights=lab[..., c].ravel(),
                                    minlength=n) / cnt for c in range(3)], 1)

    pairs = np.concatenate([
        np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], 1),
        np.stack([labels[:-1, :].ravel(), labels[1:, :].ravel()], 1)])
    pairs = np.unique(np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1), axis=0)
    adj = defaultdict(set)
    for u, v in pairs:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    ys_g, xs_g = np.mgrid[0:h, 0:w]
    seg_y = np.bincount(idx, weights=ys_g.ravel(), minlength=n) / cnt
    seg_x = np.bincount(idx, weights=xs_g.ravel(), minlength=n) / cnt

    pmax = float(seg_pw.max()) or 1.0
    score = seg_pw / pmax
    chosen = set(np.nonzero(score >= seed_frac)[0].tolist())
    if not chosen:
        chosen = {int(np.argmax(score))}
    seeds = np.array(sorted(chosen))
    # anchor colour and centre stay fixed at the seeds so growth cannot drift
    seed_col = seg_col[seeds].mean(axis=0)
    seed_xy = np.array([seg_x[seeds].mean(), seg_y[seeds].mean()])
    dist_lim = 0.45 * np.hypot(h, w)
    area = float(cnt[seeds].sum())
    limit = max_area * h * w

    frontier = {j for i in chosen for j in adj[i] if j not in chosen}
    while frontier and area < limit:
        cands = sorted(frontier, key=lambda j: np.linalg.norm(seg_col[j] - seed_col))
        grown = False
        for j in cands:
            d_col = float(np.linalg.norm(seg_col[j] - seed_col))
            d_xy = float(np.hypot(seg_x[j] - seed_xy[0], seg_y[j] - seed_xy[1]))
            if score[j] >= keep_frac or (d_col <= color_tau and d_xy <= dist_lim):
                chosen.add(j)
                area += float(cnt[j])
                grown = True
                break
        if not grown:
            break
        frontier = {j for i in chosen for j in adj[i] if j not in chosen}

    mask = np.isin(labels, list(chosen)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    nlab, lab_cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, nlab)
            if stats[i, cv2.CC_STAT_AREA] > 0.002 * h * w]
    if not keep:
        return [(0, 0, w, h)], np.ones((h, w), np.float32), labels
    mask = np.isin(lab_cc, keep).astype(np.uint8)
    # one box per surviving component keeps the cost proportional to the
    # object, not to the bounding box of everything that moves
    boxes = merge_boxes([(max(0, stats[i, cv2.CC_STAT_LEFT] - pad),
                          max(0, stats[i, cv2.CC_STAT_TOP] - pad),
                          min(w, stats[i, cv2.CC_STAT_LEFT] +
                              stats[i, cv2.CC_STAT_WIDTH] + pad),
                          min(h, stats[i, cv2.CC_STAT_TOP] +
                              stats[i, cv2.CC_STAT_HEIGHT] + pad))
                         for i in keep])
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 9.0)
    return boxes, np.clip(soft, 0, 1), labels


def merge_boxes(boxes):
    """Fuse overlapping boxes so no pixel is magnified twice."""
    boxes = [tuple(int(v) for v in b) for b in boxes]
    changed = True
    while changed:
        changed, out = False, []
        for b in boxes:
            for i, o in enumerate(out):
                if not (b[2] <= o[0] or b[0] >= o[2] or b[3] <= o[1] or b[1] >= o[3]):
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]),
                              max(b[2], o[2]), max(b[3], o[3]))
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


# ---------------------------------------------------------------------------
def run_mode(Y, fps, band, alpha, boxes, mask, mirror=32, scales=3, orients=4):
    """Magnify each padded crop and blend it back through `mask`."""
    T, H, W = Y.shape
    out = Y.copy()
    dt, px = 0.0, 0
    for x0, y0, x1, y1 in boxes:
        mx0, my0 = max(0, x0 - mirror), max(0, y0 - mirror)
        mx1, my1 = min(W, x1 + mirror), min(H, y1 + mirror)
        crop = Y[:, my0:my1, mx0:mx1]
        t0 = time.perf_counter()
        crop_m = magnify(crop, fps, band, alpha, scales, orients)
        dt += time.perf_counter() - t0
        px += crop.shape[1] * crop.shape[2]
        edge = feather_mask(crop.shape[1], crop.shape[2], border=mirror)[None]
        w_local = mask[my0:my1, mx0:mx1][None] * edge
        out[:, my0:my1, mx0:mx1] = crop * (1 - w_local) + crop_m * w_local
    return out, dt, px


def metrics(Y, Y_full, Y_mode, mask_bin):
    """Energy captured by the ROI + in-ROI fidelity against the baseline."""
    d_full = Y_full - Y
    e_tot = float((d_full ** 2).sum())
    e_in = float((d_full ** 2 * mask_bin[None]).sum())
    captured = e_in / max(e_tot, 1e-12)
    core = cv2.erode(mask_bin.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if core.sum() < 100:
        core = mask_bin > 0.5
    a = d_full[:, core].ravel()
    b = (Y_mode - Y)[:, core].ravel()
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rng = float(Y_full[:, core].max() - Y_full[:, core].min()) or 1.0
    nrmse = float(np.sqrt(np.mean((Y_full[:, core] - Y_mode[:, core]) ** 2)) / rng)
    return captured, r, nrmse


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--band", nargs=2, type=float, required=True)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--skip", type=int, default=0, help="leading frames to drop")
    ap.add_argument("--segments", type=int, default=400)
    ap.add_argument("--seed-frac", type=float, default=0.30)
    ap.add_argument("--keep-frac", type=float, default=0.08)
    ap.add_argument("--color-tau", type=float, default=14.0)
    ap.add_argument("--max-area", type=float, default=0.35,
                    help="stop region growing past this frame fraction")
    ap.add_argument("--label", default="clip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    band = tuple(args.band)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for _ in range(args.skip):
        cap.read()
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
    t_scan = time.perf_counter() - t0

    t0 = time.perf_counter()
    box_b, mask_b = bbox_roi(power, 4, (H, W))
    t_sel_b = time.perf_counter() - t0
    t0 = time.perf_counter()
    boxes_s, mask_s, labels = segment_roi(
        bgr[0], power, 4, n_segments=args.segments, seed_frac=args.seed_frac,
        keep_frac=args.keep_frac, color_tau=args.color_tau,
        max_area=args.max_area)
    t_sel_s = time.perf_counter() - t0
    print(f"  prescan {t_scan:.2f}s | bbox select {t_sel_b:.3f}s | "
          f"segment select {t_sel_s:.2f}s ({int(labels.max())+1} superpixels, "
          f"{len(boxes_s)} box(es))")

    t0 = time.perf_counter()
    Y_full = magnify(Y, fps, band, args.alpha)
    t_full = time.perf_counter() - t0
    print(f"  full   : {t_full:7.2f} s  {T*H*W/1e6:6.1f} Mpx")

    Y_b, t_b, px_b = run_mode(Y, fps, band, args.alpha, [box_b], mask_b)
    Y_s, t_s, px_s = run_mode(Y, fps, band, args.alpha, boxes_s, mask_s)

    bin_b = (mask_b > 0.5).astype(np.float32)
    bin_s = (mask_s > 0.5).astype(np.float32)
    rows = []
    for name, t_sel, t_mag, px, mb, Ym in (
            ("full", 0.0, t_full, T * H * W, np.ones((H, W), np.float32), Y_full),
            ("bbox", t_scan + t_sel_b, t_b, T * px_b, bin_b, Y_b),
            ("segment", t_scan + t_sel_s, t_s, T * px_s, bin_s, Y_s)):
        cap_e, r, nrmse = metrics(Y, Y_full, Ym, mb)
        rows.append(dict(mode=name, t_sel=t_sel, t_mag=t_mag,
                         t_total=t_sel + t_mag, mpx=px / 1e6,
                         area=float(mb.mean()), captured=cap_e,
                         r_motion=r, nrmse=nrmse,
                         speedup=t_full / (t_sel + t_mag)))
        print(f"  {name:7s}: {t_sel + t_mag:7.2f} s  {px/1e6:6.1f} Mpx  "
              f"area {100*mb.mean():5.1f}%  captured {100*cap_e:5.1f}%  "
              f"r={r:.4f}  speedup x{t_full/(t_sel+t_mag):.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        dict(label=args.label, video=Path(args.video).name, frames=T,
             size=[W, H], fps=fps, band=band, alpha=args.alpha, rows=rows),
        indent=2))

    # -- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0), facecolor=BG)
    for a_ in ax:
        a_.set_facecolor(BG)
        a_.tick_params(colors=TEXT, labelsize=7)
        for sp in a_.spines.values():
            sp.set_color(FAINT)

    rgb0 = cv2.cvtColor(bgr[0], cv2.COLOR_BGR2RGB)
    ax[0].imshow(mark_boundaries(rgb0, labels, color=(0.85, 0.47, 0.34)))
    ax[0].set_title(f"SLIC superpixels ({int(labels.max())+1})", color=TEXT, fontsize=9)
    ax[0].axis("off")

    ax[1].imshow(cv2.resize(power, (W, H)), cmap="plasma")
    ax[1].add_patch(plt.Rectangle((box_b[0], box_b[1]), box_b[2]-box_b[0],
                                  box_b[3]-box_b[1], fill=False,
                                  edgecolor="#5aa9d6", lw=1.6, label="bbox ROI"))
    ax[1].contour(bin_s, levels=[0.5], colors=["#a3c95a"], linewidths=1.6)
    ax[1].set_title("band power + bbox (blue) / segment (green)",
                    color=TEXT, fontsize=9)
    ax[1].axis("off")

    over = rgb0.copy()
    over[bin_s > 0.5] = (0.55 * over[bin_s > 0.5] +
                         0.45 * np.array([163, 201, 90])).astype(np.uint8)
    cv2.rectangle(over, (box_b[0], box_b[1]), (box_b[2], box_b[3]), (90, 169, 214), 2)
    for bx in boxes_s:
        cv2.rectangle(over, (bx[0], bx[1]), (bx[2], bx[3]), (163, 201, 90), 1)
    ax[2].imshow(over)
    ax[2].set_title(f"segment mask {100*rows[2]['area']:.0f}% / "
                    f"{len(boxes_s)} crop(s) {100*rows[2]['mpx']/rows[0]['mpx']:.0f}% px"
                    f"  vs bbox {100*rows[1]['area']:.0f}%", color=TEXT, fontsize=9)
    ax[2].axis("off")

    names = [r["mode"] for r in rows]
    xs = np.arange(3)
    ax[3].bar(xs - 0.2, [r["t_total"] for r in rows], 0.4, color=ACCENT,
              label="time (s)")
    ax[3].set_ylabel("wall-clock (s)", color=TEXT, fontsize=8)
    ax2 = ax[3].twinx()
    ax2.bar(xs + 0.2, [100 * r["captured"] for r in rows], 0.4, color="#a3c95a",
            label="captured motion (%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("captured motion energy (%)", color=TEXT, fontsize=8)
    ax2.tick_params(colors=TEXT, labelsize=7)
    ax[3].set_xticks(xs)
    ax[3].set_xticklabels(names, color=TEXT)
    ax[3].set_title(f"{args.label}: cost vs captured motion", color=TEXT, fontsize=9)
    ax[3].legend(fontsize=7, loc="upper left", facecolor=BG, labelcolor=TEXT,
                 edgecolor=FAINT)
    ax2.legend(fontsize=7, loc="upper right", facecolor=BG, labelcolor=TEXT,
               edgecolor=FAINT)
    fig.tight_layout()
    fig.savefig(out / f"{args.label}_roi_modes.png", dpi=110, facecolor=BG)
    plt.close(fig)

    # -- side-by-side video ---------------------------------------------
    import imageio.v2 as imageio

    def to_bgr(Ys):
        z = yuv.copy()
        z[..., 0] = np.clip(Ys, 0, 255).astype(np.uint8)
        return np.stack([cv2.cvtColor(f, cv2.COLOR_YUV2BGR) for f in z])

    tiles = []
    for name, stack, draw in (("original", bgr, None),
                              ("full CSP", to_bgr(Y_full), None),
                              ("bbox ROI", to_bgr(Y_b), "box"),
                              ("segment ROI", to_bgr(Y_s), "mask")):
        s = stack.copy()
        for f in s:
            if draw == "box":
                cv2.rectangle(f, (box_b[0], box_b[1]), (box_b[2], box_b[3]),
                              (214, 169, 90), 2)
            elif draw == "mask":
                cnts, _ = cv2.findContours((bin_s > 0.5).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(f, cnts, -1, (90, 201, 163), 2)
            cv2.putText(f, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (87, 119, 217), 2, cv2.LINE_AA)
        tiles.append(s)
    top = np.concatenate(tiles[:2], axis=2)
    bot = np.concatenate(tiles[2:], axis=2)
    grid = np.concatenate([top, bot], axis=1)
    imageio.mimsave(out / f"{args.label}_roi_modes.mp4",
                    [f[..., ::-1] for f in grid], fps=int(round(fps)), quality=7)
    imageio.mimsave(out / f"{args.label}_roi_modes.gif",
                    [f[..., ::-1] for f in grid[::2, ::2, ::2]],
                    fps=max(1, int(round(fps / 2))), loop=0)
    print(f"  wrote {out}/{args.label}_roi_modes.png/.mp4/.gif, metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
