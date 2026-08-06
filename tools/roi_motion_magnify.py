"""ROI-selective phase-based motion magnification — cost/fidelity study.

Implements phase-based video motion magnification on a non-subsampled
complex steerable pyramid (self-inverting FFT-domain filters,
sum |H|^2 = 1, per Simoncelli's construction) and compares two modes:

  full  — magnify the entire frame (baseline)
  roi   — two-pass: a cheap low-resolution pre-scan maps per-pixel
          band-power at the analysis band, the largest connected hotspot
          becomes the ROI (padded); only that crop is magnified and is
          feather-blended back into the original frames.

Both modes share the identical magnification core, so wall-clock and
memory differences are attributable to pixel count alone. The script
reports processing time, processed-pixel ratio, in-ROI fidelity of the
ROI result against the full-frame baseline (correlation / NRMSE), and
boundary-blend artifact level, plus side-by-side output video and a
summary figure.

Usage:
  python tools/roi_motion_magnify.py --video test_vib_6hz.mp4 \
      --band 5 7 --alpha 15 --out docs/ods_restoration/roi_magnify
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

BG = "#1f1e1c"
ACCENT = "#d97757"
TEXT = "#a8a49e"
FAINT = "#7d7a74"


# ---------------------------------------------------------------------------
# Complex steerable pyramid (non-subsampled, FFT domain)
# ---------------------------------------------------------------------------
def csp_filters(h: int, w: int, n_scales: int = 3, n_orient: int = 4):
    """Self-inverting filter bank: hi/lo residuals + scales x orientations.

    Radial: telescoping octave bands built from smooth sin/cos steps
    (u^2 + sqrt(1-u^2)^2 = 1 partners). Angular: cos^(n-1) lobes restricted
    to a half-plane (analytic signal). The whole bank is normalized
    NUMERICALLY so that hi^2 + lo^2 + sum(|B|^2 + |B_mirror|^2) = 1 at
    every frequency — exact reconstruction via out = IFFT(F(hi^2+lo^2)) +
    sum 2*Re[IFFT(FFT(B'_b) conj(H_b))]. The mirror-lobe sum is symmetric
    in xi -> -xi, so pointwise normalization keeps the identity exact.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.clip(np.sqrt(fx ** 2 + fy ** 2) / 0.5, 1e-9, None)   # Nyquist=1
    theta = np.arctan2(fy, fx)
    log_r = np.log2(r)

    def step(cut_log):        # smooth 0->1 over one octave below the cutoff
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
            A = np.where(np.abs(ang) < np.pi / 2,
                         np.cos(ang) ** (n_orient - 1), 0.0)
            bands.append((rad * A).astype(np.float32))

    total = hi ** 2 + lo ** 2
    for B in bands:
        Bm = np.roll(np.abs(B)[::-1, ::-1], (1, 1), axis=(0, 1))  # B(-xi)
        total = total + B ** 2 + Bm ** 2
    norm = np.sqrt(np.clip(total, 1e-12, None)).astype(np.float32)
    hi = (hi / norm).astype(np.float32)
    lo = (lo / norm).astype(np.float32)
    bands = [(b / norm).astype(np.float32) for b in bands]
    return bands, lo, hi


def magnify(frames_y: np.ndarray, fps: float, band: tuple[float, float],
            alpha: float, n_scales: int = 3, n_orient: int = 4) -> np.ndarray:
    """Phase-based magnification of a (T, H, W) float32 luma stack."""
    T, h, w = frames_y.shape
    bands, lo, hi = csp_filters(h, w, n_scales, n_orient)
    F = np.fft.fft2(frames_y, axes=(1, 2)).astype(np.complex64)

    f = np.fft.rfftfreq(T, 1.0 / fps)
    keep = ((f >= band[0]) & (f <= band[1])).astype(np.float32)

    out = np.fft.ifft2(F * (hi ** 2 + lo ** 2)[None], axes=(1, 2)).real
    for H in bands:
        B = np.fft.ifft2(F * H[None], axes=(1, 2))          # (T,h,w) complex
        ref = B[:1]
        dphi = np.angle(B * np.conj(ref))                    # phase vs frame0
        DP = np.fft.rfft(dphi, axis=0)
        dphi_f = np.fft.irfft(DP * keep[:, None, None], n=T, axis=0)
        Bm = B * np.exp(1j * alpha * dphi_f).astype(np.complex64)
        out += 2 * np.fft.ifft2(np.fft.fft2(Bm, axes=(1, 2)) *
                                np.conj(H)[None], axes=(1, 2)).real
        del B, dphi, DP, dphi_f, Bm
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# ROI selection + blending
# ---------------------------------------------------------------------------
def hotspot_roi(frames_y: np.ndarray, fps: float, band: tuple[float, float],
                scale: int = 4, pad: int = 48):
    """Low-res per-pixel band-power map -> largest hotspot bbox (padded)."""
    T, h, w = frames_y.shape
    small = frames_y[:, ::scale, ::scale]
    S = np.fft.rfft(small - small.mean(0), axis=0)
    f = np.fft.rfftfreq(T, 1.0 / fps)
    m = (f >= band[0]) & (f <= band[1])
    power = (np.abs(S[m]) ** 2).sum(axis=0)
    thr = power.mean() + 1.5 * power.std()
    mask = (power > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return (0, 0, w, h), power
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[k, :4]
    x0 = max(0, x * scale - pad)
    y0 = max(0, y * scale - pad)
    x1 = min(w, (x + bw) * scale + pad)
    y1 = min(h, (y + bh) * scale + pad)
    return (x0, y0, x1, y1), power


def feather_mask(h: int, w: int, border: int = 32) -> np.ndarray:
    """Raised-cosine window that fades to 0 at the crop edges."""
    def ramp(n):
        win = np.ones(n, np.float32)
        b = min(border, n // 2)
        t = 0.5 * (1 - np.cos(np.pi * np.arange(b) / b))
        win[:b] = t
        win[n - b:] = t[::-1]
        return win
    return ramp(h)[:, None] * ramp(w)[None, :]


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--band", nargs=2, type=float, default=[5.0, 7.0])
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--scales", type=int, default=3)
    ap.add_argument("--orients", type=int, default=4)
    ap.add_argument("--out", default="docs/ods_restoration/roi_magnify")
    args = ap.parse_args(argv)
    band = tuple(args.band)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    rgb = np.stack(frames)                               # (T,H,W,3) BGR
    yuv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2YUV) for f in frames])
    Y = yuv[..., 0].astype(np.float32)
    T, H, W = Y.shape
    print(f"[1] video: {T} frames {W}x{H} @ {fps:g} fps, "
          f"band {band[0]}-{band[1]} Hz, alpha {args.alpha}")

    # -- full-frame baseline ------------------------------------------------
    t0 = time.perf_counter()
    Y_full = magnify(Y, fps, band, args.alpha, args.scales, args.orients)
    t_full = time.perf_counter() - t0
    print(f"[2] full-frame magnify: {t_full:.2f} s "
          f"({T * H * W / 1e6:.1f} Mpx processed)")

    # -- ROI mode -----------------------------------------------------------
    t0 = time.perf_counter()
    (x0, y0, x1, y1), power = hotspot_roi(Y, fps, band)
    t_scan = time.perf_counter() - t0
    rw, rh = x1 - x0, y1 - y0
    frac = rw * rh / (H * W)
    print(f"[3] ROI auto-select: x[{x0},{x1}] y[{y0},{y1}] "
          f"({rw}x{rh}, {100 * frac:.1f}% of frame), scan {t_scan:.2f} s")

    mirror = 32                                          # anti-ringing pad
    mx0, my0 = max(0, x0 - mirror), max(0, y0 - mirror)
    mx1, my1 = min(W, x1 + mirror), min(H, y1 + mirror)
    crop = Y[:, my0:my1, mx0:mx1]
    t0 = time.perf_counter()
    crop_m = magnify(crop, fps, band, args.alpha, args.scales, args.orients)
    t_roi = time.perf_counter() - t0
    print(f"[4] ROI magnify: {t_roi:.2f} s "
          f"({T * crop.shape[1] * crop.shape[2] / 1e6:.1f} Mpx) "
          f"-> speedup x{t_full / t_roi:.2f} (pixel ratio "
          f"x{(H * W) / (crop.shape[1] * crop.shape[2]):.2f})")

    fm = feather_mask(crop.shape[1], crop.shape[2], border=mirror)[None]
    Y_roi = Y.copy()
    Y_roi[:, my0:my1, mx0:mx1] = crop * (1 - fm) + crop_m * fm

    # -- quantitative comparison -------------------------------------------
    ix0, iy0 = x0 + mirror, y0 + mirror                 # interior (no feather)
    ix1, iy1 = x1 - mirror, y1 - mirror
    a = Y_full[:, iy0:iy1, ix0:ix1].ravel()
    b = Y_roi[:, iy0:iy1, ix0:ix1].ravel()
    r_fid = float(np.corrcoef(a, b)[0, 1])
    nrmse = float(np.sqrt(np.mean((a - b) ** 2)) / (a.max() - a.min() + 1e-9))
    # motion-signal fidelity: compare the magnification deltas, not raw luma
    da = (Y_full - Y)[:, iy0:iy1, ix0:ix1].ravel()
    db = (Y_roi - Y)[:, iy0:iy1, ix0:ix1].ravel()
    r_mot = float(np.corrcoef(da, db)[0, 1])
    # boundary artifact: ROI-result deviation from original just OUTSIDE ROI
    ring = np.zeros((H, W), bool)
    rx0, ry0 = max(0, x0 - 16), max(0, y0 - 16)
    rx1, ry1 = min(W, x1 + 16), min(H, y1 + 16)
    ring[ry0:ry1, rx0:rx1] = True
    ring[y0:y1, x0:x1] = False
    art = float(np.sqrt(np.mean((Y_roi[:, ring] - Y[:, ring]) ** 2)))
    print(f"[5] fidelity in ROI: luma r={r_fid:.4f}, NRMSE={100 * nrmse:.2f}%,"
          f" motion-delta r={r_mot:.4f}; boundary RMS dev {art:.3f} gray-lvl")

    # -- outputs ------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    def to_bgr(Ys):
        z = yuv.copy()
        z[..., 0] = np.clip(Ys, 0, 255).astype(np.uint8)
        return np.stack([cv2.cvtColor(f, cv2.COLOR_YUV2BGR) for f in z])

    full_bgr, roi_bgr = to_bgr(Y_full), to_bgr(Y_roi)
    roi_draw = roi_bgr.copy()
    for f in roi_draw:
        cv2.rectangle(f, (x0, y0), (x1, y1), (87, 119, 217), 2)
    lab = [("original", rgb), ("full CSP", full_bgr), ("ROI CSP", roi_draw)]
    tiles = []
    for name, stack in lab:
        s = stack.copy()
        for f in s:
            cv2.putText(f, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (87, 119, 217), 2, cv2.LINE_AA)
        tiles.append(s)
    side = np.concatenate(tiles, axis=2)                # horizontal
    imageio.mimsave(out / "roi_magnify_compare.mp4",
                    [f[..., ::-1] for f in side], fps=int(fps), quality=7)
    imageio.mimsave(out / "roi_magnify_compare.gif",
                    [f[..., ::-1] for f in side[::2, :, ::2, :][:, ::2]],
                    fps=max(1, int(fps / 2)), loop=0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), facecolor=BG)
    for a_ in axes:
        a_.set_facecolor(BG)
        a_.tick_params(colors=TEXT, labelsize=8)
        for sp in a_.spines.values():
            sp.set_color(FAINT)
    axes[0].imshow(power, cmap="plasma")
    axes[0].add_patch(plt.Rectangle((x0 / 4, y0 / 4), (x1 - x0) / 4,
                                    (y1 - y0) / 4, fill=False,
                                    edgecolor="#5aa9d6", lw=1.5))
    axes[0].set_title(f"pre-scan band power (1/4 res) + ROI "
                      f"({100 * frac:.0f}% of frame)", color=TEXT, fontsize=9)
    bars = axes[1].bar(["full", "ROI"], [t_full, t_roi + t_scan],
                       color=[FAINT, ACCENT])
    axes[1].bar(["ROI"], [t_scan], color="#5aa9d6",
                label=f"pre-scan {t_scan:.2f}s")
    axes[1].set_title(f"wall-clock (s) — speedup x{t_full / (t_roi + t_scan):.2f}"
                      f" incl. scan", color=TEXT, fontsize=9)
    axes[1].bar_label(bars, fmt="%.1fs", color=TEXT, fontsize=8)
    axes[1].legend(fontsize=7, facecolor=BG, labelcolor=TEXT, edgecolor=FAINT)
    err = np.abs(Y_full - Y_roi).mean(axis=0)
    im = axes[2].imshow(err, cmap="plasma")
    fig.colorbar(im, ax=axes[2]).ax.tick_params(colors=TEXT, labelsize=7)
    axes[2].set_title(f"|full - ROI| mean abs diff — in-ROI motion r={r_mot:.3f}",
                      color=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roi_magnify_summary.png", dpi=110, facecolor=BG)
    print(f"[6] wrote {out}/roi_magnify_compare.mp4/.gif, roi_magnify_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
