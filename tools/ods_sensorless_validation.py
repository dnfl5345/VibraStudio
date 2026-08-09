"""Validate a video ODS without a reference accelerometer.

There is no sensor to compare against, so every check here is an internal
consistency test — each one asks a question that a genuine structural
deflection answers differently from residual camera motion.

  static      How much analysis-line motion survives on a region that cannot
              be moving? Camera motion moves the background and the machine
              alike; a deflection shape does not. Reported as the ratio
              machine/static, which must clear 2x for anything downstream to
              mean much. This is the cheapest check and it rules out the most.

  split-half  Restore the shape from the first half of the clip and from the
              second half independently, and correlate them (MAC). A shape
              that will not reproduce across halves of its own recording is
              noise. Passing is necessary, not sufficient: both halves share
              the same camera, so a steady camera motion passes too.

  repeat      The strong one, and the reason to shoot twice. Restore from two
              *separate recordings* of the same machine state and correlate.
              Camera motion is uncorrelated between recordings — a tripod
              bumped at a different moment, a different building-sway phase —
              while the machine's deflection shape is not. A high MAC across
              independent recordings is evidence no camera artefact can
              easily produce. Needs `pair`.

  order       For rotating machinery, response concentrates at integer orders
              of running speed. Given --rpm, this reports how far the
              dominant peak sits from 1x and what share of the band energy is
              order-locked. Building sway, hand shake, and rolling shutter do
              not know the shaft speed, so an order-locked peak is a piece of
              ground truth that costs nothing to obtain.

Usage:
  python tools/ods_sensorless_validation.py screen a.mp4 b.mp4 ... [--rpm 1800]
  python tools/ods_sensorless_validation.py pair run1.mp4 run2.mp4 \
      --roi x0,y0,x1,y1 --ref-roi "..." [--rpm 1800]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ods_pipe", str(Path(__file__).resolve().parent / "ods_from_motion_vectors.py"))
ods = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ods)

MIN_FRAMES = 48          # below this a split-half test has no resolution


def mac(a, b) -> float:
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    return float(abs(np.vdot(a, b)) ** 2
                 / max((np.vdot(a, a).real * np.vdot(b, b).real), 1e-30))


def band_power_map(path: str, rotate: int = 0, max_frames: int = 240,
                   step: int = 8):
    """Coarse per-pixel band amplitude, used to find what moves and what does not."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, ref = cap.read()
    if not ok:
        raise IOError("empty video")
    if rotate:
        ref = cv2.rotate(ref, ods._ROT[rotate])
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    h, w = ref_g.shape
    gy, gx = np.mgrid[0:h:step, 0:w:step]
    U, V = [], []
    while len(U) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if rotate:
            fr = cv2.rotate(fr, ods._ROT[rotate])
        f = cv2.calcOpticalFlowFarneback(
            ref_g, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None,
            0.5, 3, 21, 3, 7, 1.5, 0)
        U.append(f[gy, gx, 0].copy())
        V.append(-f[gy, gx, 1].copy())
    cap.release()
    U, V = np.array(U), np.array(V)
    T = len(U)
    if T < 8:
        raise ValueError(f"{path}: only {T} usable frames")
    win = np.hanning(T)[:, None, None]
    Fu = np.fft.rfft((U - U.mean(0)) * win, axis=0) * (2.0 / np.hanning(T).sum())
    Fv = np.fft.rfft((V - V.mean(0)) * win, axis=0) * (2.0 / np.hanning(T).sum())
    freqs = np.fft.rfftfreq(T, 1.0 / fps)
    spec = (np.abs(Fu) ** 2 + np.abs(Fv) ** 2).sum(axis=(1, 2))
    lo = max(1, int(np.searchsorted(freqs, 0.5)))
    spec[:lo] = 0.0
    k = int(np.argmax(spec))
    amp = np.sqrt(np.abs(Fu[k]) ** 2 + np.abs(Fv[k]) ** 2)
    return {"fps": fps, "frames": T, "size": (w, h), "freqs": freqs,
            "spec": spec, "k": k, "f0": float(freqs[k]), "amp": amp,
            "gx": gx, "gy": gy, "step": step}


def auto_regions(bp) -> tuple[tuple, list]:
    """Bounding box of the most active area, and static boxes from the quietest."""
    amp, gx, gy = bp["amp"], bp["gx"], bp["gy"]
    hot = amp >= np.quantile(amp, 0.92)
    ys, xs = np.nonzero(hot)
    roi = (int(gx[0, xs.min()]), int(gy[ys.min(), 0]),
           int(gx[0, xs.max()]) + 1, int(gy[ys.max(), 0]) + 1)
    # quietest cells, taken as four corner-ish boxes so the affine fit has a
    # lever arm across the frame rather than being anchored in one corner
    w, h = bp["size"]
    quiet = amp <= np.quantile(amp, 0.35)
    refs = []
    for qx0, qy0, qx1, qy1 in [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                               (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]:
        sel = ((gx >= qx0) & (gx < qx1) & (gy >= qy0) & (gy < qy1) & quiet)
        if sel.sum() < 6:
            continue
        ys, xs = np.nonzero(sel)
        refs.append((int(gx[0, xs.min()]), int(gy[ys.min(), 0]),
                     int(gx[0, xs.max()]) + 1, int(gy[ys.max(), 0]) + 1))
    return roi, refs


def held_out_box(bp, roi, refs):
    """Quietest region that is neither the structure nor a reference box.

    Held out on purpose: measuring the leftover on a region the detrend was
    fitted to would only report how well least squares fits its own data.
    """
    amp, gx, gy = bp["amp"], bp["gx"], bp["gy"]
    x0, y0, x1, y1 = roi
    free = ~((gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1))
    for rx0, ry0, rx1, ry1 in refs or []:
        free &= ~((gx >= rx0) & (gx < rx1) & (gy >= ry0) & (gy < ry1))
    if free.sum() < 8:
        return None
    quiet = free & (amp <= np.quantile(amp[free], 0.4))
    ys, xs = np.nonzero(quiet if quiet.sum() >= 8 else free)
    return (int(gx[0, xs.min()]), int(gy[ys.min(), 0]),
            int(gx[0, xs.max()]) + 1, int(gy[ys.max(), 0]) + 1)


def region_amp(path, box, refs, frames, rotate, freq) -> float:
    """Analysis-line amplitude in a box, after the same scene-motion removal
    the restoration uses. Measuring this on the raw field instead would
    flatter any clip whose structure happens to sit at the hot end of a
    camera gradient -- the ratio would describe the gradient, not the machine.
    """
    mv = ods.MotionVectors.from_video(
        path, grid=(16, 10), max_frames=frames, roi=box,
        ref_roi=refs if refs else None, rotate=rotate, flow_mode="direct",
        detrend="affine" if refs else "none")
    _, U, V, _, _ = ods.spectral_ods(mv, freq=freq)
    return float(np.sqrt(np.abs(U) ** 2 + np.abs(V) ** 2).mean())


def profile(path, roi, refs, frames, rotate=0, freq=None, skip=0):
    """Restored transverse station profile from one clip (or one time window)."""
    mv = ods.MotionVectors.from_video(
        path, grid=(24, 12), max_frames=skip + frames, roi=roi,
        ref_roi=refs if refs else None, rotate=rotate, flow_mode="direct",
        detrend="affine" if refs else "none")
    if skip:
        mv = ods.MotionVectors(mv.points, mv.u[skip:], mv.v[skip:], mv.fps)
    f0, Uk, Vk, _, _ = ods.spectral_ods(mv, freq=freq)
    _, _, rings = ods.restore_cylinder_ods(mv, Uk, Vk, rocking=True)
    return float(f0), np.asarray(rings["dZ"]), float(rings["r2"])


def order_check(bp, rpm: float) -> dict:
    """How close the dominant peak is to 1x, and how order-locked the band is."""
    f1 = rpm / 60.0
    freqs, spec = bp["freqs"], bp["spec"]
    if f1 <= 0 or f1 >= freqs[-1]:
        return {"rpm": rpm, "f1": f1, "resolvable": False}
    df = freqs[1] - freqs[0]
    dev = (bp["f0"] - f1) / f1
    tot = spec.sum()
    locked = 0.0
    for n in range(1, int(freqs[-1] / f1) + 1):
        sel = np.abs(freqs - n * f1) <= max(1.5 * df, 0.02 * n * f1)
        locked += spec[sel].sum()
    return {"rpm": rpm, "f1": f1, "resolvable": True, "peak": bp["f0"],
            "peak_dev_from_1x": float(dev),
            "order_locked_fraction": float(locked / max(tot, 1e-30)),
            "bin_hz": float(df)}


def screen_one(path: str, rotate: int, rpm: float | None,
               roi=None, refs=None, check_roi=None) -> dict:
    bp = band_power_map(path, rotate=rotate)
    auto = roi is None
    if auto:
        roi, refs = auto_regions(bp)
    elif refs is None:
        _, refs = auto_regions(bp)
    if refs and isinstance(refs[0], int):
        refs = [refs]
    n = bp["frames"]
    probe = check_roi or held_out_box(bp, roi, refs)
    a_in = region_amp(path, roi, refs, n, rotate, bp["f0"])
    a_out = (region_amp(path, probe, refs, n, rotate, bp["f0"])
             if probe else float("nan"))
    row = {"clip": Path(path).name, "size": list(bp["size"]),
           "fps": bp["fps"], "frames": n, "f0": bp["f0"],
           "roi": list(roi), "refs": [list(r) for r in refs], "auto": auto,
           "check_roi": list(probe) if probe else None,
           "amp_active": a_in, "amp_static": a_out,
           "static_ratio": a_in / max(a_out, 1e-9)}
    if rpm:
        row["order"] = order_check(bp, rpm)
    if n >= MIN_FRAMES:
        half = n // 2
        f0, p1, r1 = profile(path, roi, refs, half, rotate=rotate)
        _, p2, r2 = profile(path, roi, refs, half, rotate=rotate,
                            freq=f0, skip=half)
        row["split_half_mac"] = mac(p1, p2)
        row["r2"] = 0.5 * (r1 + r2)
    else:
        row["split_half_mac"] = None
        row["note"] = f"only {n} frames; split-half needs >= {MIN_FRAMES}"
    row["verdict"] = verdict(row)
    return row


def verdict(row) -> str:
    if not np.isfinite(row["static_ratio"]):
        # no region left that is neither the structure nor a reference box
        return "INCONCLUSIVE (no held-out static region available)"
    if row["static_ratio"] < 2.0:
        return "REJECT (not separable from scene motion)"
    if row["split_half_mac"] is None:
        return "INCONCLUSIVE (too short)"
    if row["split_half_mac"] < 0.90:
        return "REJECT (does not reproduce within its own recording)"
    if row.get("auto"):
        # Auto regions pick the *most* and *least* active cells, which on a
        # clip whose dominant field is a camera gradient means the two ends
        # of that gradient -- not machine vs background. Combined with a
        # split-half test that a steady camera passes perfectly, this route
        # can clear a clip that a hand-picked static region rejects outright.
        return "UNCONFIRMED (auto regions; give --roi/--ref-roi to decide)"
    return "USABLE (needs a repeat recording to confirm)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["screen", "pair"])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--rpm", type=float, default=None,
                    help="known running speed; enables the order-lock check")
    ap.add_argument("--roi", default=None,
                    help="x0,y0,x1,y1 structure region; without it both modes "
                         "guess from band power, which is triage only")
    ap.add_argument("--ref-roi", default=None,
                    help="';'-separated static boxes")
    ap.add_argument("--check-roi", default=None,
                    help="static region held OUT of --ref-roi, used to bound "
                         "how much scene motion is left; guessed if omitted")
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args(argv)

    def parse(s):
        if not s:
            return None
        b = [tuple(int(v) for v in x.split(",")) for x in s.split(";")]
        return b[0] if len(b) == 1 else b

    if args.mode == "screen":
        rows = []
        for c in args.clips:
            try:
                row = screen_one(c, args.rotate, args.rpm,
                                 roi=parse(args.roi), refs=parse(args.ref_roi),
                                 check_roi=parse(args.check_roi))
            except Exception as e:                       # noqa: BLE001
                rows.append({"clip": Path(c).name, "error": str(e),
                             "verdict": "ERROR"})
                print(f"  {Path(c).name:38s} ERROR {e}")
                continue
            rows.append(row)
            sh = ("  n/a " if row["split_half_mac"] is None
                  else f"{row['split_half_mac']:.3f}")
            print(f"  {row['clip']:38s} {row['size'][0]}x{row['size'][1]} "
                  f"{row['fps']:5.1f}fps {row['frames']:4d}fr  "
                  f"f0 {row['f0']:6.2f} Hz  active/static "
                  f"{row['static_ratio']:5.2f}x  split-half MAC {sh}  "
                  f"{row['verdict']}")
            if "order" in row and row["order"].get("resolvable"):
                o = row["order"]
                print(f"      order check @ {o['rpm']:.0f} rpm "
                      f"(1x = {o['f1']:.2f} Hz): peak is "
                      f"{100*o['peak_dev_from_1x']:+.1f}% from 1x, "
                      f"{100*o['order_locked_fraction']:.1f}% of band energy "
                      f"order-locked")
        out = {"mode": "screen", "rows": rows}
    else:
        if len(args.clips) != 2:
            ap.error("pair mode takes exactly two clips")
        a, b = args.clips
        roi, refs = parse(args.roi), parse(args.ref_roi)
        if roi is None:
            bp = band_power_map(a, rotate=args.rotate)
            roi, auto = auto_regions(bp)
            refs = refs or auto
            print(f"  auto ROI {roi}, {len(refs)} reference boxes")
        na = int(cv2.VideoCapture(a).get(cv2.CAP_PROP_FRAME_COUNT))
        nb = int(cv2.VideoCapture(b).get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(na, nb, 240)
        fa, pa, ra = profile(a, roi, refs, n, rotate=args.rotate)
        fb, pb, rb = profile(b, roi, refs, n, rotate=args.rotate, freq=fa)
        m = mac(pa, pb)
        print(f"  {Path(a).name}: f0 {fa:.3f} Hz, model R2 {ra:.3f}")
        print(f"  {Path(b).name}: f0 {fb:.3f} Hz, model R2 {rb:.3f}")
        print(f"  repeat MAC across independent recordings: {m:.3f}")
        print("  " + ("PASS — a shape that reproduces across separate "
                      "recordings is hard to explain as camera motion"
                      if m >= 0.90 else
                      "FAIL — the two recordings do not agree, so neither "
                      "shape is trustworthy"))
        out = {"mode": "pair", "clips": [Path(a).name, Path(b).name],
               "f0": [fa, fb], "r2": [ra, rb], "repeat_mac": m,
               "roi": list(roi), "frames": n}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
