"""Ground-truth test: can the ODS pipeline tell a machine apart from its camera?

Takes a real still frame, warps it with a *known* deflection shape confined to
a machine region, adds a *known* whole-scene motion at the same frequency (a
roll about the optical axis plus a translation — the signature a rocking
camera leaves), and writes the result as a video. The restoration pipeline is
then run on that video and the station profile it recovers is scored against
the shape that was injected.

The scene motion is a roll rather than a horizontal shear on purpose: a roll
has a vertical component that competes directly with the transverse
deflection being measured. A purely horizontal disturbance leaves the two
orthogonal, and the pipeline passes the test without ever having to separate
them.

Two numbers matter:

  MAC   between recovered and true transverse station profile (1 = identical
        shape, and unlike a model-fit R^2 it cannot be inflated by absorbing
        camera motion, because the truth is known)
  gain  recovered peak amplitude / true peak amplitude (1 = correct scale;
        >1 means scene motion is being reported as machine deflection)

Run with --sweep to get both against camera-motion amplitude, which is the
plot that says how much scene motion each detrend mode tolerates.

Usage:
  python tools/ods_synthetic_validation.py --base frame.png --out docs/ods_validation
  python tools/ods_synthetic_validation.py --base frame.png --sweep --out docs/...
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

BG, ACCENT, TEXT, FAINT = "#1f1e1c", "#d97757", "#a8a49e", "#7d7a74"
GREEN, BLUE, RED = "#a3c95a", "#5aa9d6", "#e05a5a"


def true_shape(xn: np.ndarray) -> np.ndarray:
    """Cantilever-like transverse profile, 0 at the drive end, 1 at the free end."""
    return xn ** 2 * (3.0 - 2.0 * xn) / 1.0


def synthesize(base: np.ndarray, machine: tuple[int, int, int, int],
               out_path: Path, n_frames: int = 180, fps: float = 30.0,
               freq: float = 6.0, amp: float = 2.0, cam_t: float = 1.0,
               cam_g: float = 6.0, cam_phase: float = 0.7) -> dict:
    """Write a clip whose motion is known exactly.

    amp   peak transverse deflection of the machine's free end, px
    cam_t whole-scene translation amplitude, px
    cam_g whole-scene rotation amplitude, px per 100 px from the frame
          centre. Modelled as a roll about the optical axis,
          u = -w(y-yc), v = +w(x-xc), so the scene motion has a *vertical*
          component that competes directly with the deflection being
          measured. A shear applied only horizontally would leave the two
          orthogonal and the test would pass for the wrong reason.
    """
    h, w = base.shape[:2]
    x0, y0, x1, y1 = machine
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    inside = ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1))
    xn = np.clip((xx - x0) / max(x1 - x0, 1), 0.0, 1.0)
    shape = np.where(inside, true_shape(xn), 0.0).astype(np.float32)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for n in range(n_frames):
        t = n / fps
        s = np.sin(2 * np.pi * freq * t)
        sc = np.sin(2 * np.pi * freq * t + cam_phase)
        # machine: transverse (image-y) deflection only, scaled by the shape
        dy = -amp * shape * s
        dx = np.zeros_like(dy)
        # camera: uniform shift + roll about the frame centre
        wc = cam_g / 100.0 * sc
        dx = dx + cam_t * sc - wc * (yy - h / 2.0)
        dy = dy + wc * (xx - w / 2.0)
        # remap samples the source, so displace by -d
        vw.write(cv2.remap(base, (xx - dx).astype(np.float32),
                           (yy - dy).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT))
    vw.release()
    return {"freq": freq, "amp": amp, "cam_t": cam_t, "cam_g": cam_g,
            "frames": n_frames, "fps": fps, "machine": list(machine)}


def mac(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    num = abs(np.vdot(a, b)) ** 2
    return float(num / max(np.vdot(a, a).real * np.vdot(b, b).real, 1e-30))


def score(video: Path, machine, ref_boxes, truth, detrend: str,
          stations: int = 16) -> dict:
    mv = ods.MotionVectors.from_video(
        str(video), grid=(28, 14), max_frames=truth["frames"], roi=machine,
        ref_roi=ref_boxes if detrend != "none" else None,
        flow_mode="direct", detrend=detrend)
    f0, Uk, Vk, _, _ = ods.spectral_ods(mv, freq=truth["freq"])
    _, _, rings = ods.restore_cylinder_ods(mv, Uk, Vk, rocking=True)
    dZ = np.asarray(rings["dZ"])
    xs = np.linspace(0.0, 1.0, len(dZ))
    ref = true_shape(xs)
    # sign/phase-free comparison, as ODS are defined up to a complex scale
    m = mac(dZ, ref)
    gain = float(np.abs(dZ).max() / truth["amp"])
    return {"detrend": detrend, "f0": float(f0), "mac": m, "gain": gain,
            "peak_px": float(np.abs(dZ).max()), "r2": float(rings["r2"]),
            "profile": np.abs(dZ).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="still image to warp")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--machine", default=None,
                    help="x0,y0,x1,y1 machine region (default: central band)")
    ap.add_argument("--amp", type=float, default=2.0,
                    help="true free-end deflection, px")
    ap.add_argument("--cam-shear", type=float, default=1.0,
                    help="camera roll amplitude, px per 100 px from centre")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep camera roll and plot MAC/gain against it")
    ap.add_argument("--shears", default="0,0.25,0.5,1.0,2.0",
                    help="--sweep values, px per 100 px. Keep the resulting "
                         "whole-frame camera displacement (printed per row) "
                         "inside the optical-flow capture range, or the "
                         "breakdown you measure is Farneback's, not the "
                         "method's. The real motor clip sits near 0.45")
    ap.add_argument("--out", default="docs/ods_validation")
    args = ap.parse_args(argv)

    base = cv2.imread(args.base)
    if base is None:
        raise SystemExit(f"cannot read {args.base}")
    if args.rotate:
        base = cv2.rotate(base, ods._ROT[args.rotate])
    h, w = base.shape[:2]
    machine = (tuple(int(v) for v in args.machine.split(","))
               if args.machine else (int(w * .2), int(h * .1),
                                     int(w * .7), int(h * .55)))
    # reference boxes: static scene, straddling the frame so the affine
    # gradient is measured over a real lever arm rather than extrapolated
    mx0, my0, mx1, my1 = machine
    refs = [(0, my1 + 20, w // 3, h), (int(w * .82), 0, w, my0 + 40),
            (0, 0, w // 8, max(my0 - 10, 20))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shears = ([float(s) for s in args.shears.split(",")] if args.sweep
              else [args.cam_shear])
    rows = []
    for g in shears:
        clip = out / f"synth_shear{g:g}.mp4"
        truth = synthesize(base, machine, clip, amp=args.amp, cam_g=g,
                           cam_t=0.5 * g if g else 0.0)
        cam_px = 0.5 * g + g * max(h, w) / 200.0
        for det in ("none", "mean", "affine"):
            r = score(clip, machine, refs, truth, det)
            r["cam_shear"], r["cam_px"] = g, cam_px
            rows.append(r)
            print(f"  shear {g:5.2f} px/100px (scene {cam_px:5.2f} px)  "
                  f"detrend {det:6s}  MAC {r['mac']:.3f}  "
                  f"gain {r['gain']:5.2f}  peak {r['peak_px']:5.2f} px  "
                  f"model R2 {r['r2']:.3f}")
    macs = np.array([r["mac"] for r in rows])
    r2s = np.array([r["r2"] for r in rows])
    if macs.std() > 0 and r2s.std() > 0:
        print(f"\n  correlation between model R^2 and true MAC over "
              f"{len(rows)} runs: r = {np.corrcoef(r2s, macs)[0, 1]:+.3f}"
              "   (if this is not strongly positive, R^2 is not telling you "
              "whether the shape is right)")
    (out / "validation.json").write_text(json.dumps(
        {"truth_amp": args.amp, "machine": list(machine), "rows": rows},
        indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
    cols = {"none": RED, "mean": BLUE, "affine": GREEN}
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(FAINT)
        ax.grid(alpha=0.15, color=FAINT)
    for det, c in cols.items():
        sel = [r for r in rows if r["detrend"] == det]
        axes[0].plot([r["cam_shear"] for r in sel], [r["mac"] for r in sel],
                     "o-", color=c, label=det)
        axes[1].plot([r["cam_shear"] for r in sel], [r["gain"] for r in sel],
                     "o-", color=c, label=det)
        axes[2].plot([r["cam_shear"] for r in sel], [r["r2"] for r in sel],
                     "o-", color=c, label=det)
    axes[0].set_ylabel("MAC vs true shape  (1 = correct)", color=TEXT)
    axes[1].set_ylabel("amplitude gain  (1 = correct)", color=TEXT)
    axes[1].axhline(1.0, color=TEXT, lw=0.8, ls="--")
    axes[2].set_ylabel("cylinder-model fit $R^2$", color=TEXT)
    axes[2].set_title("$R^2$ does not track the left panel", color=TEXT,
                      fontsize=10)
    for ax in axes:
        ax.set_xlabel("injected camera roll (px per 100 px)", color=TEXT)
        ax.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, edgecolor=FAINT)
    fig.suptitle("Known deflection shape + known camera motion: what the "
                 "pipeline recovers", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=120, facecolor=BG)
    print(f"wrote {out/'validation.png'}, {out/'validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
