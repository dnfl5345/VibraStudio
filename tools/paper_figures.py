"""Paper figures for docs/paper/draft.md.

Three figures, each computed from data rather than transcribed:

  fig1  identifiability   -- rerun of tools/ods_identifiability.py
  fig2  controlled expt   -- docs/ods_validation/validation.json
  fig3  field case        -- recomputed from the motor clip (needs --video)

Light surface, because these are destined for a manuscript rather than the
repo's dark docs. Categorical hues are the first three slots of the
validated default palette, which clear the all-pairs CVD and normal-vision
floors; the aqua slot sits below 3:1 on the light surface, so every series
carries a direct label or legend entry rather than relying on colour alone.

Usage:
  python tools/paper_figures.py --out docs/paper/figs [--video motor.mp4]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_here / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# validated palette, light surface
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "#dedcd6"


def _style(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3, width=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1 — the model spans the camera field
# ---------------------------------------------------------------------------
def fig1(out: Path):
    ident = _load("ods_identifiability")
    pts = ident.grid_points()
    x, z = pts[:, 0], pts[:, 1]
    xc, zc = x.mean(), z.mean()
    rng = np.random.default_rng(3)
    A, t = rng.normal(size=(2, 2)) * 0.01, rng.normal(size=2)

    fields = [
        ("translation", np.full_like(x, 1.0), np.full_like(x, 0.6)),
        ("shear $\\partial u/\\partial z$", 0.01 * (z - zc), np.zeros_like(x)),
        ("roll", -0.01 * (z - zc), 0.01 * (x - xc)),
        ("general affine",
         t[0] + A[0, 0] * (x - xc) + A[0, 1] * (z - zc),
         t[1] + A[1, 0] * (x - xc) + A[1, 1] * (z - zc)),
        ("shear $\\partial v/\\partial x$", np.zeros_like(x), 0.01 * (x - xc)),
        ("zoom", 0.01 * (x - xc), 0.01 * (z - zc)),
    ]
    cam = [(n, ident.fit(pts, u.astype(complex), v.astype(complex))["r2"])
           for n, u, v in fields]
    bend = ident.fit(pts, np.zeros_like(x, dtype=complex),
                     ((x / x.max()) ** 2).astype(complex))["r2"]

    counts = (4, 6, 10, 16, 24)
    conv = {}
    for name, u, v in fields:
        if name not in ("roll", "shear $\\partial v/\\partial x$", "zoom"):
            continue
        row = []
        for n in counts:
            dummy = np.zeros((4, len(pts)))
            mv = ident.ods.MotionVectors(pts, dummy, dummy, 30.0)
            _, _, r = ident.ods.restore_cylinder_ods(
                mv, u.astype(complex), v.astype(complex), n_layers=n,
                rocking=True)
            row.append(r["r2"])
        conv[name] = row

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.2, 4.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.25, 1]})
    _style(axa, grid_axis="x")
    _style(axb)

    names = [n for n, _ in cam][::-1]
    vals = [r for _, r in cam][::-1]
    ypos = np.arange(len(names))
    axa.barh(ypos, vals, height=0.62, color=BLUE, zorder=3,
             edgecolor=SURFACE, linewidth=1.6)
    for y, val in zip(ypos, vals):
        axa.text(val + 0.012, y, f"{val:.4f}", va="center", ha="left",
                 fontsize=8.5, color=INK)
    axa.axvline(bend, color=ORANGE, linewidth=2.0, zorder=4,
                label=f"a real deflection shape fits at {bend:.4f}")
    axa.legend(fontsize=8.5, frameon=False, labelcolor=ORANGE,
               loc="lower left", bbox_to_anchor=(0.0, -0.02),
               handlelength=1.4)
    axa.set_yticks(ypos)
    axa.set_yticklabels(names, fontsize=9, color=INK)
    axa.set_xlim(0, 1.13)
    axa.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axa.set_xlabel("model fit $R^2$  (camera motion only, no machine deflection)",
                   fontsize=9, color=INK2)
    axa.set_title("(a) What the structural model does to pure camera motion",
                  fontsize=10.5, color=INK, loc="left", pad=10)

    # stagger the direct labels: roll and dv/dx both converge onto 1.0 and
    # would otherwise print on top of each other
    dy = {"roll": 9, "shear $\\partial v/\\partial x$": -11, "zoom": 0}
    for (name, row), c in zip(conv.items(), (BLUE, AQUA, ORANGE)):
        axb.plot(counts, row, "o-", color=c, linewidth=2.0, markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
                 label=name)
        axb.annotate(name, (counts[-1], row[-1]), textcoords="offset points",
                     xytext=(10, dy.get(name, 0)), fontsize=8.5, color=c,
                     va="center")
    axb.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axb.text(4.2, 1.005, "perfect fit", fontsize=8, color=MUTED, va="bottom")
    axb.set_xticks(counts)
    axb.set_xlim(3.4, 34)
    axb.set_ylim(0.75, 1.03)
    axb.set_xlabel("axial stations in the fit", fontsize=9, color=INK2)
    axb.set_ylabel("model fit $R^2$", fontsize=9, color=INK2)
    axb.set_title("(b) Refining the fit: which shortfalls are real",
                  fontsize=10.5, color=INK, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out / "fig1_identifiability.png", dpi=200, facecolor=SURFACE)
    print(f"  fig1: camera fields {min(v for _, v in cam):.4f}–"
          f"{max(v for _, v in cam):.4f}, real bending {bend:.4f}")
    return {"camera": dict(cam), "bending": bend,
            "convergence": {k: v for k, v in conv.items()},
            "station_counts": list(counts)}


# ---------------------------------------------------------------------------
# Figure 2 — controlled experiment
# ---------------------------------------------------------------------------
def fig2(out: Path, val_json: Path):
    d = json.loads(val_json.read_text())
    rows = d["rows"]
    modes = ("none", "mean", "affine")
    cols = {"none": ORANGE, "mean": BLUE, "affine": AQUA}
    label = {"none": "no compensation", "mean": "mean only",
             "affine": "affine (proposed)"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor=SURFACE)
    _style(axa)
    _style(axb)

    for m in modes:
        sel = sorted([r for r in rows if r["detrend"] == m],
                     key=lambda r: r["cam_px"])
        axa.plot([r["cam_px"] for r in sel], [r["mac"] for r in sel], "o-",
                 color=cols[m], linewidth=2.0, markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3,
                 label=label[m])
    axa.axhline(0.9, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    axa.text(0.06, 0.905, "MAC 0.9", fontsize=8, color=MUTED, va="bottom")
    axa.set_xlabel("injected whole-scene camera motion (px)", fontsize=9,
                   color=INK2)
    axa.set_ylabel("MAC against the injected shape  (1 = correct)",
                   fontsize=9, color=INK2)
    axa.set_ylim(0, 1.05)
    axa.legend(fontsize=8.5, frameon=False, labelcolor=INK, loc="lower left")
    axa.set_title("(a) What unremoved scene motion costs",
                  fontsize=10.5, color=INK, loc="left", pad=10)

    r2 = np.array([r["r2"] for r in rows])
    mac = np.array([r["mac"] for r in rows])
    rr = float(np.corrcoef(r2, mac)[0, 1])
    for m in modes:
        sel = [r for r in rows if r["detrend"] == m]
        axb.scatter([r["r2"] for r in sel], [r["mac"] for r in sel], s=64,
                    color=cols[m], zorder=3, edgecolor=SURFACE, linewidth=1.6,
                    label=label[m])
    worst = min(rows, key=lambda r: r["mac"])
    best = max((r for r in rows if r["detrend"] == "affine"),
               key=lambda r: r["mac"] if r["cam_px"] > 3 else -1)
    # both callouts point inward, away from the title and the axis label
    for r, txt, off, ha in ((worst, "shape destroyed", (-26, 46), "right"),
                            (best, "shape nearly right", (18, -62), "left")):
        axb.annotate(f"{txt}\nMAC {r['mac']:.3f} · $R^2$ {r['r2']:.3f}",
                     (r["r2"], r["mac"]), textcoords="offset points",
                     xytext=off, fontsize=8.5, ha=ha,
                     color=INK, linespacing=1.35,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    axb.set_xlabel("model fit $R^2$  (what the pipeline reports)", fontsize=9,
                   color=INK2)
    axb.set_ylabel("MAC against truth  (whether it is right)", fontsize=9,
                   color=INK2)
    axb.set_ylim(0, 1.05)
    axb.legend(fontsize=8.5, frameon=False, labelcolor=INK, loc="lower left")
    axb.set_title(f"(b) The reported statistic vs the truth   $r = {rr:+.3f}$",
                  fontsize=10.5, color=INK, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out / "fig2_controlled.png", dpi=200, facecolor=SURFACE)
    print(f"  fig2: {len(rows)} runs, corr(R2, MAC) = {rr:+.3f}")
    return {"n_runs": len(rows), "corr_r2_mac": rr}


# ---------------------------------------------------------------------------
# Figure 3 — field case
# ---------------------------------------------------------------------------
REGIONS = {                     # rotated frame, (x0, y0, x1, y1), y down
    "motor casing": ((135, 25, 330, 330), "machine"),
    "gearbox": ((330, 20, 475, 320), "machine"),
    "blue pulley": ((495, 255, 655, 515), "machine"),
    "wall": ((430, 0, 560, 70), "static"),
    "floor": ((560, 205, 690, 330), "static"),
}
REF3 = [(0, 400, 260, 600), (600, 0, 690, 200), (0, 0, 90, 175)]
# compensation quality sweep -> (held-out static residual px, best-fit R2),
# as measured in docs/ods_validation/README.md
SWEEP = [("one corner box", 3.96, 0.975), ("plate only", 0.965, 0.730),
         ("plate + top-right", 0.749, 0.534), ("plate + both tops", 0.330, 0.353)]


def measure_fig3(video: str):
    """Ten optical-flow passes over the clip — the expensive part of fig3."""
    ods = _load("ods_from_motion_vectors")
    meas, f0 = {}, None
    for detr in ("none", "affine"):
        for name, (roi, kind) in REGIONS.items():
            mv = ods.MotionVectors.from_video(
                video, grid=(16, 10), max_frames=180, roi=roi,
                ref_roi=REF3 if detr == "affine" else None, rotate=90,
                flow_mode="direct", detrend=detr)
            fk, U, V, _, _ = ods.spectral_ods(mv, freq=f0)
            f0 = f0 or float(fk)
            amp = float(np.sqrt(np.abs(U) ** 2 + np.abs(V) ** 2).mean())
            meas.setdefault(name, {"kind": kind})[detr] = amp
        print(f"    fig3 {detr}: " + ", ".join(
            f"{n} {meas[n][detr]:.3f}" for n in REGIONS))
    return f0, meas


def fig3(out: Path, f0: float, meas: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.2), facecolor=SURFACE)
    _style(axa)
    _style(axb)

    names = list(REGIONS)
    xs = np.arange(len(names))
    raw = [meas[n]["none"] for n in names]
    comp = [meas[n]["affine"] for n in names]
    axa.bar(xs - 0.2, raw, 0.38, color=ORANGE, zorder=3, edgecolor=SURFACE,
            linewidth=1.6, label="raw field")
    axa.bar(xs + 0.2, comp, 0.38, color=BLUE, zorder=3, edgecolor=SURFACE,
            linewidth=1.6, label="after affine compensation")
    for xi, (a, b) in enumerate(zip(raw, comp)):
        axa.text(xi - 0.2, a + 0.03, f"{a:.2f}", ha="center", fontsize=8,
                 color=INK)
        axa.text(xi + 0.2, b + 0.03, f"{b:.2f}", ha="center", fontsize=8,
                 color=INK)
    axa.set_xticks(xs)
    axa.set_xticklabels(
        [n + ("\n(static)" if REGIONS[n][1] == "static" else "")
         for n in names], fontsize=8.5, color=INK)
    for lbl, n in zip(axa.get_xticklabels(), names):
        if REGIONS[n][1] == "static":
            lbl.set_color(ORANGE)
    axa.set_ylabel(f"{f0:.2f} Hz amplitude (px)", fontsize=9, color=INK2)
    axa.set_ylim(0, max(raw) * 1.18)
    axa.legend(fontsize=8.5, frameon=False, labelcolor=INK)
    # Careful with this claim: in the RAW field only the wall out-moves the
    # casing (the floor does not). What holds for every static region is the
    # post-compensation comparison, so that is what the title asserts.
    axa.set_title("(a) After compensation, static regions still out-move "
                  "the machine", fontsize=10.5, color=INK, loc="left", pad=10)

    resid = [s[1] for s in SWEEP]
    r2s = [s[2] for s in SWEEP]
    axb.plot(resid, r2s, "o-", color=BLUE, linewidth=2.0, markersize=8,
             markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    for (lbl, xr, yr), off in zip(SWEEP, ((6, 12), (6, -16), (6, -16), (8, 12))):
        axb.annotate(lbl, (xr, yr), textcoords="offset points", xytext=off,
                     fontsize=8.5, color=INK,
                     ha="right" if off[0] < 0 else "left")
    axb.set_xscale("log")
    # default log ticks collide at this range; label the measured points
    axb.set_xticks([s[1] for s in SWEEP])
    axb.set_xticklabels([f"{s[1]:.2f}" for s in SWEEP], fontsize=8.5)
    axb.minorticks_off()
    axb.set_xlim(0.26, 6.2)
    axb.set_xlabel("residual on a held-out static region (px)  →  worse",
                   fontsize=9, color=INK2)
    axb.set_ylabel("model fit $R^2$", fontsize=9, color=INK2)
    axb.set_ylim(0.25, 1.12)
    # arrow parallel to the trend but offset above it, so it reads as a
    # gloss on the line rather than another data series
    axb.annotate("", xy=(3.3, 1.03), xytext=(0.42, 0.62),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8,
                                 alpha=0.85))
    axb.text(1.05, 0.90, "the worse the answer,\nthe better the fit",
             fontsize=9, color=ORANGE, ha="center", va="center",
             linespacing=1.35)
    axb.set_title("(b) The inversion, on real data", fontsize=10.5, color=INK,
                  loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(out / "fig3_field.png", dpi=200, facecolor=SURFACE)
    return {"freq": f0, "regions": meas, "sweep": SWEEP}


def fig3_entry(out: Path, video: str | None, prev: dict):
    """Measure if a clip is given, otherwise redraw from the cached numbers.

    Layout fixes to this figure should not cost ten more optical-flow passes.
    """
    cached = prev.get("fig3")
    if video:
        f0, meas = measure_fig3(video)
    elif cached:
        f0, meas = cached["freq"], cached["regions"]
        print(f"    fig3: redrawing from cached measurements ({f0:.2f} Hz)")
    else:
        print("  fig3: skipped (no --video and no cached measurements)")
        return None
    return fig3(out, f0, meas)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/figs")
    ap.add_argument("--video", default=None,
                    help="motor clip; fig3 is skipped without it")
    ap.add_argument("--validation", default="docs/ods_validation/validation.json")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset (fig1,fig2,fig3) to redraw; "
                         "the rest of figures.json is carried over, so a "
                         "label tweak does not re-run fig3's optical flow")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only.split(",")) if args.only else {"fig1", "fig2", "fig3"}

    prev = {}
    fj = out / "figures.json"
    if fj.exists():
        prev = json.loads(fj.read_text())
    summary = dict(prev)
    if "fig1" in want:
        summary["fig1"] = fig1(out)
    if "fig2" in want:
        summary["fig2"] = fig2(out, Path(args.validation))
    if "fig3" in want:
        got = fig3_entry(out, args.video, prev)
        if got:
            summary["fig3"] = got
    fj.write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
