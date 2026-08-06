"""Put every ROI strategy on one cost/fidelity plane.

Reads a clip's roi_segment_benchmark metrics.json (full / bbox / segment)
and its lod_magnify lod_metrics.json (graded-resolution sweep) and plots
speedup against how much of the full-frame motion each strategy actually
reproduces.

A hard ROI reproduces the motion perfectly inside its box and not at all
outside, so its whole-frame motion correlation is
    r_whole ~= r_in * sqrt(captured_energy),
which is the number plotted for bbox/segment; the LOD runs measure their
whole-frame correlation directly.

Usage: python tools/roi_pareto.py docs/roi_segment/motor docs/roi_segment/motor_lod out.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BG, ACCENT, TEXT, FAINT = "#1f1e1c", "#d97757", "#a8a49e", "#7d7a74"
GREEN, BLUE, RED = "#a3c95a", "#5aa9d6", "#e05a5a"


def main(argv):
    roi_dir, lod_dir, out_png = Path(argv[0]), Path(argv[1]), Path(argv[2])
    roi = json.loads((roi_dir / "metrics.json").read_text())
    lod = json.loads((lod_dir / "lod_metrics.json").read_text())

    pts = []
    for row in roi["rows"]:
        r_whole = (1.0 if row["mode"] == "full"
                   else row["r_motion"] * np.sqrt(row["captured"]))
        pts.append((row["mode"], row["speedup"], float(r_whole)))
    for w in lod["sweep"]:
        pts.append((f"LOD 1/{w['coarse']}", w["speedup"], w["r_all"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(FAINT)
    style = {"full": (TEXT, "s"), "bbox": (BLUE, "o"), "segment": (GREEN, "^")}
    for name, sp_, r in pts:
        c, m = style.get(name, (ACCENT, "D"))
        ax.scatter(sp_, r, s=90, color=c, marker=m, zorder=3,
                   edgecolor=BG, linewidth=1.2)
        ax.annotate(name, (sp_, r), textcoords="offset points",
                    xytext=(8, -4), color=TEXT, fontsize=9)
    lodp = sorted([p for p in pts if p[0].startswith("LOD")], key=lambda p: p[1])
    ax.plot([p[1] for p in lodp], [p[2] for p in lodp], "-", color=ACCENT,
            alpha=0.6, zorder=2, label="LOD sweep (far-field 1/s)")
    ax.set_xlabel("speedup vs full frame  →  cheaper", color=TEXT, fontsize=10)
    ax.set_ylabel("whole-frame motion reproduced (correlation)", color=TEXT,
                  fontsize=10)
    ax.set_title(f"{roi['label']}: ROI strategies on the cost/fidelity plane\n"
                 f"({roi['size'][0]}x{roi['size'][1]}, {roi['frames']} fr, "
                 f"{roi['band'][0]}-{roi['band'][1]} Hz)", color=TEXT, fontsize=11)
    ax.grid(alpha=0.15, color=FAINT)
    ax.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, edgecolor=FAINT)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, facecolor=BG)
    for name, sp_, r in sorted(pts, key=lambda p: -p[1]):
        print(f"  {name:10s} speedup x{sp_:5.2f}   motion reproduced {r:.3f}")
    print(f"wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
