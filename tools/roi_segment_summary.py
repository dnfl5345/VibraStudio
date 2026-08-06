"""Collect roi_segment_benchmark runs into one table and figure.

Usage: python tools/roi_segment_summary.py docs/roi_segment/*/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BG, ACCENT, TEXT, FAINT = "#1f1e1c", "#d97757", "#a8a49e", "#7d7a74"
GREEN, BLUE = "#a3c95a", "#5aa9d6"


def main(argv):
    runs = [json.loads(Path(p).read_text()) for p in argv]
    if not runs:
        print("no metrics.json given")
        return 1

    lines = ["| clip | mode | time (s) | speedup | Mpx | ROI area | captured motion | r (in-ROI) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        for row in r["rows"]:
            lines.append(
                f"| {r['label']} ({r['size'][0]}x{r['size'][1]}, {r['frames']}fr,"
                f" {r['band'][0]}-{r['band'][1]} Hz) | {row['mode']} |"
                f" {row['t_total']:.1f} | x{row['speedup']:.2f} |"
                f" {row['mpx']:.1f} | {100*row['area']:.1f}% |"
                f" {100*row['captured']:.1f}% |"
                f" {row['r_motion']:.4f} |")
    table = "\n".join(lines)
    out = Path("docs/roi_segment/README.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# ROI selection strategies for phase-based motion magnification\n\n"
        "`full` magnifies every pixel; `bbox` keeps the largest band-power "
        "hotspot rectangle; `segment` grows SLIC superpixels from motion seeds "
        "into colour-similar neighbours and magnifies one crop per component.\n\n"
        "**captured motion** = share of the full-frame magnified-motion energy "
        "that falls inside the ROI; the remainder is motion the ROI drops.\n\n"
        + table + "\n")
    print(table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 4.2),
                             facecolor=BG, squeeze=False)
    for ax, r in zip(axes[0], runs):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(FAINT)
        names = [w["mode"] for w in r["rows"]]
        xs = np.arange(len(names))
        t = [w["t_total"] for w in r["rows"]]
        b1 = ax.bar(xs - 0.2, t, 0.4, color=ACCENT, label="wall-clock (s)")
        ax.bar_label(b1, fmt="%.0f", color=TEXT, fontsize=8)
        ax.set_ylabel("wall-clock (s)", color=TEXT, fontsize=9)
        ax2 = ax.twinx()
        b2 = ax2.bar(xs + 0.2, [100 * w["captured"] for w in r["rows"]], 0.4,
                     color=GREEN, label="captured motion (%)")
        ax2.bar_label(b2, fmt="%.0f%%", color=TEXT, fontsize=8)
        ax2.set_ylim(0, 118)
        ax2.set_ylabel("captured motion energy (%)", color=TEXT, fontsize=9)
        ax2.tick_params(colors=TEXT, labelsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, color=TEXT)
        ax.set_title(f"{r['label']} — {r['size'][0]}x{r['size'][1]}, "
                     f"{r['frames']} fr, {r['band'][0]}-{r['band'][1]} Hz",
                     color=TEXT, fontsize=10)
        ax.legend(fontsize=7, loc="upper center", facecolor=BG, labelcolor=TEXT,
                  edgecolor=FAINT)
    fig.tight_layout()
    fig.savefig("docs/roi_segment/roi_modes_summary.png", dpi=110, facecolor=BG)
    print("wrote docs/roi_segment/README.md, docs/roi_segment/roi_modes_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
