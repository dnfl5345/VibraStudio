# ROI selection strategies for phase-based motion magnification

`full` magnifies every pixel; `bbox` keeps the largest band-power hotspot rectangle; `segment` grows SLIC superpixels from motion seeds into colour-similar neighbours and magnifies one crop per component.

**captured motion** = share of the full-frame magnified-motion energy that falls inside the ROI; the remainder is motion the ROI drops.

| clip | mode | time (s) | speedup | Mpx | ROI area | captured motion | r (in-ROI) |
|---|---|---|---|---|---|---|---|
| motor (600x690, 120fr, 5.0-7.0 Hz) | full | 302.5 | x1.00 | 49.7 | 100.0% | 100.0% | 1.0000 |
| motor (600x690, 120fr, 5.0-7.0 Hz) | bbox | 65.2 | x4.64 | 14.6 | 21.7% | 40.7% | 0.9739 |
| motor (600x690, 120fr, 5.0-7.0 Hz) | segment | 278.9 | x1.08 | 40.7 | 35.9% | 63.2% | 0.9590 |
| hands (640x480, 120fr, 0.3-1.0 Hz) | full | 210.1 | x1.00 | 36.9 | 100.0% | 100.0% | 1.0000 |
| hands (640x480, 120fr, 0.3-1.0 Hz) | bbox | 145.1 | x1.45 | 27.0 | 57.9% | 96.0% | 0.9992 |
| hands (640x480, 120fr, 0.3-1.0 Hz) | segment | 164.9 | x1.27 | 27.2 | 35.8% | 94.1% | 0.9930 |

## What the numbers say

**Segmentation fixes coverage, not cost.** On the motor clip the hotspot
rectangle magnifies only 40.7% of the moving content — it sits on the
textured cooling fins and drops the smooth flange and shaft that move with
them. Growing SLIC superpixels from those seeds into colour-similar
neighbours pulls the whole casing in and raises captured motion to 63.2%.

**Cost follows the bounding box, not the mask.** A steerable pyramid needs
a rectangle, so a diagonal object pays for the empty corners of its box:
the motor mask covers 35.9% of the frame but its crop is 81.9% of the
pixels (fill ratio 0.44), which collapses the speedup from x4.64 to x1.08.
The hands clip shows the same ratio (mask 35.8%, crop 73.7%, fill 0.49).

**When the object already fills the frame, no ROI scheme saves much.** On
the hands clip the hotspot box alone already captures 96.0% of the motion,
and segmentation costs more pixels for slightly less coverage — bbox is
strictly better there. Tiling the mask does not rescue this at these frame
sizes: with a 32 px anti-ringing pad, 192 px tiles carry ~1.8x overhead, so
the tiled cost exceeds the single box for objects this large.

**Fidelity is never the problem.** Both ROI modes reproduce the full-frame
result inside the ROI to r >= 0.96 (bbox 0.974 / 0.999, segment 0.959 /
0.993), so the choice is purely coverage vs cost.

Practical reading: use segmentation when the goal is *selectivity* —
suppressing background magnification, or restricting ODS analysis to the
structure — and reach for a cheaper transform (Riesz pyramid, band-limited
temporal filtering, pyramid-level pruning) when the goal is speed. ROI
cropping only pays as a speed measure when the object is small relative to
the frame and its bounding box is well filled.

## Prior art

Selective, segmentation-driven magnification is not new. *Interest Region
Based Motion Magnification* (Springer, 2017) extracts the object with
kernel K-means plus superpixel scribbles and alpha matting before Eulerian
magnification, motivated by noise rejection and user focus rather than
compute. In the vibration domain, vibration-guided superpixel segmentation
has been used to place virtual sensor grids, and notes that ROI extraction
lowers computational complexity and improves robustness. What is less
common in that literature is a head-to-head cost/coverage benchmark of the
selection strategies themselves, which is what this study measures.
