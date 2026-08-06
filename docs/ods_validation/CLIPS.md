# Finding a clip that can be validated without an accelerometer

`docs/paper/README.md` names a reference sensor as the biggest gap. This is
the survey of what can be done without one, and of which clips are reachable.

## Validating without a sensor

Four internal-consistency checks, implemented in
`tools/ods_sensorless_validation.py`. Each asks a question a real deflection
answers differently from residual camera motion.

| check | what it rules out | cost |
|---|---|---|
| **static** | scene motion, directly | one clip |
| **split-half** | non-stationarity, flow noise | one clip |
| **repeat** | camera artefacts, near-conclusively | **two recordings** |
| **order** | anything that does not know the shaft speed | one clip + known rpm |

**repeat** is the one worth changing how you record for. Restore the shape
independently from two *separate recordings* of the same machine state and
correlate them. Camera motion is uncorrelated between recordings — a tripod
nudged at a different moment, a different phase of building sway — while the
machine's deflection shape is not. A high MAC across independent recordings
is evidence no camera artefact easily produces.

**order** is nearly free for rotating machinery. Response concentrates at
integer multiples of running speed; 1800 rpm means exactly 30.0 Hz. Building
sway, hand shake, and rolling shutter do not know the shaft speed, so a peak
landing on 1× is ground truth that costs nothing but writing the rpm down.

**split-half is necessary, not sufficient** — see below.

## What the screening found

```
python tools/ods_sensorless_validation.py screen *.mp4
```

| clip | size | fps | frames | f₀ | active/static | split-half MAC | verdict |
|---|---|---|---|---|---|---|---|
| ods_10.0Hz.mp4 | 320×240 | 30 | 59 | 0.51 Hz | n/a | 0.827 | REJECT (no reproduction) |
| random_real_video_clip.mp4 | 640×480 | 20 | 23 | 1.74 Hz | 1.98× | — | REJECT (not separable) |
| node_link_test_input_small.mp4 | 320×240 | 20 | 11 | 1.82 Hz | 1.25× | — | REJECT (not separable) |
| weld_hand_dual_…120310.mp4 | 640×480 | 15 | 240 | 0.56 Hz | 2.36× | 0.012 | REJECT (no reproduction) |
| weld_hand_dual_…122220.mp4 | 640×480 | 15 | 240 | 0.62 Hz | — | 0.283 | REJECT (no reproduction) |
| **test_vib_6hz.mp4** (hand-picked regions) | 690×600 | 30 | 119 | 6.05 Hz | **0.44×** | 1.000 | **REJECT (not separable)** |

Ratios are measured after the affine scene-motion removal — see below for why
that matters. `ods_10.0Hz.mp4` reports no ratio because its active region
leaves no held-out cells; the split-half result decides it anyway. The
`…122220` ratio is omitted rather than guessed: its screening run had not
finished, and split-half already rejects it.

**Nothing reachable passes.** Every clip fails on the scene-motion ratio, on
reproduction within its own recording, or on being too short.

### Two ways this screening lied, and what fixed them

The motor clip is the control: `--check-roi` in the main pipeline already
rejects it at 0.41×. The first version of this screening tool passed it at
2.93× with a **perfect split-half MAC of 1.000**. Two separate causes.

**The ratio was measured on the raw field.** `static_ratio` compared band
amplitudes *before* the affine scene-motion removal that the restoration
itself applies. On a clip whose dominant field is a smooth camera gradient,
that compares the hot end of the gradient against the cold end — it describes
the gradient, not the machine. Measuring both regions *after* the same
detrend brings the screening into agreement with the pipeline:

| measured | motor ROI | held-out static | ratio | verdict |
|---|---|---|---|---|
| raw field (wrong) | — | — | 2.00× | USABLE |
| after affine detrend | 0.30 px | 0.68 px | **0.44×** | **REJECT** |

against the main pipeline's independent 0.41×. Same conclusion, arrived at
separately.

**Auto regions pick the ends of the gradient, not machine vs background.**
The routine takes the most-active cells as the structure and the least-active
as static. When the dominant field *is* a gradient, those are its two ends.
The tool now reports `UNCONFIRMED (auto regions; give --roi/--ref-roi to
decide)` rather than `USABLE` whenever regions were guessed — automatic
screening is triage, not a verdict.

**Split-half stays at 1.000 either way, and that is the point.** Both halves
share the same camera and the same steady disturbance, so a purely
camera-driven "shape" reproduces perfectly. MAC 1.000 is not a strong result
here; it is the signature of something *too* stationary. Split-half cannot
distinguish a steady camera from a steady machine, which is precisely why the
**repeat** check — two separate recordings, uncorrelated camera motion — is
the one worth changing how you record for.

## Clips that would settle it — not reachable from here

Drive holds `1800rpm_1600fps_*.mp4`, which is the right material:

| file | why |
|---|---|
| `R1_1`, `R1_2`, `R1_3`, `R2_1`, `R9_1` | independent repeat recordings → the **repeat** check |
| `R7_boltinghard` / `R7_boltingsoft` | controlled change of mount stiffness → paired contrast |
| 1800 rpm | exactly 30.0 Hz → the **order** check |
| 1600 fps | no rolling shutter at these speeds, ample order resolution |

They are ~175 MB each. The Drive MCP download caps at 10 MB and
`drive.google.com` is refused by this environment's network policy (403 on
CONNECT), so they cannot be fetched here.

`random_real_video_clip.mp4` is, despite its name, a monochrome high-speed
frame of that same rig — plain dark background, side view, horizontal axis,
two accelerometers on top, rigid stand. It is the ideal subject for the
cylinder template. Only 23 usable frames survive in this excerpt.

**To unblock:** a 2-second trim of any repeat pair is enough — at 1600 fps
that is 3200 frames, far more than the 240 the pipeline uses.

```bash
ffmpeg -i 1800rpm_1600fps_R1_1.mp4 -t 2 -c copy R1_1_trim.mp4   # well under 10 MB
ffmpeg -i 1800rpm_1600fps_R1_2.mp4 -t 2 -c copy R1_2_trim.mp4
```

Then:

```bash
python tools/ods_sensorless_validation.py pair R1_1_trim.mp4 R1_2_trim.mp4 \
    --roi <machine box> --ref-roi "<static box>;<static box>" --rpm 1800
```

## Not usable

* `ods_10.0Hz.mp4`, `node_link_test_input_small.mp4` — an outdoor condenser
  unit; the first already has a node-link overlay rendered into it, so it is
  an *output*, not an input.
* `weld_hand_dual_*` — 15 fps caps the analysis at 7.5 Hz, below the 8–12 Hz
  band where hand tremor lives, and both clips fail split-half badly (MAC
  0.012 / 0.283). They are a genuine independent-repeat *pair* of the same
  setup, which is the right experimental design — the frame rate is what
  disqualifies them.
* `VIB_IB_H/V/A`, `VIB_Dyno_NDE_H` on Drive are real vibration channels but
  exist only as PNG plots, from a dyno campaign with no matching video.
