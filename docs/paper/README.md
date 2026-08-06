# Can this be written up? An honest assessment

Short answer: **not as "3D ODS restoration from single-view motion vectors."**
That framing is neither novel enough nor supported by the current results. But
one piece of the work is publishable, and it is the piece that came out of
the failure.

## What the literature already has

| claim we might make | already published |
|---|---|
| Extract ODS from video optical flow | Machynia et al., *Sensors* 2021 — ODS straight from optical flow with automatic masking and frame-wise fusion, validated on cantilever beams |
| Visualise/magnify ODS with phase-based methods | Chen et al. (MIT) motion-magnification modal identification; phase-based ODS magnification to ×400 |
| Full-field **3D** ODS from cameras | Gorjup, Slavič, Boltežar — frequency-domain triangulation; multiview Spectral Optical Flow Imaging. Multiple views are used *because* one 2D view cannot resolve optical-axis motion |
| ODS of rotating machinery from high-speed video | rotational/order-based ODS analysis with high-speed cameras |
| Remove camera motion with static reference regions | standard in vision-based SHM: affine/homography estimated on reference templates; camera ego-motion is known to be able to dominate the measured vibration |
| Commercial ODS-from-video | RDI Iris M, gfaitech WaveCam, etestinc |

So: the pipeline, the geometry template, and the camera-motion fix are all
prior art. Building them is useful engineering for this repository. It is not
a paper.

## What is not in the literature, as far as this survey found

The searches did not turn up a paper making this specific point:

> When a single-view video ODS is fitted to a **parametric structural model**
> (beam stations, ring ovalization, rocking), the goodness-of-fit of that
> model is *anti-correlated* with the correctness of the result, because
> residual camera motion is a smooth low-order field that the structural
> basis represents better than the true deflection does.

That is a falsifiable, quantitative, and — on this data — strongly supported
claim, and it is a trap that anyone building an ODS-from-video pipeline can
fall into, because R² or fit residual is the obvious thing to report when
there is no reference sensor.

Supporting evidence already in hand (`docs/ods_validation/`):

* On the real motor clip, R² runs 0.975 → 0.353 as scene-motion removal goes
  from bad to good; the held-out static residual runs 3.96 px → 0.330 px over
  the same range. The best answer has the worst R².
* R² climbs monotonically as the assumed machine axis rotates *away* from the
  visible axis (0.833 at −15° → 0.943 at +51°, undetrended).
* A controlled test (`tools/ods_synthetic_validation.py`) with a known
  deflection shape and a known injected camera roll puts the correlation
  between model R² and true MAC at **r = −0.162** over 15 runs. The clearest
  pair: MAC 0.060 (shape destroyed) reports R² 0.752, while MAC 0.920 (shape
  nearly right) reports R² 0.458.
* The same test bounds the damage: ~2 px of unremoved scene motion halves
  shape fidelity (MAC 0.982 → 0.611), ~4 px destroys it (0.268), and affine
  removal holds MAC ≥ 0.92 out to ~4 px. The real clip carries ~2.9 px.
* A separate calibration result worth reporting on its own: with the camera
  perfectly still and MAC 0.982, the pipeline still under-reports amplitude
  by ~43% (gain 0.58).

## What a paper would need that we do not have

1. **A reference sensor.** Two accelerometers are visible in the motor clip;
   their signals are not in this repository. Without them every claim about
   the real clip rests on video alone. This is the single biggest gap.
2. **More than one real rig.** One clip, one frequency, one camera. The
   synthetic sweep generalises the *mechanism*, not the field conditions.
3. **A clip where the ODS is actually recoverable.** The current motor clip
   is a negative result — useful for the argument, useless as a demonstration
   that the restoration works. A tripod-mounted, higher-amplitude,
   global-shutter recording would give the positive case.
4. **A stated detection threshold.** "Machine ≥ 2× held-out static region" is
   currently a heuristic. It needs a derivation or an empirical ROC.

## Recommended shape, if pursued

Target: a short paper or technical note in a measurement/SHM venue.

> **Title.** Goodness-of-fit is not validity: residual camera motion in
> model-based operating deflection shapes from video
>
> 1. Video ODS pipelines increasingly fit measured fields to structural
>    models. Fit quality is the natural self-check when no reference sensor
>    is available.
> 2. Show analytically that a low-order structural basis spans the affine
>    fields produced by camera rotation, so fit quality is not identifiable
>    against ego-motion.
> 3. Controlled experiment: known shape + known injected ego-motion, sweep
>    amplitude, report MAC and gain against truth alongside R².
> 4. Field case: the motor clip, showing the anti-correlation on real data
>    and the negative result it implies.
> 5. Propose the held-out-static-region test as the check to report instead,
>    with the threshold characterised on the controlled data.

That is honest, it is supported by what has been measured, and steps 3–5 are
already implemented here. Steps 1–2 are writing; the accelerometer data is
what would make step 4 defensible.

**What should not be claimed:** that this work restores 3D ODS from a single
view better than existing methods, or that the motor animation shows the
machine's deflection shape. On the evidence, it does not.
