# RETRACTED — motor 6 Hz restoration

The animation and overlay in this directory were produced before the
scene-motion checks existed, and they do not show the machine's deflection
shape. Keep them only as the "before" side of the comparison.

Four things were wrong:

* the clip is stored rotated 90° CW and was analysed sideways (no `--rotate`);
* the machine axis was hand-set to +15.3° in the stored frame when the data
  puts it at −76.2°, so axial and transverse were swapped;
* motion vectors came from accumulated frame-to-frame flow, which drifts
  0.616 px median / 4.442 px p90 over 120 frames against a ~2.5 px signal;
* most importantly, **83% of the 6 Hz band energy is a whole-scene field that
  moves the static wall and floor more than it moves the machine.** The
  `--ref-roi` in use subtracted only that field's mean, leaving its gradient
  untouched.

With all four corrected, the machine retains 0.295 px at 6.05 Hz while a
held-out static region retains 0.714 px — a ratio of 0.41×. The tool now
reports this and refuses to call the result an ODS:

```
[1c] scene-motion sanity check at 6.050 Hz: machine 0.295 px vs held-out
     static region 0.714 px  (ratio 0.41x)
     WARNING: ... treat it as unresolved, not as an ODS.
```

The corrected run is in `../motor_6hz_checked/`. It is a **negative result**:
this clip does not support a deflection shape at 6 Hz.

Any figure quoting a drive-end → free-end amplitude growth for this rig
(previously "0.85 px → 2.4 px vertical bending") should be discarded. That
profile was the camera's gradient, and the 2.4 px end of it was additionally
inflated by flow drift.

Full analysis: [`../../ods_validation/`](../../ods_validation/).
