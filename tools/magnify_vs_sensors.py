"""Motion magnification of a test video, checked against the accelerometers.

Thin command-line wrapper around vibrastudio.dsp.magnify (same as
``VibraStudio.exe --magnify``). See that module for the method.

  python tools/magnify_vs_sensors.py --video D:/20261102_3D_ODS_Analysis/2026-11-02_02_nomoving.mp4
      --recording dist/VibraStudio/recordings/trig_20261102_180614_007.npz
      --rig rigs/cantilever_8triax.json --mains 60 --out D:/20261102_3D_ODS_Analysis/analysis/magnify
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibrastudio.__main__ import _magnify  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_magnify(sys.argv[1:]))
