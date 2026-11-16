"""Self-check of the 3D ODS video math on synthetic data (no hardware, no Qt).

Two nodes move at 20 Hz along known global directions; a sensor is mounted
turned (printed axes map to -X/-Z/-Y like the cantilever rig). The check
asserts that shapes come back in global axes with the right amplitude and
phase, that time integration recovers the displacement, and that a short
video renders.

    .venv312\\Scripts\\python.exe tools\\check_ods_video.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vibrastudio.config import ChannelConfig                      # noqa: E402
from vibrastudio.dsp import ods_video as ov                         # noqa: E402
from vibrastudio.geometry import Geometry                           # noqa: E402

fs, T, f0 = 2048.0, 8.0, 20.17          # between FFT bins on purpose
t = np.arange(int(fs * T)) / fs
w = 2 * np.pi * f0
d_mm = {1: np.array([0.0, 0.10, 0.0]), 2: np.array([0.02, -0.25, 0.0])}   # global mm amplitude per node
# acceleration (g) = -w^2 d
acc = {n: -(w ** 2) * v[:, None] * 1e-3 / ov.G * np.sin(w * t)[None] for n, v in d_mm.items()}

geo = Geometry("test", np.array([[0, 0, 0], [0, 0, 0.25], [0, 0, 0.5]], float), [(0, 1), (1, 2)],
               {"shape": "rig"}, fixed=[0], node_labels=["Clamp", "A", "B"])
M = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float).T   # printed x->-X, y->-Z, z->-Y
chans, rows = [], []
for n in (1, 2):
    printed = M.T @ acc[n]                                   # what the sensor reads
    for j, ax in enumerate("XYZ"):
        rows.append(printed[j])
        chans.append(ChannelConfig(f"file/ch{len(rows) - 1}", f"N{n}_{ax}", point=n + 1,
                                   axis_vec=list(M[:, j]), sensor_axis=ax))
data = np.array(rows) + 1e-4 * np.random.default_rng(0).standard_normal((len(rows), t.size))

model = ov.model_from(geo, chans, data.shape[0])
node_acc = ov.node_acceleration(data, model)
ods = ov.SpectralODS(node_acc, fs, 0.5)
shape, fb, coh, ref = ods.shape(f0)
ok = True
for n in (1, 2):
    got = np.abs(shape[n]); want = np.abs(d_mm[n])
    err = np.max(np.abs(got - want))
    print(f"node {n}: |shape| {np.round(got, 4)} mm, expected {want} (err {err:.4f})")
    ok &= err < 0.02 * np.abs(d_mm[2]).max() + 1e-3
rel = shape[2, 1] / shape[1, 1]
print(f"phase node2/node1 Y: {np.degrees(np.angle(rel)):+.1f} deg (expected 180), f {fb:.3f} Hz (true {f0}), coh {coh:.3f}")
ok &= abs(fb - f0) < 0.02
ok &= abs(abs(np.degrees(np.angle(rel))) - 180) < 2

disp = ov.integrate_displacement(node_acc, fs, (3, 3, 3), 400)
mid = slice(int(fs), int(fs * (T - 1)))
amp = disp[2, 1, mid].std() * np.sqrt(2)
print(f"time integration: node 2 Y amplitude {amp:.4f} mm (expected 0.25)")
ok &= abs(amp - 0.25) < 0.005

try:
    from vibrastudio.ui.ods_video_render import Strip, write_video
    frames = ov.tone_frames(model, [dict(f=fb, shape=shape, label="check", note="synthetic")], 10, 1.0)
    band = (ods.freq > 1) & (ods.freq < 100)
    out = Path(tempfile.gettempdir()) / "vibrastudio_ods_check.mp4"
    res = write_video(frames, model, Strip("spectrum", ods.freq[band], ods.velocity_spectrum()[:, band]),
                      out, (640, 360), 10)
    print(f"render: {res} ({res.stat().st_size} bytes)")
    ok &= res.stat().st_size > 1000
except Exception as exc:  # noqa: BLE001
    print(f"render FAILED: {exc}")
    ok = False

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
