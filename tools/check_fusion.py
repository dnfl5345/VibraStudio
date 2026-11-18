r"""Self-check of dsp.fusion on a synthetic cantilever (no hardware, no video).

Run: .venv312\Scripts\python.exe tools\check_fusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vibrastudio.dsp import fusion as fu  # noqa: E402

rng = np.random.default_rng(1)
L = 50.0
zs = np.array([0.7, 7.3, 13.6, 20.7, 28.0, 35.1, 42.2, 49.0])
zv = np.linspace(1.0, 49.0, 40)
q = np.array([140, 15 * np.exp(0.2j), -110 * np.exp(0.05j), 12 * np.exp(1j)])      # um: rigid, rotation, 2 modes


def truth(z):
    return fu.beam_basis(z, L, 2) @ q


def noise(n, s):
    return s * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def err(fit, z):
    return float(np.linalg.norm(fit(z) - truth(z)) / np.linalg.norm(truth(z)))


zt = np.linspace(0, L, 200)
ok = True


def check(name, value, limit):
    global ok
    good = value < limit
    ok &= good
    print(f"{'ok  ' if good else 'FAIL'} {name}: {value:.4f} (limit {limit})")


ys = truth(zs) + noise(8, 0.05)
f8 = fu.fit_sensors(zs, ys, L)
check(f"8 sensors, {f8.n_modes} bending shapes chosen: error over the beam", err(f8, zt), 0.02)
ok &= f8.n_modes == 2

g = 1.07 * np.exp(-2.1j)
cam = fu.beam_basis(zv, L, 0) @ np.array([1.5, 0.8j])
yv = g * truth(zv) + noise(40, 0.4)
two = [1, 7]
s2 = fu.fit_sensors(zs[two], ys[two], L)
free = fu.fit_fusion(zs[two], ys[two], zv, yv, L, None, np.full(40, 0.4))
check("2 sensors + video, gain found from the data", err(free, zt), 0.03)
check("  the gain itself", abs(free.gain / g - 1), 0.03)
print(f"     (2 sensors alone: {err(s2, zt):.3f})")
ok &= err(s2, zt) > 5 * err(free, zt)

known = fu.fit_fusion(zs[two], ys[two], zv, yv + cam, L, None, np.full(40, 0.4), gain=g, camera=True)
check("2 sensors + video with camera shake, known gain", err(known, zt), 0.03)
check("  camera motion recovered", float(np.linalg.norm(known.camera - [1.5, 0.8j]) / 1.7), 0.5)

three = [0, 4, 7]
ph = fu.fit_fusion(zs[three], ys[three], zv, yv + cam, L, None, np.full(40, 0.4), gain_abs=1.07, camera=True)
check("3 sensors + video with camera shake, only |gain| known", err(ph, zt), 0.03)

bad = yv.copy(); bad[[3, 37]] *= 1.6
seed = np.zeros(40, bool); seed[::5] = True
keep = fu.screen_patches(zv, bad, seed, L)
ok &= (not keep[3]) and (not keep[37]) and keep.sum() >= 36
print(f"{'ok  ' if (not keep[3] and not keep[37]) else 'FAIL'} screening drops the two biased patches, keeps {keep.sum()}/40")

# hammer test: clamp at rest, camera sees only the upper two thirds; one sensor at the tip
qc = np.array([0, 0, 0.2, 1.0, 0.1j])
zu = np.linspace(16, 49, 32)
tc = lambda z: fu.beam_basis(z, L, 3) @ qc
part = fu.fit_fusion(zs[[7]], tc(zs[[7]]), zu, 0.6 * tc(zu) + noise(32, 0.004), L, None, np.full(32, 0.004), rigid="none")
low = zs[:3]
check("clamped model, view 16-49 cm + tip sensor: error at the 3 sensors below the view",
      float(np.linalg.norm(part(low) - tc(low)) / np.linalg.norm(tc(low))), 0.1)

# custom shapes (e.g. finite-element modes): same data, basis given as a function
fe = lambda z: fu.beam_basis(z, L, 3)[:, 2:]
cust = fu.fit_fusion(zs[[7]], tc(zs[[7]]), zu, 0.6 * tc(zu) + noise(32, 0.004), L, None, np.full(32, 0.004), basis=fe)
check("custom basis: same test", float(np.linalg.norm(cust(low) - tc(low)) / np.linalg.norm(tc(low))), 0.1)

a = fu.alias(132.3, 230.819)
ok &= abs(a[0] - 98.519) < 1e-3 and a[1] and fu.alias(18.9, 230.819) == (18.9, False)
print(f"alias 132.3 Hz at 230.819 fps -> {a[0]:.3f} Hz, conjugated {a[1]}")
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
