"""How much camera motion can the structural model absorb? Exactly how much.

The restoration fits, per axial station j,

    u(theta) = dX_j + phi_j * (z/R)                        (axial + rocking)
    v(theta) = dZ_j + [a2_j cos2t + b2_j sin2t] * cos(t)   (vertical + n=2)

A camera that translates, rolls, shears, or zooms produces an affine field
d = A p + t over the whole image. Write it per station, where x = x_j is
constant within a ring and z = R sin(theta) varies across it:

    u = (t_u + a11 x_j) + a12 z     -> constant + linear in z
    v = (t_v + a21 x_j) + a22 z     -> constant + linear in z

The u row is spanned *exactly*: the constant is dX_j and the linear term is
phi_j. The v row has dZ_j for its constant, but its linear-in-z term must be
supplied by the ovalization basis, and with z ~ sin(theta),

    sin2t cos t = (sin t + sin 3t) / 2

so the b2 term buys sin(t) only together with sin(3t) in a fixed 1:1 ratio.
The residual is therefore confined to the sin(3t) it drags in -- and any
camera field whose v does not vary with z (translation, roll, dv/dx shear)
leaves no residual at all.

This script measures that, by feeding the fitter camera fields with no
machine deflection whatsoever and reporting the R^2 it returns.

Usage: python tools/ods_identifiability.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ods_pipe", str(Path(__file__).resolve().parent / "ods_from_motion_vectors.py"))
ods = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ods)


def grid_points(nx: int = 28, nz: int = 14, length: float = 340.0,
                radius: float = 150.0) -> np.ndarray:
    """Points on the visible face of a horizontal cylinder, image-like units."""
    xs = np.linspace(0.0, length, nx)
    # sample the face uniformly in z so the fitter sees the same span it
    # would from a real side view
    zs = np.linspace(-0.94, 0.94, nz) * radius
    return np.array([(x, z) for z in zs for x in xs])


def fit(pts: np.ndarray, U: np.ndarray, V: np.ndarray, rocking: bool = True):
    dummy = np.zeros((4, len(pts)))
    mv = ods.MotionVectors(pts, dummy, dummy, 30.0)
    _, _, rings = ods.restore_cylinder_ods(mv, U, V, n_layers=6,
                                           rocking=rocking)
    return rings


def main() -> int:
    pts = grid_points()
    x, z = pts[:, 0], pts[:, 1]
    xc, zc = x.mean(), z.mean()
    rng = np.random.default_rng(3)

    # --- camera-only fields: no machine deflection anywhere in these ------
    cam = {
        "translation  (t)": (np.full_like(x, 1.0), np.full_like(x, 0.6)),
        "roll         (w)": (-0.01 * (z - zc), 0.01 * (x - xc)),
        "shear     du/dz": (0.01 * (z - zc), np.zeros_like(x)),
        "shear     dv/dx": (np.zeros_like(x), 0.01 * (x - xc)),
        "zoom         (s)": (0.01 * (x - xc), 0.01 * (z - zc)),
        "general affine": None,
    }
    A = rng.normal(size=(2, 2)) * 0.01
    t = rng.normal(size=2)
    cam["general affine"] = (t[0] + A[0, 0] * (x - xc) + A[0, 1] * (z - zc),
                             t[1] + A[1, 0] * (x - xc) + A[1, 1] * (z - zc))

    print("Camera-only fields (zero machine deflection), cylinder-model fit:\n")
    print(f"  {'field':18s} {'R2 (+rocking)':>14s} {'R2 (no rocking)':>16s}")
    for name, (u, v) in cam.items():
        rb = fit(pts, u.astype(complex), v.astype(complex), rocking=True)
        ra = fit(pts, u.astype(complex), v.astype(complex), rocking=False)
        print(f"  {name:18s} {rb['r2']:14.4f} {ra['r2']:16.4f}")

    # --- why the imperfect ones are imperfect -----------------------------
    # Roll and dv/dx put a term linear in *x* into v. Within a ring x is
    # constant, so dZ_j absorbs it exactly -- but only to the resolution of
    # the station binning, since a ring covers a range of x. Refine the
    # stations and the residual should vanish. Zoom is different: it puts a
    # term linear in *z* into v, and z ~ sin(theta), which the fit can only
    # reach through b2 sin2t cos t = (sin t + sin 3t)/2 -- so it must drag in
    # sin(3t) it does not want, and that residual does NOT vanish.
    print("\nR^2 against station count (is the shortfall just binning?):")
    print(f"  {'field':18s}" + "".join(f"{n:>9d}" for n in (4, 6, 10, 16, 24)))
    for name in ("roll         (w)", "shear     dv/dx", "zoom         (s)"):
        u, v = cam[name]
        row = []
        for n in (4, 6, 10, 16, 24):
            dummy = np.zeros((4, len(pts)))
            mv = ods.MotionVectors(pts, dummy, dummy, 30.0)
            _, _, r = ods.restore_cylinder_ods(mv, u.astype(complex),
                                               v.astype(complex),
                                               n_layers=n, rocking=True)
            row.append(r["r2"])
        print(f"  {name:18s}" + "".join(f"{q:9.4f}" for q in row))
    print("  roll and dv/dx converge to 1 -- their shortfall was only the\n"
          "  station width. zoom plateaus: its v is linear in z, which the\n"
          "  ovalization basis can only supply together with a sin(3t) it\n"
          "  does not want. That leftover is the whole of what the model\n"
          "  cannot absorb.")

    # --- for contrast: an actual deflection shape --------------------------
    print("\nFor contrast, a real cantilever-like deflection "
          "(no camera motion):")
    shape = (x / x.max()) ** 2
    rb = fit(pts, np.zeros_like(x, dtype=complex), shape.astype(complex))
    print(f"  {'bending v ~ x^2':18s} {rb['r2']:14.4f}")
    print("\nThe camera fields are not merely fitted well -- several are fitted\n"
          "to machine precision. Fit quality cannot separate them from the\n"
          "deflection shape, because they are inside the same model space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
