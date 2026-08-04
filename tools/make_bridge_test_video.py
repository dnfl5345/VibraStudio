"""Render a synthetic bridge video for end-to-end pipeline validation.

A textured girder span between two abutments, plus two piers, vibrating
in its first vertical bending mode (half-sine between supports) with a
weaker second mode — the classic traffic-driven response seen in bridge
motion-amplification footage. The deck layer is warped per column and
composited over a static background, so dense optical flow has real
texture to track and the injected motion is known exactly.

Usage: python tools/make_bridge_test_video.py out.mp4
       (defaults: 640x360, 30 fps, 150 frames, 2.2 Hz, 6 px mid-span)
"""

from __future__ import annotations

import sys

import cv2
import imageio.v2 as imageio
import numpy as np

W, H, FPS, T = 640, 360, 30.0, 150
F1, AMP = 2.2, 6.0                    # Hz, px at mid-span
X0, X1 = 40, 600                      # span between abutments
DECK_Y, DECK_H = 175, 26              # deck band position / thickness


def main(out_path: str = "bridge_synth.mp4") -> None:
    rng = np.random.default_rng(3)

    bg = np.full((H, W, 3), 200, np.uint8)          # sky
    bg[240:, :] = (120, 150, 170)                   # water
    cv2.rectangle(bg, (0, 150), (X0, 245), (90, 95, 105), -1)      # abutments
    cv2.rectangle(bg, (X1, 150), (W, 245), (90, 95, 105), -1)
    for px in (215, 425):                                          # piers
        cv2.rectangle(bg, (px - 12, DECK_Y + DECK_H), (px + 12, 300),
                      (100, 105, 115), -1)
    noise = rng.integers(-12, 12, (H, W, 1))
    bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)

    # textured deck layer with alpha
    deck = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), np.uint8)
    tex = rng.integers(60, 140, (DECK_H, X1 - X0, 3)).astype(np.uint8)
    tex[:4] = (70, 75, 85)
    tex[-4:] = (55, 60, 70)
    for k in range(0, X1 - X0, 28):                 # girder panel lines
        tex[:, k:k + 3] = (45, 50, 60)
    deck[DECK_Y:DECK_Y + DECK_H, X0:X1] = tex
    mask[DECK_Y:DECK_Y + DECK_H, X0:X1] = 255

    xs = np.arange(W, dtype=np.float32)
    s01 = np.clip((xs - X0) / (X1 - X0), 0, 1)
    mode1 = np.sin(np.pi * s01).astype(np.float32)
    mode2 = np.sin(2 * np.pi * s01).astype(np.float32)
    map_x = np.tile(xs, (H, 1))
    ys = np.arange(H, dtype=np.float32)[:, None]

    frames = []
    for n in range(T):
        t = n / FPS
        w = (AMP * mode1 * np.sin(2 * np.pi * F1 * t)
             + 0.25 * AMP * mode2 * np.sin(2 * np.pi * 1.9 * F1 * t + 0.7))
        map_y = (ys - w[None, :]).astype(np.float32)   # deck moves up by +w
        deck_w = cv2.remap(deck, map_x, map_y, cv2.INTER_LINEAR)
        mask_w = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        m = (mask_w > 127)[..., None]
        frame = np.where(m, deck_w, bg)
        frames.append(frame[..., ::-1].copy())
    imageio.mimsave(out_path, frames, fps=int(FPS), quality=8)
    print(f"wrote {out_path}: {T} frames @ {FPS:g} fps, "
          f"mode1 {F1} Hz {AMP} px mid-span, span x=[{X0},{X1}], "
          f"deck y=[{DECK_Y},{DECK_Y + DECK_H}]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bridge_synth.mp4")
