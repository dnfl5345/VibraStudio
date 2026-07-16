"""Generate the VibraStudio application icon (.ico) — coral waveform on dark."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)

    s = size / 256.0
    # Rounded dark tile
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 48 * s, 48 * s)
    p.fillPath(tile, QColor("#262320"))

    # Decaying sine burst (vibration signature)
    import math

    pen = QPen(QColor("#e08b6d"), max(1.0, 14 * s), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    path = QPainterPath()
    n = 160
    for i in range(n + 1):
        x = 36 * s + (184 * s) * i / n
        t = i / n
        y = 128 * s - (86 * s) * math.exp(-2.2 * t) * math.sin(2 * math.pi * 3.0 * t)
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    p.drawPath(path)

    # Baseline dots (measurement points)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#5bbf9a"))
    for fx in (0.15, 0.45, 0.75):
        p.drawEllipse(QPointF((36 + 184 * fx) * s, 208 * s), 9 * s, 9 * s)

    p.end()
    return img


def main() -> int:
    QGuiApplication(sys.argv)
    out_dir = Path(__file__).parent
    imgs = {sz: draw(sz) for sz in (16, 24, 32, 48, 64, 128, 256)}
    png = out_dir / "vibrastudio_256.png"
    imgs[256].save(str(png))

    # QImage cannot write .ico directly; assemble manually (PNG-compressed entries).
    import io
    import struct

    entries = []
    data_blobs = []
    for sz, im in imgs.items():
        from PySide6.QtCore import QBuffer, QIODevice

        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        im.save(buf, "PNG")
        data_blobs.append(bytes(buf.data()))
        entries.append(sz)

    ico = out_dir / "vibrastudio.ico"
    with ico.open("wb") as f:
        f.write(struct.pack("<HHH", 0, 1, len(entries)))
        offset = 6 + 16 * len(entries)
        for sz, blob in zip(entries, data_blobs):
            b = sz if sz < 256 else 0
            f.write(struct.pack("<BBBBHHII", b, b, 0, 0, 1, 32, len(blob), offset))
            offset += len(blob)
        for blob in data_blobs:
            f.write(blob)
    print(f"wrote {ico}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
