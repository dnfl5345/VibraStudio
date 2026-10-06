"""DAQ Health view: is data arriving in real time on every input?

Top row answers the stream question (delivered rate vs nominal, driver
backlog, stall time); the grid answers it per input, laid out the way the
hardware is — one row per module, one cell per BNC input — so a dead or
miswired channel can be found on the rack at a glance. Tapping a sensor
ticks its cell, which is the only way to tell an idle sensor from an empty
input on NI-9230 modules.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QTableWidget, QVBoxLayout,
    QHeaderView, QAbstractItemView,
)

from ..config import ChannelConfig
from ..daq.engine import AcquisitionEngine, NidaqBackend, split_physical
from ..dsp import health as hl
from . import theme
from .views import BaseView, toolbar_row

IEPE_SETTLE_S = 22.0
REFRESH_S = 0.4

STATE_COLOR = {
    hl.LIVE: theme.GREEN,
    hl.QUIET: theme.TEXT_FAINT,
    hl.HUM: theme.YELLOW,
    hl.COMMON: "#e0915a",
    hl.CLIP: theme.RED,
    hl.SETTLING: "#5aa9d6",
    hl.NO_DATA: theme.RED,
}
VERDICT_COLOR = {"OK": theme.GREEN, "WARN": theme.YELLOW, "FAIL": theme.RED}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


class _Tile(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        cap = QLabel(title.upper())
        cap.setObjectName("CardTitle")
        self.value = QLabel("—")
        self.value.setStyleSheet(f"color:{theme.TEXT}; font-size:20px; font-weight:700;")
        self.note = QLabel("")
        self.note.setStyleSheet(f"color:{theme.TEXT_MUTED}; font-size:11px;")
        lay.addWidget(cap)
        lay.addWidget(self.value)
        lay.addWidget(self.note)

    def set(self, value: str, note: str = "", color: str | None = None) -> None:
        self.value.setText(value)
        self.note.setText(note)
        self.value.setStyleSheet(
            f"color:{color or theme.TEXT}; font-size:20px; font-weight:700;")


class HealthView(BaseView):
    def __init__(self) -> None:
        super().__init__()
        self.channel_cfgs: list[ChannelConfig] = []
        self._cells: list[QLabel] = []          # per channel
        self._groups: list = []                 # per channel module key
        self._where: list[tuple[str, int, int]] = []
        self._tapped: list[bool] = []
        self._baseline: np.ndarray | None = None
        self._last: list[hl.ChannelHealth] = []
        self._last_refresh = 0.0
        self._verdict = ("—", "")
        self._engine: AcquisitionEngine | None = None

        self.t_stream = _Tile("Stream")
        self.t_rate = _Tile("Delivered rate")
        self.t_backlog = _Tile("Driver backlog")
        self.t_gap = _Tile("Longest gap")
        self.t_states = _Tile("Inputs")
        tiles = QHBoxLayout()
        tiles.setContentsMargins(8, 8, 8, 0)
        tiles.setSpacing(10)
        for t in (self.t_stream, self.t_rate, self.t_backlog, self.t_gap):
            tiles.addWidget(t, stretch=1)
        tiles.addWidget(self.t_states, stretch=2)

        self.reset_btn = QPushButton("Reset tap marks")
        self.save_btn = QPushButton("Save report")
        self.hint = QLabel("Start streaming, wait for IEPE settling (~20 s), then tap "
                           "each sensor: its cell gets a ✓.")
        legend = QLabel("  ".join(
            f'<span style="color:{STATE_COLOR[s]}">■</span> {s}'
            for s in (hl.LIVE, hl.QUIET, hl.COMMON, hl.HUM, hl.CLIP, hl.SETTLING, hl.NO_DATA)))
        legend.setTextFormat(Qt.RichText)

        self.table = QTableWidget(0, 1)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(46)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(tiles)
        lay.addLayout(toolbar_row(legend, "stretch", self.hint))
        lay.addWidget(self.table, stretch=1)
        lay.addLayout(toolbar_row(self.reset_btn, self.save_btn, "stretch"))

        self.reset_btn.clicked.connect(self._reset_taps)
        self.save_btn.clicked.connect(self._save_report)

    # -- layout ------------------------------------------------------------
    def set_channel_configs(self, cfgs: list[ChannelConfig]) -> None:
        self.channel_cfgs = list(cfgs)
        self.set_channels([c.label for c in cfgs])

        self._where = []
        for i, c in enumerate(cfgs):
            chassis, mod, ai = split_physical(c.physical_channel)
            if mod == 0:                      # simulation / file channels
                chassis, mod, ai = "", i // 3 + 1, i % 3
            self._where.append((chassis, mod, ai))
        self._groups = [(w[0], w[1]) for w in self._where]

        rows: list[tuple[str, int]] = []
        for key in self._groups:
            if key not in rows:
                rows.append(key)
        n_ai = max((w[2] for w in self._where), default=0) + 1

        self.table.clear()
        self.table.setRowCount(len(rows))
        self.table.setColumnCount(1 + n_ai)
        self.table.setHorizontalHeaderLabels(
            ["Module"] + [f"ai{k}" for k in range(n_ai)])
        # Cell widgets do not drive ResizeToContents, so size the module
        # column explicitly ("2196A44 · Mod8" must not be clipped).
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 170)

        for r, (chassis, mod) in enumerate(rows):
            short = chassis.split("-")[-1] if chassis else "ch"
            head = QLabel(f"<b>{short}</b> · Mod{mod}" if chassis else f"<b>{short}</b> {3*mod-2}–{3*mod}")
            head.setTextFormat(Qt.RichText)
            head.setContentsMargins(10, 0, 14, 0)
            head.setStyleSheet(f"color:{theme.TEXT_MUTED};")
            self.table.setCellWidget(r, 0, head)

        self._cells = []
        for i, (chassis, mod, ai) in enumerate(self._where):
            cell = QLabel(cfgs[i].label)
            cell.setTextFormat(Qt.RichText)
            cell.setContentsMargins(10, 2, 8, 2)
            self.table.setCellWidget(rows.index((chassis, mod)), 1 + ai, cell)
            self._cells.append(cell)

        self._tapped = [False] * len(cfgs)
        self._baseline = None
        self._last = []
        for i in range(len(cfgs)):
            self._paint(i, None)

    def _paint(self, i: int, h: hl.ChannelHealth | None) -> None:
        label = self.channel_cfgs[i].label
        tick = '  <span style="color:%s">✓ tap</span>' % theme.GREEN if self._tapped[i] else ""
        if h is None:
            self._cells[i].setText(
                f'<b>{label}</b><br><span style="color:{theme.TEXT_FAINT}">—</span>')
            self._cells[i].setStyleSheet("")
            self._cells[i].setToolTip("")
            return
        color = STATE_COLOR[h.state]
        self._cells[i].setText(
            f'<b>{label}</b> &nbsp;<span style="color:{color}">{h.state}</span>{tick}<br>'
            f'<span style="color:{theme.TEXT_MUTED}">{1e3 * h.rms:.2f} mg rms · '
            f'{h.f_dom:.0f} Hz</span>')
        self._cells[i].setStyleSheet(f"background-color:{_rgba(color, 0.13)};")
        self._cells[i].setToolTip(f"{self.channel_cfgs[i].physical_channel}\n"
                                  f"{hl.STATE_HELP[h.state]}")

    # -- live update ---------------------------------------------------------
    def update_view(self, engine: AcquisitionEngine) -> None:
        self._engine = engine
        now = time.perf_counter()
        if now - self._last_refresh < REFRESH_S or not self._cells:
            return
        self._last_refresh = now

        fs, chunk = engine.fs, engine.acq.chunk_size
        rate = engine.measured_rate()
        age = engine.block_age()
        verdict, reason = hl.stream_verdict(fs, rate, engine.backlog, age, chunk)
        if engine.blocks < 3:
            verdict, reason = "—", "waiting for the first blocks"
        self._verdict = (verdict, reason)
        self.t_stream.set(verdict, reason, VERDICT_COLOR.get(verdict))
        self.t_rate.set(f"{rate:,.0f} S/s",
                        f"nominal {fs:,.0f} · {100 * (rate / fs - 1):+.2f} %" if rate else "")
        self.t_backlog.set(f"{1e3 * engine.backlog / fs:.0f} ms",
                           f"{engine.backlog} samples · peak {engine.max_backlog}")
        self.t_gap.set(f"{1e3 * engine.max_gap:.0f} ms",
                       f"block every {1e3 * chunk / fs:.0f} ms · last {1e3 * age:.0f} ms ago")

        elapsed = now - engine.t_first if engine.t_first else 0.0
        hardware_iepe = (isinstance(engine.backend, NidaqBackend)
                         and engine.acq.coupling == "iepe")
        settling = hardware_iepe and elapsed < IEPE_SETTLE_S

        data = engine.ring.latest(int(fs))          # 1 s -> 1 Hz resolution
        if data.shape[0] != len(self._cells) or data.shape[1] < 256:
            return
        result = hl.assess_channels(
            data, fs, self._groups, full_scale=abs(engine.acq.max_val),
            settling=settling)

        # Tap response: a transient well above the channel's own baseline.
        if not settling:
            recent = data[:, -int(0.4 * fs):]
            pk = np.max(np.abs(recent - recent.mean(axis=1, keepdims=True)), axis=1)
            rms = np.array([h.rms for h in result])
            if self._baseline is None or self._baseline.size != rms.size:
                self._baseline = rms.copy()
            hit = pk > np.maximum(8.0 * self._baseline, 0.02)
            for i in np.nonzero(hit)[0]:
                self._tapped[i] = True
            calm = ~hit
            self._baseline[calm] = 0.9 * self._baseline[calm] + 0.1 * rms[calm]

        self._last = result
        counts: dict[str, int] = {}
        for i, h in enumerate(result):
            counts[h.state] = counts.get(h.state, 0) + 1
            self._paint(i, h)
        order = (hl.LIVE, hl.QUIET, hl.COMMON, hl.HUM, hl.CLIP, hl.SETTLING, hl.NO_DATA)
        self.t_states.set(
            f"{len(result)} channels",
            "   ".join(f"{s} {counts[s]}" for s in order if s in counts)
            + f"   ·   tapped {sum(self._tapped)}")
        if settling:
            self.hint.setText(f"IEPE settling… {max(0.0, IEPE_SETTLE_S - elapsed):.0f} s")
        else:
            self.hint.setText("Tap each sensor: its cell gets a ✓. QUIET = idle sensor or empty input.")

    def _reset_taps(self) -> None:
        self._tapped = [False] * len(self._tapped)
        self._baseline = None
        for i, h in enumerate(self._last):
            self._paint(i, h)

    # -- export ----------------------------------------------------------------
    def report(self) -> dict:
        e = self._engine
        stream = {}
        if e is not None:
            stream = {
                "nominal_rate_hz": e.fs,
                "measured_rate_hz": e.measured_rate(),
                "backlog_samples": e.backlog,
                "max_backlog_samples": e.max_backlog,
                "max_gap_s": e.max_gap,
                "blocks": e.blocks,
                "samples_per_channel": e.ring.total,
            }
        return {
            "verdict": self._verdict[0], "reason": self._verdict[1],
            "stream": stream,
            "channels": [
                {"label": c.label, "physical_channel": c.physical_channel,
                 "state": h.state, "rms_g": h.rms, "peak_g": h.peak, "mean_g": h.mean,
                 "f_dom_hz": h.f_dom, "hum_share": h.hum_share, "tapped": self._tapped[i]}
                for i, (c, h) in enumerate(zip(self.channel_cfgs, self._last))
            ],
        }

    def _save_report(self) -> None:
        if not self._last:
            self.hint.setText("nothing to save — start streaming first")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs") / f"health_{ts}"
        out.mkdir(parents=True, exist_ok=True)
        rep = self.report()
        (out / "summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        with (out / "channels.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "physical_channel", "state", "rms_g", "peak_g",
                        "mean_g", "f_dom_hz", "hum_share", "tapped"])
            for c in rep["channels"]:
                w.writerow([c["label"], c["physical_channel"], c["state"],
                            f"{c['rms_g']:.6e}", f"{c['peak_g']:.6e}", f"{c['mean_g']:.6e}",
                            f"{c['f_dom_hz']:.2f}", f"{c['hum_share']:.3f}", int(c["tapped"])])
        self.window().grab().save(str(out / "screenshot.png"))
        self.hint.setText(f"saved → {out}")
