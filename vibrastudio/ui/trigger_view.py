"""Trigger tab: record automatically when band-limited level crosses a threshold.

Use it to capture an externally driven shaker test without babysitting the
Record button: set the band where the excitation lives, a threshold, and how
long to keep — every run lands in recordings/ with its pre-trigger lead-in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView, QAbstractItemView,
)

from ..dsp import trigger as trg
from . import theme
from .views import BaseView, toolbar_row

STATE_STYLE = {
    trg.DISARMED: (theme.TEXT_FAINT, "DISARMED"),
    trg.SETTLING: ("#5aa9d6", "SETTLING — IEPE bias charging"),
    trg.ARMED: (theme.GREEN, "ARMED — waiting for excitation"),
    trg.CAPTURING: (theme.RED, "● RECORDING"),
}


def _spin(lo, hi, val, suffix, decimals=2, step=None):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(val)
    s.setSuffix(suffix)
    if step:
        s.setSingleStep(step)
    s.setKeyboardTracking(False)
    return s


class TriggerView(BaseView):
    start_stream_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.ctrl = trg.TriggerController(Path("recordings"))
        self._engine = None
        self._shown_events = 0
        self._arm_pending = False

        # -- condition ----------------------------------------------------------
        self.source_box = QComboBox()
        self.source_box.setMinimumWidth(110)
        self.lo_spin = _spin(0, 50000, 0, " Hz", 1, 10)
        self.hi_spin = _spin(0, 50000, 0, " Hz", 1, 10)
        for s in (self.lo_spin, self.hi_spin):
            s.setSpecialValueText("open")
        self.metric_box = QComboBox()
        self.metric_box.addItems(["RMS", "Peak"])
        self.win_spin = _spin(10, 2000, 100, " ms", 0, 10)
        self.thr_spin = _spin(0.0001, 500, 0.05, " g", 4, 0.01)

        # -- capture -----------------------------------------------------------
        self.pre_spin = _spin(0, 10, 1.0, " s", 2, 0.5)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Fixed length", "Until quiet"])
        self.dur_spin = _spin(0.1, 600, 10, " s", 1, 1)
        self.quiet_spin = _spin(0.1, 60, 2, " s", 1, 0.5)
        self.max_spin = _spin(1, 600, 120, " s", 0, 10)
        self.rearm_chk = QCheckBox("Re-arm")
        self.rearm_chk.setChecked(True)
        self.csv_chk = QCheckBox("also CSV")

        self.arm_btn = QPushButton("⚡ Arm")
        self.arm_btn.setObjectName("Primary")
        self.arm_btn.setCheckable(True)
        self.force_btn = QPushButton("Trigger now")
        self.state_lbl = QLabel("DISARMED")
        self.state_lbl.setStyleSheet(f"font-size:15px; font-weight:700; color:{theme.TEXT_FAINT};")
        self.level_lbl = QLabel("—")
        self.level_lbl.setStyleSheet(f"color:{theme.TEXT}; font-size:13px; font-weight:600;")
        self.level_lbl.setMinimumWidth(560)   # bold stylesheet font is wider than the size hint

        # -- level plot ---------------------------------------------------------
        self.plot_w = pg.PlotWidget()
        self.plot = self.plot_w.getPlotItem()
        theme.style_plot(self.plot, "", "Time (s)", "Band level (g)")
        # Keep axis numbers in g so they read the same as the threshold label.
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.curve = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=1.6))
        self.thr_line = pg.InfiniteLine(pos=self.thr_spin.value(), angle=0, movable=True,
                                        pen=pg.mkPen(theme.YELLOW, width=1.5, style=Qt.DashLine),
                                        label="threshold {value:.4g} g",
                                        labelOpts={"color": theme.YELLOW, "position": 0.06})
        self.rearm_line = pg.InfiniteLine(pos=self.thr_spin.value() * 0.8, angle=0,
                                          pen=pg.mkPen(theme.TEXT_FAINT, width=1, style=Qt.DotLine))
        self.plot.addItem(self.thr_line)
        self.plot.addItem(self.rearm_line)
        self.capture_region = pg.LinearRegionItem(movable=False,
                                                  brush=pg.mkBrush(224, 90, 90, 40),
                                                  pen=pg.mkPen(None))
        self.capture_region.setVisible(False)
        self.plot.addItem(self.capture_region)
        self.event_marks = pg.ScatterPlotItem(symbol="t", size=12,
                                              brush=pg.mkBrush(theme.RED), pen=None)
        self.plot.addItem(self.event_marks)

        # -- event log ------------------------------------------------------------
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["#", "Time", "Channel", "Level @ trig (g)", "Peak (g)", "Length (s)", "File"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(190)
        self.hint = QLabel("Signals outside the band are ignored.")
        self.hint.setToolTip("Set the band to where the shaker excites (e.g. its sweep range).")
        self.hint.setStyleSheet(f"color:{theme.TEXT_MUTED};")

        status = QHBoxLayout()
        status.setContentsMargins(10, 4, 10, 0)
        status.addWidget(self.state_lbl)
        status.addSpacing(24)
        status.addWidget(self.level_lbl)
        status.addStretch(1)
        status.addWidget(self.hint)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Source", self.source_box, "Band", self.lo_spin, "–",
                                  self.hi_spin, "Level", self.metric_box, "over",
                                  self.win_spin, "Threshold", self.thr_spin, "stretch"))
        lay.addLayout(toolbar_row("Pre-trigger", self.pre_spin, "Stop", self.mode_box,
                                  self.dur_spin, self.quiet_spin, "Max", self.max_spin,
                                  self.rearm_chk, self.csv_chk, "stretch",
                                  self.force_btn, self.arm_btn))
        lay.addLayout(status)
        lay.addWidget(self.plot_w, stretch=1)
        lay.addWidget(self.table)

        # -- wiring ----------------------------------------------------------------
        self.arm_btn.toggled.connect(self._arm_toggled)
        self.force_btn.clicked.connect(self.ctrl.force)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.thr_line.sigPositionChangeFinished.connect(
            lambda ln: self.thr_spin.setValue(max(1e-4, float(ln.value()))))
        for w in (self.lo_spin, self.hi_spin, self.win_spin, self.thr_spin, self.pre_spin,
                  self.dur_spin, self.quiet_spin, self.max_spin):
            w.valueChanged.connect(self._apply)
        for w in (self.source_box, self.metric_box, self.mode_box):
            w.currentIndexChanged.connect(self._apply)
        for w in (self.rearm_chk, self.csv_chk):
            w.toggled.connect(self._apply)
        self._mode_changed(0)

    # -- settings ------------------------------------------------------------------
    def settings(self) -> trg.TriggerSettings:
        return trg.TriggerSettings(
            source=self.source_box.currentIndex() - 1,
            f_lo=self.lo_spin.value(), f_hi=self.hi_spin.value(),
            metric="rms" if self.metric_box.currentIndex() == 0 else "peak",
            window_s=self.win_spin.value() / 1000.0,
            threshold=self.thr_spin.value(), pre_s=self.pre_spin.value(),
            stop_mode="duration" if self.mode_box.currentIndex() == 0 else "quiet",
            duration_s=self.dur_spin.value(), quiet_s=self.quiet_spin.value(),
            max_s=self.max_spin.value(), rearm=self.rearm_chk.isChecked(),
            write_csv=self.csv_chk.isChecked())

    def _apply(self, *_) -> None:
        self.ctrl.configure(self.settings(), self.labels)
        thr = self.thr_spin.value()
        self.thr_line.blockSignals(True)
        self.thr_line.setValue(thr)
        self.thr_line.blockSignals(False)
        self.rearm_line.setValue(thr * self.ctrl.settings.hysteresis)

    def _mode_changed(self, idx: int) -> None:
        fixed = idx == 0
        self.dur_spin.setVisible(fixed)
        self.quiet_spin.setVisible(not fixed)
        self.dur_spin.setToolTip("Length kept after the trigger point")
        self.quiet_spin.setToolTip("Stop after the level stays below the re-arm line this long")

    def _channels_changed(self) -> None:
        cur = self.source_box.currentText()
        self.source_box.blockSignals(True)
        self.source_box.clear()
        self.source_box.addItems(["Any channel"] + self.labels)
        i = self.source_box.findText(cur)
        self.source_box.setCurrentIndex(i if i >= 0 else 0)
        self.source_box.blockSignals(False)
        self._apply()

    # -- arming ------------------------------------------------------------------------
    def _arm_toggled(self, on: bool) -> None:
        if on:
            if self._engine is None or not self._engine.running:
                # Arming implies acquisition: start the stream, arm on the next tick.
                self._arm_pending = True
                self.start_stream_requested.emit()
                return
            self._apply()
            self.ctrl.arm(self._engine)
        else:
            self._arm_pending = False
            self.ctrl.disarm()
        self.arm_btn.setText("■ Disarm" if on else "⚡ Arm")

    def stream_stopped(self) -> None:
        self.ctrl.disarm()
        self._engine = None
        if self.arm_btn.isChecked():
            self.arm_btn.blockSignals(True)
            self.arm_btn.setChecked(False)
            self.arm_btn.blockSignals(False)
            self.arm_btn.setText("⚡ Arm")

    def badge(self) -> str:
        st = self.ctrl.state
        if st == trg.CAPTURING:
            return "  [TRIG REC]"
        if st in (trg.ARMED, trg.SETTLING):
            return "  [TRIG ARMED]"
        return ""

    # -- per tick ------------------------------------------------------------------------
    def process(self, engine) -> None:
        """Called on every GUI tick, whichever tab is visible."""
        self._engine = engine
        if getattr(self, "_arm_pending", False) and engine is not None and engine.running:
            self._arm_pending = False
            self._apply()
            self.ctrl.arm(engine)
            self.arm_btn.setText("■ Disarm")
        self.ctrl.process(engine)
        if not self.ctrl.settings.rearm and self.ctrl.state == trg.DISARMED \
                and self.arm_btn.isChecked() and self.ctrl.events:
            self.arm_btn.blockSignals(True)
            self.arm_btn.setChecked(False)
            self.arm_btn.blockSignals(False)
            self.arm_btn.setText("⚡ Arm")
        self._log_new_events()

    def _log_new_events(self) -> None:
        events = sorted(self.ctrl.events, key=lambda ev: ev["n"])
        for ev in events[self._shown_events:]:
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [str(ev["n"]), ev["time"], ev["channel"], f"{ev['level']:.4g}",
                    f"{ev['peak']:.4g}", f"{ev['duration_s']:.2f}", Path(ev["file"]).name]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 6:
                    item.setToolTip(ev["file"])
                self.table.setItem(r, c, item)
            self.table.scrollToBottom()
        self._shown_events = len(events)

    def update_view(self, engine) -> None:
        """Redraw the level trace (visible tab only)."""
        ctrl = self.ctrl
        color, text = STATE_STYLE.get(ctrl.state, (theme.TEXT, ctrl.state))
        if ctrl.state == trg.CAPTURING and ctrl._cap is not None and engine is not None:
            secs = (engine.ring.total - ctrl._cap["trigger_abs"]) / engine.fs
            text = f"● RECORDING  {secs:5.1f} s"
        if ctrl.saving:
            text += f"   · saving {ctrl.saving}"
        if ctrl.message:
            text += f"   · {ctrl.message}"
        self.state_lbl.setText(text)
        self.state_lbl.setStyleSheet(f"font-size:15px; font-weight:700; color:{color};")

        if ctrl.history and engine is not None:
            arr = np.array([(e, l) for e, l, _ in ctrl.history])
            now = engine.ring.total
            t = (arr[:, 0] - now) / engine.fs
            self.curve.setData(t, arr[:, 1])
            self.plot.setXRange(-trg.TriggerController.HISTORY_S, 0, padding=0.01)
            ch = ctrl.last_channel
            name = self.labels[ch] if 0 <= ch < len(self.labels) else "—"
            band = ctrl._meter.band if ctrl._meter is not None else (0, 0)
            self.level_lbl.setText(
                f"level {ctrl.last_level:.4g} g on {name}   ·   band {band[0]:.0f}–{band[1]:.0f} Hz")
            marks = [((ev["trigger_abs"] - now) / engine.fs, ev["level"]) for ev in ctrl.events
                     if (ev["trigger_abs"] - now) / engine.fs > -trg.TriggerController.HISTORY_S]
            self.event_marks.setData([m[0] for m in marks], [m[1] for m in marks])
            if ctrl.state == trg.CAPTURING and ctrl._cap is not None:
                self.capture_region.setRegion(((ctrl._cap["trigger_abs"] - now) / engine.fs, 0))
                self.capture_region.setVisible(True)
            else:
                self.capture_region.setVisible(False)
