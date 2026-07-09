"""ODS (Operational Deflection Shape) view with animated mode-shape display.

Amplitude comes from averaged auto-spectra, relative phase from averaged
cross-spectra against the reference channel — the standard spectral-ODS
method used by LabVIEW SVMS / commercial ODS packages.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView,
)

from ..config import ChannelConfig
from ..dsp.analysis import StreamingAnalyzer
from . import theme
from .views import BaseView, toolbar_row


class OdsView(BaseView):
    def __init__(self) -> None:
        super().__init__()
        self.channel_cfgs: list[ChannelConfig] = []

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.5, 100000.0)
        self.freq_spin.setValue(11.6)
        self.freq_spin.setDecimals(2)
        self.freq_spin.setSuffix(" Hz")
        self.peak_box = QComboBox()
        self.peak_box.setMinimumWidth(130)
        self.pick_btn = QPushButton("Find Peaks")
        self.hold_btn = QPushButton("Freeze")
        self.hold_btn.setCheckable(True)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 40)
        self.speed.setValue(12)
        self.speed.setFixedWidth(110)
        self.scale = QSlider(Qt.Horizontal)
        self.scale.setRange(1, 100)
        self.scale.setValue(40)
        self.scale.setFixedWidth(110)
        self.info = QLabel("—")

        # Two beams: X-direction and Z-direction deflection.
        self.gw = pg.GraphicsLayoutWidget()
        self.p_x = self.gw.addPlot(row=0, col=0)
        self.p_z = self.gw.addPlot(row=0, col=1)
        for p, name in ((self.p_x, "X direction"), (self.p_z, "Z direction")):
            theme.style_plot(p, f"Deflection shape — {name}",
                             "Deflection (normalized)", "Position (normalized)")
            p.setXRange(-1.4, 1.4)
            p.setYRange(-0.05, 1.1)
            p.addLine(x=0, pen=pg.mkPen(theme.TEXT_FAINT, style=Qt.DashLine))
        self.undeformed_x = self.p_x.plot(pen=pg.mkPen(theme.BORDER, width=1))
        self.undeformed_z = self.p_z.plot(pen=pg.mkPen(theme.BORDER, width=1))
        self.shape_x = self.p_x.plot(pen=pg.mkPen(theme.ACCENT, width=2.5),
                                     symbol="o", symbolSize=9,
                                     symbolBrush=pg.mkBrush(theme.ACCENT_HOVER),
                                     symbolPen=None)
        self.shape_z = self.p_z.plot(pen=pg.mkPen("#5aa9d6", width=2.5),
                                     symbol="o", symbolSize=9,
                                     symbolBrush=pg.mkBrush("#7ec2e8"), symbolPen=None)
        self.trace_x = self.p_x.plot(pen=pg.mkPen(theme.ACCENT, width=1, style=Qt.DotLine))
        self.trace_z = self.p_z.plot(pen=pg.mkPen("#5aa9d6", width=1, style=Qt.DotLine))

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Ch", "Pt", "Dir", "g RMS", "Phase°"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumWidth(360)

        body = QHBoxLayout()
        body.addWidget(self.gw, stretch=1)
        body.addWidget(self.table)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Frequency", self.freq_spin, self.pick_btn,
                                  "Peak list", self.peak_box, self.hold_btn,
                                  "Speed", self.speed, "Scale", self.scale,
                                  "stretch", self.info))
        lay.addLayout(body)

        self._phase = 0.0
        self._amp: np.ndarray | None = None      # per enabled channel
        self._phi: np.ndarray | None = None
        self._f_actual = 0.0
        self._analyzer: StreamingAnalyzer | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._animate)
        self.timer.start()
        self.peak_box.currentTextChanged.connect(self._peak_selected)
        self.pick_btn.clicked.connect(self._find_peaks)

    # ------------------------------------------------------------------
    def set_channel_configs(self, cfgs: list[ChannelConfig]) -> None:
        self.channel_cfgs = list(cfgs)
        self.set_channels([c.label for c in cfgs])
        self.table.setRowCount(len(cfgs))
        for i, c in enumerate(cfgs):
            for j, val in enumerate([c.label, str(c.point), c.direction, "—", "—"]):
                item = QTableWidgetItem(val)
                if j == 0:
                    item.setForeground(pg.mkColor(self.color(i)))
                self.table.setItem(i, j, item)

    def update_view(self, analyzer: StreamingAnalyzer) -> None:
        """Refresh ODS amplitudes/phases from the current averaged spectra."""
        self._analyzer = analyzer
        if analyzer.navg < 2 or self.hold_btn.isChecked():
            return
        f_actual, amp, phase = analyzer.ods_at(self.freq_spin.value())
        self._f_actual, self._amp, self._phi = f_actual, amp, phase
        for i in range(min(len(self.channel_cfgs), amp.size)):
            self.table.item(i, 3).setText(f"{amp[i]:.4g}")
            self.table.item(i, 4).setText(f"{np.degrees(phase[i]):.1f}")
        self.info.setText(f"bin: {f_actual:.2f} Hz   avg: {analyzer.navg}")

    def _find_peaks(self) -> None:
        if self._analyzer is None or self._analyzer.navg == 0:
            return
        from scipy.signal import find_peaks
        a = self._analyzer
        mean_amp = np.mean(a.amplitude_spectrum("rms"), axis=0)
        skip = max(1, int(2.0 / max(a.freq[1], 1e-9)))
        pk, _ = find_peaks(mean_amp[skip:], distance=5)
        pk += skip
        if pk.size == 0:
            return
        top = pk[np.argsort(mean_amp[pk])[-8:]][::-1]
        top = top[np.argsort(a.freq[top])]
        self.peak_box.blockSignals(True)
        self.peak_box.clear()
        self.peak_box.addItems([f"{a.freq[i]:.2f} Hz" for i in top])
        self.peak_box.blockSignals(False)
        if self.peak_box.count():
            self.peak_box.setCurrentIndex(0)
            self._peak_selected(self.peak_box.currentText())

    def _peak_selected(self, text: str) -> None:
        try:
            self.freq_spin.setValue(float(text.split()[0]))
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------
    def _geometry(self, direction: str) -> tuple[np.ndarray, list[int]]:
        """Sorted (positions, channel_indices) for one direction."""
        entries = [(c.position, i) for i, c in enumerate(self.channel_cfgs)
                   if c.direction == direction and c.enabled]
        entries.sort()
        if not entries:
            return np.array([]), []
        pos = np.array([e[0] for e in entries])
        idx = [e[1] for e in entries]
        return pos, idx

    def _animate(self) -> None:
        if self._amp is None or self._phi is None:
            return
        self._phase += 0.015 * self.speed.value()
        gain = self.scale.value() / 40.0
        # Normalize both directions by the same global peak so relative
        # X/Z deflection magnitudes stay physically comparable.
        global_peak = float(np.max(self._amp)) or 1.0
        for direction, shape_curve, undeformed, trace in (
            ("X", self.shape_x, self.undeformed_x, self.trace_x),
            ("Z", self.shape_z, self.undeformed_z, self.trace_z),
        ):
            pos, idx = self._geometry(direction)
            if len(idx) == 0:
                shape_curve.setData([], [])
                continue
            amps = self._amp[idx]
            phis = self._phi[idx]
            rel = amps / global_peak
            defl = gain * rel * np.cos(self._phase + phis)
            envelope = gain * rel
            # Clamped base at position 0.
            y = np.concatenate(([0.0], pos))
            x = np.concatenate(([0.0], defl))
            shape_curve.setData(x, y)
            undeformed.setData([0.0, 0.0], [0.0, 1.0])
            xe = np.concatenate(([0.0], envelope))
            trace.setData(np.concatenate((xe, [np.nan], -xe)),
                          np.concatenate((y, [np.nan], y)))
