"""Analysis views: time series, spectrum, spectrogram, octave, FRF, metrics."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView,
)

from ..config import CHANNEL_COLORS
from ..dsp import analysis as dsp
from . import theme


def toolbar_row(*widgets) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(8, 6, 8, 2)
    row.setSpacing(8)
    for w in widgets:
        if w == "stretch":
            row.addStretch(1)
        elif isinstance(w, str):
            row.addWidget(QLabel(w))
        else:
            row.addWidget(w)
    return row


class BaseView(QWidget):
    """Common state: channel labels/colors and visibility mask."""

    def __init__(self) -> None:
        super().__init__()
        self.labels: list[str] = []
        self.visible: list[bool] = []

    def set_channels(self, labels: list[str]) -> None:
        self.labels = list(labels)
        self.visible = [True] * len(labels)
        self._channels_changed()

    def set_visible(self, mask: list[bool]) -> None:
        self.visible = list(mask)

    def _channels_changed(self) -> None:  # override
        pass

    @staticmethod
    def color(i: int) -> str:
        return CHANNEL_COLORS[i % len(CHANNEL_COLORS)]


# ======================================================================
class TimeSeriesView(BaseView):
    """Rolling multi-channel time series (overlay or stacked)."""

    def __init__(self) -> None:
        super().__init__()
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.05, 30.0)
        self.window_spin.setValue(2.0)
        self.window_spin.setSuffix(" s")
        self.window_spin.setDecimals(2)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Overlay", "Stacked"])
        self.mode_box.setCurrentText("Stacked")
        self.legend_chk = QCheckBox("Legend")
        self.legend_chk.setChecked(True)

        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        theme.style_plot(self.plot, "", "Time (s)", "Acceleration (g)")
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.legend = self.plot.addLegend(offset=(-10, 10), labelTextColor=theme.TEXT_MUTED)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Window", self.window_spin, "Layout", self.mode_box,
                                  self.legend_chk, "stretch"))
        lay.addWidget(self.plot_widget)
        self.curves: list[pg.PlotDataItem] = []
        self.legend_chk.toggled.connect(lambda on: self.legend.setVisible(on))

    def _channels_changed(self) -> None:
        self.plot.clear()
        self.legend.clear()
        self.curves = []
        for i, lbl in enumerate(self.labels):
            c = self.plot.plot(pen=pg.mkPen(self.color(i), width=1.0), name=lbl)
            self.curves.append(c)

    def update_view(self, data: np.ndarray, fs: float, t_end: float) -> None:
        if data.size == 0 or not self.curves:
            return
        n = data.shape[1]
        t = t_end - (n - 1 - np.arange(n)) / fs
        stacked = self.mode_box.currentText() == "Stacked"
        vis_idx = [i for i in range(len(self.curves)) if self.visible[i]]
        if stacked and vis_idx:
            spans = [max(1e-6, float(np.ptp(data[i]))) for i in vis_idx]
            step = 1.1 * max(spans)
        else:
            step = 0.0
        slot = 0
        for i, curve in enumerate(self.curves):
            if not self.visible[i]:
                curve.setVisible(False)
                continue
            curve.setVisible(True)
            offset = (len(vis_idx) - 1 - slot) * step if stacked else 0.0
            curve.setData(t, data[i] + offset)
            slot += 1
        self.plot.setXRange(t[0], t[-1], padding=0.01)
        self.plot.enableAutoRange(axis="y")


# ======================================================================
class SpectrumView(BaseView):
    """Averaged FFT spectrum with units, integration, dB and peak markers."""

    def __init__(self) -> None:
        super().__init__()
        self.quantity_box = QComboBox()
        self.quantity_box.addItems([
            "Acceleration (g)", "Acceleration (m/s²)",
            "Velocity (mm/s)", "Displacement (µm)", "PSD (g²/Hz)",
        ])
        self.scale_box = QComboBox()
        self.scale_box.addItems(["RMS", "Peak"])
        self.db_chk = QCheckBox("dB")
        self.logx_chk = QCheckBox("Log f")
        self.peaks_spin = QSpinBox()
        self.peaks_spin.setRange(0, 10)
        self.peaks_spin.setValue(4)
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(10.0, 100000.0)
        self.fmax_spin.setValue(5000.0)
        self.fmax_spin.setSuffix(" Hz")
        self.reset_btn = QPushButton("Reset Avg")

        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        theme.style_plot(self.plot, "", "Frequency (Hz)", "Amplitude")
        self.legend = self.plot.addLegend(offset=(-10, 10), labelTextColor=theme.TEXT_MUTED)

        # Crosshair cursor with readout.
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.TEXT_FAINT, style=Qt.DashLine))
        self.cursor_label = QLabel("—")
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.scene().sigMouseMoved.connect(self._mouse_moved)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Quantity", self.quantity_box, "Scale", self.scale_box,
                                  self.db_chk, self.logx_chk, "Peaks", self.peaks_spin,
                                  "F max", self.fmax_spin, self.reset_btn,
                                  "stretch", self.cursor_label))
        lay.addWidget(self.plot_widget)
        self.curves: list[pg.PlotDataItem] = []
        self.peak_items: list = []
        self._freq = np.array([])
        self._spectra = np.zeros((0, 0))

    def _channels_changed(self) -> None:
        self.plot.clear()
        self.legend.clear()
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.curves = []
        self.peak_items = []
        for i, lbl in enumerate(self.labels):
            c = self.plot.plot(pen=pg.mkPen(self.color(i), width=1.0), name=lbl)
            self.curves.append(c)

    def _mouse_moved(self, pos) -> None:
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = self.plot.vb.mapSceneToView(pos)
        fx = 10 ** p.x() if self.logx_chk.isChecked() else p.x()
        self.vline.setPos(p.x())
        if self._freq.size:
            idx = int(np.argmin(np.abs(self._freq - fx)))
            vis = [i for i in range(len(self.curves)) if self.visible[i]]
            if vis:
                best = max(vis, key=lambda i: self._spectra[i, idx])
                self.cursor_label.setText(
                    f"{self._freq[idx]:.1f} Hz | {self.labels[best]}: {self._spectra[best, idx]:.4g}"
                )

    def update_view(self, analyzer: dsp.StreamingAnalyzer) -> None:
        if analyzer.navg == 0 or not self.curves:
            return
        q = self.quantity_box.currentText()
        scaling = "rms" if self.scale_box.currentText() == "RMS" else "peak"
        freq = analyzer.freq
        if q.startswith("PSD"):
            spec = analyzer.psd()
            unit = "g²/Hz"
        else:
            amp = analyzer.amplitude_spectrum(scaling)
            if "m/s²" in q:
                spec, unit = amp * dsp.G_TO_MS2, "m/s²"
            elif q.startswith("Velocity"):
                spec, unit = analyzer.integrate_amplitude(amp, 1), "mm/s"
            elif q.startswith("Displacement"):
                spec, unit = analyzer.integrate_amplitude(amp, 2), "µm"
            else:
                spec, unit = amp, "g"

        f_max = min(self.fmax_spin.value(), freq[-1])
        mask = freq <= f_max
        fplot = freq[mask]
        db = self.db_chk.isChecked()
        logx = self.logx_chk.isChecked()
        self._freq = fplot
        disp = spec[:, mask]
        if db:
            disp = 20.0 * np.log10(np.maximum(disp, 1e-12))
        self._spectra = disp

        if logx:
            valid = fplot > 0
            xs = np.log10(fplot[valid])
        else:
            valid = np.ones(fplot.size, dtype=bool)
            xs = fplot[valid]

        for i, curve in enumerate(self.curves):
            if not self.visible[i]:
                curve.setVisible(False)
                continue
            curve.setVisible(True)
            curve.setData(xs, disp[i][valid])

        self.plot.setLabel("left", f"{q.split(' (')[0]} ({'dB re ' + unit if db else unit})",
                           color=theme.TEXT_FAINT)

        # Peak markers on the strongest visible channel.
        for it in self.peak_items:
            self.plot.removeItem(it)
        self.peak_items = []
        n_peaks = self.peaks_spin.value()
        vis = [i for i in range(len(self.curves)) if self.visible[i]]
        if n_peaks > 0 and vis and fplot.size > 16:
            power = spec[vis][:, mask]
            best_ch = vis[int(np.argmax(np.max(power, axis=1)))]
            y = disp[best_ch]
            from scipy.signal import find_peaks
            skip = max(1, int(2.0 / max(fplot[1] - fplot[0], 1e-9)))  # ignore < 2 Hz
            pk, _ = find_peaks(y[skip:], distance=5)
            pk += skip
            if pk.size:
                top = pk[np.argsort(y[pk])[-n_peaks:]]
                for idx in top:
                    x = np.log10(fplot[idx]) if logx else fplot[idx]
                    txt = pg.TextItem(f"{fplot[idx]:.1f} Hz", color=theme.TEXT,
                                      anchor=(0.5, 1.2))
                    txt.setPos(x, y[idx])
                    dot = pg.ScatterPlotItem([x], [y[idx]], size=6,
                                             brush=pg.mkBrush(theme.ACCENT), pen=None)
                    self.plot.addItem(dot)
                    self.plot.addItem(txt)
                    self.peak_items += [dot, txt]


# ======================================================================
class SpectrogramView(BaseView):
    """Rolling STFT spectrogram (waterfall) of one selected channel."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_box = QComboBox()
        self.win_box = QComboBox()
        self.win_box.addItems(["512", "1024", "2048", "4096"])
        self.win_box.setCurrentText("1024")
        self.span_spin = QDoubleSpinBox()
        self.span_spin.setRange(1.0, 30.0)
        self.span_spin.setValue(8.0)
        self.span_spin.setSuffix(" s")

        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        theme.style_plot(self.plot, "", "Time (s)", "Frequency (Hz)")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        cmap = pg.colormap.get("inferno")
        self.bar = pg.ColorBarItem(colorMap=cmap, label="dB")
        self.bar.setImageItem(self.img, insert_in=self.plot)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Channel", self.channel_box, "FFT", self.win_box,
                                  "Span", self.span_spin, "stretch"))
        lay.addWidget(self.plot_widget)

    def _channels_changed(self) -> None:
        cur = self.channel_box.currentIndex()
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(self.labels)
        self.channel_box.setCurrentIndex(max(0, cur) if cur < len(self.labels) else 0)
        self.channel_box.blockSignals(False)

    def update_view(self, data: np.ndarray, fs: float) -> None:
        ch = self.channel_box.currentIndex()
        if data.size == 0 or ch < 0 or ch >= data.shape[0]:
            return
        span = int(self.span_spin.value() * fs)
        y = data[ch, -span:]
        f, t, sdb = dsp.compute_spectrogram(y, fs, nperseg=int(self.win_box.currentText()))
        if sdb.size == 0:
            return
        vmax = float(np.max(sdb))
        self.img.setImage(sdb.T, autoLevels=False)
        self.bar.setLevels((vmax - 80.0, vmax))
        if t.size and f.size:
            self.img.setRect(0.0, 0.0, float(t[-1]), float(f[-1]))
        self.plot.setLabel("bottom", "Time (s)", color=theme.TEXT_FAINT)


# ======================================================================
class OctaveView(BaseView):
    """1/1 and 1/3 octave band levels of one selected channel."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_box = QComboBox()
        self.fraction_box = QComboBox()
        self.fraction_box.addItems(["1/3 octave", "1/1 octave"])
        self.db_chk = QCheckBox("dB")
        self.db_chk.setChecked(True)

        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        theme.style_plot(self.plot, "", "Band center (Hz)", "Level")
        self.bars: pg.BarGraphItem | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Channel", self.channel_box, "Bands", self.fraction_box,
                                  self.db_chk, "stretch"))
        lay.addWidget(self.plot_widget)

    def _channels_changed(self) -> None:
        cur = self.channel_box.currentIndex()
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(self.labels)
        self.channel_box.setCurrentIndex(max(0, cur) if cur < len(self.labels) else 0)
        self.channel_box.blockSignals(False)

    def update_view(self, analyzer: dsp.StreamingAnalyzer) -> None:
        ch = self.channel_box.currentIndex()
        if analyzer.navg == 0 or ch < 0 or ch >= analyzer.n_ch:
            return
        fraction = 3 if self.fraction_box.currentIndex() == 0 else 1
        psd = analyzer.psd()[ch]
        centers, rms = dsp.octave_band_levels(
            analyzer.freq, psd, fraction=fraction, f_max=analyzer.fs / 2
        )
        if centers.size == 0:
            return
        if self.db_chk.isChecked():
            vals = 20.0 * np.log10(np.maximum(rms, 1e-9) / 1e-6)  # dB re 1 µg
            ylab = "Level (dB re 1 µg RMS)"
        else:
            vals = rms
            ylab = "Band RMS (g)"
        x = np.arange(centers.size)
        if self.bars is not None:
            self.plot.removeItem(self.bars)
        self.bars = pg.BarGraphItem(x=x, height=vals, width=0.8,
                                    brush=pg.mkBrush(theme.ACCENT), pen=pg.mkPen(theme.BORDER))
        self.plot.addItem(self.bars)
        ticks = [(i, f"{c:.0f}" if c < 1000 else f"{c/1000:.3g}k")
                 for i, c in enumerate(centers)]
        step = max(1, len(ticks) // 18)
        self.plot.getAxis("bottom").setTicks([ticks[::step]])
        self.plot.setLabel("left", ylab, color=theme.TEXT_FAINT)
        self.plot.enableAutoRange()


# ======================================================================
class FrfView(BaseView):
    """Transmissibility FRF (H1) + phase + coherence vs reference channel."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_box = QComboBox()
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(10.0, 100000.0)
        self.fmax_spin.setValue(1000.0)
        self.fmax_spin.setSuffix(" Hz")
        self.info_label = QLabel("")

        self.layout_widget = pg.GraphicsLayoutWidget()
        self.p_mag = self.layout_widget.addPlot(row=0, col=0)
        self.p_phase = self.layout_widget.addPlot(row=1, col=0)
        self.p_coh = self.layout_widget.addPlot(row=2, col=0)
        theme.style_plot(self.p_mag, "FRF Magnitude |H1| (transmissibility)", "", "|H|")
        theme.style_plot(self.p_phase, "Phase", "", "deg")
        theme.style_plot(self.p_coh, "Coherence", "Frequency (Hz)", "γ²")
        self.p_phase.setXLink(self.p_mag)
        self.p_coh.setXLink(self.p_mag)
        self.p_coh.setYRange(0, 1.05)
        self.p_phase.setYRange(-190, 190)
        self.c_mag = self.p_mag.plot(pen=pg.mkPen(theme.ACCENT, width=1.2))
        self.c_phase = self.p_phase.plot(pen=pg.mkPen("#5aa9d6", width=1.1))
        self.c_coh = self.p_coh.plot(pen=pg.mkPen("#7fbf7a", width=1.1))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Response channel", self.channel_box,
                                  "F max", self.fmax_spin, "stretch", self.info_label))
        lay.addWidget(self.layout_widget)

    def _channels_changed(self) -> None:
        cur = self.channel_box.currentIndex()
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(self.labels)
        self.channel_box.setCurrentIndex(max(0, cur) if cur < len(self.labels) else 0)
        self.channel_box.blockSignals(False)

    def update_view(self, analyzer: dsp.StreamingAnalyzer, ref_label: str) -> None:
        ch = self.channel_box.currentIndex()
        if analyzer.navg < 2 or ch < 0 or ch >= analyzer.n_ch:
            return
        h1, coh = analyzer.frf()
        mask = analyzer.freq <= min(self.fmax_spin.value(), analyzer.freq[-1])
        f = analyzer.freq[mask]
        self.c_mag.setData(f, np.abs(h1[ch])[mask])
        self.c_phase.setData(f, np.degrees(np.angle(h1[ch]))[mask])
        self.c_coh.setData(f, coh[ch][mask])
        self.info_label.setText(f"ref: {ref_label}   avg: {analyzer.navg}")


# ======================================================================
class MetricsView(BaseView):
    """Live per-channel scalar metrics table (LabVIEW 'Vibration Level' style)."""

    COLS = ["Channel", "RMS (g)", "Peak (g)", "Pk-Pk (g)", "Crest", "Kurtosis",
            "Skew", "DC (g)", "Dominant (Hz)"]

    def __init__(self) -> None:
        super().__init__()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        hint = QLabel("Crest > 6 or Kurtosis > 8 flagged as impulsive (possible bearing damage).")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(self.table)
        lay.addWidget(hint)

    def _channels_changed(self) -> None:
        self.table.setRowCount(len(self.labels))
        for i, lbl in enumerate(self.labels):
            item = QTableWidgetItem(lbl)
            item.setForeground(pg.mkColor(self.color(i)))
            self.table.setItem(i, 0, item)

    def update_view(self, data: np.ndarray, analyzer: dsp.StreamingAnalyzer) -> None:
        if data.size == 0:
            return
        amp = analyzer.amplitude_spectrum("rms") if analyzer.navg else None
        for i in range(min(data.shape[0], self.table.rowCount())):
            m = dsp.channel_metrics(data[i])
            dom = 0.0
            if amp is not None and analyzer.freq.size > 4:
                skip = max(1, int(2.0 / max(analyzer.freq[1], 1e-9)))
                dom = float(analyzer.freq[skip + int(np.argmax(amp[i][skip:]))])
            vals = [m["rms"], m["peak"], m["pkpk"], m["crest"], m["kurtosis"],
                    m["skew"], m["mean"], dom]
            impulsive = m["crest"] > 6.0 or m["kurtosis"] > 8.0
            for j, v in enumerate(vals, start=1):
                item = self.table.item(i, j)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(i, j, item)
                item.setText(f"{v:.4g}")
                if j in (5, 6) and impulsive:
                    item.setForeground(pg.mkColor(theme.RED))
                else:
                    item.setForeground(pg.mkColor(theme.TEXT))


# ======================================================================
class EnvelopeView(BaseView):
    """Envelope (demodulation) spectrum for bearing-fault detection."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_box = QComboBox()
        self.lo_spin = QDoubleSpinBox()
        self.lo_spin.setRange(10.0, 50000.0)
        self.lo_spin.setValue(500.0)
        self.lo_spin.setSuffix(" Hz")
        self.hi_spin = QDoubleSpinBox()
        self.hi_spin.setRange(20.0, 100000.0)
        self.hi_spin.setValue(5000.0)
        self.hi_spin.setSuffix(" Hz")
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(10.0, 2000.0)
        self.fmax_spin.setValue(500.0)
        self.fmax_spin.setSuffix(" Hz")
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(0.5, 20.0)
        self.len_spin.setValue(4.0)
        self.len_spin.setSuffix(" s")

        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        theme.style_plot(self.plot, "Envelope Spectrum (band-passed Hilbert demodulation)",
                         "Frequency (Hz)", "Envelope amplitude (g)")
        self.curve = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=1.2))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Channel", self.channel_box, "Band", self.lo_spin,
                                  "–", self.hi_spin, "Display F max", self.fmax_spin,
                                  "Length", self.len_spin, "stretch"))
        lay.addWidget(self.plot_widget)

    def _channels_changed(self) -> None:
        cur = self.channel_box.currentIndex()
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(self.labels)
        self.channel_box.setCurrentIndex(max(0, cur) if cur < len(self.labels) else 0)
        self.channel_box.blockSignals(False)

    def update_view(self, data: np.ndarray, fs: float) -> None:
        ch = self.channel_box.currentIndex()
        if data.size == 0 or ch < 0 or ch >= data.shape[0]:
            return
        n = int(self.len_spin.value() * fs)
        y = data[ch, -n:]
        freq, spec = dsp.envelope_spectrum(y, fs, self.lo_spin.value(), self.hi_spin.value())
        if freq.size == 0:
            return
        mask = (freq >= 1.0) & (freq <= self.fmax_spin.value())
        self.curve.setData(freq[mask], spec[mask])
        self.plot.enableAutoRange()
