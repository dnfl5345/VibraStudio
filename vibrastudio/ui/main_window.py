"""Main application window: wires DAQ engine, analyzer, recorder and views."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QMainWindow, QMessageBox, QTabWidget, QWidget,
)

from .. import __app_name__, __version__
from ..config import AppConfig, channels_from_detected, default_channel_template
from ..daq.engine import AcquisitionEngine, discover_ai_channels
from ..dsp.analysis import StreamingAnalyzer
from ..recorder import StreamRecorder
from . import theme
from .ods_view import OdsView
from .panels import Sidebar
from .views import (
    EnvelopeView, FrfView, MetricsView, OctaveView, SpectrogramView,
    SpectrumView, TimeSeriesView,
)

GUI_TICK_MS = 60


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle(f"{__app_name__} {__version__} — NI cDAQ Real-Time Vibration Analysis")
        self.resize(1720, 1000)

        self.engine: AcquisitionEngine | None = None
        self.analyzer: StreamingAnalyzer | None = None
        self.recorder = StreamRecorder(Path("recordings"))
        self._read_mark = 0
        self._t_start = 0.0

        # -- layout ------------------------------------------------------
        self.sidebar = Sidebar(cfg)
        self.tabs = QTabWidget()
        self.view_time = TimeSeriesView()
        self.view_spec = SpectrumView()
        self.view_sgram = SpectrogramView()
        self.view_octave = OctaveView()
        self.view_env = EnvelopeView()
        self.view_frf = FrfView()
        self.view_ods = OdsView()
        self.view_metrics = MetricsView()
        self.tabs.addTab(self.view_time, "Time Series")
        self.tabs.addTab(self.view_spec, "FFT Spectrum")
        self.tabs.addTab(self.view_sgram, "Spectrogram")
        self.tabs.addTab(self.view_octave, "Octave")
        self.tabs.addTab(self.view_env, "Envelope")
        self.tabs.addTab(self.view_frf, "FRF / Coherence")
        self.tabs.addTab(self.view_ods, "ODS / Modes")
        self.tabs.addTab(self.view_metrics, "Metrics")

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.sidebar)
        lay.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready — select source and press Start Streaming")

        # -- wiring --------------------------------------------------------
        self.sidebar.start_requested.connect(self.start_streaming)
        self.sidebar.stop_requested.connect(self.stop_streaming)
        self.sidebar.record_toggled.connect(self.toggle_record)
        self.sidebar.analysis_changed.connect(self.apply_analysis_settings)
        self.sidebar.channels_changed.connect(self.refresh_channel_dependent_ui)
        self.sidebar.detect_requested.connect(self.detect_hardware)
        self.sidebar.default_requested.connect(self.load_default_channels)
        self.sidebar.save_config.connect(self.save_config)
        self.sidebar.load_config.connect(self.load_config)

        self.view_spec.reset_btn.clicked.connect(
            lambda: self.analyzer.reset_averaging() if self.analyzer else None)

        self.timer = QTimer(self)
        self.timer.setInterval(GUI_TICK_MS)
        self.timer.timeout.connect(self.on_tick)

        # Populate channel table (auto-detect hardware if present).
        if not cfg.channels:
            detected = discover_ai_channels()
            if detected:
                cfg.channels = channels_from_detected(detected)
            else:
                cfg.channels = default_channel_template()
                self.sidebar.source_box.setCurrentIndex(1)  # simulation
        self.sidebar.load_from_config(cfg)

    # ------------------------------------------------------------------
    def refresh_channel_dependent_ui(self) -> None:
        self.sidebar.apply_to_config(self.cfg)
        enabled = self.cfg.enabled_channels()
        labels = [c.label for c in enabled]
        for view in (self.view_time, self.view_spec, self.view_sgram,
                     self.view_octave, self.view_env, self.view_frf, self.view_metrics):
            view.set_channels(labels)
        self.view_ods.set_channel_configs(enabled)

    def detect_hardware(self) -> None:
        detected = discover_ai_channels()
        if not detected:
            QMessageBox.warning(self, "Detect", "No NI-DAQmx AI channels found.\n"
                                "Check NI MAX / device power, or use Simulation mode.")
            return
        self.cfg.channels = channels_from_detected(detected)
        self.sidebar.channel_table.load(self.cfg.channels)
        self.statusBar().showMessage(f"Detected {len(detected)} AI channels — mapped first {len(self.cfg.channels)}")

    def load_default_channels(self) -> None:
        self.cfg.channels = default_channel_template()
        self.sidebar.channel_table.load(self.cfg.channels)

    # ------------------------------------------------------------------
    def start_streaming(self) -> None:
        self.sidebar.apply_to_config(self.cfg)
        enabled = self.cfg.enabled_channels()
        if not enabled:
            QMessageBox.warning(self, "Channels", "Enable at least one channel.")
            return

        try:
            self.engine = AcquisitionEngine(enabled, self.cfg.acquisition)
            self.engine.start()
        except Exception as exc:
            self.engine = None
            QMessageBox.critical(
                self, "DAQ Error",
                f"Failed to start acquisition:\n{exc}\n\n"
                "Tip: switch Source to 'Simulation' to run without hardware.")
            return

        self.analyzer = StreamingAnalyzer(len(enabled), self.cfg.acquisition.sample_rate)
        self.apply_analysis_settings()
        self.refresh_channel_dependent_ui()
        self._read_mark = 0
        self._t_start = time.perf_counter()
        self.sidebar.set_running(True)
        mode = "Simulation" if self.cfg.acquisition.simulate else "Hardware"
        self.statusBar().showMessage(
            f"Streaming {len(enabled)} ch @ {self.cfg.acquisition.sample_rate:.0f} Hz ({mode})")
        self.timer.start()

    def stop_streaming(self) -> None:
        self.timer.stop()
        if self.sidebar.record_btn.isChecked():
            self.sidebar.record_btn.setChecked(False)
        if self.engine is not None:
            self.engine.stop()
        self.sidebar.set_running(False)
        self.sidebar.set_stat("State", "Stopped")
        self.statusBar().showMessage("Stopped")

    def apply_analysis_settings(self) -> None:
        if self.analyzer is None:
            return
        self.sidebar.apply_to_config(self.cfg)
        a = self.cfg.analysis
        self.analyzer.configure(
            nfft=a.fft_size, window=a.window, averaging=a.averaging,
            avg_count=a.avg_count, overlap=a.overlap,
            ref_index=self.cfg.reference_index(),
        )

    # ------------------------------------------------------------------
    def toggle_record(self, on: bool) -> None:
        if self.engine is None or not self.engine.running:
            if on:
                self.sidebar.record_btn.setChecked(False)
                QMessageBox.information(self, "Record", "Start streaming first.")
            return
        if on:
            names = [c.label for c in self.cfg.enabled_channels()]
            self.recorder.start(self.engine.fs, names)
            self.engine.add_listener(self.recorder.feed)
            self.sidebar.record_btn.setText("●  Recording…")
        else:
            self.engine.remove_listener(self.recorder.feed)
            path = self.recorder.stop(write_csv=self.sidebar.csv_chk.isChecked())
            self.sidebar.record_btn.setText("●  Record")
            if path is not None:
                self.statusBar().showMessage(f"Recording saved: {path}")

    # ------------------------------------------------------------------
    def on_tick(self) -> None:
        engine, analyzer = self.engine, self.analyzer
        if engine is None or analyzer is None:
            return
        if engine.error:
            self.statusBar().showMessage(f"DAQ error: {engine.error}")
            self.stop_streaming()
            return

        # Feed the analyzer with everything new since the last tick.
        block, self._read_mark = engine.ring.read_since(
            self._read_mark, max_samples=8 * analyzer.nfft)
        if block.size:
            analyzer.push(block)

        fs = engine.fs
        total = engine.ring.total
        elapsed = time.perf_counter() - self._t_start
        self.sidebar.set_stat("State", "● Streaming" + ("  [REC]" if self.recorder.recording else ""))
        self.sidebar.set_stat("Samples/ch", f"{total:,}")
        self.sidebar.set_stat("Elapsed", f"{elapsed:.1f} s")
        self.sidebar.set_stat("Buffer", f"{min(100, 100*total/engine.ring.capacity):.0f} %")
        self.sidebar.set_stat("Avg count", str(analyzer.navg))

        idx = self.tabs.currentIndex()
        widget = self.tabs.currentWidget()
        t_now = total / fs
        if widget is self.view_time:
            n = int(self.view_time.window_spin.value() * fs)
            self.view_time.update_view(engine.ring.latest(n), fs, t_now)
        elif widget is self.view_spec:
            self.view_spec.update_view(analyzer)
        elif widget is self.view_sgram:
            n = int(self.view_sgram.span_spin.value() * fs)
            self.view_sgram.update_view(engine.ring.latest(n), fs)
        elif widget is self.view_octave:
            self.view_octave.update_view(analyzer)
        elif widget is self.view_env:
            n = int(self.view_env.len_spin.value() * fs)
            self.view_env.update_view(engine.ring.latest(n), fs)
        elif widget is self.view_frf:
            ref = self.cfg.enabled_channels()[self.cfg.reference_index()].label
            self.view_frf.update_view(analyzer, ref)
        elif widget is self.view_ods:
            self.view_ods.update_view(analyzer)
        elif widget is self.view_metrics:
            n = int(2.0 * fs)
            self.view_metrics.update_view(engine.ring.latest(n), analyzer)
        _ = idx

        # Reset-averaging button on the spectrum view.
        # (connected lazily so the analyzer instance is current)

    # ------------------------------------------------------------------
    def save_config(self) -> None:
        self.sidebar.apply_to_config(self.cfg)
        path, _ = QFileDialog.getSaveFileName(self, "Save configuration",
                                              "vibrastudio_config.json", "JSON (*.json)")
        if path:
            self.cfg.save(path)
            self.statusBar().showMessage(f"Config saved: {path}")

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "",
                                              "JSON (*.json)")
        if path:
            try:
                self.cfg = AppConfig.load(path)
            except Exception as exc:
                QMessageBox.critical(self, "Config", f"Failed to load config:\n{exc}")
                return
            self.sidebar.load_from_config(self.cfg)
            self.statusBar().showMessage(f"Config loaded: {path}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_streaming()
        super().closeEvent(event)
