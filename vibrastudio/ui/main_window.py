"""Main application window: wires DAQ engine, analyzer, recorder and views."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QMainWindow, QMessageBox, QTabWidget, QWidget,
)

from .. import __app_name__, __version__
from ..config import (
    AppConfig, ChannelConfig, channels_from_detected, default_channel_template,
)
from ..daq.engine import (
    AcquisitionEngine, ConnectResult, auto_connect, unreachable_chassis,
)
from ..daq.playback import PlaybackBackend, RecordingData, load_recording
from ..dsp.analysis import StreamingAnalyzer
from ..recorder import StreamRecorder
from . import theme
from .health_view import HealthView
from .trigger_view import TriggerView
from .impact_view import ImpactView
from .ods_view import OdsView
from .panels import Sidebar
from .views import (
    EnvelopeView, FrfView, MetricsView, OctaveView, SpectrogramView,
    SpectrumView, TimeSeriesView,
)

GUI_TICK_MS = 60


class MainWindow(QMainWindow):
    _connect_finished = Signal(object, bool)  # (ConnectResult, was_auto)

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
        self.view_impact = ImpactView()
        self.view_metrics = MetricsView()
        self.view_health = HealthView()
        self.view_trigger = TriggerView()
        self.tabs.addTab(self.view_time, "Time Series")
        self.tabs.addTab(self.view_spec, "FFT Spectrum")
        self.tabs.addTab(self.view_sgram, "Spectrogram")
        self.tabs.addTab(self.view_octave, "Octave")
        self.tabs.addTab(self.view_env, "Envelope")
        self.tabs.addTab(self.view_frf, "FRF / Coherence")
        self.tabs.addTab(self.view_ods, "ODS / Modes")
        self.tabs.addTab(self.view_impact, "Impact Test")
        self.tabs.addTab(self.view_trigger, "Trigger")
        self.tabs.addTab(self.view_metrics, "Metrics")
        self.tabs.addTab(self.view_health, "DAQ Health")

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
        self.sidebar.open_file_requested.connect(self.open_recording)
        self.sidebar.source_box.currentIndexChanged.connect(self._on_source_changed)
        self.view_ods.channels_mapped.connect(self._on_channels_mapped)
        self.view_impact.beam_preset_requested.connect(self.apply_beam_preset)
        self.view_impact.mode_frequency_selected.connect(self._show_mode_in_ods)
        self.view_trigger.start_stream_requested.connect(self._start_for_trigger)
        self.view_trigger.open_recording_requested.connect(self._open_capture)
        self.playback: RecordingData | None = None

        self.view_spec.reset_btn.clicked.connect(
            lambda: self.analyzer.reset_averaging() if self.analyzer else None)

        self.timer = QTimer(self)
        self.timer.setInterval(GUI_TICK_MS)
        self.timer.timeout.connect(self.on_tick)

        self._connect_finished.connect(self._on_connect_finished)
        self._connecting = False

        # Populate the table immediately, then look for hardware in the
        # background (network chassis reservation can take a few seconds).
        if not cfg.channels:
            cfg.channels = default_channel_template()
        self.sidebar.load_from_config(cfg)
        self.view_ods.set_geometry_dict(cfg.geometry)
        if not cfg.acquisition.simulate:
            QTimer.singleShot(300, lambda: self.detect_hardware(auto=True))

    # ------------------------------------------------------------------
    def refresh_channel_dependent_ui(self) -> None:
        self.sidebar.apply_to_config(self.cfg)
        enabled = self.cfg.enabled_channels()
        labels = [c.label for c in enabled]
        for view in (self.view_time, self.view_spec, self.view_sgram,
                     self.view_octave, self.view_env, self.view_frf, self.view_metrics):
            view.set_channels(labels)
        self.view_ods.set_channel_configs(enabled)
        self.view_impact.set_channel_configs(enabled)
        self.view_health.set_channel_configs(enabled)
        self.view_trigger.set_channels(labels)
        self.sidebar.set_channel_count(len(enabled), len(self.cfg.channels))

    # -- device auto-connect ---------------------------------------------
    def detect_hardware(self, auto: bool = False) -> None:
        """Scan for DAQ hardware; reserve network chassis if needed (async)."""
        if self._connecting:
            return
        self._connecting = True
        self.sidebar.detect_btn.setEnabled(False)
        self.sidebar.set_device_info("Connecting… scanning NI-DAQmx devices")
        self.statusBar().showMessage("Connecting to DAQ hardware…")

        def worker() -> None:
            try:
                result = auto_connect(reserve=True, prefer_iepe=True)
            except Exception as exc:  # defensive: never kill the GUI thread
                result = ConnectResult(messages=[f"auto-connect crashed: {exc}"])
            self._connect_finished.emit(result, auto)

        threading.Thread(target=worker, daemon=True, name="daq-connect").start()

    def _on_connect_finished(self, res: ConnectResult, was_auto: bool) -> None:
        self._connecting = False
        self.sidebar.detect_btn.setEnabled(True)

        if not res.ok:
            self.sidebar.set_device_info(res.summary(), connected=False)
            if was_auto:
                # Silent fallback at startup: keep the app usable without HW.
                self.sidebar.source_box.setCurrentIndex(1)
                self.statusBar().showMessage(
                    "No DAQ hardware found — Simulation mode (press Connect DAQ to retry)")
            else:
                QMessageBox.warning(
                    self, "Connect DAQ",
                    "No NI-DAQmx AI channels found.\n\n"
                    + "\n".join(res.messages[-3:] or ["Check device power / Ethernet link."])
                    + "\n\nYou can keep working in Simulation mode.")
            return

        # Success: map channels and switch to hardware source (idle only).
        if self.engine is None or not self.engine.running:
            self.cfg.channels = channels_from_detected(res.channels)
            self.sidebar.channel_table.load(self.cfg.channels)
            self.sidebar.source_box.setCurrentIndex(0)
        self.sidebar.set_device_info(res.summary(), connected=True)
        note = " · chassis reserved" if res.reserved else ""
        if res.added:
            note += f" · new chassis registered: {', '.join(res.added)}"
        self.statusBar().showMessage(
            f"DAQ connected: {len(res.chassis)} chassis, "
            f"{len(res.channels)} AI channels mapped{note}")

    def load_default_channels(self) -> None:
        self.cfg.channels = default_channel_template()
        self.sidebar.channel_table.load(self.cfg.channels)

    def _on_channels_mapped(self, mapped: list) -> None:
        """ODS auto-map rewrote node/direction/labels — sync the table.

        Merge by physical channel: auto-map renames labels (S<node>_<axis>),
        so labels can no longer identify rows."""
        by_phys = {c.physical_channel: c for c in mapped}
        for ch in self.cfg.channels:
            m = by_phys.get(ch.physical_channel)
            if m is not None:
                ch.label = m.label
                ch.point, ch.direction, ch.position = m.point, m.direction, m.position
        self.sidebar.channel_table.load(self.cfg.channels)
        self.statusBar().showMessage(
            f"Channels auto-mapped onto {self.view_ods.geometry.name} "
            f"({self.view_ods.geometry.n_nodes} nodes)")

    # -- impact test helpers -----------------------------------------------
    def apply_beam_preset(self) -> None:
        """Physical rig: 50 cm free cantilever, S1..S4 at 10/20/30/40 cm from
        the tip (positions from the clamped base: 0.8/0.6/0.4/0.2), and the
        9th channel used as the impact hammer.

        Also builds the matching ODS geometry: base(fixed) - S4..S1 - tip,
        node ids 1..6 bottom-up, so the 3D deflection animation uses the
        true sensor heights and a clamped base."""
        import numpy as np

        from ..geometry import Geometry
        from .impact_view import FREE_LEN_CM, SENSOR_FROM_TIP_CM

        # sensor point -> beam node (bottom-up: 1=base, 2=S4 ... 5=S1, 6=tip)
        point_to_node = {1: 5, 2: 4, 3: 3, 4: 2}
        for ch in self.cfg.channels:
            if 1 <= ch.point <= len(SENSOR_FROM_TIP_CM):
                d = SENSOR_FROM_TIP_CM[ch.point - 1]
                ch.position = (FREE_LEN_CM - d) / FREE_LEN_CM
                ch.point = point_to_node.get(ch.point, ch.point)
        if len(self.cfg.channels) >= 9:
            ham = self.cfg.channels[8]
            ham.label = "HAMMER"
            ham.enabled = True
            ham.is_reference = False
            ham.point = 6            # free tip (hit location)
            ham.position = 1.0
        self.sidebar.channel_table.load(self.cfg.channels)

        heights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        nodes = np.array([[0.0, 0.0, z] for z in heights])
        geo = Geometry("beam-rig", nodes, [(i, i + 1) for i in range(5)],
                       {"shape": "beam-rig", "free_len_cm": FREE_LEN_CM},
                       fixed=[0])
        self.view_ods.geometry = geo
        self.view_ods._user_geometry = True
        self.view_ods._rebuild_static()
        self.statusBar().showMessage(
            "Beam preset: S1..S4 at 10/20/30/40 cm from tip (nodes 5..2), "
            "base fixed, ch9 = HAMMER at tip")

    def _show_mode_in_ods(self, fn: float) -> None:
        self.view_ods.freq_spin.setValue(fn)
        self.tabs.setCurrentWidget(self.view_ods)
        self.statusBar().showMessage(f"ODS frequency set to {fn:.2f} Hz")

    # -- file playback -----------------------------------------------------
    def _on_source_changed(self, index: int) -> None:
        if index == 2 and self.playback is None:
            self.open_recording()

    def open_recording(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open recording", "recordings",
                "Recordings (*.npz *.csv);;All files (*)")
        if not path:
            return
        try:
            rec = load_recording(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Recording", f"Failed to load file:\n{exc}")
            return

        self.playback = rec
        self.cfg.channels = self._channels_from_labels(rec.channels)
        self.sidebar.channel_table.load(self.cfg.channels)
        self.sidebar.source_box.setCurrentIndex(2)
        self.sidebar.fs_box.setCurrentText(f"{rec.fs:g}")
        self.sidebar.file_label.setText(
            f"{Path(rec.path).name} — {len(rec.channels)} ch · "
            f"{rec.fs:g} Hz · {rec.duration:.1f} s")
        self.statusBar().showMessage(
            f"Recording loaded: {Path(rec.path).name} ({rec.duration:.1f} s)")

        # Start replaying immediately — an idle screen after picking a file
        # reads as "the file did not load".
        if self.engine is not None and self.engine.running:
            self.stop_streaming()
        self.start_streaming()

    @staticmethod
    def _channels_from_labels(labels: list[str]) -> list[ChannelConfig]:
        """Rebuild channel geometry from recorded labels like 'S3_Z'."""
        import re

        out: list[ChannelConfig] = []
        n_points = max(1, len(labels) // 2)
        for i, label in enumerate(labels):
            m = re.match(r"^S(\d+)[_-]?([XYZ])$", label.strip(), re.IGNORECASE)
            point = int(m.group(1)) if m else i // 2 + 1
            direction = m.group(2).upper() if m else ("X" if i % 2 == 0 else "Z")
            out.append(ChannelConfig(
                physical_channel=f"file/ch{i}", label=label, enabled=True,
                point=point, direction=direction,
                position=point / max(n_points, point),
                sensitivity_mv_g=100.0, is_reference=(i == 0)))
        return out

    # ------------------------------------------------------------------
    def start_streaming(self) -> None:
        self.sidebar.apply_to_config(self.cfg)
        is_playback = self.sidebar.source_box.currentIndex() == 2

        if is_playback:
            if self.playback is None:
                self.open_recording()
                if self.playback is None:
                    return
            # Sample rate always comes from the file.
            self.cfg.acquisition.sample_rate = self.playback.fs
            self.sidebar.fs_box.setCurrentText(f"{self.playback.fs:g}")

        enabled = self.cfg.enabled_channels()
        if not enabled:
            QMessageBox.warning(self, "Channels", "Enable at least one channel.")
            return

        backend = None
        if is_playback:
            rows = [i for i, c in enumerate(self.cfg.channels) if c.enabled]
            if max(rows) >= self.playback.data.shape[0]:
                QMessageBox.warning(self, "Playback",
                                    "Channel table no longer matches the loaded file — reloading it.")
                self.cfg.channels = self._channels_from_labels(self.playback.channels)
                self.sidebar.channel_table.load(self.cfg.channels)
                return
            backend = PlaybackBackend(self.playback, self.cfg.acquisition.chunk_size,
                                      channel_rows=rows)
        elif not self.cfg.acquisition.simulate:
            # A chassis that was unplugged stays registered in DAQmx with its
            # old module list; starting a task with it fails (-201401).
            dead = unreachable_chassis([c.physical_channel for c in enabled])
            if dead:
                n_dead = sum(1 for c in enabled
                             if any(c.physical_channel.startswith(d + "Mod") for d in dead))
                answer = QMessageBox.question(
                    self, "Chassis not reachable",
                    "These chassis do not answer on the network:\n  "
                    + "\n  ".join(dead)
                    + f"\n\nDisable their {n_dead} channels and start with the rest?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if answer != QMessageBox.Yes:
                    return
                for c in self.cfg.channels:
                    if any(c.physical_channel.startswith(d + "Mod") for d in dead):
                        c.enabled = False
                self.sidebar.channel_table.load(self.cfg.channels)
                enabled = self.cfg.enabled_channels()
                if not enabled:
                    QMessageBox.warning(self, "Channels", "No reachable channels left.")
                    return

        try:
            self.engine = AcquisitionEngine(enabled, self.cfg.acquisition, backend=backend)
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
        mode = ("Playback" if is_playback
                else "Simulation" if self.cfg.acquisition.simulate else "Hardware")
        self.statusBar().showMessage(
            f"Streaming {len(enabled)} ch @ {self.cfg.acquisition.sample_rate:.0f} Hz ({mode})")
        self.timer.start()

    def _open_capture(self, path: str) -> None:
        """Re-analyze a trigger capture in every tab via File playback."""
        if self.engine is not None and self.engine.running:
            self.stop_streaming()
        self.open_recording(path)

    def _start_for_trigger(self) -> None:
        if self.engine is None or not self.engine.running:
            self.start_streaming()
        if self.engine is None or not self.engine.running:
            self.view_trigger.stream_stopped()     # start failed: un-press Arm

    def stop_streaming(self) -> None:
        self.view_trigger.stream_stopped()
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
        if isinstance(engine.backend, PlaybackBackend):
            pb = engine.backend
            pct = 100.0 * pb.position / max(1, pb.n)
            state = f"▶ Playback {pct:.0f}%" + (f" (loop {pb.loops + 1})" if pb.loops else "")
        else:
            state = "● Streaming"
        # The trigger watches every new sample, whichever tab is visible.
        self.view_trigger.process(engine)
        self.sidebar.set_stat("State", state + ("  [REC]" if self.recorder.recording else "")
                              + self.view_trigger.badge())
        self.sidebar.set_stat("Samples/ch", f"{total:,}")
        self.sidebar.set_stat("Elapsed", f"{elapsed:.1f} s")
        self.sidebar.set_stat("Buffer", f"{min(100, 100*total/engine.ring.capacity):.0f} %")
        self.sidebar.set_stat("Avg count", str(analyzer.navg))
        rate = engine.measured_rate()
        self.sidebar.set_stat(
            "Rate", f"{rate:,.0f} S/s ({100 * (rate / fs - 1):+.1f} %)" if rate else "—")

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
        elif widget is self.view_impact:
            self.view_impact.update_view(engine)
        elif widget is self.view_trigger:
            self.view_trigger.update_view(engine)
        elif widget is self.view_health:
            self.view_health.update_view(engine)
        elif widget is self.view_metrics:
            n = int(2.0 * fs)
            self.view_metrics.update_view(engine.ring.latest(n), analyzer)
        _ = idx

        # Reset-averaging button on the spectrum view.
        # (connected lazily so the analyzer instance is current)

    # ------------------------------------------------------------------
    def save_config(self) -> None:
        self.sidebar.apply_to_config(self.cfg)
        self.cfg.geometry = self.view_ods.geometry_dict()
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
            self.view_ods.set_geometry_dict(self.cfg.geometry)
            self.statusBar().showMessage(f"Config loaded: {path}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_streaming()
        super().closeEvent(event)
