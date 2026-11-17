"""Dialog: motion magnification of a test video, compared with the sensors.

Pick the camera video and the recording taken at the same time (sine drive);
the rig maps the sensors into global axes. The run finds the frame rate,
selects the sensor blocks and beam regions by itself, sweeps the
amplification factor alpha and shows how far the magnified video agrees with
the accelerometers. See vibrastudio.dsp.magnify.

Optional: sensor-guided synthesis (vibrastudio.dsp.synth). For each frequency
line the factor alpha is taken from the sensors, the video magnification is
checked against them, and the still frame is warped by the shape the sensors
give, so lines the camera cannot resolve are still shown.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

STAGES = {"read": "Reading video", "magnify": "Magnifying (phase phasors)", "alpha": "Alpha sweep",
          "video": "Writing overlay video", "synth": "Sensor-guided synthesis",
          "synth_video": "Writing synthesis video"}


class _Worker(QThread):
    log = Signal(str)
    progress = Signal(str, int, int)
    done = Signal(object, str)
    failed = Signal(str)

    def __init__(self, video, recording, rig, opts, out) -> None:
        super().__init__()
        self.video, self.recording, self.rig, self.opts, self.out = video, recording, rig, opts, out
        self.cancel = False

    def _prog(self, stage: str, i: int, n: int) -> bool:
        self.progress.emit(stage, i, n)
        return not self.cancel

    def run(self) -> None:
        try:
            from ..daq.playback import load_recording
            from ..dsp import magnify as mg
            from ..dsp import ods_video as ov
            from .magnify_render import (overlay_video, save_results, summary_figure, synthesis_figure,
                                         synthesis_video)
            from .ods_video_render import resolve_mapping

            rec = load_recording(self.recording)
            chans, geo, how = resolve_mapping(rec, self.rig, "rig")
            self.log.emit(f"sensor map: {how}")
            model = ov.model_from(geo, chans, rec.data.shape[0])
            acc = ov.node_acceleration(rec.data, model, base_follows=False)
            r = mg.run(self.video, model, acc, rec.fs, log=self.log.emit, progress=self._prog, **self.opts)
            out = Path(self.out)
            out.mkdir(parents=True, exist_ok=True)
            save_results(r, out, self.video, self.recording)
            summary_figure(r, out, f"{Path(self.video).name}  vs  {Path(self.recording).name}")
            overlay_video(r, out, progress=lambda i, n: self._prog("video", i, n))
            if r.synth:
                synthesis_figure(r, out, f"{Path(self.video).name}  vs  {Path(self.recording).name}")
                synthesis_video(r, out, progress=lambda i, n: self._prog("synth_video", i, n))
            self.done.emit(r, str(out))
        except InterruptedError:
            self.failed.emit("Cancelled.")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MagnifyDialog(QDialog):
    def __init__(self, rig=None, recording_path: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Motion magnification vs sensors")
        self.resize(1250, 900)
        self.rig = rig
        self.worker: _Worker | None = None
        self.out_dir = ""

        self.video = QLineEdit()
        bv = QPushButton("Browse…"); bv.clicked.connect(self._browse_video)
        self.rec = QLineEdit(recording_path)
        br = QPushButton("Browse…"); br.clicked.connect(self._browse_rec)
        self.rig_label = QLabel(f"Rig: {rig.name}" if rig else
                                "No rig applied — use Sensor Setup… first (sensor positions and axes).")
        self.rig_label.setWordWrap(True)
        self.fps_mode = QComboBox()
        self.fps_mode.addItems(["Auto from light flicker (60 Hz mains)", "Auto from light flicker (50 Hz mains)",
                                "Fixed frame rate"])
        self.fps = QDoubleSpinBox(); self.fps.setRange(1, 100000); self.fps.setValue(240); self.fps.setSuffix(" fps")
        self.fps_hint = QDoubleSpinBox(); self.fps_hint.setRange(0, 100000); self.fps_hint.setValue(0)
        self.fps_hint.setSpecialValueText("none"); self.fps_hint.setSuffix(" fps")
        self.fps_hint.setToolTip("Rough frame rate (e.g. the camera setting) to choose among flicker aliases")
        self.alphas = QLineEdit("auto")
        self.alphas.setToolTip("auto, or amplification factors separated by commas, e.g. 5, 10, 20, 40, 80")
        self.target = QDoubleSpinBox(); self.target.setRange(1, 200); self.target.setValue(12)
        self.target.setSuffix(" px")
        self.target.setToolTip("auto alphas are centred where the largest block motion is drawn this big")
        self.synth = QCheckBox("Sensor-guided synthesis per frequency")
        self.synth.setToolTip("Alpha from the sensors for every line; the video magnification is checked against "
                              "the sensors, and the still frame is warped by the sensor shape")
        self.synth_freqs = QLineEdit("auto")
        self.synth_freqs.setToolTip("auto (drive + strongest sensor lines), or Hz separated by commas, "
                                    "e.g. 5, 18.9, 37.8. Lines above half the frame rate are allowed.")
        self.synth_source = QComboBox()
        self.synth_source.addItems(["Auto", "Sensors only", "Sensors + video (fusion)"])
        self.synth_source.setToolTip("Auto: fusion when fewer than 4 sensors are used. The video is only used "
                                     "at lines where it passes the check.")
        self.synth_sensors = QLineEdit(); self.synth_sensors.setPlaceholderText("all sensors")
        self.synth_sensors.setToolTip("Labels the shape may use, e.g. S1, S8. The others then test the result.")
        self.synth_px = QDoubleSpinBox(); self.synth_px.setRange(1, 40); self.synth_px.setValue(10)
        self.synth_px.setSuffix(" px")
        self.synth_px.setToolTip("Every line is drawn with this much motion at its largest sensor")
        self.out = QLineEdit()
        bo = QPushButton("…"); bo.clicked.connect(self._browse_out)

        g = QGroupBox("Inputs")
        f = QFormLayout(g)
        for lab, w, b in (("Camera video", self.video, bv), ("Recording (same test)", self.rec, br)):
            row = QHBoxLayout(); row.addWidget(w, 1); row.addWidget(b); f.addRow(lab, row)
        f.addRow("", self.rig_label)
        r2 = QHBoxLayout(); r2.addWidget(self.fps_mode, 1); r2.addWidget(self.fps); r2.addWidget(QLabel("hint"))
        r2.addWidget(self.fps_hint)
        f.addRow("Frame rate", r2)
        r3 = QHBoxLayout(); r3.addWidget(self.alphas, 1); r3.addWidget(QLabel("auto target")); r3.addWidget(self.target)
        f.addRow("Alpha", r3)
        r5 = QHBoxLayout(); r5.addWidget(self.synth)
        for lab, w, st in (("lines", self.synth_freqs, 2), ("shape from", self.synth_source, 0),
                           ("sensors", self.synth_sensors, 1), ("draw", self.synth_px, 0)):
            r5.addWidget(QLabel(lab)); r5.addWidget(w, st)
        f.addRow("Synthesis", r5)
        r4 = QHBoxLayout(); r4.addWidget(self.out, 1); r4.addWidget(bo)
        f.addRow("Output folder", r4)
        note = QLabel("Video and recording must show the same steady sine drive. Camera and DAQ are not "
                      "synchronized: one complex gain aligns them (time offset), the rest is compared point by point.")
        note.setWordWrap(True); note.setStyleSheet("color: gray")
        f.addRow(note)

        self.logbox = QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(500)
        self.image = QLabel("Results appear here"); self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(330); self.image.setStyleSheet("background: white; color: gray")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Sensor", "Height cm", "Sensor µm", "Video µm", "Video/Sensor",
                                              "Phase °"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        split = QSplitter(Qt.Horizontal)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("Per sensor (horizontal image motion at the drive frequency)")); lv.addWidget(self.table)
        lv.addWidget(QLabel("Log")); lv.addWidget(self.logbox)
        split.addWidget(left); split.addWidget(self.image); split.setSizes([450, 800])

        self.bar = QProgressBar(); self.stage = QLabel("")
        self.btn_run = QPushButton("Run"); self.btn_run.setDefault(True)
        self.btn_video = QPushButton("Open video"); self.btn_folder = QPushButton("Open folder")
        self.btn_synth = QPushButton("Open synthesis")
        self.btn_close = QPushButton("Close")
        for b in (self.btn_video, self.btn_synth, self.btn_folder):
            b.setEnabled(False)
        self.btn_synth.clicked.connect(lambda: self._open(Path(self.out_dir) / "synthesis_vs_video.mp4"))
        self.btn_run.clicked.connect(self._run)
        self.btn_video.clicked.connect(lambda: self._open(Path(self.out_dir) / "magnify_vs_sensors.mp4"))
        self.btn_folder.clicked.connect(lambda: self._open(Path(self.out_dir)))
        self.btn_close.clicked.connect(self.close)
        acts = QHBoxLayout()
        acts.addWidget(self.stage); acts.addWidget(self.bar, 1)
        for b in (self.btn_run, self.btn_video, self.btn_synth, self.btn_folder, self.btn_close):
            acts.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addWidget(g); lay.addWidget(split, 1); lay.addLayout(acts)
        self.fps_mode.currentIndexChanged.connect(self._fps_mode)
        self.video.editingFinished.connect(self._suggest_out)
        self.synth.toggled.connect(self._synth_mode)
        self._fps_mode(0)
        self._synth_mode(False)

    # -- inputs ------------------------------------------------------------
    def _fps_mode(self, i: int) -> None:
        self.fps.setEnabled(i == 2)
        self.fps_hint.setEnabled(i != 2)

    def _synth_mode(self, on: bool) -> None:
        for w in (self.synth_freqs, self.synth_source, self.synth_sensors, self.synth_px):
            w.setEnabled(on)

    def _browse_video(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Camera video", "", "Video (*.mp4 *.avi *.mov *.mkv);;All (*)")
        if p:
            self.video.setText(p); self._suggest_out()

    def _browse_rec(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Recording", self.rec.text() or "recordings", "Recordings (*.npz *.csv)")
        if p:
            self.rec.setText(p)

    def _browse_out(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text())
        if p:
            self.out.setText(p)

    def _suggest_out(self) -> None:
        v = self.video.text().strip()
        if v and not self.out.text().strip():
            self.out.setText(str(Path(v).with_name(Path(v).stem + "_magnify")))

    def _opts(self) -> dict:
        i = self.fps_mode.currentIndex()
        o = dict(target_px=self.target.value())
        if i == 2:
            o["fps"] = self.fps.value()
        else:
            o["mains"] = 60.0 if i == 0 else 50.0
            o["fps_hint"] = self.fps_hint.value() or None
        t = self.alphas.text().strip().lower()
        if t and t != "auto":
            o["alphas"] = [float(v) for v in t.replace(";", ",").split(",") if v.strip()]
        if self.synth.isChecked():
            from ..dsp.synth import SynthOptions
            t = self.synth_freqs.text().strip().lower()
            freqs = None if t in ("", "auto") else [float(v) for v in t.replace(";", ",").split(",") if v.strip()]
            use = [v.strip() for v in self.synth_sensors.text().replace(";", ",").split(",") if v.strip()]
            o["synth"] = SynthOptions(freqs=freqs, target_px=self.synth_px.value(),
                                      source=("auto", "sensors", "fusion")[self.synth_source.currentIndex()],
                                      use_sensors=use or None)
        return o

    # -- run -------------------------------------------------------------
    def _run(self) -> None:
        if self.worker is not None:
            self.worker.cancel = True
            self.btn_run.setText("Cancelling…")
            return
        video, rec = self.video.text().strip(), self.rec.text().strip()
        if not (video and Path(video).exists() and rec and Path(rec).exists()):
            QMessageBox.warning(self, "Magnification", "Choose an existing camera video and recording.")
            return
        if self.rig is None:
            QMessageBox.warning(self, "Magnification",
                                "Apply a rig first (ODS tab → Sensor Setup…): positions and axes of the sensors.")
            return
        try:
            opts = self._opts()
        except ValueError:
            QMessageBox.warning(self, "Magnification",
                                "Alpha and synthesis lines: 'auto' or numbers separated by commas.")
            return
        self._suggest_out()
        self.logbox.clear(); self.table.setRowCount(0)
        self.worker = _Worker(video, rec, self.rig, opts, self.out.text().strip())
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.btn_run.setText("Cancel")
        self.worker.start()

    def _on_progress(self, stage: str, i: int, n: int) -> None:
        self.stage.setText(STAGES.get(stage, stage))
        if n:
            self.bar.setMaximum(n); self.bar.setValue(i)
        else:
            self.bar.setMaximum(0)

    def _finish(self) -> None:
        self.worker = None
        self.btn_run.setText("Run")
        self.bar.setMaximum(1); self.bar.setValue(1)

    def _on_done(self, r, out: str) -> None:
        self._finish()
        self.out_dir = out
        self.stage.setText(f"Done: recommended α = {r.alpha_recommended:g}")
        self.btn_video.setEnabled(True); self.btn_folder.setEnabled(True)
        self.btn_synth.setEnabled(bool(r.synth))
        self.table.setRowCount(len(r.per_sensor))
        for i, row in enumerate(r.per_sensor):
            vals = [row["sensor"], f"{row['height_cm']:g}", f"{row['sensor_um']:.1f}", f"{row['video_um']:.1f}",
                    "—" if row["small"] else f"{row['ratio']:.2f}", "—" if row["small"] else f"{row['phase_deg']:+.1f}"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))
        for ln in r.synth:
            self.logbox.appendPlainText(
                f"{ln.name}: α {ln.alpha:,.0f}, video magnification {'valid' if ln.valid else 'NOT valid'}, "
                f"synthesis from {ln.source}")
        pm = QPixmap(str(Path(out) / ("synthesis_shapes.png" if r.synth else "magnify_summary.png")))
        self.image.setPixmap(pm.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logbox.appendPlainText(f"written to {out}")

    def _on_failed(self, msg: str) -> None:
        self._finish()
        self.stage.setText(msg)
        if msg != "Cancelled.":
            QMessageBox.warning(self, "Magnification", msg)

    @staticmethod
    def _open(p: Path) -> None:
        if p.exists() and hasattr(os, "startfile"):
            os.startfile(str(p))  # noqa: S606 - user's own file

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.cancel = True
            self.worker.wait(10000)
        super().closeEvent(e)
