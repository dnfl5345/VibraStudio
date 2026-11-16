"""Dialog: export a 3D ODS video (3D / front / side / top) from a recording.

Modes
  Resonance peaks   shapes at the strongest peaks, each animated a few seconds
  Frequencies       shapes at frequencies you type (e.g. 7, 46.1, 128)
  Frequency sweep   frequency slides through a band on one amplitude scale,
                    so the shape grows and turns at resonances
  Time response     the measured motion itself (FFT-integrated displacement),
                    in slow motion: impacts, start-up, beating
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from ..daq.playback import load_recording
from ..dsp.ods_video import model_from
from .ods_video_render import (
    Renderer, VideoOptions, build_frames, ffmpeg_exe, resolve_mapping, write_video,
)

MODES = [("Resonance peaks", "peaks"), ("Frequencies", "list"),
         ("Frequency sweep", "sweep"), ("Time response", "time")]
SIZES = [("1920 × 1080", (1920, 1080)), ("1280 × 720", (1280, 720))]
MAPS = [("Auto (rig if it matches)", "auto"), ("Rig, by physical channel", "rig"),
        ("File's own channel map", "file")]


def _spin(lo, hi, val, dec=2, suffix="", step=1.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val); s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    return s


class _Worker(QThread):
    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, rec, model, opts, out, name) -> None:
        super().__init__()
        self.rec, self.model, self.opts, self.out, self.name = rec, model, opts, out, name
        self.cancel = False

    def run(self) -> None:
        try:
            frames, strip = build_frames(self.rec.data, self.rec.fs, self.model, self.opts, self.name)

            def cb(i: int, n: int) -> bool:
                if i % 5 == 0:
                    self.progress.emit(i, n)
                return not self.cancel

            path = write_video(frames, self.model, strip, self.out, self.opts.size, self.opts.fps, cb,
                               stills=(len(frames) // 2,))
            self.done.emit(str(path))
        except InterruptedError:
            Path(self.out).unlink(missing_ok=True)
            self.failed.emit("Cancelled.")
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(exc))


class OdsVideoDialog(QDialog):
    def __init__(self, rig=None, recording_path: str = "", recordings_dir: str = "recordings",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export 3D ODS video")
        self.setMinimumWidth(760)
        self.rig = rig
        self.rec = None
        self.model = None
        self._dir = recordings_dir
        self.worker: _Worker | None = None

        # source
        self.path = QLineEdit(recording_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.map_box = QComboBox()
        self.map_box.addItems([m[0] for m in MAPS])
        self.map_info = QLabel("—")
        self.map_info.setWordWrap(True)
        src = QGroupBox("Recording")
        f = QFormLayout(src)
        row = QHBoxLayout(); row.addWidget(self.path, 1); row.addWidget(browse)
        f.addRow("File", row)
        f.addRow("Sensor map", self.map_box)
        f.addRow("", self.map_info)

        # mode + parameters
        self.mode = QComboBox()
        self.mode.addItems([m[0] for m in MODES])
        self.stack = QStackedWidget()
        self.fmin = _spin(0.1, 20000, 3.0, suffix=" Hz")
        self.fmax = _spin(0.1, 20000, 500.0, suffix=" Hz")
        self.npk = QSpinBox(); self.npk.setRange(1, 20); self.npk.setValue(6)
        self.each = _spin(0.5, 30, 3.0, 1, " s")
        p_peaks = QWidget(); fp = QFormLayout(p_peaks)
        fp.addRow("Band from", self.fmin); fp.addRow("to", self.fmax)
        fp.addRow("Number of peaks", self.npk); fp.addRow("Seconds per shape", self.each)
        self.freqs = QLineEdit("7.0, 46.1, 128.1")
        self.each2 = _spin(0.5, 30, 3.0, 1, " s")
        p_list = QWidget(); fl = QFormLayout(p_list)
        fl.addRow("Frequencies (Hz)", self.freqs); fl.addRow("Seconds per shape", self.each2)
        self.s1 = _spin(0.1, 20000, 3.0, suffix=" Hz")
        self.s2 = _spin(0.1, 20000, 300.0, suffix=" Hz")
        self.ssec = _spin(2, 300, 20.0, 1, " s")
        p_sweep = QWidget(); fs_ = QFormLayout(p_sweep)
        fs_.addRow("From", self.s1); fs_.addRow("to", self.s2); fs_.addRow("Video length", self.ssec)
        self.slow = _spin(1, 1000, 20.0, 1, " ×")
        self.lo_x = _spin(0.1, 5000, 3.0, 1, " Hz")
        self.lo_y = _spin(0.1, 5000, 3.0, 1, " Hz")
        self.lo_z = _spin(0.1, 5000, 3.0, 1, " Hz")
        self.hi = _spin(1, 20000, 400.0, 0, " Hz")
        p_time = QWidget(); ft = QFormLayout(p_time)
        ft.addRow("Slow motion", self.slow)
        lo_row = QHBoxLayout()
        for lab, w in (("X", self.lo_x), ("Y", self.lo_y), ("Z", self.lo_z)):
            lo_row.addWidget(QLabel(lab)); lo_row.addWidget(w)
        ft.addRow("Low cut per axis", lo_row)
        ft.addRow("High cut", self.hi)
        hint = QLabel("Tip: a hammer hit right on a sensor makes a slow, symmetric wobble after "
                      "integration. Raise that axis' low cut above it if the structure has no mode there.")
        hint.setWordWrap(True); hint.setStyleSheet("color: gray")
        ft.addRow(hint)
        for w in (p_peaks, p_list, p_sweep, p_time):
            self.stack.addWidget(w)
        self.mode.currentIndexChanged.connect(self.stack.setCurrentIndex)

        self.t0 = _spin(0, 1e6, 0.0, 3, " s", 0.1)
        self.t1 = _spin(0, 1e6, 0.0, 3, " s", 0.1)
        self.t1.setSpecialValueText("end")
        self.res = _spin(0.05, 20, 0.5, 2, " Hz", 0.1)
        from PySide6.QtWidgets import QCheckBox
        self.base_chk = QCheckBox("Clamp moves with a sensor right next to it (shaker base)")
        self.base_chk.setChecked(True)
        self.base_chk.setToolTip("Fixed nodes stay at rest, unless a sensor sits within 5 % of the span:\n"
                                 "then the clamp is taken to move with it (base excitation).")
        par = QGroupBox("What to show")
        fpar = QFormLayout(par)
        fpar.addRow("Mode", self.mode)
        fpar.addRow(self.stack)
        win = QHBoxLayout(); win.addWidget(self.t0); win.addWidget(QLabel("to")); win.addWidget(self.t1)
        fpar.addRow("Use data from", win)
        fpar.addRow("Spectral resolution", self.res)
        fpar.addRow(self.base_chk)

        # output
        self.out = QLineEdit()
        ob = QPushButton("…"); ob.clicked.connect(self._browse_out)
        self.size = QComboBox(); self.size.addItems([s[0] for s in SIZES])
        self.fps = QSpinBox(); self.fps.setRange(10, 60); self.fps.setValue(30)
        outg = QGroupBox("Video")
        fo = QFormLayout(outg)
        r2 = QHBoxLayout(); r2.addWidget(self.out, 1); r2.addWidget(ob)
        fo.addRow("Save as", r2)
        r3 = QHBoxLayout(); r3.addWidget(self.size); r3.addWidget(QLabel("fps")); r3.addWidget(self.fps)
        r3.addStretch(1)
        fo.addRow("Size", r3)
        if ffmpeg_exe() is None:
            fo.addRow(QLabel("ffmpeg not found: the video is saved as an animated GIF."))

        # preview + actions
        self.preview = QLabel("Preview appears here")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet("background: white; color: gray; border: 1px solid #555")
        self.bar = QProgressBar(); self.bar.setVisible(False)
        self.status = QLabel("")
        self.btn_prev = QPushButton("Preview")
        self.btn_go = QPushButton("Export")
        self.btn_go.setDefault(True)
        self.btn_close = QPushButton("Close")
        self.btn_prev.clicked.connect(self._preview)
        self.btn_go.clicked.connect(self._export)
        self.btn_close.clicked.connect(self._close)
        acts = QHBoxLayout()
        acts.addWidget(self.status, 1); acts.addWidget(self.btn_prev)
        acts.addWidget(self.btn_go); acts.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        left = QVBoxLayout(); left.addWidget(src); left.addWidget(par); left.addWidget(outg)
        top.addLayout(left, 1)
        lay.addLayout(top)
        lay.addWidget(self.preview, 1)
        lay.addWidget(self.bar)
        lay.addLayout(acts)

        self.path.editingFinished.connect(self._load)
        self.map_box.currentIndexChanged.connect(lambda *_: self._remap())
        self.mode.currentIndexChanged.connect(lambda *_: self._suggest_out())
        if recording_path:
            self._load()

    # -- source --------------------------------------------------------------
    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Recording", self._dir, "Recordings (*.npz *.csv)")
        if p:
            self.path.setText(p)
            self._load()

    def _load(self) -> None:
        p = self.path.text().strip()
        if not p:
            return
        try:
            self.rec = load_recording(p)
        except Exception as exc:  # noqa: BLE001
            self.rec = None
            self.map_info.setText(f"Cannot read file: {exc}")
            return
        self.t1.setMaximum(self.rec.duration); self.t0.setMaximum(self.rec.duration)
        self.fmax.setMaximum(self.rec.fs / 2); self.s2.setMaximum(self.rec.fs / 2)
        self._remap()
        self._suggest_out()

    def _remap(self) -> None:
        if self.rec is None:
            return
        prefer = MAPS[self.map_box.currentIndex()][1]
        chans, geo, how = resolve_mapping(self.rec, self.rig, prefer)
        self.model = model_from(geo, chans, self.rec.data.shape[0])
        if prefer == "rig" and not how.startswith("rig"):
            how = "no rig match — " + how
        self.map_info.setText(
            f"{Path(self.rec.path).name}: {len(self.rec.channels)} ch, {self.rec.fs:g} Hz, "
            f"{self.rec.duration:.1f} s.  Map: {how}; {len(self.model.rows)} inputs on "
            f"{self.model.n_nodes} nodes ({geo.name}).")

    def _suggest_out(self) -> None:
        if self.rec is None:
            return
        mode = MODES[self.mode.currentIndex()][1]
        p = Path(self.rec.path)
        self.out.setText(str(p.with_name(f"{p.stem}_ods_{mode}.mp4")))

    def _browse_out(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save video", self.out.text(), "MP4 (*.mp4);;GIF (*.gif)")
        if p:
            self.out.setText(p)

    # -- options ---------------------------------------------------------------
    def options(self) -> VideoOptions:
        mode = MODES[self.mode.currentIndex()][1]
        o = VideoOptions(mode=mode, size=SIZES[self.size.currentIndex()][1], fps=float(self.fps.value()),
                         t0=self.t0.value(), t1=(self.t1.value() or None), resolution_hz=self.res.value(),
                         base_follows=self.base_chk.isChecked())
        if mode == "peaks":
            o.fmin, o.fmax, o.n_peaks, o.seconds_each = (self.fmin.value(), self.fmax.value(),
                                                         self.npk.value(), self.each.value())
        elif mode == "list":
            try:
                o.freqs = tuple(float(v) for v in self.freqs.text().replace(";", ",").split(",") if v.strip())
            except ValueError:
                raise ValueError("Frequencies: numbers separated by commas, e.g. 7, 46.1, 128")
            o.seconds_each = self.each2.value()
        elif mode == "sweep":
            o.fmin, o.fmax, o.sweep_seconds = self.s1.value(), self.s2.value(), self.ssec.value()
        else:
            o.slow, o.hi_hz = self.slow.value(), self.hi.value()
            o.lo_xyz = (self.lo_x.value(), self.lo_y.value(), self.lo_z.value())
        return o

    def _ready(self) -> bool:
        if self.rec is None:
            self._load()
        if self.rec is None or self.model is None:
            QMessageBox.warning(self, "3D ODS video", "Choose a recording first.")
            return False
        return True

    # -- actions -------------------------------------------------------------
    def _preview(self) -> None:
        if not self._ready():
            return
        try:
            o = self.options()
            frames, strip = build_frames(self.rec.data, self.rec.fs, self.model, o, Path(self.rec.path).stem)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "3D ODS video", str(exc))
            return
        fr = frames[len(frames) // 2]
        img = np.ascontiguousarray(Renderer(self.model, strip, (1280, 720)).draw(fr))
        q = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(q).scaled(self.preview.size(), Qt.KeepAspectRatio,
                                                           Qt.SmoothTransformation))
        self.status.setText(f"{len(frames)} frames = {len(frames) / o.fps:.1f} s of video")

    def _export(self) -> None:
        if self.worker is not None:
            self.worker.cancel = True
            return
        if not self._ready():
            return
        try:
            o = self.options()
        except ValueError as exc:
            QMessageBox.warning(self, "3D ODS video", str(exc))
            return
        out = self.out.text().strip()
        if not out:
            return
        self.worker = _Worker(self.rec, self.model, o, out, Path(self.rec.path).stem)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.bar.setVisible(True); self.bar.setValue(0)
        self.btn_go.setText("Cancel"); self.btn_prev.setEnabled(False)
        self.status.setText("Rendering…")
        self.worker.start()

    def _on_progress(self, i: int, n: int) -> None:
        self.bar.setMaximum(n); self.bar.setValue(i)
        self.status.setText(f"Rendering frame {i} / {n}")

    def _finish(self) -> None:
        self.worker = None
        self.bar.setVisible(False)
        self.btn_go.setText("Export"); self.btn_prev.setEnabled(True)

    def _on_done(self, path: str) -> None:
        self._finish()
        self.status.setText(f"Saved: {path}")
        still = Path(path).with_name(f"{Path(path).stem}_still0.png")
        if still.exists():
            pm = QPixmap(str(still))
            self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        box = QMessageBox(QMessageBox.Information, "3D ODS video", f"Saved:\n{path}", parent=self)
        open_btn = box.addButton("Open", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is open_btn:
            import os
            if hasattr(os, "startfile"):
                os.startfile(path)  # noqa: S606 - user's own file

    def _on_failed(self, msg: str) -> None:
        self._finish()
        self.status.setText(msg)
        if msg != "Cancelled.":
            QMessageBox.warning(self, "3D ODS video", msg)

    def _close(self) -> None:
        if self.worker is not None:
            self.worker.cancel = True
            self.worker.wait(5000)
        self.reject()

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.cancel = True
            self.worker.wait(5000)
        super().closeEvent(e)
