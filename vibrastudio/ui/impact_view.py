"""Impact (modal hammer) test view: hit-location guide, triggered capture,
averaged H1 FRF/coherence and modal identification for free-vibration modes."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView,
)

from ..config import ChannelConfig
from ..daq.engine import AcquisitionEngine
from ..dsp.impact import ImpactAccumulator, identify_modes, impact_guide
from . import theme
from .views import BaseView, toolbar_row

# Physical rig defaults: 60 cm beam, bottom 10 cm clamped in the jig ->
# 50 cm free cantilever; sensors S1..S4 at 10/20/30/40 cm from the free tip.
FREE_LEN_CM = 50.0
SENSOR_FROM_TIP_CM = [10.0, 20.0, 30.0, 40.0]


class ImpactView(BaseView):
    beam_preset_requested = Signal()
    mode_frequency_selected = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.channel_cfgs: list[ChannelConfig] = []
        self._armed = False
        self._mark = 0
        self._pending: int | None = None   # absolute trigger sample index
        self._deadline = 0                 # absolute index where capture ends
        self._dead_until = 0
        self.acc: ImpactAccumulator | None = None
        self.hits = 0
        self.rejects = 0
        self._last_force: np.ndarray | None = None
        self._last_resp: np.ndarray | None = None
        self.modes: list[dict] = []

        # -- controls row 1: channels & trigger ---------------------------
        self.hammer_box = QComboBox()
        self.hammer_box.setMinimumWidth(90)
        self.hammer_sens = QDoubleSpinBox()
        self.hammer_sens.setRange(0.01, 10000.0)
        self.hammer_sens.setValue(2.25)
        self.hammer_sens.setSuffix(" mV/N")
        self.trig_spin = QDoubleSpinBox()
        self.trig_spin.setRange(0.1, 10000.0)
        self.trig_spin.setValue(5.0)
        self.trig_spin.setSuffix(" N")
        self.pre_spin = QDoubleSpinBox()
        self.pre_spin.setRange(1.0, 500.0)
        self.pre_spin.setValue(50.0)
        self.pre_spin.setSuffix(" ms")
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(0.2, 10.0)
        self.len_spin.setValue(2.0)
        self.len_spin.setSuffix(" s")
        self.reject_chk = QCheckBox("Reject double-hit")
        self.reject_chk.setChecked(True)
        self.preset_btn = QPushButton("Beam preset (4 sens @10cm)")

        # -- controls row 2: run ------------------------------------------
        self.arm_btn = QPushButton("⚡ Arm Trigger")
        self.arm_btn.setObjectName("Primary")
        self.arm_btn.setCheckable(True)
        self.clear_btn = QPushButton("Clear Averages")
        self.dir_box = QComboBox()
        self.dir_box.addItems(["All dirs", "X only", "Y only", "Z only"])
        self.save_btn = QPushButton("💾 Save Results")
        self.status = QLabel("idle — start streaming, then Arm")

        # -- impact location guide (left) ---------------------------------
        self.guide = pg.PlotWidget()
        self.guide.setFixedWidth(300)
        gp = self.guide.getPlotItem()
        theme.style_plot(gp, "Impact location guide", "mode score", "from tip (cm)")
        gp.invertY(True)   # tip (0 cm) on top, like the physical beam
        gp.setXRange(0, 1.05)
        self._draw_guide()

        # -- capture plots (center) ----------------------------------------
        self.gw = pg.GraphicsLayoutWidget()
        self.p_time = self.gw.addPlot(row=0, col=0)
        theme.style_plot(self.p_time, "Last hit — force & response", "Time (ms)", "")
        self.c_force = self.p_time.plot(pen=pg.mkPen(theme.ACCENT, width=1.4))
        self.c_resp = self.p_time.plot(pen=pg.mkPen("#5aa9d6", width=1.0))
        self.p_frf = self.gw.addPlot(row=1, col=0)
        theme.style_plot(self.p_frf, "FRF |H1| (g/N)", "", "|H| (log)")
        self.p_frf.setLogMode(y=True)
        self.p_coh = self.gw.addPlot(row=2, col=0)
        theme.style_plot(self.p_coh, "Coherence", "Frequency (Hz)", "γ²")
        self.p_coh.setYRange(0, 1.05)
        self.p_frf.setXLink(self.p_coh)
        self.frf_curves: list[pg.PlotDataItem] = []
        self.coh_curves: list[pg.PlotDataItem] = []
        self.mode_lines: list[pg.InfiniteLine] = []

        # -- modes table + shape (right) ----------------------------------
        self.mode_table = QTableWidget(0, 3)
        self.mode_table.setHorizontalHeaderLabels(["fn (Hz)", "ζ (%)", "|H|"])
        self.mode_table.verticalHeader().setVisible(False)
        self.mode_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mode_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.mode_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mode_table.setMaximumWidth(260)
        self.ods_btn = QPushButton("Show in 3D ODS →")

        self.shape_plot = pg.PlotWidget()
        self.shape_plot.setMaximumWidth(260)
        sp = self.shape_plot.getPlotItem()
        theme.style_plot(sp, "Mode shape", "normalized", "from tip (cm)")
        sp.invertY(True)
        self.shape_x = sp.plot(pen=pg.mkPen(theme.ACCENT, width=2),
                               symbol="o", symbolSize=7,
                               symbolBrush=pg.mkBrush(theme.ACCENT_HOVER), symbolPen=None)
        self.shape_z = sp.plot(pen=pg.mkPen("#5aa9d6", width=2),
                               symbol="s", symbolSize=7,
                               symbolBrush=pg.mkBrush("#7ec2e8"), symbolPen=None)
        sp.addLine(x=0, pen=pg.mkPen(theme.TEXT_FAINT, style=Qt.DashLine))
        sp.addLegend(offset=(-5, 5), labelTextColor=theme.TEXT_MUTED)

        right = QVBoxLayout()
        right.addWidget(self.mode_table, stretch=1)
        right.addWidget(self.ods_btn)
        right.addWidget(self.shape_plot, stretch=1)

        body = QHBoxLayout()
        body.addWidget(self.guide)
        body.addWidget(self.gw, stretch=1)
        body.addLayout(right)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Hammer", self.hammer_box, self.hammer_sens,
                                  "Trigger", self.trig_spin, "Pre", self.pre_spin,
                                  "Length", self.len_spin, self.reject_chk,
                                  "stretch", self.preset_btn))
        lay.addLayout(toolbar_row(self.arm_btn, self.clear_btn,
                                  "Response", self.dir_box, self.save_btn,
                                  "stretch", self.status))
        lay.addLayout(body)

        self.arm_btn.toggled.connect(self._arm_toggled)
        self.clear_btn.clicked.connect(self._clear)
        self.preset_btn.clicked.connect(self.beam_preset_requested)
        self.mode_table.itemSelectionChanged.connect(self._mode_selected)
        self.ods_btn.clicked.connect(self._send_to_ods)
        self.save_btn.clicked.connect(self._save_results)
        self.dir_box.currentIndexChanged.connect(self._dir_changed)
        self._hit_records: list[tuple[np.ndarray, np.ndarray]] = []
        self._sel_resp: list[tuple[int, int]] = []   # (row in acc, channel idx)
        self._last_resp_idx: list[int] = []

    # -- guide -----------------------------------------------------------
    def _draw_guide(self) -> None:
        gp = self.guide.getPlotItem()
        gp.clear()
        g = impact_guide(FREE_LEN_CM, SENSOR_FROM_TIP_CM[0])
        gp.setYRange(-2, FREE_LEN_CM + 2)

        # beam line + sensors
        gp.plot([0.02, 0.02], [0.0, FREE_LEN_CM], pen=pg.mkPen(theme.BORDER, width=3))
        for i, d in enumerate(SENSOR_FROM_TIP_CM):
            gp.plot([0.02], [d], symbol="o", symbolSize=10,
                    symbolBrush=pg.mkBrush(theme.ACCENT_HOVER), symbolPen=None)
            t = pg.TextItem(f"S{i+1}", color=theme.TEXT_MUTED, anchor=(0, 0.5))
            t.setPos(0.08, d)
            gp.addItem(t)
        jig = pg.TextItem("JIG (fixed)", color=theme.TEXT_FAINT, anchor=(0, 0.5))
        jig.setPos(0.08, FREE_LEN_CM + 0.5)
        gp.addItem(jig)

        # score curve inside the hammer zone
        gp.plot(g["score"], g["d_cm"], pen=pg.mkPen("#5bbf9a", width=2))
        # avoid nodes
        for d in g["avoid_cm"]:
            gp.plot([0.02], [d], symbol="x", symbolSize=12,
                    symbolPen=pg.mkPen(theme.RED, width=2))
            t = pg.TextItem(f"avoid {d:.1f}", color=theme.RED, anchor=(0, 0.5))
            t.setPos(0.45, d)
            gp.addItem(t)
        # best point
        best = g["best_cm"]
        gp.plot([g["best_score"]], [best], symbol="star", symbolSize=18,
                symbolBrush=pg.mkBrush("#e0b25a"), symbolPen=None)
        t = pg.TextItem(f"BEST {best:.1f} cm", color="#e0b25a", anchor=(1, 1.1))
        t.setPos(g["best_score"], best)
        gp.addItem(t)

    # -- config ----------------------------------------------------------
    def set_channel_configs(self, cfgs: list[ChannelConfig]) -> None:
        self.channel_cfgs = list(cfgs)
        self.set_channels([c.label for c in cfgs])
        cur = self.hammer_box.currentText()
        self.hammer_box.blockSignals(True)
        self.hammer_box.clear()
        self.hammer_box.addItems([c.label for c in cfgs])
        labels = [c.label for c in cfgs]
        if cur in labels:
            self.hammer_box.setCurrentIndex(labels.index(cur))
        else:
            # default: a channel literally named HAMMER, else the last one
            idx = next((i for i, l in enumerate(labels) if "HAM" in l.upper()),
                       len(labels) - 1)
            self.hammer_box.setCurrentIndex(max(0, idx))
        self.hammer_box.blockSignals(False)
        self.acc = None   # channel set changed -> restart averaging

    # -- run state -------------------------------------------------------
    def _arm_toggled(self, on: bool) -> None:
        self._armed = on
        self.arm_btn.setText("⚡ Armed — hit the beam" if on else "⚡ Arm Trigger")
        if on:
            self._pending = None
            self._mark = -1   # sync to ring on next tick

    def _clear(self) -> None:
        if self.acc is not None:
            self.acc.reset()
        self.hits = 0
        self.rejects = 0
        self.modes = []
        self._hit_records = []
        self.mode_table.setRowCount(0)
        self.status.setText("averages cleared")

    def _dir_changed(self, *_):
        if self.acc is not None and self._last_resp_idx:
            self._redraw(self._last_resp_idx, self.acc.fs)

    # -- per-tick processing ---------------------------------------------
    def update_view(self, engine: AcquisitionEngine) -> None:
        if not self._armed or engine is None or not engine.running:
            return
        ring, fs = engine.ring, engine.fs
        if self._mark < 0:
            self._mark = ring.total
            self._dead_until = ring.total
        ham = self.hammer_box.currentIndex()
        if not (0 <= ham < len(self.channel_cfgs)):
            return
        n_to_force = self.channel_cfgs[ham].sensitivity_mv_g / max(
            self.hammer_sens.value(), 1e-9)   # g-units -> N
        trig_g = self.trig_spin.value() / max(n_to_force, 1e-12)

        pre_n = int(self.pre_spin.value() * 1e-3 * fs)
        cap_n = int(self.len_spin.value() * fs)

        if self._pending is None:
            block, new_mark = ring.read_since(self._mark, max_samples=ring.capacity)
            if block.size:
                start_abs = new_mark - block.shape[1]
                y = block[ham]
                above = np.nonzero(np.abs(y) >= trig_g)[0]
                for i in above:
                    abs_i = start_abs + int(i)
                    if abs_i >= self._dead_until:
                        self._pending = abs_i
                        self.status.setText("hit detected — capturing…")
                        break
                self._mark = new_mark
        if self._pending is not None and ring.total >= self._pending + cap_n:
            self._process_capture(engine, self._pending, pre_n, cap_n, ham, n_to_force)
            self._dead_until = self._pending + cap_n
            self._mark = ring.total
            self._pending = None

    def _process_capture(self, engine, trig_abs: int, pre_n: int, cap_n: int,
                         ham: int, n_to_force: float) -> None:
        ring, fs = engine.ring, engine.fs
        span = ring.total - (trig_abs - pre_n)
        window = ring.latest(span)[:, : pre_n + cap_n]
        if window.shape[1] < pre_n + cap_n // 2:
            self.status.setText("capture missed (buffer too short)")
            return

        force = window[ham] * n_to_force
        resp_idx = [i for i in range(len(self.channel_cfgs)) if i != ham]
        resp = window[resp_idx]
        resp = resp - resp[:, :pre_n].mean(axis=1, keepdims=True)

        # double-hit check: a second force peak after the main pulse
        pk = np.max(np.abs(force))
        after = np.abs(force[pre_n + int(0.05 * fs):])
        double = after.size > 0 and np.max(after) > 0.25 * pk
        if double and self.reject_chk.isChecked():
            self.rejects += 1
            self.status.setText(
                f"double hit rejected  (hits {self.hits}, rejected {self.rejects})")
            return

        # accumulate from the trigger point (keep small pre for the pulse)
        keep = slice(max(0, pre_n - int(0.005 * fs)), None)
        f_seg = force[keep]
        r_seg = resp[:, keep]
        if self.acc is None or self.acc.n_resp != len(resp_idx) or self.acc.nfft != cap_n:
            self.acc = ImpactAccumulator(fs=fs, n_resp=len(resp_idx), nfft=cap_n)
            self.hits = 0
            self._hit_records = []
        self.acc.add_hit(f_seg, r_seg)
        self.hits += 1
        if len(self._hit_records) < 32:   # keep raw captures for export
            self._hit_records.append((f_seg.copy(), r_seg.copy()))
        self._last_force = f_seg
        self._last_resp = r_seg
        self.status.setText(
            f"hit {self.hits} accepted  (peak {pk * 1:.1f} N"
            + (", double!" if double else "") + f", rejected {self.rejects})")
        self._redraw(resp_idx, fs)

    # -- drawing ----------------------------------------------------------
    def _redraw(self, resp_idx: list[int], fs: float) -> None:
        assert self.acc is not None
        # last hit time record (force + strongest response)
        if self._last_force is not None and self._last_resp is not None:
            n_show = min(int(0.25 * fs), self._last_force.size)
            t_ms = np.arange(n_show) / fs * 1e3
            self.c_force.setData(t_ms, self._last_force[:n_show])
            strongest = int(np.argmax(np.max(np.abs(self._last_resp), axis=1)))
            scale = (np.max(np.abs(self._last_force[:n_show])) /
                     max(np.max(np.abs(self._last_resp[strongest][:n_show])), 1e-12))
            self.c_resp.setData(t_ms, self._last_resp[strongest][:n_show] * scale)

        self._last_resp_idx = list(resp_idx)
        h1, coh = self.acc.frf()
        freq = self.acc.freq
        f_max = min(1000.0, freq[-1])
        mask = freq <= f_max

        # Direction filter: identify modes per excitation direction so a
        # Z-family test is not polluted by weakly excited cross-axis channels.
        want = self.dir_box.currentText()
        self._sel_resp = [
            (k, ch_i) for k, ch_i in enumerate(resp_idx)
            if want == "All dirs" or self.channel_cfgs[ch_i].direction == want[0]
        ]
        sel_rows = [k for k, _ in self._sel_resp]

        while len(self.frf_curves) < len(resp_idx):
            k = len(self.frf_curves)
            self.frf_curves.append(self.p_frf.plot(
                pen=pg.mkPen(self.color(k), width=1.0)))
            self.coh_curves.append(self.p_coh.plot(
                pen=pg.mkPen(self.color(k), width=1.0)))
        for k in range(len(self.frf_curves)):
            if k < len(resp_idx) and k in sel_rows:
                mag = np.maximum(np.abs(h1[k])[mask], 1e-12)
                self.frf_curves[k].setData(freq[mask], mag)
                self.coh_curves[k].setData(freq[mask], coh[k][mask])
            else:
                self.frf_curves[k].setData([], [])
                self.coh_curves[k].setData([], [])

        # modal identification on the selected direction subset
        h_sel = h1[sel_rows] if sel_rows else h1
        self.modes = identify_modes(freq, h_sel, f_min=3.0, f_max=f_max,
                                    exp_tau_sec=self.acc.exp_tau_sec)
        for ln in self.mode_lines:
            self.p_frf.removeItem(ln)
        self.mode_lines = []
        sel = self.mode_table.currentRow()
        self.mode_table.setRowCount(len(self.modes))
        for r, md in enumerate(self.modes):
            for c, txt in enumerate([f"{md['fn']:.2f}", f"{100*md['zeta']:.2f}",
                                     f"{md['amp']:.3g}"]):
                self.mode_table.setItem(r, c, QTableWidgetItem(txt))
            ln = pg.InfiniteLine(md["fn"], angle=90,
                                 pen=pg.mkPen(theme.YELLOW, width=1, style=Qt.DashLine))
            self.p_frf.addItem(ln)
            self.mode_lines.append(ln)
        if 0 <= sel < len(self.modes):
            self.mode_table.selectRow(sel)
        elif self.modes:
            self.mode_table.selectRow(0)

    def _mode_selected(self) -> None:
        r = self.mode_table.currentRow()
        if not (0 <= r < len(self.modes)) or self.acc is None:
            return
        shape = self.modes[r]["shape"]
        for direction, curve in (("X", self.shape_x), ("Z", self.shape_z)):
            pts = []
            for j, (_, ch_i) in enumerate(self._sel_resp):
                c = self.channel_cfgs[ch_i]
                if c.direction == direction and j < shape.size:
                    d_tip = FREE_LEN_CM * (1.0 - c.position)
                    pts.append((d_tip, float(shape[j])))
            pts.sort()
            if pts:
                curve.setData([p[1] for p in pts], [p[0] for p in pts])
            else:
                curve.setData([], [])

    def _send_to_ods(self) -> None:
        r = self.mode_table.currentRow()
        if 0 <= r < len(self.modes):
            self.mode_frequency_selected.emit(self.modes[r]["fn"])

    # -- export ------------------------------------------------------------
    def _save_results(self) -> None:
        """Write FRF/coherence, mode table, raw hits and a screenshot."""
        if self.acc is None or self.acc.navg == 0:
            self.status.setText("nothing to save — capture at least one hit")
            return
        import csv
        import json
        from datetime import datetime
        from pathlib import Path

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs") / f"impact_{ts}"
        out.mkdir(parents=True, exist_ok=True)

        resp_idx = self._last_resp_idx
        labels = [self.channel_cfgs[i].label for i in resp_idx]
        h1, coh = self.acc.frf()
        freq = self.acc.freq

        with (out / "frf.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["freq_hz"]
            for lbl in labels:
                header += [f"H_mag_gN_{lbl}", f"H_phase_deg_{lbl}", f"coherence_{lbl}"]
            w.writerow(header)
            for i in range(freq.size):
                row = [f"{freq[i]:.6f}"]
                for k in range(len(labels)):
                    row += [f"{abs(h1[k, i]):.6e}",
                            f"{np.degrees(np.angle(h1[k, i])):.3f}",
                            f"{coh[k, i]:.5f}"]
                w.writerow(row)

        with (out / "modes.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            sel_labels = [self.channel_cfgs[ch_i].label for _, ch_i in self._sel_resp]
            w.writerow(["mode", "fn_hz", "zeta_pct", "H_amp",
                        *[f"shape_{l}" for l in sel_labels]])
            for m_i, md in enumerate(self.modes, start=1):
                w.writerow([m_i, f"{md['fn']:.4f}", f"{100*md['zeta']:.4f}",
                            f"{md['amp']:.6e}",
                            *[f"{v:.5f}" for v in md["shape"]]])

        if self._hit_records:
            np.savez_compressed(
                out / "raw_hits.npz",
                force_n=np.stack([h[0] for h in self._hit_records]),
                response_g=np.stack([h[1] for h in self._hit_records]),
                fs=self.acc.fs, response_labels=np.array(labels),
            )

        meta = {
            "timestamp": ts,
            "hits_accepted": self.hits, "hits_rejected": self.rejects,
            "hammer_channel": self.hammer_box.currentText(),
            "hammer_sens_mv_n": self.hammer_sens.value(),
            "trigger_n": self.trig_spin.value(),
            "pretrigger_ms": self.pre_spin.value(),
            "capture_s": self.len_spin.value(),
            "response_filter": self.dir_box.currentText(),
            "sample_rate_hz": self.acc.fs,
            "modes": [{"fn_hz": m["fn"], "zeta_pct": 100 * m["zeta"]}
                      for m in self.modes],
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        win = self.window()
        win.grab().save(str(out / "screenshot.png"))
        self.status.setText(f"saved → {out}  (frf/modes/raw_hits/screenshot)")
