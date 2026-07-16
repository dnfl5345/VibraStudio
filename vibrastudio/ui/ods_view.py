"""ODS (Operational Deflection Shape) view with 3D structure animation.

Amplitude comes from averaged auto-spectra, relative phase from averaged
cross-spectra against the reference channel — the standard spectral-ODS
method used by LabVIEW SVMS / commercial ODS packages.

The structure is a parametric 3D geometry (beam, plate, ring, cylinder —
e.g. 12 nodes/ring x 2 layers) built in the toolbar; channels map onto
nodes via the Pt (node id) and Dir columns of the channel table.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QHeaderView,
)

from ..config import ChannelConfig
from ..dsp.analysis import StreamingAnalyzer
from ..geometry import AXIS_INDEX, Geometry, auto_map_channels, build, make_line
from . import theme
from .views import BaseView, toolbar_row

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:  # pragma: no cover - missing OpenGL driver
    gl = None
    _HAS_GL = False


def _rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    c = pg.mkColor(hex_color)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


# Shape templates: (combo label, shape key, param1 label, param1 default,
#                   param2 label, param2 default, param2 enabled)
_SHAPE_DEFS = [
    ("Line (beam)", "line", "Nodes", 5, "-", 1, False),
    ("Grid (plate)", "grid", "Cols", 3, "Rows", 3, True),
    ("Circle (ring)", "circle", "Nodes", 12, "-", 1, False),
    ("Cylinder", "cylinder", "Nodes/ring", 12, "Layers", 2, True),
]

_MAP_MODES = [("Map X+Z", ("X", "Z")), ("Map X+Y+Z", ("X", "Y", "Z")),
              ("Map Z only", ("Z",)), ("Map X only", ("X",))]


class OdsView(BaseView):
    channels_mapped = Signal(object)   # list[ChannelConfig] after auto-map

    def __init__(self) -> None:
        super().__init__()
        self.channel_cfgs: list[ChannelConfig] = []
        self.geometry: Geometry = make_line(5)
        self._user_geometry = False

        # -- analysis controls -------------------------------------------
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.5, 100000.0)
        self.freq_spin.setValue(11.6)
        self.freq_spin.setDecimals(2)
        self.freq_spin.setSuffix(" Hz")
        self.peak_box = QComboBox()
        self.peak_box.setMinimumWidth(120)
        self.pick_btn = QPushButton("Find Peaks")
        self.hold_btn = QPushButton("Freeze")
        self.hold_btn.setCheckable(True)
        self.speed = QSlider(Qt.Horizontal)
        self.speed.setRange(1, 40)
        self.speed.setValue(12)
        self.speed.setFixedWidth(100)
        self.scale = QSlider(Qt.Horizontal)
        self.scale.setRange(1, 100)
        self.scale.setValue(40)
        self.scale.setFixedWidth(100)
        self.info = QLabel("—")

        # -- geometry builder controls -------------------------------------
        self.shape_box = QComboBox()
        self.shape_box.addItems([s[0] for s in _SHAPE_DEFS])
        self.shape_box.setCurrentIndex(3)  # cylinder showcase default
        self.p1_label = QLabel("Nodes/ring")
        self.p1_spin = QSpinBox()
        self.p1_spin.setRange(2, 96)
        self.p1_spin.setValue(12)
        self.p2_label = QLabel("Layers")
        self.p2_spin = QSpinBox()
        self.p2_spin.setRange(1, 32)
        self.p2_spin.setValue(2)
        self.build_btn = QPushButton("Build Structure")
        self.map_box = QComboBox()
        self.map_box.addItems([m[0] for m in _MAP_MODES])
        self.map_btn = QPushButton("Auto-map Channels")
        self.labels_chk = QCheckBox("Node ids")
        self.labels_chk.setChecked(True)

        # -- 3D viewport -----------------------------------------------------
        if _HAS_GL:
            self.view3d = gl.GLViewWidget()
            self.view3d.setBackgroundColor(pg.mkColor(theme.PLOT_BG))
            grid = gl.GLGridItem()
            grid.setSize(3, 3)
            grid.setSpacing(0.25, 0.25)
            grid.setColor((255, 255, 255, 18))
            self.view3d.addItem(grid)
            self.gl_undeformed = gl.GLLinePlotItem(mode="lines", antialias=True,
                                                   color=_rgba(theme.TEXT_FAINT, 0.35), width=1.0)
            self.gl_deformed = gl.GLLinePlotItem(mode="lines", antialias=True,
                                                 color=_rgba(theme.ACCENT, 0.95), width=2.5)
            self.gl_nodes = gl.GLScatterPlotItem(pxMode=True)
            self.view3d.addItem(self.gl_undeformed)
            self.view3d.addItem(self.gl_deformed)
            self.view3d.addItem(self.gl_nodes)
            self._gl_labels: list = []
            viewport = self.view3d
        else:  # pragma: no cover
            viewport = QLabel("3D view unavailable — PyOpenGL not installed.\n"
                              "pip install PyOpenGL")
            viewport.setAlignment(Qt.AlignCenter)
            self.view3d = None

        # -- amplitude/phase table ------------------------------------------
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Ch", "Node", "Dir", "g RMS", "Phase°"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumWidth(340)

        body = QHBoxLayout()
        body.addWidget(viewport, stretch=1)
        body.addWidget(self.table)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(toolbar_row("Frequency", self.freq_spin, self.pick_btn,
                                  "Peaks", self.peak_box, self.hold_btn,
                                  "Speed", self.speed, "Scale", self.scale,
                                  "stretch", self.info))
        lay.addLayout(toolbar_row("Shape", self.shape_box,
                                  self.p1_label, self.p1_spin,
                                  self.p2_label, self.p2_spin,
                                  self.build_btn, self.map_box, self.map_btn,
                                  self.labels_chk, "stretch", QLabel("")))
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
        self.shape_box.currentIndexChanged.connect(self._shape_changed)
        self.build_btn.clicked.connect(self._build_geometry)
        self.map_btn.clicked.connect(self._auto_map)
        self.labels_chk.toggled.connect(lambda *_: self._rebuild_static())
        self._shape_changed(self.shape_box.currentIndex())
        self._rebuild_static()

    # -- geometry builder --------------------------------------------------
    def _shape_changed(self, idx: int) -> None:
        _, _, l1, d1, l2, d2, en2 = _SHAPE_DEFS[idx]
        self.p1_label.setText(l1)
        self.p1_spin.setValue(d1)
        self.p2_label.setText(l2)
        self.p2_spin.setValue(d2)
        self.p2_spin.setEnabled(en2)
        self.p2_label.setVisible(en2)
        self.p2_spin.setVisible(en2)

    def _build_geometry(self) -> None:
        _, shape, *_ = _SHAPE_DEFS[self.shape_box.currentIndex()]
        p1, p2 = self.p1_spin.value(), self.p2_spin.value()
        if shape == "line":
            self.geometry = build(shape, n_nodes=p1)
        elif shape == "grid":
            self.geometry = build(shape, nx=p1, ny=p2)
        elif shape == "circle":
            self.geometry = build(shape, n_nodes=p1)
        else:
            self.geometry = build(shape, n_per_ring=p1, n_layers=p2)
        self._user_geometry = True
        self._rebuild_static()
        self.info.setText(f"{self.geometry.name}: {self.geometry.n_nodes} nodes, "
                          f"{len(self.geometry.edges)} edges")

    def _auto_map(self) -> None:
        if not self.channel_cfgs:
            return
        axes = _MAP_MODES[self.map_box.currentIndex()][1]
        mapped = auto_map_channels(self.channel_cfgs, self.geometry, axes)
        self.set_channel_configs(mapped)
        self.channels_mapped.emit(mapped)

    # -- persistence ---------------------------------------------------------
    def geometry_dict(self) -> dict:
        return self.geometry.to_dict()

    def set_geometry_dict(self, d: dict) -> None:
        if not d or not d.get("nodes"):
            return
        try:
            self.geometry = Geometry.from_dict(d)
            self._user_geometry = True
            self._rebuild_static()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def set_channel_configs(self, cfgs: list[ChannelConfig]) -> None:
        self.channel_cfgs = list(cfgs)
        self.set_channels([c.label for c in cfgs])

        # Auto-grow the default beam so every mapped node exists; a user- or
        # config-built geometry is never overridden.
        max_pt = max((c.point for c in cfgs), default=0)
        if not self._user_geometry and max_pt >= 2 and max_pt != self.geometry.n_nodes:
            self.geometry = make_line(max_pt)
            self._rebuild_static()

        self.table.setRowCount(len(cfgs))
        for i, c in enumerate(cfgs):
            for j, val in enumerate([c.label, str(c.point), c.direction, "—", "—"]):
                item = QTableWidgetItem(val)
                if j == 0:
                    item.setForeground(pg.mkColor(self.color(i)))
                self.table.setItem(i, j, item)
        self._rebuild_static()

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

    # -- 3D rendering --------------------------------------------------------
    def _sensor_nodes(self) -> set[int]:
        return {c.point - 1 for c in self.channel_cfgs
                if 0 <= c.point - 1 < self.geometry.n_nodes}

    def _rebuild_static(self) -> None:
        """Redraw undeformed wireframe, node scatter, labels and camera."""
        if not _HAS_GL:
            return
        g = self.geometry
        self.gl_undeformed.setData(pos=g.edge_array())
        self._update_scatter(g.nodes)

        for item in self._gl_labels:
            self.view3d.removeItem(item)
        self._gl_labels = []
        if self.labels_chk.isChecked() and g.n_nodes <= 64:
            offset = 0.03 * g.bbox_size()
            for i in range(g.n_nodes):
                txt = gl.GLTextItem(pos=g.nodes[i] + offset,
                                    text=str(i + 1),
                                    color=pg.mkColor(theme.TEXT_MUTED))
                self.view3d.addItem(txt)
                self._gl_labels.append(txt)

        center = g.nodes.mean(axis=0) if g.n_nodes else np.zeros(3)
        self.view3d.opts["center"] = pg.Vector(*center)
        self.view3d.setCameraPosition(distance=2.2 * g.bbox_size() + 0.5,
                                      elevation=22, azimuth=-60)

    def _update_scatter(self, positions: np.ndarray) -> None:
        g = self.geometry
        if g.n_nodes == 0:
            return
        sensors = self._sensor_nodes()
        colors = np.tile(_rgba(theme.TEXT_FAINT, 0.55), (g.n_nodes, 1))
        sizes = np.full(g.n_nodes, 6.0)
        for i in sensors:
            colors[i] = _rgba(theme.ACCENT_HOVER, 1.0)
            sizes[i] = 11.0
        self.gl_nodes.setData(pos=positions, color=colors, size=sizes)

    def _animate(self) -> None:
        if not _HAS_GL or self.geometry.n_nodes == 0:
            return
        g = self.geometry
        if self._amp is None or self._phi is None or not self.channel_cfgs:
            self.gl_deformed.setData(pos=g.edge_array())
            return

        self._phase += 0.015 * self.speed.value()
        global_peak = float(np.max(self._amp)) or 1.0
        gain = (self.scale.value() / 40.0) * 0.18 * g.bbox_size()

        disp = np.zeros_like(g.nodes)
        for i, c in enumerate(self.channel_cfgs):
            if i >= self._amp.size:
                break
            node = c.point - 1
            axis = AXIS_INDEX.get(c.direction)
            if axis is None or not (0 <= node < g.n_nodes):
                continue
            rel = self._amp[i] / global_peak
            disp[node, axis] += rel * np.cos(self._phase + self._phi[i])

        deformed = g.nodes + gain * disp
        self.gl_deformed.setData(pos=g.edge_array(deformed))
        self._update_scatter(deformed)
