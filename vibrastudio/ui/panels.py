"""Sidebar control panel: acquisition, analysis and channel table cards."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QAbstractItemView,
)

from ..config import AppConfig, ChannelConfig, CHANNEL_COLORS, DIRECTIONS
import pyqtgraph as pg


def make_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    t = QLabel(title.upper())
    t.setObjectName("CardTitle")
    lay.addWidget(t)
    return card, lay


class ChannelTable(QTableWidget):
    """Editable 18-channel map: enable, label, physical, point, dir, pos, sens, ref."""

    COLS = ["On", "Label", "Physical channel", "Pt", "Dir", "Inv", "Pos", "mV/g", "Ref"]
    COL_INV, COL_POS, COL_SENS, COL_REF = 5, 6, 7, 8
    changed = Signal()

    def __init__(self) -> None:
        super().__init__(0, len(self.COLS))
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(24)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.setMinimumHeight(260)
        self.itemChanged.connect(self._on_item_changed)
        self._loading = False

    def load(self, channels: list[ChannelConfig]) -> None:
        self._loading = True
        self.setRowCount(len(channels))
        for i, ch in enumerate(channels):
            on = QTableWidgetItem()
            on.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            on.setCheckState(Qt.Checked if ch.enabled else Qt.Unchecked)
            self.setItem(i, 0, on)

            lbl = QTableWidgetItem(ch.label)
            lbl.setForeground(pg.mkColor(CHANNEL_COLORS[i % len(CHANNEL_COLORS)]))
            self.setItem(i, 1, lbl)
            self.setItem(i, 2, QTableWidgetItem(ch.physical_channel))
            self.setItem(i, 3, QTableWidgetItem(str(ch.point)))

            dir_box = QComboBox()
            dir_box.addItems(DIRECTIONS)
            dir_box.setCurrentText(ch.direction)
            dir_box.currentTextChanged.connect(lambda *_: self.changed.emit())
            self.setCellWidget(i, 4, dir_box)

            inv = QTableWidgetItem()
            inv.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            inv.setCheckState(Qt.Checked if ch.sign < 0 else Qt.Unchecked)
            inv.setToolTip("Invert axis sign (left-handed sensor frame)")
            self.setItem(i, self.COL_INV, inv)

            self.setItem(i, self.COL_POS, QTableWidgetItem(f"{ch.position:.2f}"))
            self.setItem(i, self.COL_SENS, QTableWidgetItem(f"{ch.sensitivity_mv_g:.0f}"))

            ref = QTableWidgetItem()
            ref.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            ref.setCheckState(Qt.Checked if ch.is_reference else Qt.Unchecked)
            self.setItem(i, self.COL_REF, ref)
        self._loading = False
        self.changed.emit()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        if item.column() == self.COL_REF and item.checkState() == Qt.Checked:
            # Only one reference channel at a time.
            self._loading = True
            for r in range(self.rowCount()):
                other = self.item(r, self.COL_REF)
                if other is not None and r != item.row():
                    other.setCheckState(Qt.Unchecked)
            self._loading = False
        self.changed.emit()

    def read(self) -> list[ChannelConfig]:
        out: list[ChannelConfig] = []
        for r in range(self.rowCount()):
            try:
                dir_box = self.cellWidget(r, 4)
                out.append(
                    ChannelConfig(
                        physical_channel=self.item(r, 2).text().strip(),
                        label=self.item(r, 1).text().strip() or f"CH{r+1}",
                        enabled=self.item(r, 0).checkState() == Qt.Checked,
                        point=int(float(self.item(r, 3).text())),
                        direction=dir_box.currentText() if dir_box else "Z",
                        position=float(self.item(r, self.COL_POS).text()),
                        sensitivity_mv_g=float(self.item(r, self.COL_SENS).text()),
                        is_reference=self.item(r, self.COL_REF).checkState() == Qt.Checked,
                        sign=-1 if self.item(r, self.COL_INV).checkState() == Qt.Checked else 1,
                    )
                )
            except (ValueError, AttributeError):
                continue
        return out


class Sidebar(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    record_toggled = Signal(bool)
    analysis_changed = Signal()
    channels_changed = Signal()
    detect_requested = Signal()
    default_requested = Signal()
    save_config = Signal()
    load_config = Signal()
    open_file_requested = Signal()

    SAMPLE_RATES = ["1652", "2048", "4096", "8192", "10240", "12800", "25600", "51200"]
    FFT_SIZES = ["1024", "2048", "4096", "8192", "16384", "32768"]

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(400)

        # -- header -----------------------------------------------------
        header = QVBoxLayout()
        title = QLabel("VibraStudio")
        title.setObjectName("AppTitle")
        sub = QLabel("NI cDAQ REAL-TIME VIBRATION ANALYSIS")
        sub.setObjectName("AppSubtitle")
        header.addWidget(title)
        header.addWidget(sub)

        # -- acquisition card --------------------------------------------
        acq_card, acq = make_card("Acquisition")
        grid = QGridLayout()
        grid.setVerticalSpacing(6)
        self.source_box = QComboBox()
        self.source_box.addItems(["Hardware (NI-DAQmx)", "Simulation", "File playback"])
        self.coupling_box = QComboBox()
        self.coupling_box.addItems(["IEPE accelerometer", "Voltage"])
        self.fs_box = QComboBox()
        self.fs_box.setEditable(True)
        self.fs_box.addItems(self.SAMPLE_RATES)
        self.fs_box.setCurrentText(str(int(cfg.acquisition.sample_rate)))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(64, 65536)
        self.chunk_spin.setValue(cfg.acquisition.chunk_size)
        grid.addWidget(QLabel("Source"), 0, 0)
        grid.addWidget(self.source_box, 0, 1)
        grid.addWidget(QLabel("Coupling"), 1, 0)
        grid.addWidget(self.coupling_box, 1, 1)
        grid.addWidget(QLabel("Sample rate (Hz)"), 2, 0)
        grid.addWidget(self.fs_box, 2, 1)
        grid.addWidget(QLabel("Chunk size"), 3, 0)
        grid.addWidget(self.chunk_spin, 3, 1)
        self.open_file_btn = QPushButton("Open Recording…")
        grid.addWidget(QLabel("File"), 4, 0)
        grid.addWidget(self.open_file_btn, 4, 1)
        acq.addLayout(grid)
        self.file_label = QLabel("no recording loaded")
        self.file_label.setObjectName("DeviceInfo")
        self.file_label.setWordWrap(True)
        acq.addWidget(self.file_label)

        self.start_btn = QPushButton("▶  Start Streaming")
        self.start_btn.setObjectName("Primary")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setVisible(False)
        self.record_btn = QPushButton("●  Record")
        self.record_btn.setCheckable(True)
        self.csv_chk = QCheckBox("also CSV")
        rec_row = QHBoxLayout()
        rec_row.addWidget(self.record_btn, stretch=1)
        rec_row.addWidget(self.csv_chk)
        acq.addWidget(self.start_btn)
        acq.addWidget(self.stop_btn)
        acq.addLayout(rec_row)

        # -- analysis card -----------------------------------------------
        ana_card, ana = make_card("Spectral Analysis")
        g2 = QGridLayout()
        g2.setVerticalSpacing(6)
        self.fft_box = QComboBox()
        self.fft_box.addItems(self.FFT_SIZES)
        self.fft_box.setCurrentText(str(cfg.analysis.fft_size))
        self.window_box = QComboBox()
        self.window_box.addItems(["hann", "hamming", "blackman", "blackmanharris",
                                  "flattop", "rectangular"])
        self.window_box.setCurrentText(cfg.analysis.window)
        self.avg_box = QComboBox()
        self.avg_box.addItems(["exponential", "linear", "peakhold", "none"])
        self.avg_box.setCurrentText(cfg.analysis.averaging)
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 999)
        self.avg_spin.setValue(cfg.analysis.avg_count)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 90)
        self.overlap_spin.setValue(int(cfg.analysis.overlap * 100))
        self.overlap_spin.setSuffix(" %")
        g2.addWidget(QLabel("FFT size"), 0, 0)
        g2.addWidget(self.fft_box, 0, 1)
        g2.addWidget(QLabel("Window"), 1, 0)
        g2.addWidget(self.window_box, 1, 1)
        g2.addWidget(QLabel("Averaging"), 2, 0)
        g2.addWidget(self.avg_box, 2, 1)
        g2.addWidget(QLabel("Avg count"), 3, 0)
        g2.addWidget(self.avg_spin, 3, 1)
        g2.addWidget(QLabel("Overlap"), 4, 0)
        g2.addWidget(self.overlap_spin, 4, 1)
        ana.addLayout(g2)
        self.res_label = QLabel("Δf = 1.56 Hz")
        ana.addWidget(self.res_label)

        # -- channels card -----------------------------------------------
        ch_card, chl = make_card("Channels  (18 ch map)")
        btn_row = QHBoxLayout()
        self.detect_btn = QPushButton("⟲  Connect DAQ")
        self.default_btn = QPushButton("Default 18ch")
        btn_row.addWidget(self.detect_btn)
        btn_row.addWidget(self.default_btn)
        chl.addLayout(btn_row)
        self.device_label = QLabel("No device — press Connect DAQ")
        self.device_label.setObjectName("DeviceInfo")
        self.device_label.setWordWrap(True)
        chl.addWidget(self.device_label)
        self.channel_table = ChannelTable()
        chl.addWidget(self.channel_table)

        # -- status card ---------------------------------------------------
        st_card, st = make_card("Status")
        self.status_grid = QGridLayout()
        self._stat_labels: dict[str, QLabel] = {}
        for row, key in enumerate(["State", "Samples/ch", "Elapsed", "Buffer", "Avg count"]):
            self.status_grid.addWidget(QLabel(key), row, 0)
            v = QLabel("—")
            v.setObjectName("StatValue")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._stat_labels[key] = v
            self.status_grid.addWidget(v, row, 1)
        st.addLayout(self.status_grid)

        # -- config buttons --------------------------------------------------
        cfg_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Config")
        self.loadcfg_btn = QPushButton("Load Config")
        cfg_row.addWidget(self.save_btn)
        cfg_row.addWidget(self.loadcfg_btn)

        # -- scrollable assembly ----------------------------------------------
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)
        v.addLayout(header)
        v.addWidget(acq_card)
        v.addWidget(ana_card)
        v.addWidget(ch_card)
        v.addWidget(st_card)
        v.addLayout(cfg_row)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # -- signals -------------------------------------------------------
        self.start_btn.clicked.connect(self.start_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.record_btn.toggled.connect(self.record_toggled)
        self.detect_btn.clicked.connect(self.detect_requested)
        self.default_btn.clicked.connect(self.default_requested)
        self.save_btn.clicked.connect(self.save_config)
        self.loadcfg_btn.clicked.connect(self.load_config)
        self.open_file_btn.clicked.connect(self.open_file_requested)
        self.channel_table.changed.connect(self.channels_changed)
        for w in (self.fft_box, self.window_box, self.avg_box):
            w.currentTextChanged.connect(lambda *_: self.analysis_changed.emit())
        self.avg_spin.valueChanged.connect(lambda *_: self.analysis_changed.emit())
        self.overlap_spin.valueChanged.connect(lambda *_: self.analysis_changed.emit())
        self.fft_box.currentTextChanged.connect(self._update_res)
        self.fs_box.currentTextChanged.connect(self._update_res)
        self._update_res()

    # ------------------------------------------------------------------
    def _update_res(self) -> None:
        try:
            df = float(self.fs_box.currentText()) / float(self.fft_box.currentText())
            self.res_label.setText(f"Δf = {df:.3f} Hz per bin")
        except (ValueError, ZeroDivisionError):
            pass

    def set_running(self, running: bool) -> None:
        self.start_btn.setVisible(not running)
        self.stop_btn.setVisible(running)
        for w in (self.source_box, self.coupling_box, self.fs_box, self.chunk_spin,
                  self.detect_btn, self.default_btn, self.channel_table,
                  self.open_file_btn):
            w.setEnabled(not running)

    def set_stat(self, key: str, value: str) -> None:
        if key in self._stat_labels:
            self._stat_labels[key].setText(value)

    def set_device_info(self, text: str, connected: bool = False) -> None:
        self.device_label.setText(text)
        self.device_label.setProperty("connected", connected)
        self.device_label.style().unpolish(self.device_label)
        self.device_label.style().polish(self.device_label)

    # -- config sync ------------------------------------------------------
    def apply_to_config(self, cfg: AppConfig) -> None:
        cfg.acquisition.simulate = self.source_box.currentIndex() == 1
        cfg.acquisition.coupling = "iepe" if self.coupling_box.currentIndex() == 0 else "voltage"
        try:
            cfg.acquisition.sample_rate = float(self.fs_box.currentText())
        except ValueError:
            pass
        cfg.acquisition.chunk_size = self.chunk_spin.value()
        cfg.analysis.fft_size = int(self.fft_box.currentText())
        cfg.analysis.window = self.window_box.currentText()
        cfg.analysis.averaging = self.avg_box.currentText()
        cfg.analysis.avg_count = self.avg_spin.value()
        cfg.analysis.overlap = self.overlap_spin.value() / 100.0
        cfg.channels = self.channel_table.read()

    def load_from_config(self, cfg: AppConfig) -> None:
        self.source_box.setCurrentIndex(1 if cfg.acquisition.simulate else 0)
        self.coupling_box.setCurrentIndex(0 if cfg.acquisition.coupling == "iepe" else 1)
        self.fs_box.setCurrentText(str(int(cfg.acquisition.sample_rate)))
        self.chunk_spin.setValue(cfg.acquisition.chunk_size)
        self.fft_box.setCurrentText(str(cfg.analysis.fft_size))
        self.window_box.setCurrentText(cfg.analysis.window)
        self.avg_box.setCurrentText(cfg.analysis.averaging)
        self.avg_spin.setValue(cfg.analysis.avg_count)
        self.overlap_spin.setValue(int(cfg.analysis.overlap * 100))
        self.channel_table.load(cfg.channels)
