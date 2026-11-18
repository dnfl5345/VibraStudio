"""Sensor setup: where each sensor sits and which way its printed axes point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from ..rig import AXIS_CHOICES, Rig, SensorMount, mount_matrix, rigs_dir
from . import theme

COLS = ["Sensor", "Module", "X (cm)", "Y (cm)", "Z (cm)",
        "Printed X →", "Printed Y →", "Printed Z →",
        "Tilt X°", "Tilt Y°", "Tilt Z°", "Frame"]
C_NAME, C_MOD, C_X, C_Y, C_Z, C_AX, C_AY, C_AZ, C_TX, C_TY, C_TZ, C_HAND = range(12)


def _spin(lo: float, hi: float, val: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(1)
    s.setValue(val)
    s.setButtonSymbols(QDoubleSpinBox.NoButtons)
    return s


class RigDialog(QDialog):
    def __init__(self, rig: Rig | None, modules: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sensor setup")
        self.resize(1160, 560)
        self.modules = list(modules)
        self.rig = rig or Rig(name="rig")

        frame = QLabel("Global axes: <b>+X</b> right in the front view, <b>+Y</b> away from "
                       "the viewer, <b>+Z</b> up. For each sensor pick where its printed "
                       "axes point in those global axes.")
        frame.setWordWrap(True)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(C_MOD, QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(30)

        self.clamp_spin = _spin(-1000, 1000, 0.0)
        self.clamp_box = QComboBox()
        self.clamp_box.addItems(["Clamped node at Z =", "No clamped node"])
        self.status = QLabel("")

        add_btn = QPushButton("Add sensor")
        del_btn = QPushButton("Remove")
        copy_btn = QPushButton("Copy orientation to all")
        copy_btn.setToolTip("Apply the selected row's axes and tilt to every sensor")
        sort_btn = QPushButton("Number bottom → top")
        sort_btn.setToolTip("Renumber S1, S2, ... by height, then left to right")
        load_btn = QPushButton("Load rig…")
        save_btn = QPushButton("Save rig…")
        ok_btn = QPushButton("Apply")
        ok_btn.setObjectName("Primary")
        cancel_btn = QPushButton("Cancel")

        row1 = QHBoxLayout()
        for w in (add_btn, del_btn, copy_btn, sort_btn):
            row1.addWidget(w)
        row1.addStretch(1)
        row1.addWidget(self.clamp_box)
        row1.addWidget(self.clamp_spin)
        row2 = QHBoxLayout()
        row2.addWidget(load_btn)
        row2.addWidget(save_btn)
        row2.addWidget(self.status, stretch=1)
        row2.addWidget(cancel_btn)
        row2.addWidget(ok_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(frame)
        lay.addWidget(self.table, stretch=1)
        lay.addLayout(row1)
        lay.addLayout(row2)

        add_btn.clicked.connect(self._add_row)
        del_btn.clicked.connect(self._del_row)
        copy_btn.clicked.connect(self._copy_orientation)
        sort_btn.clicked.connect(self._sort_bottom_up)
        load_btn.clicked.connect(self._load)
        save_btn.clicked.connect(self._save)
        ok_btn.clicked.connect(self._accept)
        cancel_btn.clicked.connect(self.reject)
        self._fill(self.rig)

    # -- rows -------------------------------------------------------------------
    def _fill(self, rig: Rig) -> None:
        self.table.setRowCount(0)
        for s in rig.sensors:
            self._append(s)
        clamp = next((n for n in rig.extra_nodes if n.get("fixed")), None)
        self.clamp_box.setCurrentIndex(0 if clamp or not rig.sensors else 1)
        if clamp:
            self.clamp_spin.setValue(float(clamp["position_cm"][2]))
        self._edges = [list(e) for e in rig.edges]
        self._refresh_frames()

    def _append(self, s: SensorMount) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, C_NAME, QTableWidgetItem(s.name))
        mod = QComboBox()
        mod.setEditable(True)
        mod.addItems(self.modules)
        mod.setCurrentText(s.module)
        self.table.setCellWidget(r, C_MOD, mod)
        for c, v in zip((C_X, C_Y, C_Z), s.position_cm):
            self.table.setCellWidget(r, c, _spin(-1000, 1000, float(v)))
        for c, v in zip((C_AX, C_AY, C_AZ), (s.x_to, s.y_to, s.z_to)):
            box = QComboBox()
            box.addItems(AXIS_CHOICES)
            box.setCurrentText(v)
            box.currentTextChanged.connect(lambda *_: self._refresh_frames())
            self.table.setCellWidget(r, c, box)
        for c, v in zip((C_TX, C_TY, C_TZ), s.tilt_deg):
            self.table.setCellWidget(r, c, _spin(-180, 180, float(v)))
        item = QTableWidgetItem("")
        item.setFlags(Qt.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(r, C_HAND, item)

    def _sensor(self, r: int) -> SensorMount:
        cell = self.table.cellWidget
        return SensorMount(
            name=self.table.item(r, C_NAME).text().strip() or f"S{r + 1}",
            module=cell(r, C_MOD).currentText().strip(),
            position_cm=tuple(cell(r, c).value() for c in (C_X, C_Y, C_Z)),
            x_to=cell(r, C_AX).currentText(), y_to=cell(r, C_AY).currentText(),
            z_to=cell(r, C_AZ).currentText(),
            tilt_deg=tuple(cell(r, c).value() for c in (C_TX, C_TY, C_TZ)),
        )

    def _refresh_frames(self) -> None:
        """Show right-/left-handed per sensor; flag impossible axis choices."""
        for r in range(self.table.rowCount()):
            s = self._sensor(r)
            item = self.table.item(r, C_HAND)
            try:
                hand = "right-handed" if np.linalg.det(
                    mount_matrix(s.x_to, s.y_to, s.z_to)) > 0 else "left-handed"
                item.setText(hand)
                item.setForeground(theme.pg.mkColor(
                    theme.TEXT_MUTED if hand == "right-handed" else theme.YELLOW))
            except ValueError:
                item.setText("axes clash")
                item.setForeground(theme.pg.mkColor(theme.RED))

    def _add_row(self) -> None:
        r = self.table.rowCount()
        last = self._sensor(r - 1) if r else None
        used = {self._sensor(i).module for i in range(r)}
        module = next((m for m in self.modules if m not in used), "")
        z = last.position_cm[2] + 7.0 if last else 0.0
        self._append(SensorMount(
            name=f"S{r + 1}", module=module, position_cm=(0.0, 0.0, z),
            x_to=last.x_to if last else "+X", y_to=last.y_to if last else "+Y",
            z_to=last.z_to if last else "+Z"))
        if last:
            self._edges.append([last.name, f"S{r + 1}"])
        self._refresh_frames()

    def _del_row(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        name = self._sensor(r).name
        self.table.removeRow(r)
        self._edges = [e for e in self._edges if name not in e]
        self._refresh_frames()

    def _copy_orientation(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            self.status.setText("Select the row to copy from first.")
            return
        for other in range(self.table.rowCount()):
            for c in (C_AX, C_AY, C_AZ):
                self.table.cellWidget(other, c).setCurrentText(
                    self.table.cellWidget(r, c).currentText())
            for c in (C_TX, C_TY, C_TZ):
                self.table.cellWidget(other, c).setValue(self.table.cellWidget(r, c).value())
        self._refresh_frames()

    def _sort_bottom_up(self) -> None:
        sensors = [self._sensor(r) for r in range(self.table.rowCount())]
        order = sorted(sensors, key=lambda s: (round(s.position_cm[2], 1), s.position_cm[0]))
        rename = {s.name: f"S{i + 1}" for i, s in enumerate(order)}
        self._edges = [[rename.get(a, a), rename.get(b, b)] for a, b in self._edges]
        self.table.setRowCount(0)
        for s in order:
            s.name = rename[s.name]
            self._append(s)
        self._refresh_frames()

    # -- result ---------------------------------------------------------------------
    def build(self) -> Rig:
        sensors = [self._sensor(r) for r in range(self.table.rowCount())]
        extra: list[dict] = []
        edges = [e for e in self._edges if e[0] != "Clamp" and e[1] != "Clamp"]
        linked = {n for e in edges for n in e}
        if not edges and len(sensors) > 1:          # default: a chain by height
            chain = sorted(sensors, key=lambda s: s.position_cm[2])
            edges = [[a.name, b.name] for a, b in zip(chain, chain[1:])]
            linked = {n for e in edges for n in e}
        for s in sensors:
            linked.add(s.name)
        if self.clamp_box.currentIndex() == 0 and sensors:
            z = self.clamp_spin.value()
            low = min(sensors, key=lambda s: abs(s.position_cm[2] - z))
            extra.append({"name": "Clamp", "fixed": True,
                          "position_cm": [low.position_cm[0], low.position_cm[1], z]})
            edges.insert(0, ["Clamp", low.name])
        rig = Rig(name=self.rig.name, description=self.rig.description, sensors=sensors,
                  extra_nodes=extra, edges=edges, reference=self.rig.reference,
                  hammer=dict(self.rig.hammer))
        rig.validate()
        return rig

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load rig", str(rigs_dir()), "Rig (*.json)")
        if not path:
            return
        try:
            self.rig = Rig.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load rig", f"Cannot read this rig file:\n{exc}")
            return
        self._fill(self.rig)
        self.status.setText(f"Loaded {Path(path).name}")

    def _save(self) -> None:
        try:
            rig = self.build()
        except ValueError as exc:
            QMessageBox.warning(self, "Save rig", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save rig", str(rigs_dir() / f"{rig.name}.json"),
                                              "Rig (*.json)")
        if path:
            rig.save(path)
            self.rig.path = path
            self.status.setText(f"Saved {Path(path).name}")

    def _accept(self) -> None:
        try:
            self.result_rig = self.build()
        except ValueError as exc:
            QMessageBox.warning(self, "Sensor setup", f"{exc}\nFix the highlighted row.")
            return
        self.result_rig.path = self.rig.path
        self.accept()
