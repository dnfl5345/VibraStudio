"""Test rig description: where each sensor sits and how it is turned.

A triaxial sensor reports along its own printed X/Y/Z axes, and those differ
from the structure's global axes depending on how it was mounted. A Rig
stores, per sensor, the position in global coordinates, where each printed
axis points, and the module it is wired to. From that it derives the node-link
geometry and, per channel, the unit vector of the measurement in global
coordinates, so ODS deflection is always drawn in one absolute frame.

Global frame convention for a rig photographed from the front:
    +X = right in the photo, +Y = into the photo, +Z = up.

A rig is plain JSON, so a structure read from photos can be written directly.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ChannelConfig
from .geometry import Geometry

AXIS_TO_VEC = {
    "+X": (1, 0, 0), "-X": (-1, 0, 0),
    "+Y": (0, 1, 0), "-Y": (0, -1, 0),
    "+Z": (0, 0, 1), "-Z": (0, 0, -1),
}
AXIS_CHOICES = list(AXIS_TO_VEC)
LOCAL_AXES = ("X", "Y", "Z")


def _rot(axis: int, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mount_matrix(x_to: str, y_to: str, z_to: str,
                 tilt_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """3x3 matrix whose columns are the printed X, Y, Z axes in global axes.

    ``tilt_deg`` is a mounting error about global X, Y, Z, applied after the
    nominal orientation. Raises ValueError if two axes are not perpendicular.
    """
    base = np.array([AXIS_TO_VEC[x_to], AXIS_TO_VEC[y_to], AXIS_TO_VEC[z_to]], dtype=float).T
    if abs(abs(np.linalg.det(base)) - 1.0) > 1e-9:
        raise ValueError(f"axes {x_to}, {y_to}, {z_to} are not perpendicular")
    return _rot(2, tilt_deg[2]) @ _rot(1, tilt_deg[1]) @ _rot(0, tilt_deg[0]) @ base


@dataclass
class SensorMount:
    name: str
    module: str                                  # e.g. cDAQ9189-2196A44Mod6
    position_cm: tuple[float, float, float]      # global coordinates
    x_to: str = "+X"
    y_to: str = "+Y"
    z_to: str = "+Z"
    tilt_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ai: dict = field(default_factory=lambda: {"X": 0, "Y": 1, "Z": 2})  # printed axis -> input

    def matrix(self) -> np.ndarray:
        return mount_matrix(self.x_to, self.y_to, self.z_to, tuple(self.tilt_deg))

    @property
    def handedness(self) -> str:
        return "R" if np.linalg.det(self.matrix()) > 0 else "L"


@dataclass
class Rig:
    name: str = "rig"
    description: str = ""
    sensors: list[SensorMount] = field(default_factory=list)
    # Nodes without a sensor, e.g. the clamp: {"name", "position_cm", "fixed"}
    extra_nodes: list[dict] = field(default_factory=list)
    edges: list[list[str]] = field(default_factory=list)  # by node name
    reference: str = ""                                   # label of the phase reference
    path: str = ""

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "frame": "+X right in the front photo, +Y into the photo, +Z up",
            "sensors": [
                {"name": s.name, "module": s.module, "position_cm": list(s.position_cm),
                 "x_to": s.x_to, "y_to": s.y_to, "z_to": s.z_to,
                 "tilt_deg": list(s.tilt_deg), "ai": s.ai}
                for s in self.sensors
            ],
            "extra_nodes": self.extra_nodes,
            "edges": self.edges,
            "reference": self.reference,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")
        self.path = str(path)

    @classmethod
    def load(cls, path: str | Path) -> "Rig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        rig = cls(
            name=str(d.get("name", Path(path).stem)),
            description=str(d.get("description", "")),
            sensors=[
                SensorMount(
                    name=str(s["name"]), module=str(s.get("module", "")),
                    position_cm=tuple(float(v) for v in s["position_cm"]),
                    x_to=s.get("x_to", "+X"), y_to=s.get("y_to", "+Y"), z_to=s.get("z_to", "+Z"),
                    tilt_deg=tuple(float(v) for v in s.get("tilt_deg", (0, 0, 0))),
                    ai=dict(s.get("ai", {"X": 0, "Y": 1, "Z": 2})),
                )
                for s in d.get("sensors", [])
            ],
            extra_nodes=list(d.get("extra_nodes", [])),
            edges=[list(e) for e in d.get("edges", [])],
            reference=str(d.get("reference", "")),
            path=str(path),
        )
        rig.validate()
        return rig

    def validate(self) -> None:
        names = [s.name for s in self.sensors] + [n["name"] for n in self.extra_nodes]
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique")
        for s in self.sensors:
            s.matrix()                           # raises on a bad orientation
        for a, b in self.edges:
            if a not in names or b not in names:
                raise ValueError(f"edge {a}-{b} refers to an unknown node")

    # -- derived --------------------------------------------------------------
    def node_names(self) -> list[str]:
        """Sensors first, so sensor S<n> is node <n>; sensorless nodes follow."""
        return [s.name for s in self.sensors] + [n["name"] for n in self.extra_nodes]

    def span_cm(self) -> float:
        pts = [s.position_cm for s in self.sensors] + [n["position_cm"] for n in self.extra_nodes]
        return float(np.max(np.ptp(np.array(pts, dtype=float), axis=0))) if pts else 1.0

    def to_geometry(self) -> Geometry:
        names = self.node_names()
        pts = [s.position_cm for s in self.sensors] + [n["position_cm"] for n in self.extra_nodes]
        nodes = np.array(pts, dtype=float).reshape(-1, 3) / 100.0     # metres
        index = {n: i for i, n in enumerate(names)}
        fixed = [index[n["name"]] for n in self.extra_nodes if n.get("fixed")]
        mounts = [{"node": i, "name": s.name, "matrix": s.matrix().tolist()}
                  for i, s in enumerate(self.sensors)]
        return Geometry(self.name, nodes, [(index[a], index[b]) for a, b in self.edges],
                        {"shape": "rig", "mounts": mounts}, fixed=fixed, node_labels=names)

    def to_channels(self, detected: list[str] | None = None) -> list[ChannelConfig]:
        """Channels of every sensor, plus any other detected input as AUX.

        AUX inputs are recorded but have no place on the structure (point 0).
        """
        z_top = self._z_extent()
        out: list[ChannelConfig] = []
        used: set[str] = set()
        for node, s in enumerate(self.sensors, start=1):
            R = s.matrix()
            for j, local in enumerate(LOCAL_AXES):
                if local not in s.ai:
                    continue
                vec = R[:, j]
                k = int(np.argmax(np.abs(vec)))
                phys = f"{s.module}/ai{s.ai[local]}"
                used.add(phys)
                out.append(ChannelConfig(
                    physical_channel=phys,
                    label=f"{s.name}_{'XYZ'[k]}",
                    enabled=detected is None or phys in detected,
                    point=node, direction="XYZ"[k],
                    position=z_top(s.position_cm[2]),
                    sign=1 if vec[k] >= 0 else -1,
                    axis_vec=[float(v) for v in vec],
                    sensor_axis=local,
                ))
        for n, phys in enumerate((p for p in (detected or []) if p not in used), start=1):
            out.append(ChannelConfig(physical_channel=phys, label=f"AUX{n}", enabled=True,
                                     point=0, direction="Z", position=0.0))
        if not any(c.label == self.reference and c.enabled for c in out):
            self.reference = next((c.label for c in reversed(out) if c.point and c.enabled), "")
        for c in out:
            c.is_reference = c.label == self.reference
        return out

    def _z_extent(self):
        zs = [s.position_cm[2] for s in self.sensors] + [n["position_cm"][2] for n in self.extra_nodes]
        lo, hi = (min(zs), max(zs)) if zs else (0.0, 1.0)
        return lambda z: (z - lo) / ((hi - lo) or 1.0)

    def channels_for_recording(self, channel_map: list[dict]) -> list[ChannelConfig] | None:
        """Rig mapping for a recording that stored its channel map.

        Rows are matched by physical channel, so a file recorded with another
        sensor map (e.g. a plain auto-map before the rig existed) still gets
        the rig's positions and directions. Returns None when no row matches.
        Rows get physical_channel 'file/ch<i>' like any playback channel.
        """
        base = {c.physical_channel: c for c in self.to_channels()}
        out, hits, aux = [], 0, 0
        for i, entry in enumerate(channel_map):
            c = base.get(str(entry.get("physical_channel", "")))
            if c is not None:
                hits += 1
                out.append(ChannelConfig(**{**c.__dict__, "physical_channel": f"file/ch{i}",
                                            "enabled": bool(entry.get("enabled", True))}))
            else:
                aux += 1
                out.append(ChannelConfig(physical_channel=f"file/ch{i}", label=f"AUX{aux}",
                                         point=0, direction="Z", position=0.0))
        if not hits:
            return None
        ref = next((c for c in out if c.label == self.reference and c.point), None)
        for c in out:
            c.is_reference = c is ref
        return out

    def legacy_channels(self, labels: list[str]) -> list[ChannelConfig] | None:
        """Mapping for a recording made before files stored their channel map.

        Those files number modules in order on the first chassis, so 'S3_Z'
        means Mod3, printed Z. Returns None if the labels do not fit that.
        """
        base = {c.physical_channel: c for c in self.to_channels()}
        chassis: list[str] = []
        for s in self.sensors:
            head = s.module.split("Mod")[0]
            if head not in chassis:
                chassis.append(head)
        if not chassis:
            return None
        out, aux = [], 0
        for i, label in enumerate(labels):
            m = re.match(r"^S(\d+)_([XYZ])$", label)
            if not m:
                return None
            sensor_ai = {"X": 0, "Y": 1, "Z": 2}[m.group(2)]
            c = base.get(f"{chassis[0]}Mod{int(m.group(1))}/ai{sensor_ai}")
            if c is not None:
                out.append(ChannelConfig(**{**c.__dict__, "physical_channel": f"file/ch{i}"}))
            else:
                aux += 1
                out.append(ChannelConfig(physical_channel=f"file/ch{i}", label=f"AUX{aux}",
                                         point=0, direction="Z", position=0.0))
        return out if any(c.point for c in out) else None


def rigs_dir() -> Path:
    """Folder holding the rig files shipped with the app."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "rigs"
    return Path(__file__).resolve().parent.parent / "rigs"
