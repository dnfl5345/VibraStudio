"""Configuration models for channels, acquisition and analysis settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# 18 visually distinct channel colors tuned for the dark theme.
CHANNEL_COLORS = [
    "#e08b6d", "#e0b25a", "#a3c95a", "#5bbf9a", "#5aa9d6", "#7a8ce0",
    "#a97fe0", "#d977b8", "#e05a5a", "#c98f4a", "#8fbf6a", "#4ac9a3",
    "#4a9fc9", "#5a6ee0", "#8f5ae0", "#c95aa3", "#d97757", "#b8c95a",
]

DIRECTIONS = ["X", "Y", "Z"]


@dataclass
class ChannelConfig:
    """One physical accelerometer channel and its geometry for ODS."""

    physical_channel: str
    label: str
    enabled: bool = True
    point: int = 1                # measurement point id (geometry node)
    direction: str = "Z"          # X / Y / Z
    position: float = 0.0         # normalized position along structure [0..1]
    sensitivity_mv_g: float = 100.0
    is_reference: bool = False    # reference channel for FRF / ODS phase
    sign: int = 1                 # +1/-1: flips a left-handed sensor axis

    def color(self, index: int) -> str:
        return CHANNEL_COLORS[index % len(CHANNEL_COLORS)]


@dataclass
class AcquisitionConfig:
    sample_rate: float = 12800.0
    chunk_size: int = 1024
    coupling: str = "iepe"        # "iepe" (accel chan) or "voltage"
    min_val: float = -50.0        # g (iepe) or V (voltage)
    max_val: float = 50.0
    simulate: bool = False        # force simulation backend


@dataclass
class AnalysisConfig:
    fft_size: int = 8192
    window: str = "hann"          # rectangular/hann/hamming/blackman/blackmanharris/flattop
    averaging: str = "exponential"  # none/linear/exponential/peakhold
    avg_count: int = 8
    overlap: float = 0.5          # 0.0 .. 0.9


@dataclass
class AppConfig:
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    channels: list[ChannelConfig] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)   # ODS 3D structure (see geometry.py)

    # ------------------------------------------------------------------
    def enabled_channels(self) -> list[ChannelConfig]:
        return [c for c in self.channels if c.enabled]

    def reference_index(self) -> int:
        """Index of the reference channel among *enabled* channels."""
        enabled = self.enabled_channels()
        for i, ch in enumerate(enabled):
            if ch.is_reference:
                return i
        return 0

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(
            acquisition=AcquisitionConfig(**raw.get("acquisition", {})),
            analysis=AnalysisConfig(**raw.get("analysis", {})),
            channels=[ChannelConfig(**c) for c in raw.get("channels", [])],
            geometry=dict(raw.get("geometry", {})),
        )
        return cfg


def default_channel_template(chassis: str = "cDAQ9189-233EBC6", n_channels: int = 18) -> list[ChannelConfig]:
    """Build the default 18-channel map: 9 points x (X, Z) on NI-9230 modules.

    NI-9230 exposes 3 channels per module, so 18 channels span Mod1..Mod6.
    Points are spaced evenly along a normalized structure height.
    """
    channels: list[ChannelConfig] = []
    n_points = max(1, n_channels // 2)
    for i in range(n_channels):
        module = i // 3 + 1
        ai = i % 3
        point = i // 2 + 1
        direction = "X" if i % 2 == 0 else "Z"
        channels.append(
            ChannelConfig(
                physical_channel=f"{chassis}Mod{module}/ai{ai}",
                label=f"S{point}_{direction}",
                enabled=True,
                point=point,
                direction=direction,
                position=point / n_points,
                sensitivity_mv_g=100.0,
                is_reference=(i == 0),
            )
        )
    return channels


def channels_from_detected(physical: list[str], max_channels: int = 96) -> list[ChannelConfig]:
    """Map detected physical AI channels onto sensors.

    Three-input modules (NI-9230) are treated as one triaxial sensor each:
    ai0/ai1/ai2 -> X/Y/Z of sensor S<n>, numbered across every chassis in
    the order given (two 8-slot chassis -> S1..S16, 48 channels). Any other
    layout falls back to the X/Z pair template.
    """
    physical = physical[:max_channels]

    modules: dict[str, list[str]] = {}
    for name in physical:
        modules.setdefault(name.rsplit("/", 1)[0], []).append(name)
    if modules and all(len(v) == 3 for v in modules.values()):
        n_sensors = len(modules)
        channels = []
        for s, names in enumerate(modules.values(), start=1):
            for name, axis in zip(names, "XYZ"):
                channels.append(
                    ChannelConfig(
                        physical_channel=name,
                        label=f"S{s}_{axis}",
                        enabled=True,
                        point=s,
                        direction=axis,
                        position=s / n_sensors,
                        sensitivity_mv_g=100.0,
                        is_reference=(not channels),
                    )
                )
        return channels

    n_points = max(1, len(physical) // 2) if len(physical) > 1 else 1
    channels: list[ChannelConfig] = []
    for i, name in enumerate(physical):
        point = i // 2 + 1
        direction = "X" if i % 2 == 0 else "Z"
        channels.append(
            ChannelConfig(
                physical_channel=name,
                label=f"S{point}_{direction}",
                enabled=True,
                point=point,
                direction=direction,
                position=point / n_points,
                sensitivity_mv_g=100.0,
                is_reference=(i == 0),
            )
        )
    return channels
