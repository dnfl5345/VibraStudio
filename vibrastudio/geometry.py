"""3D structure geometry for ODS: parametric templates + channel mapping.

A Geometry is a set of 3D nodes (1-based ids for display) and edges
connecting them. Templates generate common test structures from a couple of
parameters — e.g. a cylinder with 12 nodes per ring and 2 layers gives the
classic 12x2 ODS grid. Channels are mapped onto nodes via ChannelConfig.point
(node id) and ChannelConfig.direction (deflection axis at that node).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ChannelConfig

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


@dataclass
class Geometry:
    name: str
    nodes: np.ndarray                      # (n, 3) float64
    edges: list[tuple[int, int]]           # 0-based node index pairs
    params: dict = field(default_factory=dict)
    fixed: list[int] = field(default_factory=list)   # clamped node indices (0-based)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    def bbox_size(self) -> float:
        if self.n_nodes == 0:
            return 1.0
        span = np.ptp(self.nodes, axis=0)
        return float(np.max(span)) or 1.0

    def edge_array(self, nodes: np.ndarray | None = None) -> np.ndarray:
        """Interleaved (2*n_edges, 3) vertex array for GL 'lines' mode."""
        pts = self.nodes if nodes is None else nodes
        if not self.edges:
            return np.zeros((0, 3))
        idx = np.asarray(self.edges, dtype=int).reshape(-1)
        return pts[idx]

    def fix_base(self, tol_fraction: float = 0.05) -> int:
        """Mark the lowest nodes (along Z) as clamped. Planar shapes are
        left free — with every node at z=0 there is no meaningful base."""
        z = self.nodes[:, 2]
        span = float(np.ptp(z))
        if span < 1e-9:
            self.fixed = []
            return 0
        self.fixed = [int(i) for i in np.nonzero(z <= z.min() + tol_fraction * span)[0]]
        return len(self.fixed)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": self.nodes.tolist(),
            "edges": [list(e) for e in self.edges],
            "params": self.params,
            "fixed": [int(i) for i in self.fixed],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(
            name=str(d.get("name", "custom")),
            nodes=np.asarray(d.get("nodes", []), dtype=float).reshape(-1, 3),
            edges=[tuple(int(i) for i in e) for e in d.get("edges", [])],
            params=dict(d.get("params", {})),
            fixed=[int(i) for i in d.get("fixed", [])],
        )


# -- templates -----------------------------------------------------------
def make_line(n_nodes: int = 5, length: float = 1.0, ccw: bool = True) -> Geometry:
    """Vertical beam along Z (cantilever).

    ccw=True numbers bottom-up (node 1 at the base); ccw=False top-down
    (node 1 at the free tip) — match how the sensors are labeled.
    """
    n = max(2, int(n_nodes))
    z = np.linspace(0.0, length, n)
    if not ccw:
        z = z[::-1]
    nodes = np.column_stack((np.zeros(n), np.zeros(n), z))
    edges = [(i, i + 1) for i in range(n - 1)]
    return Geometry("line", nodes, edges,
                    {"shape": "line", "n_nodes": n, "length": length, "ccw": ccw})


def make_grid(nx: int = 3, ny: int = 3, width: float = 1.0, depth: float = 1.0,
              ccw: bool = True) -> Geometry:
    """Plate in the XY plane; node ids row-major (x fastest).

    ccw=True runs +x within each row; ccw=False mirrors the traversal
    (x right-to-left viewed from +Z).
    """
    nx, ny = max(2, int(nx)), max(2, int(ny))
    xs = np.linspace(-width / 2, width / 2, nx)
    if not ccw:
        xs = xs[::-1]
    ys = np.linspace(-depth / 2, depth / 2, ny)
    nodes = np.array([(x, y, 0.0) for y in ys for x in xs])
    edges: list[tuple[int, int]] = []
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if i < nx - 1:
                edges.append((k, k + 1))
            if j < ny - 1:
                edges.append((k, k + nx))
    return Geometry("grid", nodes, edges,
                    {"shape": "grid", "nx": nx, "ny": ny, "width": width,
                     "depth": depth, "ccw": ccw})


def make_circle(n_nodes: int = 12, radius: float = 0.5, ccw: bool = True) -> Geometry:
    """Ring in the XY plane, node 1 at angle 0.

    ccw=True numbers nodes counter-clockwise (viewed from +Z); ccw=False
    clockwise — match whichever way the sensors were labeled on the rig.
    """
    n = max(3, int(n_nodes))
    sgn = 1.0 if ccw else -1.0
    ang = sgn * 2 * np.pi * np.arange(n) / n
    nodes = np.column_stack((radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)))
    edges = [(i, (i + 1) % n) for i in range(n)]
    return Geometry("circle", nodes, edges,
                    {"shape": "circle", "n_nodes": n, "radius": radius, "ccw": ccw})


def make_cylinder(n_per_ring: int = 12, n_layers: int = 2,
                  radius: float = 0.5, height: float = 1.0,
                  ccw: bool = True) -> Geometry:
    """Cylinder: n_layers rings of n_per_ring nodes, stacked along Z.

    Node ids run around ring 1 first (1..n), then ring 2 (n+1..2n), etc.
    Edges: each ring closed + verticals between adjacent rings.
    ccw selects the numbering direction viewed from +Z.
    """
    n, layers = max(3, int(n_per_ring)), max(1, int(n_layers))
    sgn = 1.0 if ccw else -1.0
    ang = sgn * 2 * np.pi * np.arange(n) / n
    zs = [0.0] if layers == 1 else np.linspace(0.0, height, layers)
    nodes = np.array([
        (radius * np.cos(a), radius * np.sin(a), z) for z in np.atleast_1d(zs) for a in ang
    ])
    edges: list[tuple[int, int]] = []
    for layer in range(layers):
        base = layer * n
        edges.extend((base + i, base + (i + 1) % n) for i in range(n))
        if layer < layers - 1:
            edges.extend((base + i, base + n + i) for i in range(n))
    return Geometry("cylinder", nodes, edges,
                    {"shape": "cylinder", "n_per_ring": n, "n_layers": layers,
                     "radius": radius, "height": height, "ccw": ccw})


SHAPES = {
    "line": make_line,
    "grid": make_grid,
    "circle": make_circle,
    "cylinder": make_cylinder,
}


def build(shape: str, **params) -> Geometry:
    fn = SHAPES.get(shape)
    if fn is None:
        raise ValueError(f"Unknown shape '{shape}'")
    return fn(**params)


# -- channel mapping ------------------------------------------------------
def auto_map_channels(
    channels: list[ChannelConfig], geometry: Geometry, axes: tuple[str, ...] = ("X", "Z")
) -> list[ChannelConfig]:
    """Assign channels to nodes in id order, cycling measurement axes per node.

    With axes=("X","Z"): ch1->node1/X, ch2->node1/Z, ch3->node2/X, ...
    With axes=("Z",):    ch1->node1/Z, ch2->node2/Z, ...
    Channels beyond the last node wrap back to node 1. Labels are rewritten
    to S<node>_<axis> so the table always reads consistently with the map.
    """
    n_axes = max(1, len(axes))
    out = []
    for i, ch in enumerate(channels):
        node = (i // n_axes) % max(1, geometry.n_nodes) + 1
        ch.point = node
        ch.direction = axes[i % n_axes]
        ch.label = f"S{node}_{ch.direction}"
        ch.position = geometry.nodes[node - 1, 2] / max(geometry.bbox_size(), 1e-9)
        out.append(ch)
    return out
