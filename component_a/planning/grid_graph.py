"""Grid-graph abstraction of the occupancy grid for path planning (A2).

Representation (per the assignment brief: "graph or occupancy grid"):
the occupancy grid ITSELF is used as an implicit graph - every free cell is
a node, edges connect the 8-connected neighbourhood. 8-connectivity is
chosen over 4-connectivity because the robot is omnidirectional in heading
(differential drive can turn on the spot), so diagonal moves are physically
executable and produce shorter, more natural paths. Diagonal moves cost
sqrt(2) * resolution so path cost is a true metric distance.

Obstacle inflation by the robot radius (Minkowski sum with a disc) turns
the point-robot assumption into a safe plan on this same grid - the robot
body can later be collision-checked trivially.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from common.environment import FREE, GridWorld

# (dr, dc, cost-multiplier) for 8-connectivity
NEIGHBOURS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5),
]


@dataclass
class GridGraph:
    """Implicit 8-connected graph over the free cells of a binary grid."""

    free: np.ndarray          # bool (rows, cols): True where the robot may go
    resolution: float

    @classmethod
    def from_world(cls, world: GridWorld, robot_radius_m: float = 0.25) -> "GridGraph":
        """Build from a GridWorld, inflating obstacles by the robot radius."""
        occupied = world.grid != FREE
        radius_cells = int(np.ceil(robot_radius_m / world.resolution))
        struct = ndimage.generate_binary_structure(2, 2)
        inflated = occupied.copy()
        for _ in range(radius_cells):  # iterative disc-ish dilation
            inflated = ndimage.binary_dilation(inflated, structure=struct)
        return cls(free=~inflated, resolution=world.resolution)

    @classmethod
    def from_probability_grid(
        cls, prob: np.ndarray, resolution: float,
        occ_threshold: float = 0.65, robot_radius_m: float = 0.25,
    ) -> "GridGraph":
        """Build from a fused probability grid (used by the integration node)."""
        struct = ndimage.generate_binary_structure(2, 2)
        inflated = prob >= occ_threshold
        radius_cells = int(np.ceil(robot_radius_m / resolution))
        for _ in range(radius_cells):
            inflated = ndimage.binary_dilation(inflated, structure=struct)
        return cls(free=~inflated, resolution=resolution)

    def neighbours(self, node: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        """Yield ``(neighbour_node, metric_cost)`` pairs; corner-cutting blocked."""
        r, c = node
        out = []
        for dr, dc, mul in NEIGHBOURS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.free.shape[0] and 0 <= nc < self.free.shape[1]):
                continue
            if not self.free[nr, nc]:
                continue
            # disallow cutting diagonally past an occupied orthogonal cell
            if dr != 0 and dc != 0 and not (self.free[r + dr, c] and self.free[r, c + dc]):
                continue
            out.append(((nr, nc), mul * self.resolution))
        return out


def path_length_m(path: list[tuple[int, int]], resolution: float) -> float:
    """Metric length of a grid path (sum of edge costs)."""
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        total += resolution * (2**0.5 if (r0 != r1 and c0 != c1) else 1.0)
    return total
