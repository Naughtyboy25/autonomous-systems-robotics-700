"""Swarm arena environment for Component B (multi-agent coverage task).

A 30 m x 30 m arena (larger than Component A's: a coverage task needs room
for emergent spatial partitioning to be meaningful) at 0.15 m resolution,
with sparse obstacles so the challenge is coordination, not maze navigation.

The arena reuses :class:`common.environment.GridWorld` (ray-casting,
occupancy queries) so Components A and B share one representation layer -
a deliberate reproducibility decision: identical collision and sensing
semantics across both components.

Coverage model (justified simplification)
------------------------------------------
An agent "covers" every free cell whose centre lies within its sensing
radius, stamped onto (a) the environment's ground-truth coverage grid and
(b) the agent's own belief grid. Line-of-sight occlusion by interior
obstacles is approximated away: the obstacles are convex and thin relative
to the 2.5 m sensing disc, and the coordination algorithms under study
(consensus, PSO) are insensitive to small over-crediting at obstacle edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.environment import GridWorld


def make_swarm_arena() -> GridWorld:
    """30 m x 30 m arena with border walls and sparse interior obstacles."""
    world = GridWorld(width_m=30.0, height_m=30.0, resolution=0.15)
    world.add_border_walls(thickness_m=0.3)
    world.add_rect(5.0, 5.0, 8.0, 7.0)
    world.add_rect(14.0, 4.0, 17.0, 9.0)
    world.add_rect(22.0, 15.0, 26.0, 18.0)
    world.add_rect(6.0, 20.0, 10.0, 23.0)
    world.add_rect(18.0, 24.0, 21.0, 26.0)
    world.add_circle(12.0, 14.0, 1.2)
    world.add_circle(24.0, 10.0, 1.0)
    return world


SPAWN_POSES = [
    (3.0, 3.0, np.deg2rad(45.0)),
    (27.0, 3.0, np.deg2rad(135.0)),
    (3.0, 27.0, np.deg2rad(-135.0)),
    (27.0, 27.0, np.deg2rad(-45.0)),
    (15.0, 15.0, 0.0),          # agent 0, the coordinator, starts centrally
]


@dataclass
class SwarmEnvironment:
    """World + ground-truth coverage bookkeeping for the swarm."""

    world: GridWorld = field(default_factory=make_swarm_arena)
    sensing_radius_m: float = 2.5
    comms_radius_m: float = 6.0

    def __post_init__(self) -> None:
        free_mask = self.world.grid == 0
        self.free_cells = int(free_mask.sum())
        self.coverage_truth = np.zeros_like(free_mask, dtype=bool)
        # pre-computed sensing disc stencil (offsets in cells)
        rad = self.sensing_radius_m / self.world.resolution
        rr, cc = np.mgrid[-int(rad): int(rad) + 1, -int(rad): int(rad) + 1]
        self._stencil = (rr * rr + cc * cc) <= rad * rad

    # ------------------------------------------------------------------
    def stamp_coverage(self, belief: np.ndarray, x: float, y: float) -> None:
        """OR the agent's sensing disc into ``belief`` (truth or agent belief)."""
        r0, c0 = self.world.world_to_grid(x, y)
        r_lo, r_hi = max(0, r0 - self._stencil.shape[0] // 2), min(
            self.world.rows, r0 + self._stencil.shape[0] // 2 + 1)
        c_lo, c_hi = max(0, c0 - self._stencil.shape[1] // 2), min(
            self.world.cols, c0 + self._stencil.shape[1] // 2 + 1)
        s = self._stencil[
            r_lo - (r0 - self._stencil.shape[0] // 2): r_hi - (r0 - self._stencil.shape[0] // 2),
            c_lo - (c0 - self._stencil.shape[1] // 2): c_hi - (c0 - self._stencil.shape[1] // 2),
        ]
        region = belief[r_lo:r_hi, c_lo:c_hi]
        region[s] = True  # in-place: belief is a reference grid

    def sample_free_waypoint(self, rng: np.random.Generator) -> tuple[float, float]:
        """Uniform random FREE cell centre (baseline wandering target)."""
        while True:
            r = int(rng.integers(0, self.world.rows))
            c = int(rng.integers(0, self.world.cols))
            if self.world.grid[r, c] == 0:
                return self.world.grid_to_world(r, c)

    def coverage_fraction(self, belief: np.ndarray | None = None) -> float:
        grid = self.coverage_truth if belief is None else belief
        free = self.world.grid == 0
        return float((grid & free).sum() / self.free_cells)
