"""Shared 2-D occupancy-grid world model for the ASR700 assignment.

This module is the single source of truth for the simulated environment used
by BOTH components of the assignment:

* Component A uses it as ground truth for the LiDAR/camera simulators and as
  the map on which Dijkstra/A* plan.
* Component B reuses the same arena (scaled up) for the swarm coverage task.

Design decisions (kept here so every module inherits them)
----------------------------------------------------------
* Occupancy grid at 0.1 m resolution: coarse enough that 360-beam ray-casting
  and grid planning stay fast on a low-RAM laptop, fine enough that a
  differential-drive robot (radius ~0.2 m) can be collision-checked exactly.
* Convention: grid[row, col], world x -> col, world y -> row; cell CENTRES are
  used when converting back to world coordinates. ``1`` = occupied, ``0`` = free.
* All randomness flows through a seeded ``numpy.random.Generator`` so every
  figure in the submission is bit-for-bit reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FREE = 0
OCCUPIED = 1


@dataclass
class GridWorld:
    """A rectangular 2-D occupancy grid with metric extents."""

    width_m: float
    height_m: float
    resolution: float = 0.1
    grid: np.ndarray = field(init=False, repr=False)
    landmarks: list[tuple[float, float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rows = int(round(self.height_m / self.resolution))
        self.cols = int(round(self.width_m / self.resolution))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def add_border_walls(self, thickness_m: float = 0.2) -> None:
        """Add a wall ring around the arena (required for ray-casting)."""
        t = max(1, int(round(thickness_m / self.resolution)))
        self.grid[:t, :] = OCCUPIED
        self.grid[-t:, :] = OCCUPIED
        self.grid[:, :t] = OCCUPIED
        self.grid[:, -t:] = OCCUPIED

    def add_rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Add an axis-aligned rectangular obstacle (metric corners)."""
        r0, c0 = self.world_to_grid(x0, y0)
        r1, c1 = self.world_to_grid(x1, y1)
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))
        self.grid[r0 : r1 + 1, c0 : c1 + 1] = OCCUPIED

    def add_circle(self, cx: float, cy: float, radius: float) -> None:
        """Add a circular obstacle (represents e.g. a pillar or a barrel)."""
        rr, cc = np.ogrid[: self.rows, : self.cols]
        xs = (cc + 0.5) * self.resolution
        ys = (rr + 0.5) * self.resolution
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
        self.grid[mask] = OCCUPIED

    def add_landmark(self, x: float, y: float, color: str) -> None:
        """Register a visual landmark (used by the camera simulator only)."""
        self.landmarks.append((x, y, color))

    # ------------------------------------------------------------------
    # coordinate transforms
    # ------------------------------------------------------------------
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        row = int(y / self.resolution)
        col = int(x / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) * self.resolution, (row + 0.5) * self.resolution

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == FREE

    def is_free_world(self, x: float, y: float) -> bool:
        return self.is_free(*self.world_to_grid(x, y))

    # ------------------------------------------------------------------
    # sensing primitive
    # ------------------------------------------------------------------
    def ray_cast(self, x: float, y: float, angle: float, max_range: float) -> float:
        """March along a ray and return the distance to the first obstacle.

        A fine step of half a cell is used instead of a DDA so that thin
        walls can never be tunnelled through; the cost is negligible at our
        grid sizes and the code stays auditable for marking purposes.
        """
        step = self.resolution * 0.5
        dx, dy = np.cos(angle) * step, np.sin(angle) * step
        dist = step
        px, py = x, y
        while dist < max_range:
            px += dx
            py += dy
            row, col = self.world_to_grid(px, py)
            if not self.in_bounds(row, col):
                return dist
            if self.grid[row, col] == OCCUPIED:
                return dist
            dist += step
        return max_range


def make_assignment_arena(seed: int = 42) -> GridWorld:
    """Build the deterministic 20 m x 20 m arena used throughout Component A.

    Layout: border walls, a mix of rectangular and circular obstacles and
    three colour-coded landmarks for the camera pipeline. The layout is
    hand-designed (not random) so the report figures are stable; ``seed`` is
    accepted for API symmetry with the swarm world.
    """
    rng = np.random.default_rng(seed)
    del rng  # layout is fixed by design; seed kept for interface symmetry
    world = GridWorld(width_m=20.0, height_m=20.0, resolution=0.1)
    world.add_border_walls()
    # rectangular blocks
    world.add_rect(4.0, 4.0, 6.5, 5.5)
    world.add_rect(10.0, 8.0, 11.5, 12.5)
    world.add_rect(14.0, 3.0, 17.5, 4.2)
    world.add_rect(3.0, 12.0, 5.0, 14.5)
    world.add_rect(13.5, 14.0, 15.5, 17.0)
    # circular pillars
    world.add_circle(8.0, 15.5, 1.0)
    world.add_circle(15.5, 9.0, 0.8)
    # colour-coded landmarks for the camera/OpenCV pipeline
    world.add_landmark(2.0, 2.0, "red")
    world.add_landmark(18.0, 18.0, "green")
    world.add_landmark(18.0, 6.5, "blue")
    return world


START_POSE_A = (1.5, 1.5, np.deg2rad(45.0))  # x, y, theta for Component A
GOAL_A = (18.0, 17.0)
