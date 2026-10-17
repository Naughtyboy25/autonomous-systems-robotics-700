"""Multi-sensor fusion into a log-odds occupancy grid (Component A1).

Fusion design justification
---------------------------
Two modalities are fused: the 360-deg LiDAR (accurate range, sparse angular
sampling per obstacle edge) and the overhead camera (dense local coverage,
colour-classified obstacle evidence, no metric range per pixel until
projected). They are fused into a **log-odds occupancy grid** because:

1. Log-odds turns Bayes updates into *addition*, so an arbitrary number of
   asynchronous sensor updates can be combined in any order.
2. The inverse sensor model for each modality is written in the same
   logarithmic units, so relative trust is expressed by the magnitude of
   ``L_OCC``/``L_FREE`` per sensor: the camera (exact colour classification,
   no range noise after projection) contributes a stronger occupied update
   than the LiDAR (2 cm Gaussian range noise).
3. Unknown space stays at exactly ``p = 0.5`` until observed, which the
   planner can treat differently from confirmed-free space.

This is the standard approach of Thrun's occupancy-grid mapping and matches
what the Webots controller in ``arena.wbt`` implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.environment import GridWorld

# log-odds increments; |lidar| < |camera| encodes relative trust (see docstring)
L0 = 0.0
L_FREE_LIDAR = -0.4
L_OCC_LIDAR = 0.85
L_OCC_CAMERA = 1.2
L_CLAMP = 4.0


@dataclass
class OccupancyGridFusion:
    """Log-odds occupancy grid fusing LiDAR scans and camera detections."""

    world: GridWorld
    log_odds: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.log_odds = np.full((self.world.rows, self.world.cols), L0, dtype=np.float32)
        self.n_lidar_updates = 0
        self.n_camera_updates = 0

    # ------------------------------------------------------------------
    def integrate_lidar(self, x: float, y: float, ranges: np.ndarray, bearings: np.ndarray) -> None:
        """Inverse sensor model: cells along each beam are free up to the hit."""
        res = self.world.resolution
        for r, b in zip(ranges, bearings):
            hit = r < (8.0 - 1e-6)  # max_range sentinel handled by caller value
            steps = int(r / res)
            for i in range(steps):
                d = (i + 0.5) * res
                row, col = self.world.world_to_grid(x + d * np.cos(b), y + d * np.sin(b))
                if not self.world.in_bounds(row, col):
                    break
                self.log_odds[row, col] += L_FREE_LIDAR
            if hit:
                row, col = self.world.world_to_grid(
                    x + r * np.cos(b) - 1e-9, y + r * np.sin(b) - 1e-9
                )
                if self.world.in_bounds(row, col):
                    self.log_odds[row, col] += L_OCC_LIDAR
        self.n_lidar_updates += 1
        np.clip(self.log_odds, -L_CLAMP, L_CLAMP, out=self.log_odds)

    # ------------------------------------------------------------------
    def integrate_camera(self, detections_world: list[tuple[float, float]], radius_m: float = 0.25) -> None:
        """Stamp occupied evidence around each detected obstacle centroid."""
        for wx, wy in detections_world:
            r0, c0 = self.world.world_to_grid(wx, wy)
            rad = int(radius_m / self.world.resolution)
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr * dr + dc * dc > rad * rad:
                        continue
                    row, col = r0 + dr, c0 + dc
                    if self.world.in_bounds(row, col):
                        self.log_odds[row, col] += L_OCC_CAMERA
            self.n_camera_updates += 1
        np.clip(self.log_odds, -L_CLAMP, L_CLAMP, out=self.log_odds)

    # ------------------------------------------------------------------
    def probabilities(self) -> np.ndarray:
        """Convert log-odds back to occupancy probabilities in [0, 1]."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def occupied_mask(self, threshold: float = 0.65) -> np.ndarray:
        return self.probabilities() >= threshold

    def free_mask(self, threshold: float = 0.35) -> np.ndarray:
        return self.probabilities() <= threshold
