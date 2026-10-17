"""Simulated 2-D LiDAR sensor (Component A1 - sensor simulation).

Model
-----
The LiDAR is modelled as a full-circle (360 degrees) single-layer range scanner,
matching the behaviour of the Webots ``Lidar`` node configured later in
``webots/worlds/arena.wbt``. Each beam is ray-cast against the occupancy grid
(:meth:`common.environment.GridWorld.ray_cast`).

Noise model
-----------
Additive zero-mean Gaussian range noise (sigma = 2 cm), clipped to
``[0, max_range]``. Gaussian noise is the standard first-order model for
time-of-flight scanners (see e.g. Siegwart, Nourbakhsh & Scaramuzza,
*Introduction to Autonomous Mobile Robots*) and is deliberately conservative:
the fusion stage must be robust to small range errors.

Returns ``max_range`` on a miss so downstream code can distinguish "no return"
(``range == max_range``) from a real hit without sentinel values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.environment import GridWorld


@dataclass
class LidarSimulator:
    """Ray-cast 2-D LiDAR with Gaussian range noise."""

    world: GridWorld
    n_beams: int = 360
    max_range: float = 8.0
    fov: float = 2.0 * np.pi
    noise_std: float = 0.02
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def scan(self, x: float, y: float, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ranges, bearings)`` for a scan at pose ``(x, y, theta)``.

        ``bearings`` are world-frame absolute angles (rad); ``ranges`` in metres.
        """
        offsets = np.linspace(-self.fov / 2.0, self.fov / 2.0, self.n_beams, endpoint=False)
        bearings = theta + offsets
        ranges = np.array(
            [self.world.ray_cast(x, y, float(a), self.max_range) for a in bearings]
        )
        noise = self.rng.normal(0.0, self.noise_std, size=ranges.shape)
        # no noise on max-range returns: a miss carries no range information
        hits = ranges < self.max_range
        ranges[hits] = np.clip(ranges[hits] + noise[hits], 0.0, self.max_range)
        return ranges, bearings
