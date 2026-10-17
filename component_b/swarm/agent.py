"""Swarm agent: kinematics, sensing footprint and belief state (Component B1).

Hardware is homogeneous for all five agents (an explicit design decision:
cheap, interchangeable robots); roles are purely logical. The coordinator
(role id 0) is only a *decision-making* role - every agent runs the same
code and the swarm degrades gracefully (no single point of execution) if
the coordinator fails, which is re-examined in the ethics discussion.

Motion model: first-order heading kinematics (v, omega) with a turn-rate
limit, collision-checked against the arena grid - the same unicycle model
as Component A without the wheel-layer details (Component B studies
coordination, not low-level control).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from component_b.swarm.environment import SwarmEnvironment

MAX_TURN_RATE = np.deg2rad(90.0)   # rad/s
CRUISE_SPEED = 0.8                 # m/s
ARRIVE_RADIUS = 0.5                # m


@dataclass
class SwarmAgent:
    aid: int
    x: float
    y: float
    theta: float
    role: str = "explorer"
    comms_range_m: float = 6.0
    max_speed: float = CRUISE_SPEED
    belief: np.ndarray = field(init=False, repr=False)   # personal coverage map
    target: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        # belief allocated lazily on first attach (env owns the shape)
        self.belief = None  # type: ignore[assignment]

    def attach(self, env: SwarmEnvironment) -> None:
        self.belief = np.zeros_like(env.coverage_truth)

    # ------------------------------------------------------------------
    def step_towards(self, env: SwarmEnvironment, dt: float) -> None:
        """Move one dt towards the current waypoint with obstacle probing."""
        if self.target is None:
            return
        desired = np.arctan2(self.target[1] - self.y, self.target[0] - self.x)
        err = (desired - self.theta + np.pi) % (2 * np.pi) - np.pi
        omega = np.clip(2.5 * err, -MAX_TURN_RATE, MAX_TURN_RATE)

        # obstacle probe: slow down / steer away when the front is blocked
        front = env.world.ray_cast(self.x, self.y, self.theta, 1.2)
        if front < 0.8:
            left = env.world.ray_cast(self.x, self.y, self.theta + np.deg2rad(35), 1.2)
            right = env.world.ray_cast(self.x, self.y, self.theta - np.deg2rad(35), 1.2)
            omega += np.deg2rad(60.0) * (1.0 if left > right else -1.0)
            speed = self.max_speed * 0.4
        else:
            speed = self.max_speed * max(0.35, np.cos(min(abs(err), np.pi / 2)))

        self.theta = float((self.theta + omega * dt + np.pi) % (2 * np.pi) - np.pi)
        nx = self.x + speed * np.cos(self.theta) * dt
        ny = self.y + speed * np.sin(self.theta) * dt
        if env.world.is_free_world(nx, ny):
            self.x, self.y = nx, ny
        else:  # blocked: rotate in place next step; re-sample target if stuck
            if env.world.ray_cast(self.x, self.y, self.theta, 0.6) < 0.35:
                self.target = None

    def at_target(self) -> bool:
        if self.target is None:
            return True
        return np.hypot(self.target[0] - self.x, self.target[1] - self.y) < ARRIVE_RADIUS
