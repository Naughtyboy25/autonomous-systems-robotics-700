"""Kinematic model of the differential-drive robot (Component A3).

Unicycle model (kinematics)
---------------------------
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

The robot is commanded in body twist ``(v, omega)``; wheel speeds follow the
standard differential-drive relations with wheel radius ``R`` and track
(width) ``L``::

    omega_R = (v + omega * L / 2) / R
    omega_L = (v - omega * L / 2) / R

Actuator dynamics
-----------------
Real motors cannot change speed instantaneously, so each command passes
through a first-order lag ``tau`` (identified order of magnitude for small
ground robots like the Webots e-puck):  ``v_dot = (v_cmd - v) / tau``. This
lag is what makes PID tuning non-trivial and gives realistic overshoot /
oscillation when gains are pushed too high.

Euler integration at ``dt <= 0.02`` s keeps the local truncation error well
below sensor noise levels used elsewhere in the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class DifferentialDriveRobot:
    """Unicycle kinematics with first-order actuator lag."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    wheel_radius: float = 0.033   # m  (e-puck class)
    track_width: float = 0.16     # m
    tau_v: float = 0.15           # s linear actuator lag
    tau_omega: float = 0.10       # s angular actuator lag

    def step(self, v_cmd: float, omega_cmd: float, dt: float) -> None:
        """Advance the state by ``dt`` seconds under command ``(v_cmd, omega_cmd)``."""
        self.v += (v_cmd - self.v) / self.tau_v * dt
        self.omega += (omega_cmd - self.omega) / self.tau_omega * dt
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = float(wrap_angle(self.theta + self.omega * dt))

    def wheel_speeds(self) -> tuple[float, float]:
        """Current (left, right) wheel angular speeds in rad/s."""
        wl = (self.v - self.omega * self.track_width / 2.0) / self.wheel_radius
        wr = (self.v + self.omega * self.track_width / 2.0) / self.wheel_radius
        return wl, wr

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta
