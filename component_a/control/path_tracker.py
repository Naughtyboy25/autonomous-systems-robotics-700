"""PID path tracker for the differential-drive robot (Component A3/A4).

Structure: a lookahead point is projected onto the reference polyline at
distance ``lookahead`` ahead of the robot's closest path point (pure-pursuit
style geometric reference generation), then a PID on the *heading error* to
that point produces ``omega`` while a heuristic cruise-speed law produces
``v`` (slow in sharp turns, full speed on straights). Heading-only PID is
sufficient and standard for unicycle path tracking at constant cruise speed
(Samson & Ait-Abderrahim style feedback reduces to this at fixed speed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from component_a.control.kinematics import DifferentialDriveRobot, wrap_angle
from component_a.control.pid import PIDController


@dataclass
class PIDPathTracker:
    path: np.ndarray                 # (N, 2) polyline waypoints, metres
    heading_pid: PIDController
    lookahead_m: float = 0.6
    cruise_speed: float = 0.5        # m/s
    goal_tolerance: float = 0.15     # m

    def _project(self, x: float, y: float) -> tuple[int, float]:
        """Index of closest segment and distance to the polyline."""
        d = np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)
        return int(np.argmin(d)), float(np.min(d))

    def _lookahead_point(self, x: float, y: float) -> np.ndarray:
        i, _ = self._project(x, y)
        acc = 0.0
        for j in range(i, len(self.path) - 1):
            seg = np.hypot(*(self.path[j + 1] - self.path[j]))
            if acc + seg >= self.lookahead_m:
                t = (self.lookahead_m - acc) / seg
                return self.path[j] + t * (self.path[j + 1] - self.path[j])
            acc += seg
        return self.path[-1]

    def reached_goal(self, x: float, y: float) -> bool:
        return float(np.hypot(*(self.path[-1] - np.array([x, y])))) < self.goal_tolerance

    def command(self, robot: DifferentialDriveRobot, dt: float) -> tuple[float, float]:
        """Return ``(v_cmd, omega_cmd)`` for the current robot state."""
        target = self._lookahead_point(robot.x, robot.y)
        desired_heading = float(np.arctan2(target[1] - robot.y, target[0] - robot.x))
        omega_cmd = self.heading_pid.update(desired_heading, robot.theta, dt)
        heading_err = abs(float(wrap_angle(desired_heading - robot.theta)))
        # slow down in turns: full speed when aligned, 40% when facing away hard
        v_cmd = self.cruise_speed * max(0.4, np.cos(min(heading_err, np.pi / 2)))
        return float(v_cmd), float(omega_cmd)


def cross_track_error(path: np.ndarray, x: float, y: float) -> float:
    """Signed distance from (x, y) to the closest polyline segment."""
    best = np.inf
    for a, b in zip(path, path[1:]):
        ab = b - a
        t = np.clip(np.dot([x, y] - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
        best = min(best, float(np.hypot(*(np.array([x, y]) - (a + t * ab)))))
    return best
