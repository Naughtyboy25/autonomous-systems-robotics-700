"""Webots controller: closed-loop navigation in arena.wbt (Component A4).

Mirrors component_a/integration/closed_loop.py one-to-one, but every "sensor"
is now a real Webots device instead of a ray-cast simulator:

* LiDAR       -> Webots Lidar node (90 beams, 2*pi FOV, 8 m, 2 cm noise)
* Camera      -> Webots Camera node (160x120, 0.9 rad FOV, pitched down);
                 OpenCV blob detection classifies dark obstacles, the LiDAR
                 provides the range for the same bearing - cross-modal fusion
* Pose        -> GPS + Compass (ground-truth stand-in for a localization
                 stack such as AMCL, which is outside assignment scope)

Pipeline each cycle: sense -> fuse (log-odds grid, starting fully unknown) ->
corridor check -> A*-Euclid replan when blocked (unknown space is optimistically
free, exactly like the offline sim) -> PID heading tracker -> wheel speeds.

NOTE (conventions used, documented for reproducibility):
* ENU world: x East, z North; the 2-D maps use (x, z) as (x, y).
* heading = atan2(compass[0], compass[2]) gives 0 rad = East, pi/2 = North.
* Lidar beam i covers absolute bearing  heading - fov/2 + i*fov/N
  (Webots returns the first beam at -fov/2 of the sensor frame).
* Camera pixel u maps to bearing offset (u - w/2) * fov / w (radians).

Requires in the Webots Python: numpy, opencv-python, scipy (the same
versions pinned in the repo's pyproject.toml).
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from controller import Robot

from common.environment import GridWorld
from component_a.control.kinematics import DifferentialDriveRobot
from component_a.control.path_tracker import PIDPathTracker
from component_a.control.pid import PIDController
from component_a.perception.camera import detect_features
from component_a.perception.fusion import OccupancyGridFusion
from component_a.planning.astar import astar
from component_a.planning.grid_graph import GridGraph

LIDAR_PERIOD_S = 0.5
CAMERA_PERIOD_S = 1.0
REPLAN_PERIOD_S = 3.0
CORRIDOR_RADIUS_CELLS = 3
SAFETY_INFLATION_M = 0.30
FRONT_SLOWDOWN_M = 0.8
GOAL = (18.0, 17.0)
TUNED_PID = (5.0, 0.8, 0.4)
WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16
CRUISE_SPEED = 0.5
MAX_TIME_S = 240.0


class WebotsNavigator:
    def __init__(self) -> None:
        self.robot = Robot()
        self.dt = self.robot.getBasicTimeStep() / 1000.0
        self.timestep = int(self.robot.getBasicTimeStep())

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        self.left = self.robot.getDevice("left_wheel")
        self.right = self.robot.getDevice("right_wheel")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # coordinate frame only: empty 20 m x 20 m grid, fusion starts unknown
        self.frame = GridWorld(width_m=20.0, height_m=20.0, resolution=0.1)
        self.frame.grid[:] = 0
        self.fusion = OccupancyGridFusion(self.frame)
        self.n_beams = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.lidar_max = self.lidar.getMaxRange()
        self.cam_w, self.cam_h = self.camera.getWidth(), self.camera.getHeight()
        self.cam_fov = self.camera.getFov()

        self.state = DifferentialDriveRobot()
        self.last_scan: tuple[np.ndarray, np.ndarray] | None = None
        self.path: list[tuple[int, int]] = []
        self.tracker: PIDPathTracker | None = None
        self.last_lidar = self._last_camera = -1e9
        self.last_replan = -1e9
        self.n_replans = 0
        self.trajectory: list[tuple[float, float, float]] = []
        self.t = 0.0

    # ------------------------------------------------------------------
    def read_pose(self) -> None:
        g = self.gps.getValues()
        c = self.compass.getValues()
        self.state.x, self.state.y = g[0], g[2]          # ENU: (x, z)
        self.state.theta = float(np.arctan2(c[0], c[2]))  # 0 = East, pi/2 = North

    def read_lidar(self) -> None:
        ranges = np.clip(np.array(self.lidar.getRangeImage(), dtype=float), 0.0, self.lidar_max)
        offsets = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2,
                              self.n_beams, endpoint=False)
        bearings = self.state.theta + offsets
        self.fusion.integrate_lidar(self.state.x, self.state.y, ranges, bearings)
        self.last_scan = (ranges, bearings)

    def read_camera(self) -> None:
        if self.last_scan is None:
            return
        img = np.frombuffer(self.camera.getImage(), np.uint8)
        img = img.reshape((self.cam_h, self.cam_w, 4))[:, :, :3].copy()  # BGRA -> BGR
        feats = detect_features(img)
        ranges, bearings = self.last_scan
        pts: list[tuple[float, float]] = []
        for u, _v, _a in feats["obstacles"]:
            bearing = self.state.theta + (u - self.cam_w / 2) * self.cam_fov / self.cam_w
            # fuse camera CLASSIFICATION with the LiDAR RANGE at that bearing
            i = int(np.argmin(np.abs(((bearings - bearing + np.pi) % (2 * np.pi)) - np.pi)))
            r = ranges[i]
            if r < self.lidar_max - 1e-6:
                pts.append((self.state.x + r * np.cos(bearing),
                            self.state.y + r * np.sin(bearing)))
        if pts:
            self.fusion.integrate_camera(pts)

    # ------------------------------------------------------------------
    def plan(self) -> None:
        prob = self.fusion.probabilities()
        graph = GridGraph.from_probability_grid(
            prob, self.frame.resolution,
            occ_threshold=0.65, robot_radius_m=SAFETY_INFLATION_M,
        )
        start = self.frame.world_to_grid(self.state.x, self.state.y)
        goal = self.frame.world_to_grid(*GOAL)
        result = astar(graph, start, goal, "euclidean")
        if result.found and len(result.path) > 1:
            self.path = result.path
            world_pts = np.array([self.frame.grid_to_world(*n) for n in self.path])
            self.tracker = PIDPathTracker(
                path=world_pts, heading_pid=PIDController(*TUNED_PID),
                cruise_speed=CRUISE_SPEED,
            )

    def corridor_blocked(self, from_index: int) -> bool:
        if not self.path:
            return True
        prob = self.fusion.probabilities()
        occ = prob >= 0.65
        for idx in range(from_index, len(self.path), 3):
            r, c = self.path[idx]
            window = occ[
                max(0, r - CORRIDOR_RADIUS_CELLS): r + CORRIDOR_RADIUS_CELLS + 1,
                max(0, c - CORRIDOR_RADIUS_CELLS): c + CORRIDOR_RADIUS_CELLS + 1,
            ]
            if window.any():
                return True
        return False

    def progress_index(self) -> int:
        r, c = self.frame.world_to_grid(self.state.x, self.state.y)
        return int(np.argmin([(pr - r) ** 2 + (pc - c) ** 2 for pr, pc in self.path]))

    def front_clearance(self) -> float:
        if self.last_scan is None:
            return self.lidar_max
        ranges, bearings = self.last_scan
        rel = (bearings - self.state.theta + np.pi) % (2 * np.pi) - np.pi
        front = np.abs(rel) < np.deg2rad(20)
        if not front.any():
            return self.lidar_max
        return float(np.min(ranges[front]))

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.read_pose()
        self.plan()
        print("[nav_controller] starting navigation to", GOAL)
        while self.robot.step(self.timestep) != -1 and self.t < MAX_TIME_S:
            self.read_pose()
            self.trajectory.append((self.state.x, self.state.y, self.state.theta))

            if self.t - self.last_lidar >= LIDAR_PERIOD_S:
                self.last_lidar = self.t
                self.read_lidar()
            if self.t - self._last_camera >= CAMERA_PERIOD_S:
                self._last_camera = self.t
                self.read_camera()
            if self.tracker is None or not self.path:
                self.plan()
            else:
                blocked = self.t > 2.0 and self.corridor_blocked(self.progress_index())
                periodic = self.t - self.last_replan >= REPLAN_PERIOD_S
                if blocked or periodic:
                    if blocked:
                        self.n_replans += 1
                        print(f"[nav_controller] corridor blocked at t={self.t:.1f}s -> replan")
                    self.plan()
                    self.last_replan = self.t

            if self.tracker is not None:
                v, omega = self.tracker.command(self.state, self.dt)
                clearance = self.front_clearance()
                if clearance < FRONT_SLOWDOWN_M:
                    v *= max(0.0, clearance / FRONT_SLOWDOWN_M)
                wl = (v - omega * TRACK_WIDTH / 2) / WHEEL_RADIUS
                wr = (v + omega * TRACK_WIDTH / 2) / WHEEL_RADIUS
                self.left.setVelocity(float(np.clip(wl, -25, 25)))
                self.right.setVelocity(float(np.clip(wr, -25, 25)))

                if self.tracker.reached_goal(self.state.x, self.state.y):
                    self.left.setVelocity(0.0)
                    self.right.setVelocity(0.0)
                    print(f"[nav_controller] GOAL REACHED at t={self.t:.1f}s "
                          f"(replans={self.n_replans})")
                    self.save_trajectory()
                    return
            self.t += self.dt
        self.left.setVelocity(0.0)
        self.right.setVelocity(0.0)
        print("[nav_controller] timeout - stopping")
        self.save_trajectory()

    def save_trajectory(self) -> None:
        try:
            out = os.path.join(os.path.dirname(__file__), "nav_trajectory.csv")
            with open(out, "w", encoding="utf-8") as f:
                f.write("t_step,x_m,z_m,heading_rad\n")
                for i, (x, y, th) in enumerate(self.trajectory):
                    f.write(f"{i},{x:.4f},{y:.4f},{th:.4f}\n")
            print(f"[nav_controller] trajectory saved to {out}")
        except OSError as exc:
            print("[nav_controller] could not save trajectory:", exc)


if __name__ == "__main__":
    WebotsNavigator().run()
