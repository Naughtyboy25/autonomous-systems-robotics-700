"""Closed-loop autonomous navigation (Component A4).

Integrates the perception pipeline (A1), the A* planner (A2) and the PID
path tracker (A3) into a single sense-plan-act loop:

    sense -> fuse (log-odds) -> corridor check -> replan if blocked
          -> PID track -> actuate -> repeat

Scenario: the planner is given an OUTDATED a-priori map (the central block
and one pillar are missing). The robot must therefore discover the hidden
obstacles with its own sensors and replan online - demonstrating that the
fused perception actually drives the planning, not the a-priori map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from common.environment import GOAL_A, START_POSE_A, GridWorld, make_assignment_arena
from component_a.control.kinematics import DifferentialDriveRobot
from component_a.control.path_tracker import PIDPathTracker, cross_track_error
from component_a.control.pid import PIDController
from component_a.perception.camera import CameraSimulator, detect_features
from component_a.perception.fusion import OccupancyGridFusion
from component_a.perception.lidar import LidarSimulator
from component_a.planning.astar import astar
from component_a.planning.grid_graph import GridGraph, path_length_m

DT = 0.05                    # 20 Hz control loop
LIDAR_PERIOD_S = 0.5
CAMERA_PERIOD_S = 1.0
REPLAN_PERIOD_S = 3.0
CORRIDOR_RADIUS_CELLS = 3    # ~0.3 m safety corridor around remaining path
SAFETY_INFLATION_M = 0.30    # planner inflation on the FUSED map (adds margin)
FRONT_SLOWDOWN_M = 0.8       # cap speed when an obstacle is this close ahead
TUNED_PID = (5.0, 0.8, 0.4)


def outdated_prior_map(world: GridWorld) -> GridWorld:
    """A-priori map missing the central block and the pillar at (15.5, 9)."""
    prior = make_assignment_arena()
    prior.grid[80:126, 100:116] = 0            # clear the central block
    rr, cc = np.ogrid[: prior.rows, : prior.cols]
    xs = (cc + 0.5) * prior.resolution
    ys = (rr + 0.5) * prior.resolution
    prior.grid[(xs - 15.5) ** 2 + (ys - 9.0) ** 2 <= 1.0] = 0   # clear pillar
    assert prior.grid.shape == world.grid.shape
    return prior


@dataclass
class IntegrationResult:
    trajectory: np.ndarray                 # (N, 3) x, y, theta
    times: np.ndarray
    initial_path: list[tuple[int, int]]
    executed_paths: list[list[tuple[int, int]]] = field(default_factory=list)
    n_replans: int = 0
    mean_cte: float = 0.0
    max_cte: float = 0.0
    travelled_m: float = 0.0
    time_to_goal_s: float = 0.0
    min_clearance_m: float = 0.0
    reached: bool = False


class ClosedLoopNavigator:
    """Sense-plan-act loop over the simulated robot and sensors."""

    def __init__(self, world: GridWorld | None = None, max_time_s: float = 180.0) -> None:
        self.world = world if world is not None else make_assignment_arena()
        self.prior = outdated_prior_map(self.world)
        self.start = START_POSE_A[:2]
        self.goal = GOAL_A
        self.max_time_s = max_time_s
        # cheaper sensors than the A1 demo: 90 beams still over-sample the
        # 0.1 m grid at 8 m range; camera every second
        self.lidar = LidarSimulator(self.world, n_beams=90, noise_std=0.02, seed=7)
        self.camera = CameraSimulator(self.world, seed=7)
        self.fusion = OccupancyGridFusion(self.world)
        self.dt = DT
        # clearance lookup for evaluation (distance transform of true grid)
        self._clearance = ndimage.distance_transform_edt(
            self.world.grid == 0
        ) * self.world.resolution

    # ------------------------------------------------------------------
    def _plan(self, robot_xy: tuple[float, float]) -> list[tuple[int, int]]:
        """A*-Euclidean on the current fused map (falls back to prior map)."""
        prob = self.fusion.probabilities()
        graph = GridGraph.from_probability_grid(
            prob, self.world.resolution,
            occ_threshold=0.65, robot_radius_m=SAFETY_INFLATION_M,
        )
        start = self.world.world_to_grid(*robot_xy)
        goal = self.world.world_to_grid(*self.goal)
        result = astar(graph, start, goal, "euclidean")
        if not result.found:  # extremely conservative fallback: plan on prior
            prior_graph = GridGraph.from_world(self.prior, SAFETY_INFLATION_M)
            result = astar(prior_graph, start, goal, "euclidean")
        return result.path

    def _corridor_blocked(self, path: list[tuple[int, int]], from_index: int) -> bool:
        """True if fused occupancy blocks a ~0.3 m corridor around the path."""
        prob = self.fusion.probabilities()
        occ = prob >= 0.65
        for idx in range(from_index, len(path), 3):
            r, c = path[idx]
            window = occ[
                max(0, r - CORRIDOR_RADIUS_CELLS): r + CORRIDOR_RADIUS_CELLS + 1,
                max(0, c - CORRIDOR_RADIUS_CELLS): c + CORRIDOR_RADIUS_CELLS + 1,
            ]
            if window.any():
                return True
        return False

    def _progress_index(self, path: list[tuple[int, int]], robot_xy: tuple[float, float]) -> int:
        r, c = self.world.world_to_grid(*robot_xy)
        dists = [(pr - r) ** 2 + (pc - c) ** 2 for pr, pc in path]
        return int(np.argmin(dists))

    def _front_clearance(self, ranges: np.ndarray, bearings: np.ndarray, theta: float) -> float:
        rel = (bearings - theta + np.pi) % (2 * np.pi) - np.pi
        front = np.abs(rel) < np.deg2rad(20)
        if not front.any():
            return self.lidar.max_range
        return float(np.min(ranges[front]))

    # ------------------------------------------------------------------
    def run(self) -> IntegrationResult:
        robot = DifferentialDriveRobot(x=self.start[0], y=self.start[1], theta=START_POSE_A[2])
        path = self._plan(self.start)
        initial_path = path
        tracker = PIDPathTracker(
            path=np.array([self.world.grid_to_world(*node) for node in path]),
            heading_pid=PIDController(*TUNED_PID),
        )
        executed: list[list[tuple[int, int]]] = [path]

        traj: list[tuple[float, float, float]] = []
        times: list[float] = []
        ctes: list[float] = []
        clearances: list[float] = []
        t = 0.0
        last_lidar = last_camera = last_replan = -1e9
        last_scan: tuple[np.ndarray, np.ndarray] | None = None
        result = IntegrationResult(trajectory=np.zeros((0, 3)), times=np.zeros(0),
                                   initial_path=initial_path, executed_paths=executed)

        while t < self.max_time_s:
            x, y, theta = robot.pose
            # ---- sense + fuse ----------------------------------------
            if t - last_lidar >= LIDAR_PERIOD_S:
                last_lidar = t
                ranges, bearings = self.lidar.scan(x, y, theta)
                self.fusion.integrate_lidar(x, y, ranges, bearings)
                last_scan = (ranges, bearings)
            if t - last_camera >= CAMERA_PERIOD_S:
                last_camera = t
                frame = self.camera.render(x, y)
                feats = detect_features(frame)
                pts = [self.camera.px_to_world(cx, cy, x, y)
                       for cx, cy, _ in feats["obstacles"]]
                self.fusion.integrate_camera(pts)
            # ---- plan: corridor check / periodic replan ----------------
            i_now = self._progress_index(path, (x, y))
            blocked = t > 2.0 and self._corridor_blocked(path, i_now)
            periodic = t - last_replan >= REPLAN_PERIOD_S
            if blocked or periodic:
                if blocked:
                    result.n_replans += 1
                new_path = self._plan((x, y))
                if new_path and len(new_path) > 1:
                    path = new_path
                    executed.append(path)
                    tracker = PIDPathTracker(
                        path=np.array([self.world.grid_to_world(*node) for node in path]),
                        heading_pid=PIDController(*TUNED_PID),
                    )
                last_replan = t
            # ---- control ------------------------------------------------
            v, omega = tracker.command(robot, self.dt)
            if last_scan is not None:
                clearance = self._front_clearance(last_scan[0], last_scan[1], theta)
                if clearance < FRONT_SLOWDOWN_M:
                    v *= max(0.0, clearance / FRONT_SLOWDOWN_M)
            robot.step(v, omega, self.dt)

            traj.append(robot.pose)
            times.append(t)
            ref = tracker.path
            ctes.append(cross_track_error(ref, robot.x, robot.y))
            r, c = self.world.world_to_grid(robot.x, robot.y)
            clearances.append(float(self._clearance[r, c]))
            t += self.dt

            if tracker.reached_goal(robot.x, robot.y):
                result.reached = True
                break

        result.trajectory = np.array(traj)
        result.times = np.array(times)
        result.mean_cte = float(np.mean(ctes))
        result.max_cte = float(np.max(ctes))
        diffs = np.hypot(np.diff(result.trajectory[:, 0]), np.diff(result.trajectory[:, 1]))
        result.travelled_m = float(np.sum(diffs))
        result.time_to_goal_s = float(times[-1]) if result.reached else float("nan")
        result.min_clearance_m = float(np.min(clearances))
        return result
