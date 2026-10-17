"""Webots swarm controller (Component B1 baseline behaviour).

One identical controller for all five agents (role comes from controllerArgs,
which carries the agent id). Baseline behaviour mirrors the Python sim
``component_b/swarm/run_swarm_setup.py``:

1. waypoint following on a blueprint of the arena (agents know the static
   layout from the "site blueprint"; the open problem is who covers where,
   not mapping - see the B1 justification document),
2. LiDAR obstacle probing with the same steer-away law as
   :class:`component_b.swarm.agent.SwarmAgent`,
3. personal coverage belief (2.5 m disc footprint, 0.15 m grid) stamped at
   the GPS pose,
4. radio gossip: every 5 s the packed belief (200x200 bits -> 5 kB) is
   broadcast on channel 1; received beliefs are union-merged, reproducing
   the CommunicationNetwork.gossip() semantics over the 6 m unit-disc radio.

Conventions match the nav_controller: ENU world, (x, z) = (x, y) of the 2-D
models, heading = atan2(compass[0], compass[2]).
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from controller import Robot

from component_b.swarm.environment import SwarmEnvironment

DT_STEP = 0.04            # 25 Hz control
SENSE_PERIOD_S = 0.5
GOSSIP_PERIOD_S = 5.0
CRUISE_SPEED = 0.8
ARRIVE_RADIUS = 0.5
MAX_TURN_RATE = np.deg2rad(90.0)
WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.16


class SwarmController:
    def __init__(self) -> None:
        self.robot = Robot()
        self.dt = DT_STEP
        self.ts = int(self.robot.getBasicTimeStep())
        args = self.robot.getControllerArgs()
        self.aid = int(args[0]) if args else int(self.robot.getName().split("_")[-1])
        self.rng = np.random.default_rng(1000 + self.aid)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.ts)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.ts)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.ts)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.ts)
        self.left = self.robot.getDevice("left_wheel")
        self.right = self.robot.getDevice("right_wheel")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # arena blueprint + belief grid (identical constants to the Python sim)
        self.env = SwarmEnvironment()
        self.belief = np.zeros_like(self.env.coverage_truth)
        self.n_beams = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()
        self.x = self.y = self.theta = 0.0
        self.target: tuple[float, float] | None = None
        self.last_sense = self.last_gossip = -1e9
        self.t = 0.0

    # ------------------------------------------------------------------
    def read_pose(self) -> None:
        g = self.gps.getValues()
        c = self.compass.getValues()
        self.x, self.y = g[0], g[2]
        self.theta = float(np.arctan2(c[0], c[2]))

    def sample_waypoint(self) -> tuple[float, float]:
        return self.env.sample_free_waypoint(self.rng)

    def front_clearance(self) -> tuple[float, float]:
        """Min LiDAR range in +-20 deg of heading, and its sign bias."""
        ranges = np.clip(np.array(self.lidar.getRangeImage(), dtype=float),
                         0.0, self.lidar.getMaxRange())
        offsets = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2,
                              self.n_beams, endpoint=False)
        rel = (offsets + np.pi) % (2 * np.pi) - np.pi
        front_mask = np.abs(rel) < np.deg2rad(20)
        if not front_mask.any():
            return self.lidar.getMaxRange(), 0.0
        idx = np.where(front_mask)[0]
        return float(np.min(ranges[idx])), float(np.mean(np.sign(rel[idx])))

    # ------------------------------------------------------------------
    def drive(self) -> None:
        if self.target is None:
            self.target = self.sample_waypoint()
        desired = np.arctan2(self.target[1] - self.y, self.target[0] - self.x)
        err = (desired - self.theta + np.pi) % (2 * np.pi) - np.pi
        omega = float(np.clip(2.5 * err, -MAX_TURN_RATE, MAX_TURN_RATE))
        clearance, side_bias = self.front_clearance()
        if clearance < 0.8:
            omega += np.deg2rad(60.0) * (1.0 if side_bias > 0 else -1.0)
            speed = CRUISE_SPEED * 0.4
        else:
            speed = CRUISE_SPEED * max(0.35, np.cos(min(abs(err), np.pi / 2)))
        wl = (speed - omega * TRACK_WIDTH / 2) / WHEEL_RADIUS
        wr = (speed + omega * TRACK_WIDTH / 2) / WHEEL_RADIUS
        self.left.setVelocity(float(np.clip(wl, -25, 25)))
        self.right.setVelocity(float(np.clip(wr, -25, 25)))
        if np.hypot(self.target[0] - self.x, self.target[1] - self.y) < ARRIVE_RADIUS:
            self.target = None

    def sense(self) -> None:
        self.env.stamp_coverage(self.belief, self.x, self.y)

    def gossip(self) -> None:
        # broadcast packed belief: id header (uint16) + 200*200 bits
        header = np.array([self.aid], dtype=np.uint16).tobytes()
        self.emitter.send(header + np.packbits(self.belief).tobytes())
        while self.receiver.getQueueLength() > 0:
            try:
                data = self.receiver.getData()
                other = np.unpackbits(np.frombuffer(data[2:], dtype=np.uint8),
                                       count=self.belief.size).reshape(self.belief.shape)
                self.belief |= other.astype(bool)
            finally:
                self.receiver.nextPacket()
        cov = float((self.belief & (self.env.world.grid == 0)).sum()) / self.env.free_cells
        print(f"[swarm_{self.aid}] t={self.t:.0f}s belief coverage = {cov * 100:.1f}%")

    # ------------------------------------------------------------------
    def run(self) -> None:
        print(f"[swarm_{self.aid}] controller up (baseline wander + gossip)")
        while self.robot.step(self.ts) != -1:
            self.read_pose()
            if self.t - self.last_sense >= SENSE_PERIOD_S:
                self.last_sense = self.t
                self.sense()
            if self.t - self.last_gossip >= GOSSIP_PERIOD_S:
                self.last_gossip = self.t
                self.gossip()
            self.drive()
            self.t += self.dt


if __name__ == "__main__":
    SwarmController().run()
