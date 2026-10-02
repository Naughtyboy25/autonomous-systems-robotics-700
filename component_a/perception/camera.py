"""Synthetic overhead camera + OpenCV feature extraction (Component A1).

Camera model
------------
A downward-facing camera (as mountable on the Webots e-puck-style robot used
in ``arena.wbt``) renders the metric area around the robot into a BGR image.
The palette is deliberately high contrast (white floor, dark obstacles,
saturated landmark colours) so that classical colour segmentation suffices;
a learned detector would be overkill here and is reserved for Component B.

Justification of the classical OpenCV pipeline
----------------------------------------------
1. **Gaussian blur (5x5)** suppresses per-pixel quantisation noise from the
   rasterisation before edge/threshold operators see it.
2. **Colour segmentation (inRange)** is used instead of raw Canny on
   grayscale because obstacle/floor classes are colour-separable by design;
   it is robust to the simulated lighting (none) and cheap to compute.
3. **Morphological open then close** removes isolated speckles (open) and
   seals small gaps in obstacle silhouettes (close), giving clean contours.
4. **Contour moments** yield centroids that are converted back to metric
   bearing/range observations through the known camera geometry - this is
   exactly the information the fusion stage needs to cross-validate LiDAR.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from common.environment import FREE, GridWorld

FLOOR_BGR = (245, 245, 245)
OBSTACLE_BGR = (50, 50, 50)
LANDMARK_BGR = {"red": (30, 30, 220), "green": (30, 170, 30), "blue": (220, 80, 30)}


@dataclass
class CameraSimulator:
    """Overhead camera rendering a robot-centred patch of the world."""

    world: GridWorld
    range_m: float = 5.0          # radius of the visible disc
    px_per_m: int = 60            # 300 px across a 5 m disc diameter
    seed: int = 42

    def __post_init__(self) -> None:
        self.size_px = int(2 * self.range_m * self.px_per_m)
        self.rng = np.random.default_rng(self.seed + 7)

    # ------------------------------------------------------------------
    def render(self, x: float, y: float) -> np.ndarray:
        """Render the visible patch around ``(x, y)``; robot at image centre."""
        img = np.full((self.size_px, self.size_px, 3), FLOOR_BGR, dtype=np.uint8)
        c = self.size_px // 2
        # paint occupied cells as filled squares (metric -> pixel transform)
        r0 = max(0, int((y - self.range_m) / self.world.resolution))
        r1 = min(self.world.rows, int((y + self.range_m) / self.world.resolution) + 1)
        c0 = max(0, int((x - self.range_m) / self.world.resolution))
        c1 = min(self.world.cols, int((x + self.range_m) / self.world.resolution) + 1)
        cell_px = max(1, round(self.world.resolution * self.px_per_m))
        for r in range(r0, r1):
            for cc in range(c0, c1):
                if self.world.grid[r, cc] != FREE:
                    wx, wy = self.world.grid_to_world(r, cc)
                    px, py = self._world_to_px(wx, wy, x, y)
                    cv2.rectangle(
                        img,
                        (px - cell_px // 2, py - cell_px // 2),
                        (px + cell_px // 2, py + cell_px // 2),
                        OBSTACLE_BGR,
                        thickness=-1,
                    )
        # landmarks on top so they are never occluded by obstacle paint
        for lx, ly, color in self.world.landmarks:
            if (lx - x) ** 2 + (ly - y) ** 2 > self.range_m**2:
                continue
            px, py = self._world_to_px(lx, ly, x, y)
            if 0 <= px < self.size_px and 0 <= py < self.size_px:
                cv2.circle(img, (px, py), int(0.2 * self.px_per_m), LANDMARK_BGR[color], -1)
        # weak sensor noise so preprocessing has something real to suppress
        noise = self.rng.normal(0.0, 2.0, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def _world_to_px(self, wx: float, wy: float, rx: float, ry: float) -> tuple[int, int]:
        c = self.size_px // 2
        px = int(round(c + (wx - rx) * self.px_per_m))
        py = int(round(c - (wy - ry) * self.px_per_m))  # image y is flipped
        return px, py

    def px_to_world(self, px: float, py: float, rx: float, ry: float) -> tuple[float, float]:
        c = self.size_px // 2
        return rx + (px - c) / self.px_per_m, ry - (py - c) / self.px_per_m


def _contour_centroids(mask: np.ndarray, min_area_px: int) -> list[tuple[float, float, float]]:
    """Return ``(cx, cy, area)`` for each external contour of ``mask``."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    return out


def detect_features(img: np.ndarray) -> dict[str, list[tuple[float, ...]]]:
    """Classical OpenCV pipeline: blur -> segment -> morphology -> contours.

    Returns ``{"obstacles": [(cx, cy, area)], "red": [...], ...}`` in PIXELS.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    obstacle_mask = cv2.inRange(blurred, (0, 0, 0), (110, 110, 110))
    kernel = np.ones((3, 3), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    landmark_masks = {
        # red wraps the hue circle -> two ranges combined
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)),
            cv2.inRange(hsv, (170, 120, 120), (179, 255, 255)),
        ),
        "green": cv2.inRange(hsv, (40, 80, 80), (90, 255, 255)),
        "blue": cv2.inRange(hsv, (95, 80, 80), (135, 255, 255)),
    }
    out: dict[str, list[tuple[float, ...]]] = {
        "obstacles": _contour_centroids(obstacle_mask, min_area_px=40)
    }
    for color, mask in landmark_masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        out[color] = _contour_centroids(mask, min_area_px=20)
    return out
