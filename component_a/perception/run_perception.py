"""Component A1 driver: run the full perception pipeline and visualise it.

Pipeline: ground-truth arena -> LiDAR scan -> camera render -> OpenCV feature
extraction -> log-odds fusion (LiDAR + camera) -> evaluation against ground
truth. All figures are written to ``plots/`` and metrics to ``results/``.

Run with:  uv run python -m component_a.perception.run_perception
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.environment import START_POSE_A, GridWorld, make_assignment_arena
from component_a.perception.camera import CameraSimulator, detect_features
from component_a.perception.fusion import OccupancyGridFusion
from component_a.perception.lidar import LidarSimulator

PLOTS = Path("plots")
RESULTS = Path("results")


def robot_to_world_detections(
    features: dict[str, list[tuple[float, ...]]],
    cam: CameraSimulator,
    rx: float,
    ry: float,
) -> list[tuple[float, float]]:
    """Project detected obstacle centroids from pixels to world metres."""
    return [cam.px_to_world(cx, cy, rx, ry) for cx, cy, _ in features["obstacles"]]


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    world: GridWorld = make_assignment_arena()
    x, y, theta = START_POSE_A

    # ---- 1. sense ----------------------------------------------------
    lidar = LidarSimulator(world)
    ranges, bearings = lidar.scan(x, y, theta)

    cam = CameraSimulator(world)
    frame = cam.render(x, y)
    features = detect_features(frame)
    obstacle_pts = robot_to_world_detections(features, cam, x, y)

    # ---- 2. fuse -----------------------------------------------------
    fusion = OccupancyGridFusion(world)
    fusion.integrate_lidar(x, y, ranges, bearings)
    fusion.integrate_camera(obstacle_pts)
    prob = fusion.probabilities()

    # ---- 3. evaluate against ground truth ----------------------------
    sensed = prob != 0.5
    gt_occ = world.grid.astype(bool)
    pred_occ = fusion.occupied_mask()
    tp = int(np.sum(pred_occ & gt_occ & sensed))
    fp = int(np.sum(pred_occ & ~gt_occ & sensed))
    fn = int(np.sum(~pred_occ & gt_occ & sensed))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    # ---- 4. figures ---------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Component A1 - Perception Pipeline (LiDAR + Camera fusion)")

    axes[0, 0].imshow(world.grid, cmap="gray_r", origin="lower")
    axes[0, 0].plot(x / world.resolution, y / world.resolution, "b^", ms=10)
    axes[0, 0].set_title("Ground-truth occupancy grid")
    for row in axes.flat:
        row.set_xticks([])
        row.set_yticks([])

    hits = ranges < lidar.max_range - 1e-6
    axes[0, 1].imshow(world.grid, cmap="gray_r", origin="lower", alpha=0.4)
    axes[0, 1].scatter(
        (x + ranges[hits] * np.cos(bearings[hits])) / world.resolution,
        (y + ranges[hits] * np.sin(bearings[hits])) / world.resolution,
        s=1, c="red",
    )
    axes[0, 1].plot(x / world.resolution, y / world.resolution, "b^", ms=10)
    axes[0, 1].set_title(f"LiDAR scan ({lidar.n_beams} beams, {hits.sum()} hits)")

    axes[0, 2].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for cx, cy, *_ in features["obstacles"]:
        axes[0, 2].add_patch(plt.Circle((cx, cy), 10, fill=False, ec="lime", lw=2))
    for color in ("red", "green", "blue"):
        for cx, cy, *_ in features[color]:
            axes[0, 2].add_patch(plt.Circle((cx, cy), 12, fill=False, ec="yellow", lw=2))
    axes[0, 2].set_title(
        f"Camera + OpenCV ({len(features['obstacles'])} obstacle, "
        f"{sum(len(features[c]) for c in ('red','green','blue'))} landmark contours)"
    )

    axes[1, 0].imshow(prob, cmap="RdYlGn_r", origin="lower", vmin=0, vmax=1)
    axes[1, 0].set_title("Fused occupancy probability (LiDAR + camera)")

    lidar_only = OccupancyGridFusion(world)
    lidar_only.integrate_lidar(x, y, ranges, bearings)
    axes[1, 1].imshow(lidar_only.probabilities(), cmap="RdYlGn_r", origin="lower", vmin=0, vmax=1)
    axes[1, 1].set_title("LiDAR only (for comparison)")

    err_img = np.zeros((*world.grid.shape, 3), float)
    err_img[sensed & pred_occ & gt_occ] = (0, 0.7, 0)     # TP green
    err_img[sensed & pred_occ & ~gt_occ] = (1, 0, 0)      # FP red
    err_img[sensed & ~pred_occ & gt_occ] = (1, 0.7, 0)    # FN orange
    axes[1, 2].imshow(err_img, origin="lower")
    axes[1, 2].set_title(f"Errors: P={precision:.2f} R={recall:.2f}")

    fig.tight_layout()
    fig.savefig(PLOTS / "a1_perception_pipeline.png", dpi=150)
    plt.close(fig)
    cv2.imwrite(str(PLOTS / "a1_camera_frame.png"), frame)

    # ---- 5. metrics file ---------------------------------------------
    summary = f"""# Component A1 - Perception pipeline metrics

| Metric | Value |
|---|---|
| LiDAR beams / hits | {lidar.n_beams} / {int(hits.sum())} |
| LiDAR noise std | {lidar.noise_std} m |
| Camera resolution | {cam.size_px}x{cam.size_px} px ({cam.px_per_m} px/m) |
| Obstacle contours (OpenCV) | {len(features['obstacles'])} |
| Landmarks detected | {sum(len(features[c]) for c in ('red','green','blue'))} / {len(world.landmarks)} |
| LiDAR updates fused | {fusion.n_lidar_updates} |
| Camera updates fused | {fusion.n_camera_updates} |
| Occupancy precision | {precision:.3f} |
| Occupancy recall | {recall:.3f} |

Fusion: log-odds occupancy grid; camera evidence (|L|={1.2}) outweighs a
single LiDAR hit (|L|={0.85}) because colour classification is noise-free up
to projection quantisation, while LiDAR carries 2 cm Gaussian range noise.
"""
    (RESULTS / "a1_sensor_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
