"""Component A4 driver: run the integrated navigation system and evaluate it.

Produces:
* ``plots/a4_planned_vs_actual.png`` - planned path(s) vs executed trajectory
  on top of the fused occupancy map
* ``plots/a4_tracking_error.png``     - cross-track error and speed over time
* ``results/a4_integration_metrics.md`` - performance table + critical
  discussion of the perception/planning/control trade-offs (per the brief)

Run with:  uv run python -m component_a.integration.run_integration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.environment import GOAL_A, make_assignment_arena
from component_a.control.path_tracker import cross_track_error
from component_a.integration.closed_loop import ClosedLoopNavigator, IntegrationResult
from component_a.planning.grid_graph import path_length_m

PLOTS = Path("plots")
RESULTS = Path("results")
TUNED_PID_GAINS_TEXT = "Kp=5, Ki=0.8, Kd=0.4"


def plot_result(
    result: IntegrationResult, resolution: float, prob: np.ndarray
) -> None:
    """Planned-vs-actual trajectory on the map plus fused-occupancy overlay."""
    world = make_assignment_arena()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    fig.suptitle("Component A4 - Closed-loop navigation: planned vs actual")

    ax.imshow(world.grid, cmap="gray_r", origin="lower")
    ax.imshow(np.ma.masked_less(prob, 0.65), cmap="autumn_r", origin="lower", alpha=0.55)
    pts = np.array(result.initial_path)
    ax.plot(pts[:, 1], pts[:, 0], "--", lw=1.6, color="royalblue", label="initial plan")
    for i, p in enumerate(result.executed_paths[1:], start=1):
        pts = np.array(p)
        ax.plot(pts[:, 1], pts[:, 0], "--", lw=1.4, color="teal", label=f"replan #{i}")
    tr = result.trajectory
    ax.plot(tr[:, 0] / resolution, tr[:, 1] / resolution, "r-", lw=2, label="actual")
    ax.plot(tr[0, 0] / resolution, tr[0, 1] / resolution, "g^", ms=10, label="start")
    ax.plot(GOAL_A[0] / resolution, GOAL_A[1] / resolution, "g*", ms=16, label="goal")
    ax.set_title(f"goal reached: {result.reached} | online replans: {result.n_replans}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(PLOTS / "a4_planned_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_tracking(result: IntegrationResult, resolution: float, final_path: np.ndarray) -> None:
    """Cross-track error (vs the final plan) and speed over time."""
    cte = [cross_track_error(final_path, x, y) for x, y in result.trajectory[:, :2]]
    speed = np.hypot(np.gradient(result.trajectory[:, 0]),
                     np.gradient(result.trajectory[:, 1])) / 0.05
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(result.times, cte, "tab:blue", lw=1.5, label="cross-track error (m)")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("cross-track error (m)")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(result.times, speed, "tab:orange", alpha=0.6, label="speed (m/s)")
    ax2.set_ylabel("speed (m/s)")
    ax1.legend(loc="upper right", fontsize=8)
    fig.suptitle("Component A4 - tracking error and speed over time")
    fig.tight_layout()
    fig.savefig(PLOTS / "a4_tracking_error.png", dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    nav = ClosedLoopNavigator()
    result = nav.run()
    prob = nav.fusion.probabilities()
    final_path = np.array([nav.world.grid_to_world(*n) for n in result.executed_paths[-1]])

    planned_len = path_length_m(result.initial_path, nav.world.resolution)
    plot_result(result, nav.world.resolution, prob)
    plot_tracking(result, nav.world.resolution, final_path)

    reached = "yes" if result.reached else "NO"
    table = f"""# Component A4 - Integrated navigation performance

| Metric | Value |
|---|---|
| Goal reached | {reached} |
| Initial planned path length | {planned_len:.2f} m |
| Actually travelled distance | {result.travelled_m:.2f} m |
| Time to goal | {result.time_to_goal_s:.1f} s |
| Mean cross-track error | {result.mean_cte:.3f} m |
| Max cross-track error | {result.max_cte:.3f} m |
| Online replans (corridor blocked) | {result.n_replans} |
| Min clearance to obstacles | {result.min_clearance_m:.2f} m |

## Critical analysis: perception / planning / control trade-offs

The run starts from a deliberately outdated a-priori map: the initial plan
crosses a region where a real obstacle exists. Only the fused LiDAR+camera
occupancy (autumn overlay in the figure) reveals the blockage, which
triggers online A* replanning around it - perception accuracy is therefore
not a passive quality metric here, it is what makes the plan executable at
all. The cost of that accuracy is sensing latency: fusion lags the truth by
up to one scan period (0.5 s LiDAR / 1.0 s camera), so the planner inflates
obstacles by an extra safety margin (0.30 m) and the controller caps speed
when front clearance drops below 0.8 m. Planning optimality suffers from the
same conservatism: inflation and the optimistic treatment of unknown space
(planned as free until proven occupied) make replans more frequent than a
full-knowledge planner would need, but guarantee the robot never commits to
a corridor it cannot physically fit through. Control responsiveness closes
the loop: the tuned PID ({TUNED_PID_GAINS_TEXT}) tracks the (re)planned path
with a mean CTE of {result.mean_cte:.3f} m, but aggressive tracking of a
freshly replanned path is exactly where an over-tuned controller would
oscillate; moderate gains keep the transient benign at the cost of slightly
slower convergence. In shared human environments every one of these margins
is a direct trade between mission time and safety: slower, wider-margin
behaviour is ethically defensible even when it is sub-optimal on paper.
"""
    (RESULTS / "a4_integration_metrics.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
