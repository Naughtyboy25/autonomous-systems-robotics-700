"""Component A2 driver: benchmark Dijkstra vs A* on the assignment arena.

Compares computational efficiency (runtime, nodes explored) and path
optimality (metric length) for Dijkstra, A*-Euclidean and A*-Manhattan, and
visualises both the paths and the explored-node sets.

Run with:  uv run python -m component_a.planning.run_planning
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.environment import GOAL_A, START_POSE_A, make_assignment_arena
from component_a.planning.astar import astar
from component_a.planning.dijkstra import SearchResult, dijkstra
from component_a.planning.grid_graph import GridGraph, path_length_m

PLOTS = Path("plots")
RESULTS = Path("results")


def _draw(ax, graph: GridGraph, result: SearchResult, title: str) -> None:
    ax.imshow(~graph.free, cmap="gray_r", origin="lower")
    if result.explored:
        ex = np.array(list(result.explored))
        ax.scatter(ex[:, 1], ex[:, 0], s=1, c="deepskyblue", alpha=0.5, label="explored")
    if result.path:
        p = np.array(result.path)
        ax.plot(p[:, 1], p[:, 0], "r-", lw=2, label="path")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    world = make_assignment_arena()
    graph = GridGraph.from_world(world, robot_radius_m=0.25)
    start = world.world_to_grid(START_POSE_A[0], START_POSE_A[1])
    goal = world.world_to_grid(*GOAL_A)

    runs = {
        "Dijkstra": dijkstra(graph, start, goal),
        "A* (Euclidean)": astar(graph, start, goal, "euclidean"),
        "A* (Manhattan)": astar(graph, start, goal, "manhattan"),
    }

    # ---- figure: paths -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Component A2 - Dijkstra vs A* on the assignment arena")
    for ax, (name, res) in zip(axes, runs.items()):
        length = path_length_m(res.path, graph.resolution) if res.found else float("nan")
        _draw(
            ax, graph, res,
            f"{name}\n{res.nodes_explored} nodes | {res.runtime_s*1e3:.1f} ms | "
            f"{length:.2f} m",
        )
    fig.tight_layout()
    fig.savefig(PLOTS / "a2_paths_comparison.png", dpi=150)
    plt.close(fig)

    # ---- table ----------------------------------------------------------
    optimal = runs["Dijkstra"].cost
    lines = [
        "# Component A2 - Planner comparison (start (1.5, 1.5) -> goal (18.0, 17.0))",
        "",
        "| Algorithm | Heuristic | Nodes explored | Runtime (ms) | Path length (m) | Optimal? |",
        "|---|---|---|---|---|---|",
    ]
    for name, res in runs.items():
        length = path_length_m(res.path, graph.resolution)
        heur = {"Dijkstra": "none", "A* (Euclidean)": "euclidean",
                "A* (Manhattan)": "manhattan"}[name]
        is_opt = "yes" if abs(res.cost - optimal) < 1e-9 else "no"
        lines.append(
            f"| {name} | {heur} | {res.nodes_explored} | {res.runtime_s*1e3:.1f} "
            f"| {length:.2f} | {is_opt} |"
        )
    lines += [
        "",
        "Dijkstra guarantees optimality but explores nearly the whole reachable",
        "map because it has no goal-directed heuristic. A*-Euclidean stays",
        "optimal (admissible) while exploring a fraction of the nodes. A*-Manhattan",
        "is inadmissible on an 8-connected grid, so it trades a small risk of",
        "suboptimality for even fewer expansions.",
    ]
    table = "\n".join(lines)
    (RESULTS / "a2_planning_comparison.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
