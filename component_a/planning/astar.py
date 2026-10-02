"""A* search with switchable heuristics (A2).

Heuristics
----------
* ``euclidean``: straight-line distance. ADMISSIBLE for an 8-connected grid
  whose diagonal edge cost is sqrt(2)*r (a diagonal step covers exactly
  sqrt(2)*r of Euclidean distance, never more), so A* with this heuristic
  is optimal.
* ``manhattan``: L1 distance. NOT admissible once diagonal moves are allowed
  (it overestimates the true cost of diagonal travel), so it acts as a
  greedy-biased heuristic: typically fewer expansions but a potentially
  slightly suboptimal path - a useful demonstration of the
  admissibility/efficiency trade-off required by the assignment.
"""

from __future__ import annotations

import heapq
import time

from component_a.planning.dijkstra import SearchResult
from component_a.planning.grid_graph import GridGraph


def _heuristic(
    node: tuple[int, int], goal: tuple[int, int], resolution: float, kind: str
) -> float:
    dr = abs(node[0] - goal[0]) * resolution
    dc = abs(node[1] - goal[1]) * resolution
    if kind == "euclidean":
        return (dr * dr + dc * dc) ** 0.5
    if kind == "manhattan":
        return dr + dc
    raise ValueError(f"unknown heuristic: {kind}")


def astar(
    graph: GridGraph,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic: str = "euclidean",
) -> SearchResult:
    """A* search; identical interface to :func:`dijkstra.dijkstra`."""
    t0 = time.perf_counter()
    g: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    pq: list[tuple[float, int, tuple[int, int]]] = [
        (_heuristic(start, goal, graph.resolution, heuristic), 0, start)
    ]
    counter = 1
    explored: set[tuple[int, int]] = set()

    while pq:
        _, _, node = heapq.heappop(pq)
        if node in explored:
            continue
        explored.add(node)
        if node == goal:
            path = []
            n: tuple[int, int] | None = goal
            while n is not None:
                path.append(n)
                n = parent[n]
            return SearchResult(
                found=True, path=path[::-1], cost=g[goal],
                nodes_explored=len(explored), explored=explored,
                runtime_s=time.perf_counter() - t0,
            )
        for nbr, w in graph.neighbours(node):
            ng = g[node] + w
            if ng < g.get(nbr, float("inf")):
                g[nbr] = ng
                parent[nbr] = node
                f = ng + _heuristic(nbr, goal, graph.resolution, heuristic)
                heapq.heappush(pq, (f, counter, nbr))
                counter += 1

    return SearchResult(found=False, nodes_explored=len(explored),
                        explored=explored, runtime_s=time.perf_counter() - t0)
