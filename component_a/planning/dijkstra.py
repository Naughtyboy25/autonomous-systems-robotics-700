"""Dijkstra's algorithm on the implicit grid graph (A2).

Optimality: Dijkstra explores nodes in non-decreasing cost order, so the
first time the goal is popped from the priority queue its cost is provably
minimal (all edge costs > 0). It uses NO heuristic, hence it generally
expands far more nodes than A* on metric grids - the comparison against A*
is the point of the assignment benchmark.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from component_a.planning.grid_graph import GridGraph


@dataclass
class SearchResult:
    found: bool
    path: list[tuple[int, int]] = field(default_factory=list)
    cost: float = float("inf")
    nodes_explored: int = 0
    explored: set[tuple[int, int]] = field(default_factory=set)
    runtime_s: float = 0.0


def dijkstra(
    graph: GridGraph, start: tuple[int, int], goal: tuple[int, int]
) -> SearchResult:
    """Classic Dijkstra shortest-path search; returns path and statistics."""
    t0 = time.perf_counter()
    dist: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    pq: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, start)]
    counter = 1
    explored: set[tuple[int, int]] = set()

    while pq:
        d, _, node = heapq.heappop(pq)
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
                found=True, path=path[::-1], cost=d,
                nodes_explored=len(explored), explored=explored,
                runtime_s=time.perf_counter() - t0,
            )
        for nbr, w in graph.neighbours(node):
            nd = d + w
            if nd < dist.get(nbr, float("inf")):
                dist[nbr] = nd
                parent[nbr] = node
                heapq.heappush(pq, (nd, counter, nbr))
                counter += 1

    return SearchResult(found=False, nodes_explored=len(explored),
                        explored=explored, runtime_s=time.perf_counter() - t0)
