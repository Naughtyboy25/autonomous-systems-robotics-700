# Component A2 - Planner comparison (start (1.5, 1.5) -> goal (18.0, 17.0))

| Algorithm | Heuristic | Nodes explored | Runtime (ms) | Path length (m) | Optimal? |
|---|---|---|---|---|---|
| Dijkstra | none | 29906 | 244.7 | 24.85 | yes |
| A* (Euclidean) | euclidean | 11918 | 118.9 | 24.85 | yes |
| A* (Manhattan) | manhattan | 498 | 4.4 | 25.77 | no |

Dijkstra guarantees optimality but explores nearly the whole reachable
map because it has no goal-directed heuristic. A*-Euclidean stays
optimal (admissible) while exploring a fraction of the nodes. A*-Manhattan
is inadmissible on an 8-connected grid, so it trades a small risk of
suboptimality for even fewer expansions.