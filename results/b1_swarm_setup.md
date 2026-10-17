# Component B1 - Multi-agent simulation setup

## Agents and roles
| Agent | Start (x, y) | Role | Notes |
|---|---|---|---|
| 0 | (15, 15) | coordinator | starts centrally; same hardware/code as the rest |
| 1 | (3, 3) | explorer | corner spawn |
| 2 | (27, 3) | explorer | corner spawn |
| 3 | (3, 27) | explorer | corner spawn |
| 4 | (27, 27) | explorer | corner spawn |

Hardware is homogeneous; roles are logical (assigned by id) so any agent can
take over coordination - no single point of execution.

## Communication topology
Unit-disc radio model, radius 6 m, recomputed every 1 s. Agents exchange
their coverage beliefs (bitwise union gossip) and a scalar state vector
(position + coverage estimate). Multi-hop propagation happens naturally when
the graph reconnects. Mean connected components over the run:
4.00; mean active edges:
1.0.

## Cooperative task objective
Distributed area coverage: maximise the fraction of free cells (0.15 m
resolution) sensed by at least one agent, minimising duplicate coverage.
Ground truth after 90 s with the baseline (random waypoint) policy:
**77.3%** - this is the benchmark that the
consensus/PSO layer (B2) must beat.

## Environment and controller design justification
* 30 m x 30 m arena, sparse convex obstacles: the difficulty is *coordination*
  (avoiding duplicated work), not maze navigation.
* 0.15 m resolution keeps per-agent belief exchange (200x200 bit masks)
  within realistic radio budgets while resolving the 0.8 m robot body.
* Sensing = 2.5 m disc footprint (no line-of-sight ray per cell): obstacles
  are thin and convex, and the coordination algorithms studied in B2/B3 are
  insensitive to edge over-crediting; the simplification keeps a full swarm
  simulation well inside CPU budgets on a laptop.
* The Webots mirror (webots/worlds/swarm_arena.wbt) uses the same constants
  (wheel radius, sensing radius, comms radius) so results transfer.
