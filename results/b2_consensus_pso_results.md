# Component B2 - Consensus + PSO results (90 s horizon, 3 seeds)

| Metric | Baseline (B1) | PSO + consensus (B2) |
|---|---|---|
| Coverage at 90 s | 72.3 +/- 4.9% | 78.9 +/- 5.5% |
| Time to 50% coverage (all 3 seeds reach) | 22.1 s | 20.1 s |
| Seeds reaching 75% in 90 s | 1/3 (mean 76.1 s) | 2/3 (mean 69.2 s) |
| Mean comms edges (seed 42) | - | 1.6 |
| Final consensus error (seed 42) | - | 0.0371 |

## Emergent behaviour
PSO waypoint allocation produces emergent spatial partitioning: each agent's
fitness penalises candidate waypoints within 3.0 m of peers'
known targets, so agents spread into complementary regions WITHOUT any
central assignment (targets are known only via gossip, which itself depends
on the distance-limited radio). The leader-election trace shows the
coordinator role held by agent 2 (highest
gossiped coverage count) at the end of the run; earlier per-component
elections resolve once the graph reconnects.

## Convergence
* Average consensus reaches agreement (state std -> 0) on both static and
  switching graphs (b2_consensus_convergence.png); during the run the
  error spikes when the graph fragments and decays whenever connectivity
  returns - the classic behaviour of consensus over intermittently
  connected swarms.
* PSO converges in 10-15 iterations per waypoint decision (see
  b2_pso_emergent.png right panel); early calls optimise for distance
  spreading (empty belief), later calls for residual uncovered pockets.
