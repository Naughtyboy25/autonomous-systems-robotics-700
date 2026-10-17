"""Component B1 driver: multi-agent simulation setup + baseline behaviour.

Defines and demonstrates the multi-agent system before coordination
algorithms arrive in B2: 5 homogeneous agents (1 coordinator + 4
explorers) roam the 30 m x 30 m arena towards random waypoints while
building personal coverage beliefs and gossiping them over the
distance-limited network. Outputs:

* ``plots/b1_swarm_setup.png``      - arena, agents, comms graph, coverage
* ``results/b1_swarm_setup.md``     - roles, topology, objective, justification

Run with:  uv run python -m component_b.swarm.run_swarm_setup
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from component_b.swarm.agent import SwarmAgent
from component_b.swarm.communication import CommunicationNetwork
from component_b.swarm.environment import SPAWN_POSES, SwarmEnvironment

PLOTS = Path("plots")
RESULTS = Path("results")
DT = 0.05
SENSE_PERIOD_S = 0.5
COMMS_PERIOD_S = 1.0
T_END = 90.0
N_AGENTS = 5
SEED = 42


def build_swarm(env: SwarmEnvironment) -> list[SwarmAgent]:
    agents = []
    for i, (x, y, th) in enumerate(SPAWN_POSES):
        role = "coordinator" if i == 0 else "explorer"
        a = SwarmAgent(aid=i, x=x, y=y, theta=th, role=role)
        a.attach(env)
        agents.append(a)
    return agents


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    env = SwarmEnvironment()
    agents = build_swarm(env)
    net = CommunicationNetwork()

    cov_hist, edges_hist, comps_hist = [], [], []
    beliefs_hist = []
    t = 0.0
    last_sense = last_comms = -1e9

    while t < T_END:
        for a in agents:
            if a.at_target():
                a.target = env.sample_free_waypoint(rng)
            a.step_towards(env, DT)
        if t - last_sense >= SENSE_PERIOD_S:
            last_sense = t
            for a in agents:
                env.stamp_coverage(a.belief, a.x, a.y)
                env.stamp_coverage(env.coverage_truth, a.x, a.y)
            cov_hist.append((t, env.coverage_fraction(),
                             float(np.mean([env.coverage_fraction(a.belief) for a in agents]))))
        if t - last_comms >= COMMS_PERIOD_S:
            last_comms = t
            adj = net.adjacency(agents)
            net.gossip(agents, adj)
            edges_hist.append((t, net.n_edges(adj)))
            comps_hist.append((t, net.n_components(adj)))
            beliefs_hist.append(adj.copy())
        t += DT

    # ---- figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    fig.suptitle("Component B1 - multi-agent setup (5 agents, distance-limited comms)")

    ax = axes[0]
    ax.imshow(env.world.grid, cmap="gray_r", origin="lower")
    final_adj = beliefs_hist[-1]
    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            if final_adj[i, j]:
                ax.plot([agents[i].x / 0.15, agents[j].x / 0.15],
                        [agents[i].y / 0.15, agents[j].y / 0.15], "c-", alpha=0.7, lw=1.5)
    for a in agents:
        style = "r^" if a.role == "coordinator" else "bo"
        ax.plot(a.x / 0.15, a.y / 0.15, style, ms=9, label=a.role if a.aid == 0 or a.aid == 1 else None)
    ax.set_title(f"Final configuration (t={T_END:.0f} s, edges={edges_hist[-1][1]})")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ch = np.array(cov_hist)
    ax.plot(ch[:, 0], ch[:, 1] * 100, "k-", label="true coverage")
    ax.plot(ch[:, 0], ch[:, 2] * 100, "g--", label="mean belief (post-gossip)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("coverage (%)")
    ax.set_title("Area coverage vs time (baseline random waypoints)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    eh = np.array(edges_hist)
    ph = np.array(comps_hist)
    ax.plot(eh[:, 0], eh[:, 1], "b-", label="comms edges")
    ax.plot(ph[:, 0], ph[:, 1], "r-", label="connected components")
    ax.set_xlabel("t (s)")
    ax.set_title("Network connectivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS / "b1_swarm_setup.png", dpi=150)
    plt.close(fig)

    # ---- setup document -------------------------------------------------
    doc = f"""# Component B1 - Multi-agent simulation setup

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
{np.mean([c for _, c in comps_hist]):.2f}; mean active edges:
{np.mean([e for _, e in edges_hist]):.1f}.

## Cooperative task objective
Distributed area coverage: maximise the fraction of free cells (0.15 m
resolution) sensed by at least one agent, minimising duplicate coverage.
Ground truth after {T_END:.0f} s with the baseline (random waypoint) policy:
**{env.coverage_fraction()*100:.1f}%** - this is the benchmark that the
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
"""
    (RESULTS / "b1_swarm_setup.md").write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
