"""Distance-limited communication network for the swarm (Component B1).

Topology (justification)
------------------------
A unit-disc radio model: two agents communicate iff they are within
``comms_radius_m`` (6 m) of each other. This is the standard first-order
model for low-cost swarm radios (WiFi/BLE class) and - unlike a global
broadcast - forces the swarm to maintain *spatial coherence* to keep
information flowing, which is exactly the emergent behaviour we want to
study. The resulting graph is recomputed every communication tick and may
disconnect; gossip flooding (pairwise belief merging below) then carries
information through multi-hop chains whenever connectivity returns.

Message semantics for B1: pairwise union (OR) of coverage beliefs plus the
scalar state vector each agent broadcasts (position, coverage estimate).
This mirrors the Webots Emitter/Receiver packets used in swarm_arena.wbt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from component_b.swarm.agent import SwarmAgent


@dataclass
class CommunicationNetwork:
    comms_radius_m: float = 6.0

    def adjacency(self, agents: list[SwarmAgent]) -> np.ndarray:
        """Boolean adjacency matrix of the current distance-limited graph."""
        n = len(agents)
        pos = np.array([[a.x, a.y] for a in agents])
        d = np.hypot(pos[:, None, 0] - pos[None, :, 0],
                     pos[:, None, 1] - pos[None, :, 1])
        adj = (d <= self.comms_radius_m) & ~np.eye(n, dtype=bool)
        return adj

    def n_edges(self, adj: np.ndarray) -> int:
        return int(adj.sum() // 2)

    def n_components(self, adj: np.ndarray) -> int:
        """Connected-component count (union-find, tiny n)."""
        n = adj.shape[0]
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] and find(i) != find(j):
                    parent[find(i)] = find(j)
        return len({find(i) for i in range(n)})

    def gossip(self, agents: list[SwarmAgent], adj: np.ndarray) -> int:
        """Merge coverage beliefs across every active edge; return # merges."""
        merges = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if adj[i, j]:
                    bi, bj = agents[i].belief, agents[j].belief
                    if not np.array_equal(bi | bj, bi):
                        bi |= bj
                        merges += 1
                    bj |= bi
        return merges
