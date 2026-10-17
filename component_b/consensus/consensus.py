"""Consensus and leader election over the swarm comms graph (Component B2).

Average consensus
-----------------
Each agent i holds a scalar state x_i (here: its local coverage estimate).
At every communication tick it performs one synchronous update over its
CURRENT neighbours (Olfati-Saber & Murray, 2004):

    x_i(t+1) = x_i + sum_{j in N_i(t)} w_ij * (x_j - x_i)

with Metropolis-Hastings weights w_ij = 1 / (1 + max(d_i, d_j)), which makes
W doubly stochastic for ANY graph - the swarm-average is preserved exactly
and, as long as the union of graphs over time is connected frequently, all
states converge to that average. No central node is involved.

Leader election
---------------
Flood-max on a SCALAR key that encodes the pair as
``key = coverage_count * 10 000 + agent_id`` (exact in float64 for these
ranges, ids < 10 000): every agent keeps the maximum of its own key and its
neighbours' keys each tick, so the strongest key in a connected component
propagates along multi-hop chains within (component diameter) ticks.
Ties on coverage are broken by the higher id - deterministic and
reproducible. A persistent ``merged`` memory (own key vs propagated max)
keeps the leader knowledge across graph fragmentation.
"""


from __future__ import annotations

import numpy as np


def encode_election_key(coverage_count: float, aid: int) -> float:
    """(coverage, id) pair packed into one exactly-comparable scalar."""
    return coverage_count * 10_000.0 + aid


def decode_election_key(key: float) -> tuple[float, int]:
    return key // 10_000.0, int(key % 10_000.0)


def metropolis_weights(adj: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings edge weights for a boolean adjacency matrix."""
    deg = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        w = 1.0 / (1.0 + np.maximum(deg[:, None], deg[None, :]))
    return w * adj


def consensus_step(adj: np.ndarray, values: np.ndarray) -> np.ndarray:
    """One synchronous average-consensus update (in-place semantics free)."""
    w = metropolis_weights(adj)
    neighbour_pull = (w * (values[None, :] - values[:, None])).sum(axis=1)
    return values + neighbour_pull


def consensus_error(values: np.ndarray) -> float:
    """Spread of states around the swarm mean (0 when fully agreed)."""
    return float(np.std(values))


def election_step(adj: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """One flood-max hop: each agent keeps max(own key, neighbours' keys)."""
    new_keys = keys.copy()
    for i in range(adj.shape[0]):
        for j in np.where(adj[i])[0]:
            if keys[j] > new_keys[i]:
                new_keys[i] = keys[j]
    return new_keys


def elected_leader(keys: np.ndarray) -> int:
    """Agent id of the component-wide elected coordinator."""
    return int(np.max(keys) % 10_000.0)
