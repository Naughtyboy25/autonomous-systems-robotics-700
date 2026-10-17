"""Particle Swarm Optimisation for distributed waypoint allocation (B2).

Each agent runs a small PSO *locally* whenever it needs a new waypoint.
Particles are candidate (x, y) waypoints; the fitness measures how much
UNCOVERED area a candidate would add to the agent's belief map, minus a
repulsion penalty near the targets the agent knows its peers are heading
to (received via gossip). The personal best is the agent's own memory; the
global best is shared across neighbours through the same radio - so the
swarm intelligence layer inherits the decentralised topology of B1.

Why PSO (not ACO) for coverage: the objective is a continuous 2-D placement
problem, exactly the domain PSO's velocity-position updates handle well;
ACO's discrete pheromone graph adds state we do not need at this scale.

Standard gbest PSO with inertia weight (Shi & Eberhart, 1998):
    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)
    x <- x + v           (clamped to bounds and |v| <= vmax)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

FitnessFn = Callable[[np.ndarray], np.ndarray]  # (n, 2) -> (n,)


@dataclass
class PSO:
    fitness: FitnessFn
    bounds: tuple[float, float]                 # square arena [lo, hi]
    n_particles: int = 12
    iters: int = 15
    w: float = 0.7
    c1: float = 1.4
    c2: float = 1.4
    vmax: float = 6.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        """Run PSO; return (best_position, best_fitness, per-iter best history)."""
        lo, hi = self.bounds
        x = self.rng.uniform(lo, hi, size=(self.n_particles, 2))
        v = np.zeros_like(x)
        fit = self.fitness(x)
        pbest, pbest_fit = x.copy(), fit.copy()
        g_idx = int(np.argmax(pbest_fit))
        gbest, gbest_fit = pbest[g_idx].copy(), float(pbest_fit[g_idx])
        history: list[float] = []

        for _ in range(self.iters):
            r1 = self.rng.uniform(size=(self.n_particles, 1))
            r2 = self.rng.uniform(size=(self.n_particles, 1))
            v = (self.w * v
                 + self.c1 * r1 * (pbest - x)
                 + self.c2 * r2 * (gbest[None, :] - x))
            v = np.clip(v, -self.vmax, self.vmax)
            x = np.clip(x + v, lo, hi)
            fit = self.fitness(x)
            better = fit > pbest_fit
            pbest[better] = x[better]
            pbest_fit[better] = fit[better]
            g_idx = int(np.argmax(pbest_fit))
            if pbest_fit[g_idx] > gbest_fit:
                gbest = pbest[g_idx].copy()
                gbest_fit = float(pbest_fit[g_idx])
            history.append(gbest_fit)
        return gbest, gbest_fit, history
