"""Multi-Objective Particle Swarm Optimisation (MOPSO).

NumPy port of the original GPU particle-swarm optimiser. Global attractors are the best
solution in each of three exceedance regions (low / centre / high); local attractors are
the best solution in each of ``p_bins`` exceedance bins. Bounds are enforced by reflection
(velocity reversed and damped). ``partial`` restricts how many parameters each particle
moves per generation.
"""

import numpy as np

from .base import BaseOptimizer


class MOPSO(BaseOptimizer):
    """Particle-swarm implementation of the GPU generate/select hooks.

    Parameters
    ----------
    inertia : float
        Velocity carry-over factor.
    c1, c2, c3 : float
        Strength of local attractors, global attractors and random perturbation.
    p_bins : int
        Number of exceedance bins for local attractors.
    partial : float
        Fraction of parameters perturbed per particle each generation (``<= 1``).
    """

    def __init__(self, inertia=0.3, c1=0.1, c2=0.1, c3=0.001, p_bins=10, partial=1.0):
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.p_bins = p_bins
        self.partial = partial

    def _on_initialise(self, population):
        n = self.population_
        self._c3 = self.c3 if self.c3 != 0 else 1e-5
        self._partial_count = max(int(np.round(self.n_params_ * self.partial)), 1)
        self._p_best = np.arange(n)
        self._g_best = np.arange(n)
        self._velocities = np.zeros((n, self.n_params_))
        edges = np.hstack((-np.inf, np.linspace(0.0, 1.0, num=self.p_bins)))
        self._bins = np.vstack((edges[:-1], edges[1:]))

    @staticmethod
    def _euclidean(reference, points):
        return np.sqrt(np.sum((points - reference) ** 2, axis=1))

    def _update_attractors(self, fit):
        exceedance = fit[:, 0]
        sorted_idx = np.argsort(exceedance)

        # Global attractors: best in each of three exceedance regions.
        regions = np.array_split(sorted_idx, 3)
        borders = np.arange(0, 4) / 3.0
        borders[-1] = np.inf
        for r in range(3):
            idx = np.where((exceedance >= borders[r]) & (exceedance < borders[r + 1]))[0]
            if idx.size == 0:
                continue
            if r == 0:
                target = np.array((0.0, np.min(fit[idx, 1])))
            elif r == 1:
                target = fit[idx[np.argmin(fit[idx, 1])]]
            else:
                target = np.array((1.0, np.min(fit[idx, 1])))
            dist = self._euclidean(target, fit[idx])
            self._g_best[regions[r]] = idx[np.argmin(dist)]

        # Local attractors: best (min loss) in each exceedance bin.
        for b in range(self.p_bins):
            idx = np.where((exceedance > self._bins[0, b]) & (exceedance <= self._bins[1, b]))[0]
            if idx.size:
                self._p_best[idx] = idx[np.argmin(fit[idx, 1])]
        idx = np.where(exceedance >= self._bins[1, self.p_bins - 1])[0]
        if idx.size:
            self._p_best[idx] = idx[np.argmin(fit[idx, 1])]

    def generate(self, fit, population):
        self._update_attractors(fit)
        p_best = population[self._p_best]
        g_best = population[self._g_best]
        n, v = self.population_, self.n_params_

        r1 = self.rng_.uniform(0.0, self.c1, (n, 1)).repeat(v, 1)
        r2 = self.rng_.uniform(0.0, self.c2, (n, 1)).repeat(v, 1)
        r3 = self.rng_.normal(0.0, self._c3, (n, v))

        if self.partial < 1:
            mask = np.zeros((n, v))
            for i in range(n):
                cols = self.rng_.permutation(v)[: self._partial_count]
                mask[i, cols] = 1.0
            r1, r2, r3 = r1 * mask, r2 * mask, r3 * mask

        self._velocities *= self.inertia
        candidate_velocities = (
            self._velocities
            + r1 * (p_best - population)
            + r2 * (g_best - population)
            + r3
        )
        candidates, to_reflect = self.enforce_bounds(population + candidate_velocities)
        candidate_velocities[to_reflect] *= -0.5
        self._joint_velocities = np.vstack((self._velocities, candidate_velocities))
        return candidates

    def select(self, fit, front_levels, crowd, population):
        keep = []
        available = self.population_
        max_front = int(np.nanmax(front_levels))
        level = 0
        while available > 0 and level <= max_front:
            idx = np.where(front_levels == level)[0]
            if idx.size <= available:
                keep.extend(idx.tolist())
            else:
                # Keep the least-crowded first; break ties by lower loss.
                order = np.lexsort((fit[idx, 1], crowd[idx]))[::-1]
                keep.extend(idx[order[:available]].tolist())
            available = self.population_ - len(keep)
            level += 1

        keep = np.asarray(keep, dtype=int)
        self._velocities = self._joint_velocities[keep]
        return keep
