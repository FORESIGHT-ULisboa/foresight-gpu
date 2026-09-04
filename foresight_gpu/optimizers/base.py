"""Optimizer interface and the single-generation engine step.

The **estimator drives the loop** and calls :func:`evolve` once per generation so it can
early-stop between generations. :func:`evolve` performs the shared engine work — evaluate,
(optional) exceedance penalty, non-domination sort, crowding, selection — while the
MOEA-specific parts (``generate`` / ``select``) are delegated to a :class:`BaseOptimizer`.

A :class:`BaseOptimizer` subclasses :class:`sklearn.base.BaseEstimator` for ``optimizer__*``
parameter routing during hyperparameter search.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from ..crowding import phenotype_crowding
from ..metrics.probabilistic import to_custom_log_space


class BaseOptimizer(BaseEstimator, ABC):
    """Population-based multi-objective optimiser (generate / select hooks)."""

    def initialise(self, low, high, population, rng):
        """Draw an initial population uniformly within ``[low, high]`` (search space).

        Subclasses may override :meth:`_on_initialise` to set up per-run state.
        """
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        self.low_ = low
        self.high_ = high
        self.n_params_ = low.size
        self.population_ = int(population)
        self.rng_ = rng
        pop = rng.uniform(0.0, 1.0, (self.population_, self.n_params_)) * (high - low) + low
        self._on_initialise(pop)
        return pop

    def _on_initialise(self, population):
        """Hook for subclasses to initialise per-run state (e.g. velocities)."""

    def enforce_bounds(self, population):
        """Clip to the search bounds; return ``(bounded, changed_mask)``."""
        bounded = np.clip(population, self.low_, self.high_)
        return bounded, bounded != population

    @abstractmethod
    def generate(self, fit, population):
        """Return a matrix of candidate parameter sets (search space)."""

    @abstractmethod
    def select(self, fit, front_levels, crowd, population):
        """Return indices of the joint population to keep (length = ``population_``)."""


def selection_crowding(fit, front_levels):
    """Crowding distance for selection, with the GPU edge-emphasis weighting.

    Combines NSGA-II phenotype crowding on (exceedance, loss) with a factor that favours
    solutions near the exceedance extremes (``|to_custom_log_space(eta)|``).
    """
    distance = phenotype_crowding(fit[:, 0], fit[:, 1], fronts=front_levels)
    with np.errstate(invalid="ignore"):
        distance = distance * np.abs(to_custom_log_space(fit[:, 0]))
    return distance


def evolve(optimizer, sorter, population, fit, simulations, evaluate, penalize=None):
    """Advance the population by one generation.

    Parameters
    ----------
    optimizer : BaseOptimizer
        Provides ``generate`` / ``select`` and holds MOEA state.
    sorter : DominanceSorter
        Non-domination sorter (e.g. double-Pareto).
    population, fit, simulations : ndarray
        Current population ``[N, n_params]``, its objective matrix ``[N, 2]`` (column 0 =
        exceedance, column 1 = loss), and simulations ``[n_samples, N]``.
    evaluate : callable
        ``evaluate(params) -> (simulations, fit)`` for candidate parameter sets.
    penalize : callable, optional
        Maps a ranking-objective matrix to a penalised copy (the force-non-exceedance term).

    Returns
    -------
    tuple
        ``(population, fit, simulations, rejected_fit)`` after selection.
    """
    candidates = optimizer.generate(fit, population)
    candidate_sims, candidate_fit = evaluate(candidates)

    joint_population = np.vstack([population, candidates])
    joint_fit = np.vstack([fit, candidate_fit])
    joint_simulations = np.hstack([simulations, candidate_sims])

    ranking_fit = penalize(joint_fit.copy()) if penalize is not None else joint_fit
    front_levels = sorter.front_levels(ranking_fit)
    crowd = selection_crowding(ranking_fit, front_levels)

    keep = optimizer.select(ranking_fit, front_levels, crowd, joint_population)
    keep = np.asarray(keep, dtype=int)
    rejected = np.ones(joint_population.shape[0], dtype=bool)
    rejected[keep] = False

    return (
        joint_population[keep],
        joint_fit[keep],
        joint_simulations[:, keep],
        joint_fit[rejected],
    )
