"""Non-exceedance: the probabilistic axis of the GPU objective space.

``non_exceedance(sim, obs)`` is the fraction of the series in which the **simulation
exceeds the observation**, computed per particle:

    eta = mean(sim > obs, axis=0)

A model that always sits above the observations scores 1; one that always undershoots
scores 0. This is the quantity the double-Pareto sort mirrors around 0.5, and the level at
which :class:`~foresight_gpu.ensemble.ParetoEnsemble` places each model when it aggregates
the population into non-exceedance-probability bands. The convention (``sim > obs``) is
kept identical to the original GPU implementation so the band aggregation and predictive
QQ diagnostics stay consistent.
"""

import numpy as np


def non_exceedance(sim, obs):
    """Fraction of the series where each simulation exceeds the observation.

    Parameters
    ----------
    sim : ndarray
        Simulations, shape ``[n_samples]`` or ``[n_samples, n_particles]``.
    obs : ndarray
        Observations, shape ``[n_samples]``.

    Returns
    -------
    float or ndarray
        Non-exceedance fraction per particle, in ``[0, 1]`` (scalar if ``sim`` is 1-D).
    """
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float).ravel()
    squeeze = sim.ndim == 1
    if squeeze:
        sim = sim[:, None]
    eta = np.mean(sim > obs[:, None], axis=0)
    return float(eta[0]) if squeeze else eta
