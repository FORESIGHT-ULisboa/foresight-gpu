"""Fast, vectorised deterministic error metrics.

Every function accepts

* ``sim`` — simulations, shape ``[n_samples]`` or ``[n_samples, n_particles]``,
* ``obs`` — observations, shape ``[n_samples]``,

and reduces along ``axis=0``, returning one value **per particle** (shape
``[n_particles]``), or a scalar when ``sim`` is 1-D. This population-wise vectorisation is
what makes them cheap to call on the whole swarm every generation — deliberately distinct
from ``forecast_performance``'s pandas-per-series metrics, which they match numerically
(``pandas`` standard deviations use ``ddof=1``, mirrored here).

Definitions follow the companion ``forecast_performance`` package exactly, including its
KGE' formulation ``gamma = CV_obs / CV_sim`` (see its docstring); the cross-check tests
assert equivalence per column.
"""

import numpy as np

from .base import Metric


def _prepare(sim, obs):
    """Return ``sim`` as ``[n, P]``, ``obs`` as ``[n, 1]`` and a squeeze flag."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float).ravel()
    squeeze = sim.ndim == 1
    if squeeze:
        sim = sim[:, None]
    return sim, obs[:, None], squeeze


def _finish(values, squeeze):
    return float(values[0]) if squeeze else values


def _std1(a, axis=0):
    """Sample standard deviation (``ddof=1``) to match pandas semantics."""
    return np.std(a, axis=axis, ddof=1)


def _pearson(sim, obs):
    """Pearson correlation between each simulation column and the observations."""
    sim_c = sim - sim.mean(axis=0)
    obs_c = obs - obs.mean(axis=0)
    num = np.sum(sim_c * obs_c, axis=0)
    den = np.sqrt(np.sum(sim_c**2, axis=0) * np.sum(obs_c**2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def _nse(sim, obs):
    """Nash-Sutcliffe efficiency (perfect = 1, higher is better)."""
    s, o, squeeze = _prepare(sim, obs)
    denom = np.sum((o - o.mean()) ** 2)
    num = np.sum((o - s) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, 1.0 - num / denom)
    return _finish(out, squeeze)


def _kge(sim, obs):
    """Kling-Gupta efficiency (perfect = 1, higher is better)."""
    s, o, squeeze = _prepare(sim, obs)
    r = _pearson(s, o)
    obs_mean, obs_std = o.mean(), _std1(o)
    beta = s.mean(axis=0) / obs_mean
    alpha = _std1(s, axis=0) / obs_std
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return _finish(out, squeeze)


def _kge_prime(sim, obs):
    """Modified Kling-Gupta efficiency KGE' (perfect = 1, higher is better).

    Follows ``forecast_performance``: ``gamma = (std_obs / mean_obs) / (std_sim / mean_sim)``.
    """
    s, o, squeeze = _prepare(sim, obs)
    r = _pearson(s, o)
    obs_mean, obs_std = o.mean(), _std1(o)
    sim_mean = s.mean(axis=0)
    beta = sim_mean / obs_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = (obs_std / obs_mean) / (_std1(s, axis=0) / sim_mean)
        out = 1.0 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)
    return _finish(out, squeeze)


def _mae(sim, obs):
    """Mean absolute error (perfect = 0, lower is better)."""
    s, o, squeeze = _prepare(sim, obs)
    return _finish(np.mean(np.abs(o - s), axis=0), squeeze)


def _mse(sim, obs):
    """Mean squared error (perfect = 0, lower is better)."""
    s, o, squeeze = _prepare(sim, obs)
    return _finish(np.mean(np.square(o - s), axis=0), squeeze)


def _rmse(sim, obs):
    """Root mean squared error (perfect = 0, lower is better)."""
    s, o, squeeze = _prepare(sim, obs)
    return _finish(np.sqrt(np.mean(np.square(o - s), axis=0)), squeeze)


nse = Metric("nse", _nse, greater_is_better=True, aliases=("NSE",))
kge = Metric("kge", _kge, greater_is_better=True, aliases=("KGE",))
kge_prime = Metric(
    "kge_prime", _kge_prime, greater_is_better=True, aliases=("KGEprime", "kgeprime")
)
mae = Metric("mae", _mae, greater_is_better=False, aliases=("MAE",))
mse = Metric("mse", _mse, greater_is_better=False, aliases=("MSE",))
rmse = Metric("rmse", _rmse, greater_is_better=False, aliases=("RMSE",))

# PascalCase aliases (same objects) for backward compatibility with FORESIGHT code.
NSE = nse
KGE = kge
KGEprime = kge_prime
MAE = mae
MSE = mse
RMSE = rmse

#: All public deterministic metrics, in display order.
DETERMINISTIC = [nse, kge, kge_prime, mae, mse, rmse]
