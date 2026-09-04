"""Probabilistic diagnostics for GPU predictions (Renard et al., 2010, WRR).

These operate on the **aggregated band matrix** produced by
:class:`~foresight_gpu.ensemble.ParetoEnsemble` (shape ``[n_samples, n_quantiles]``) and
the observations. They are used for reporting, for the predictive QQ plot, and — via
:mod:`foresight_gpu.scoring` — as the held-out criterion for early stopping and model
selection. They run once per evaluation (not per particle per generation), so the small
Python loop in :func:`predictive_pvalues` is acceptable.

Ported faithfully from the original GPU implementation to preserve validated behaviour,
including the custom log-space used to interpolate the distribution tails.
"""

import warnings

import numpy as np
from scipy.interpolate import interp1d


def to_custom_log_space(x):
    """Map probabilities in ``[0, 1]`` to a log space symmetric about 0.5.

    Spreads the tails so extreme quantiles interpolate sensibly.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        y = np.empty_like(x)
        low = x < 0.5
        y[low] = np.log(x[low]) - np.log(0.5)
        high = ~low
        y[high] = -np.log(1 - x[high]) + np.log(0.5)
    return y


def from_custom_log_space(y):
    """Inverse of :func:`to_custom_log_space`."""
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore"):
        x = np.empty_like(y)
        low = y < 0
        x[low] = np.exp(y[low] + np.log(0.5))
        high = ~low
        x[high] = 1 - np.exp(-y[high] + np.log(0.5))
    return x


def predictive_pvalues(aggregated, targets, quantiles):
    """PIT p-values of the observations within the predicted distribution.

    Parameters
    ----------
    aggregated : ndarray
        Band values, shape ``[n_samples, n_quantiles]`` (may contain NaN gaps).
    targets : ndarray
        Observations, shape ``[n_samples]``.
    quantiles : sequence of float
        Non-exceedance-probability levels of the columns of ``aggregated``.

    Returns
    -------
    ndarray
        p-value per observation in ``[0, 1]`` (NaN where it cannot be evaluated).
    """
    targets = np.asarray(targets, dtype=float).ravel()
    bands = to_custom_log_space(np.asarray(quantiles, dtype=float)[::-1])
    pvalues = np.full(targets.shape[0], np.nan)

    for i in range(targets.shape[0]):
        finite = np.isfinite(aggregated[i, :])
        agg = aggregated[i, finite]
        band = bands[finite]
        sims, idxs = np.unique(agg, return_index=True)
        if sims.size < 2:
            continue
        if targets[i] < sims[0]:
            pvalues[i] = np.inf
        elif targets[i] > sims[-1]:
            pvalues[i] = -np.inf
        else:
            pvalues[i] = interp1d(
                sims, band[idxs], kind="linear", assume_sorted=True
            )(targets[i])

    pvalues = np.clip(from_custom_log_space(pvalues), 0.0, 1.0)
    return 1.0 - pvalues


def reliability(pvalues):
    """Reliability index alpha (1 = perfectly reliable).

    ``alpha = 1 - 2 * mean(|sorted p-values - Uniform[0, 1]|)``.
    """
    pv = np.sort(np.asarray(pvalues, dtype=float))
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        return np.nan
    uniform = np.linspace(0.0, 1.0, pv.size)
    return float(1.0 - 2.0 * np.mean(np.abs(uniform - pv)))


def renard_metrics(pvalues, aggregated, band_probabilities):
    """Full Renard-2010 diagnostic bundle.

    Parameters
    ----------
    pvalues : ndarray
        Predictive p-values (see :func:`predictive_pvalues`).
    aggregated : ndarray
        Band values, shape ``[n_samples, n_quantiles]``.
    band_probabilities : ndarray
        Probability mass of each band (bound widths), shape ``[n_quantiles]``.

    Returns
    -------
    dict
        ``alpha`` (reliability), ``xi`` (fraction of observations inside the range),
        ``pi`` (resolution / sharpness), ``pi_rel`` (relative resolution), ``sigma``
        (mean predictive standard deviation).
    """
    pv = np.sort(np.asarray(pvalues, dtype=float))
    pv = pv[np.isfinite(pv)]
    if pv.size == 0:
        return {k: np.nan for k in ("alpha", "xi", "pi", "pi_rel", "sigma")}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            uniform = np.linspace(0.0, 1.0, pv.size)
            alpha = 1.0 - 2.0 * np.mean(np.abs(uniform - pv))

            outside = np.zeros_like(pv)
            outside[(pv == 0) | (pv == 1)] = 1
            xi = 1.0 - np.mean(outside)

            agg = np.where(np.isnan(aggregated), 0.0, aggregated)
            band_probabilities = np.asarray(band_probabilities, dtype=float)
            represented = np.sum(band_probabilities)
            weights = np.tile(band_probabilities, (agg.shape[0], 1))
            ex = np.sum(agg * weights, axis=1) / represented
            ex2 = np.sum(np.square(agg) * weights, axis=1) / represented
            std = np.sqrt(ex2 - np.square(ex))
            pi = float(np.nanmean(1.0 / std))
            pi_rel = float(np.nanmean(ex / std))
            sigma = float(np.nanmean(std))

    return {"alpha": float(alpha), "xi": float(xi), "pi": pi, "pi_rel": pi_rel, "sigma": sigma}
