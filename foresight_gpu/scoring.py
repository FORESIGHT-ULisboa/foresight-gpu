"""Probabilistic scorers for model selection (``cross_val_score`` / ``GridSearchCV``).

Unlike ``sklearn.metrics.make_scorer`` (which only sees ``predict``), these reach the
**probabilistic** output via the fitted :class:`~foresight_gpu.ensemble.ParetoEnsemble`.
They follow the sklearn scorer contract ``scorer(estimator, X, y) -> float`` with **higher =
better**. Early stopping does not use them: it always monitors the validation hypervolume.

All four go through :func:`score_ensemble`. ``"hypervolume"`` reads the fitted **front**
(:meth:`ParetoEnsemble.score_hypervolume`) rather than the aggregated bands, which is why
it branches before ``predict_quantiles`` is ever called.
"""

import numpy as np
from forecast_performance.metrics.probabilistic import crps as fp_crps

from .metrics.probabilistic import predictive_pvalues, renard_metrics


def probabilistic_diagnostics(ensemble, agg, y):
    """Reliability alpha, resolution pi and mean CRPS of a band matrix.

    ``agg`` is ``ensemble``'s post-processed band matrix on the rows of ``y`` (from
    ``predict_quantiles``, or ``_bands_from_sims`` when the simulations are already in hand).
    """
    y = np.asarray(y, dtype=float).ravel()
    pvalues = predictive_pvalues(agg, y, ensemble.quantiles)
    renard = renard_metrics(pvalues, agg, ensemble.band_probabilities)
    return {
        "reliability": renard["alpha"],
        "resolution": renard["pi"],
        "crps": _crps(ensemble, agg, y),
    }


def score_ensemble(ensemble, X, y, scoring="crps"):
    """Score a :class:`ParetoEnsemble` on held-out data (higher = better).

    ``scoring`` may be ``"hypervolume"`` (the double-Pareto front indicator, in
    ``[0, 1]``), ``"reliability"`` (Renard alpha), ``"resolution"`` (pi), ``"crps"``
    (negated, via ``forecast_performance``) or a callable ``scoring(ensemble, X, y) -> float``.

    ``"hypervolume"`` uses the metric, reference, interpolation and objective space the
    ensemble was built with; call :meth:`ParetoEnsemble.score_hypervolume` directly to
    override any of them.
    """
    if callable(scoring):
        return float(scoring(ensemble, X, y))

    # Before predict_quantiles on purpose: the front indicator needs neither the aggregated
    # bands nor the p-values, so branching here keeps it the cheapest of the four.
    if scoring == "hypervolume":
        return float(ensemble.score_hypervolume(X, y))
    if scoring not in ("reliability", "resolution", "crps"):
        raise ValueError(
            f"Unknown scoring {scoring!r}; use 'hypervolume', 'reliability', "
            f"'resolution', 'crps' or a callable."
        )

    diag = probabilistic_diagnostics(ensemble, ensemble.predict_quantiles(X), y)
    return -diag["crps"] if scoring == "crps" else float(diag[scoring])


def _crps(ensemble, agg, y):
    """Mean CRPS over the rows with a complete band set."""
    mask = np.isfinite(agg).all(axis=1)
    if not mask.any():
        return float("nan")
    value = fp_crps(agg[mask], np.asarray(ensemble.quantiles, dtype=float),
                    y[mask], "probabilistic")
    return float(np.nanmean(value))


# --- sklearn-compatible scorers: scorer(estimator, X, y) -------------------------------

def reliability_scorer(estimator, X, y):
    """Renard reliability alpha of the estimator's predictive distribution."""
    return score_ensemble(estimator.ensemble_, X, y, "reliability")


def resolution_scorer(estimator, X, y):
    """Renard resolution pi (sharpness) of the estimator's predictive distribution."""
    return score_ensemble(estimator.ensemble_, X, y, "resolution")


def crps_scorer(estimator, X, y):
    """Negated mean CRPS (higher = better) via ``forecast_performance``."""
    return score_ensemble(estimator.ensemble_, X, y, "crps")


def hypervolume_scorer(estimator, X, y):
    """Double-Pareto hypervolume of the fitted ensemble's front, in ``[0, 1]``."""
    return score_ensemble(estimator.ensemble_, X, y, "hypervolume")


_SCORERS = {
    "hypervolume": hypervolume_scorer,
    "reliability": reliability_scorer,
    "resolution": resolution_scorer,
    "crps": crps_scorer,
}


def make_gpu_scorer(name):
    """Return a named sklearn-compatible scorer (``scorer(estimator, X, y)``)."""
    try:
        return _SCORERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scorer {name!r}; choose from {sorted(_SCORERS)}."
        ) from None
