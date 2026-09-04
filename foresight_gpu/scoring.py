"""Probabilistic scorers for model selection and early stopping.

Unlike ``sklearn.metrics.make_scorer`` (which only sees ``predict``), these reach the
**probabilistic** output via the fitted :class:`~foresight_gpu.ensemble.ParetoEnsemble`.
They follow the sklearn scorer contract ``scorer(estimator, X, y) -> float`` with **higher =
better**, so they drop straight into ``cross_val_score(..., scoring=...)`` and
``GridSearchCV``. The same logic backs the estimator's internal early-stopping monitor via
:func:`score_ensemble`.
"""

import numpy as np

from .metrics.probabilistic import predictive_pvalues, reliability, renard_metrics


def score_ensemble(ensemble, X, y, scoring="crps"):
    """Score a :class:`ParetoEnsemble` on held-out data (higher = better).

    ``scoring`` may be ``"reliability"`` (Renard alpha), ``"resolution"`` (pi),
    ``"crps"`` (negated, via ``forecast_performance``) or a callable
    ``scoring(ensemble, X, y) -> float``.
    """
    if callable(scoring):
        return float(scoring(ensemble, X, y))

    y = np.asarray(y, dtype=float).ravel()
    agg = ensemble.predict_quantiles(X)
    pvalues = predictive_pvalues(agg, y, ensemble.quantiles)

    if scoring == "reliability":
        return float(reliability(pvalues))
    if scoring == "resolution":
        return float(renard_metrics(pvalues, agg, ensemble.band_probabilities)["pi"])
    if scoring == "crps":
        return -float(_crps(ensemble, agg, y))
    raise ValueError(
        f"Unknown scoring {scoring!r}; use 'reliability', 'resolution', 'crps' or a callable."
    )


def _crps(ensemble, agg, y):
    """Mean CRPS via the companion ``forecast_performance`` package."""
    try:
        from performance.metrics.probabilistic import crps as fp_crps
    except Exception as exc:  # pragma: no cover - optional path
        raise ImportError(
            "scoring='crps' requires forecast_performance (imports as `performance`)."
        ) from exc
    mask = np.isfinite(agg).all(axis=1)
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


_SCORERS = {
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
