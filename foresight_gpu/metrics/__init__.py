"""Metrics for ``foresight_gpu``.

* Deterministic error metrics (:mod:`.deterministic`) — vectorised over the population,
  used as the per-particle **training** loss (via :meth:`Metric.loss`).
* :func:`.exceedance.non_exceedance` — the probabilistic objective axis.
* :func:`.regularization.lp_penalty` — L-p parameter penalty.
* Probabilistic diagnostics (:mod:`.probabilistic`) — reliability/resolution and predictive
  p-values, used for reporting and as the early-stopping / CV **scoring** criterion.

Resolve a metric from a name/alias (case-insensitive) or a handle with :func:`get_metric`.
"""

from .base import Metric
from .deterministic import (
    DETERMINISTIC,
    KGE,
    KGEprime,
    MAE,
    MSE,
    NSE,
    RMSE,
    kge,
    kge_prime,
    mae,
    mse,
    nse,
    rmse,
)
from .exceedance import non_exceedance
from .regularization import lp_penalty
from .probabilistic import (
    from_custom_log_space,
    predictive_pvalues,
    reliability,
    renard_metrics,
    to_custom_log_space,
)


def _build_registry(metrics):
    """Map every metric name *and* alias (lowercased) to its :class:`Metric`."""
    registry = {}
    for metric in metrics:
        registry[metric.__name__.lower()] = metric
        for alias in metric.aliases:
            registry[alias.lower()] = metric
    return registry


#: Case-insensitive name/alias -> Metric.
DETERMINISTIC_METRICS = _build_registry(DETERMINISTIC)


def get_metric(metric):
    """Resolve ``metric`` (a :class:`Metric` handle or a name/alias string).

    Raises
    ------
    ValueError
        If the name is not a known deterministic metric.
    """
    if isinstance(metric, Metric):
        return metric
    key = str(metric).lower()
    try:
        return DETERMINISTIC_METRICS[key]
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: "
            f"{sorted({m.__name__ for m in DETERMINISTIC})}"
        ) from None


__all__ = [
    "Metric",
    "get_metric",
    "DETERMINISTIC",
    "DETERMINISTIC_METRICS",
    "nse", "kge", "kge_prime", "mae", "mse", "rmse",
    "NSE", "KGE", "KGEprime", "MAE", "MSE", "RMSE",
    "non_exceedance",
    "lp_penalty",
    "predictive_pvalues",
    "reliability",
    "renard_metrics",
    "to_custom_log_space",
    "from_custom_log_space",
]
