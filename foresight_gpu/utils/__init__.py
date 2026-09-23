"""Helper utilities (not core regressor machinery).

Feature engineering, screening, and plotting live here so the core package stays a general
scikit-learn regressor.
"""

from .features import LagFeatures, OudinPET, PeriodicFeatures, RollingSum
from .plotting import (
    plot_double_pareto_front,
    plot_hypervolume,
    plot_qq,
    plot_timeseries,
)
from .screening import prepare_arrays, screen_initial_population

__all__ = [
    "PeriodicFeatures",
    "LagFeatures",
    "RollingSum",
    "OudinPET",
    "plot_timeseries",
    "plot_qq",
    "plot_double_pareto_front",
    "plot_hypervolume",
    "prepare_arrays",
    "screen_initial_population",
]
