"""foresight_gpu — Generalized Pareto Uncertainty (GPU).

A scikit-learn-style probabilistic regressor that turns a family of deterministic models
into a reliable predictive distribution via a double / mirrored Pareto front over
non-exceedance and an error metric.
"""

__version__ = "0.6.0"

from .estimator import GPURegressor
from .ensemble import DEFAULT_QUANTILES, ParetoEnsemble
from .models import BaseForwardModel, GR4JModel, HYPEModel, MLPModel, MLPModelOpenCL
from .optimizers import MOPSO, BaseOptimizer
from .domination import (
    DEFAULT_HV_LOG_PENALTY,
    DEFAULT_HV_PENALTY,
    HV_CLIP_WARN_FRACTION,
    DominanceSorter,
    DoubleParetoSorter,
    default_hv_penalty,
    double_pareto_hypervolume,
    hypervolume,
)
from .metrics import (
    get_metric,
    kge,
    kge_prime,
    mae,
    mse,
    nse,
    rmse,
)
from . import scoring

__all__ = [
    "__version__",
    "GPURegressor",
    "ParetoEnsemble",
    "DEFAULT_QUANTILES",
    "BaseForwardModel",
    "MLPModel",
    "GR4JModel",
    "HYPEModel",
    "MLPModelOpenCL",
    "MOPSO",
    "BaseOptimizer",
    "DominanceSorter",
    "DoubleParetoSorter",
    "hypervolume",
    "double_pareto_hypervolume",
    "default_hv_penalty",
    "DEFAULT_HV_PENALTY",
    "DEFAULT_HV_LOG_PENALTY",
    "HV_CLIP_WARN_FRACTION",
    "get_metric",
    "nse",
    "kge",
    "kge_prime",
    "mae",
    "mse",
    "rmse",
    "scoring",
]
