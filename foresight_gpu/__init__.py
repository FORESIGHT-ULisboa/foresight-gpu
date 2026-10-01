"""foresight_gpu — Generalized Pareto Uncertainty (GPU).

A scikit-learn-style probabilistic regressor that turns a family of deterministic models
into a reliable predictive distribution via a double / mirrored Pareto front over
non-exceedance and an error metric.
"""

__version__ = "0.8.0"

from .estimator import GPURegressor
from .ensemble import DEFAULT_QUANTILES, ParetoEnsemble
from .models import BaseForwardModel, GR4JModel, MLPModel, MLPModelOpenCL
from .optimizers import MOPSO, BaseOptimizer
from .domination import (
    DEFAULT_HV_LOG_REFERENCE,
    DEFAULT_HV_REFERENCE,
    HV_CLIP_WARN_FRACTION,
    HV_REFERENCE_MARGIN,
    DominanceSorter,
    DoubleParetoSorter,
    default_hv_reference,
    double_pareto_hypervolume,
    hypervolume,
    reference_nadir,
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
    "MLPModelOpenCL",
    "MOPSO",
    "BaseOptimizer",
    "DominanceSorter",
    "DoubleParetoSorter",
    "hypervolume",
    "double_pareto_hypervolume",
    "default_hv_reference",
    "reference_nadir",
    "DEFAULT_HV_REFERENCE",
    "DEFAULT_HV_LOG_REFERENCE",
    "HV_CLIP_WARN_FRACTION",
    "HV_REFERENCE_MARGIN",
    "get_metric",
    "nse",
    "kge",
    "kge_prime",
    "mae",
    "mse",
    "rmse",
    "scoring",
]
