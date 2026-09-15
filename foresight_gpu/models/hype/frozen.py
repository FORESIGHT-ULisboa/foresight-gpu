"""Freezing a fitted HYPE ensemble into a portable, executable-free artifact.

A fitted :class:`~foresight_gpu.ensemble.ParetoEnsemble` holds the model by reference, and a
:class:`~.model.HYPEModel` keeps only the *path* to its template folder. So an unpickled
estimator can still predict, but only on a machine where that folder exists and HYPE runs -
and every prediction costs one HYPE run per retained member.

:func:`freeze` removes both constraints. Since the retained front is a fixed set of parameter
vectors and the simulation window is fixed too, the whole predictive object collapses to one
table of shape ``[n_steps, n_models]``. Prediction becomes an index lookup: no executable, no
template, no worker folders, and a pickle small enough to hand to an operational system.
"""

import numpy as np

from ...ensemble import ParetoEnsemble
from ..base import BaseForwardModel
from .dates import resolve_indices


class TableModel(BaseForwardModel):
    """A pre-simulated lookup table wearing the forward-model contract.

    Parameters
    ----------
    table : ndarray
        Simulations, shape ``[n_steps, n_models]``.
    t0 : int
        Day ordinal of the first row.
    date_column : int
        Column of ``X`` holding the day ordinal.

    Notes
    -----
    Its "parameters" are model **indices**, not physical values: one column per retained
    member. The physical parameters are kept on the frozen ensemble as ``hype_params`` for
    reporting.
    """

    scales_inputs = False
    scales_outputs = False

    def __init__(self, table=None, t0=0, date_column=0):
        self.table = table
        self.t0 = t0
        self.date_column = date_column

    def n_parameters(self, n_features):
        return 1

    def parameter_bounds(self, n_features):
        n_models = 0 if self.table is None else np.asarray(self.table).shape[1]
        return np.zeros(1), np.full(1, max(n_models - 1, 0))

    def regularizable_mask(self, n_features):
        return np.zeros(1, dtype=bool)

    def forward(self, X, params):
        table = np.asarray(self.table, dtype=float)
        X = np.asarray(X, dtype=float)
        rows = resolve_indices(X[:, int(self.date_column)], self.t0, table.shape[0])
        columns = np.rint(
            np.atleast_2d(np.asarray(params, dtype=float))[:, 0]
        ).astype(np.int64)
        if columns.size and (columns.min() < 0 or columns.max() >= table.shape[1]):
            raise ValueError(
                f"Model index out of range for a table with {table.shape[1]} column(s)."
            )
        return table[np.ix_(rows, columns)]


def freeze(ensemble, dates=None):
    """Return a copy of ``ensemble`` backed by a pre-simulated table.

    Parameters
    ----------
    ensemble : ParetoEnsemble
        A fitted ensemble whose model is a :class:`~.model.HYPEModel`.
    dates : array-like or None
        Window to bake in. ``None`` uses the model's full simulation window, so the frozen
        object predicts anywhere the original could.

    Returns
    -------
    ParetoEnsemble
        Same bands, same exceedances, but with a :class:`TableModel` inside. Carries
        ``hype_params`` (the physical parameters) and ``hype_window`` for reference.

    Notes
    -----
    This runs the retained members once over the window - the last HYPE runs the ensemble
    ever needs. With a warm cache from ``fit`` it usually costs nothing.
    """
    from .dates import as_X, from_ordinals

    model = ensemble.model
    if not hasattr(model, "_ensure_layout"):
        raise TypeError(
            f"freeze expects an ensemble built on a HYPEModel, got {type(model).__name__}."
        )
    model._ensure_layout()

    if dates is None:
        t0, n_steps = model._t0, model._n_steps
        X = from_ordinals(np.arange(t0, t0 + n_steps)).astype("datetime64[D]")
        X = as_X(X)
    else:
        X = as_X(dates)
        t0 = int(np.rint(X[:, 0].min()))
        n_steps = int(np.rint(X[:, 0].max())) - t0 + 1
        if X.shape[0] != n_steps:
            raise ValueError("freeze needs a contiguous daily window of dates.")

    table = model.forward(X, ensemble.params)

    frozen = ParetoEnsemble(
        model=TableModel(table=table, t0=t0, date_column=0),
        params=np.arange(table.shape[1], dtype=float).reshape(-1, 1),
        exceedances=ensemble.exceedances,
        x_scaler=ensemble.x_scaler,
        y_scaler=ensemble.y_scaler,
        quantiles=ensemble.quantiles,
        band_width=ensemble.band_width,
        min_models=ensemble.min_models,
        force_positive=ensemble.force_positive,
    )
    frozen.hype_params = np.asarray(ensemble.params, dtype=float)
    frozen.hype_window = (int(t0), int(n_steps))
    return frozen
