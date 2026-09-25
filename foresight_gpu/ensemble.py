"""The fitted GPU artifact: turning a Pareto front into a predictive distribution.

After the optimiser converges, the retained front is a set of deterministic models, each
with an **observed exceedance** on the training data. :class:`ParetoEnsemble` implements the
GPU prediction rule (WRR draft, Eq. 4): to estimate the value at non-exceedance probability
``p``, aggregate (median of) the models whose observed exceedance lies within ``band_width``
of ``p``. Bands are then monotonically ordered and gaps interpolated in a custom log space so
the result is a valid, sharp inverse CDF.

This object is what :attr:`GPURegressor.ensemble_` holds; it is pickle-serialisable.
"""

import warnings

import numpy as np

from .domination import (
    DEFAULT_HV_LOG_PENALTY,
    DEFAULT_HV_PENALTY,
    HV_CLIP_WARN_FRACTION,
    DoubleParetoSorter,
    double_pareto_hypervolume,
)
from .metrics import get_metric
from .metrics.exceedance import non_exceedance
from .metrics.probabilistic import (
    predictive_pvalues,
    to_custom_log_space,
)

#: Default non-exceedance-probability levels (must lie strictly in (0, 1)).
DEFAULT_QUANTILES = [
0.01, 0.025, 0.05, 0.15, 0.25, 0.35, 0.45,
0.55, 0.65, 0.75, 0.85, 0.95, 0.975, 0.99,
]


def band_bounds(quantiles, band_width):
    """Lower/upper exceedance bounds ``[2, n_quantiles]`` for each probability band."""
    q = np.asarray(quantiles, dtype=float)
    ext = np.unique(np.concatenate([q, [0.0, 1.0]]))
    interval = ext[1:] - ext[:-1]
    interval[1:-1] /= 2.0
    rolled = np.minimum(interval[:-1], interval[1:])  # == pandas rolling(2).min().dropna()
    interval = np.minimum(band_width, rolled)
    return np.vstack([q - interval, q + interval])


def aggregate_by_band(simulations, exceedances, bounds, min_models, force_positive):
    """Median of the models whose exceedance falls in each band (WRR Eq. 4)."""
    aggregated = np.full((simulations.shape[0], bounds.shape[1]), np.nan)
    for i in range(bounds.shape[1]):
        idx = np.where((exceedances >= bounds[0, i]) & (exceedances <= bounds[1, i]))[0]
        if idx.size >= min_models:
            aggregated[:, i] = np.median(simulations[:, idx], axis=1)
    if force_positive:
        aggregated = np.maximum(0.0, aggregated)
    return aggregated


def post_process_bands(aggregated, quantiles):
    """Sort bands, drop inversions/near-duplicates, and interpolate gaps (custom log space)."""
    aggregated = aggregated.copy()
    bands = to_custom_log_space(np.asarray(quantiles, dtype=float))
    for i in range(aggregated.shape[0]):
        row = aggregated[i, :]
        finite = ~np.isnan(row)
        if finite.sum() >= 2:
            row[finite] = np.sort(row[finite])
            prev = row[-1]
            for j in range(len(row) - 2, 0, -1):
                if np.isnan(prev):
                    prev = row[j]
                elif not np.isnan(row[j]):
                    if np.round(prev, 6) <= np.round(row[j], 6):
                        row[j] = np.nan
                    else:
                        prev = row[j]
            keep = ~np.isnan(row)
            if keep.sum() > 1:
                aggregated[i, :] = np.interp(bands, bands[keep], row[keep])
            else:
                aggregated[i, :] = np.nan
        else:
            aggregated[i, :] = np.nan
    return aggregated


class ParetoEnsemble:
    """A converged GPU front, ready to make probabilistic predictions.

    Parameters
    ----------
    model : BaseForwardModel
        The (fitted-configuration) forward model.
    params : ndarray
        Retained front parameter sets in *model* space, shape ``[n_models, n_params]``.
    exceedances : ndarray
        Observed training exceedance of each model, shape ``[n_models]``.
    quantiles : sequence of float
        Default non-exceedance-probability levels.
    band_width : float
        Half-width of the exceedance window used to select models per band.
    min_models : int
        Minimum models required to populate a band.
    force_positive : bool
        Clip predictions at zero.
    metric : str or Metric or None
        The metric this front was calibrated on — the *default* for
        :meth:`score_hypervolume` and :meth:`front_objectives`, never a lock: pass
        ``metric=`` to re-read the same front under any other metric. Stored as given;
        ``Metric`` reduces to a plain ``str`` on pickling and is re-resolved with
        ``get_metric`` on use.
    hv_penalty : float or None
        Hypervolume ceiling ``P``, **already resolved** (scale applied, ``"climatology"``
        expanded), in the units of ``hv_space``. ``None`` -> the default for that space.
    hv_interpolation : {"step", "linear"}
        Default integration rule for :meth:`score_hypervolume`.
    hv_space : {"linear", "log10"}
        Default objective space for :meth:`score_hypervolume`. ``"log10"`` integrates
        ``log10(loss)`` in a symmetric ``[-P, P]`` box.
    """

    def __init__(self, model, params, exceedances,
                 quantiles=DEFAULT_QUANTILES, band_width=0.025, min_models=1,
                 force_positive=False, metric=None, hv_penalty=None,
                 hv_interpolation="step", hv_space="linear"):
        self.model = model
        self.params = np.asarray(params, dtype=float)
        self.exceedances = np.asarray(exceedances, dtype=float)
        self.quantiles = list(quantiles)
        self.band_width = band_width
        self.min_models = min_models
        self.force_positive = force_positive
        self.metric = metric
        self.hv_penalty = hv_penalty
        self.hv_interpolation = hv_interpolation
        self.hv_space = hv_space
        self.band_bounds = band_bounds(self.quantiles, band_width)
        self.band_probabilities = self.band_bounds[1] - self.band_bounds[0]

    @property
    def n_models(self):
        return self.params.shape[0]

    def _simulate(self, X):
        """Run every retained model on ``X``; return ``(sims, valid_mask)``."""
        X = np.asarray(X, dtype=float)
        valid = np.isfinite(X).all(axis=1)
        return self.model.forward(X[valid], self.params), valid

    def predict_quantiles(self, X, quantiles=None, post_process=True):
        """Predicted band values, shape ``[n_samples, n_quantiles]`` (NaN where invalid)."""
        if quantiles is None:
            quantiles = self.quantiles
            bounds = self.band_bounds
        else:
            quantiles = list(quantiles)
            bounds = band_bounds(quantiles, self.band_width)

        sims, valid = self._simulate(X)
        agg = aggregate_by_band(
            sims, self.exceedances, bounds, self.min_models, self.force_positive
        )
        if post_process:
            agg = post_process_bands(agg, quantiles)

        out = np.full((valid.shape[0], len(quantiles)), np.nan)
        out[valid] = agg
        return out

    def predict(self, X):
        """Point estimate: the band nearest non-exceedance 0.5, shape ``[n_samples]``."""
        agg = self.predict_quantiles(X)
        idx = int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))
        return agg[:, idx]

    def inverse_cdf(self, X, p):
        """Value at a single non-exceedance probability ``p`` (WRR Eq. 4).

        Post-processing (which needs >= 2 bands to order/interpolate) is skipped for a
        single level.
        """
        return self.predict_quantiles(X, quantiles=[p], post_process=False)[:, 0]

    def predictive_pvalues(self, X, y):
        """PIT p-values of ``y`` within the predicted distribution (for the QQ plot)."""
        agg = self.predict_quantiles(X)
        return predictive_pvalues(agg, np.asarray(y, dtype=float).ravel(), self.quantiles)

    # -- front geometry ----------------------------------------------------------------

    def _resolve_metric(self, metric):
        metric = metric if metric is not None else getattr(self, "metric", None)
        if metric is None:
            raise ValueError(
                "This ParetoEnsemble carries no metric; pass one explicitly, e.g. "
                "ensemble.score_hypervolume(X, y, 'nse')."
            )
        return get_metric(metric)

    def front_objectives(self, X, y, metric=None):
        """Double-Pareto objectives of every retained model on ``(X, y)``.

        The public route for plotting or interrogating the front;
        :meth:`score_hypervolume` integrates exactly these arrays.

        Parameters
        ----------
        X, y : ndarray
            Any window — the front is re-evaluated on it, so this is not restricted to the
            data the ensemble was fitted on.
        metric : str or Metric, optional
            Defaults to the calibration metric stored on the ensemble.

        Returns
        -------
        eta : ndarray ``[n_models]``
            Non-exceedance on ``(X, y)`` — recomputed, *not* the training
            :attr:`exceedances`.
        loss : ndarray ``[n_models]``
            Raw minimisation loss, with non-finite values mapped to ``+inf``.
        front : ndarray of int
            Front-0 indices, eta-ordered.

        Notes
        -----
        The loss is computed from the simulations, so the Lp regularisation term (which
        penalises parameters, not held-out fit) cannot reach it by construction.

        Rows are dropped where **either** ``X`` or ``y`` is non-finite. ``_simulate``
        masks only ``X``; a single NaN in ``y`` would otherwise make ``metric.loss`` NaN
        for every particle, clipping them all to ``P`` and reporting ``hv == 0`` silently.
        """
        metric = self._resolve_metric(metric)
        y = np.asarray(y, dtype=float).ravel()
        sims, valid = self._simulate(X)
        if y.shape[0] != valid.shape[0]:
            raise ValueError(f"X has {valid.shape[0]} rows but y has {y.shape[0]}.")
        yv = y[valid]
        finite = np.isfinite(yv)
        if not finite.all():
            sims, yv = sims[finite], yv[finite]
        if yv.size == 0:
            raise ValueError("No finite (X, y) rows to score.")

        loss = metric.loss(sims, yv)
        # Mirrors what double_pareto_hypervolume does internally before sorting, so a
        # precomputed ``front=`` is the one it would have found itself.
        loss = np.where(np.isfinite(loss), loss, np.inf)
        eta = non_exceedance(sims, yv)
        front = np.asarray(
            DoubleParetoSorter().sort(np.column_stack([eta, loss]))[0], dtype=int
        )
        return eta, loss, front

    def score_hypervolume(self, X, y, metric=None, *, penalty=None, interpolation=None,
                          space=None, details=False):
        """Double-Pareto hypervolume of this front on ``(X, y)``, in ``[0, 1]``.

        Higher = better. Unlike the band-based scorers this measures the **front**, so it
        re-simulates the retained models rather than calling ``predict_quantiles`` — which
        is also why it is cheaper than they are.

        Parameters
        ----------
        X, y : ndarray
            The window to score on.
        metric : str or Metric, optional
            Defaults to the calibration metric. Passing another re-reads the *same* fitted
            front under it, which is how a front calibrated on NSE can be examined under
            KGE or MAE after the fact.
        penalty : float, optional
            Ceiling ``P``, in the units of ``space``. Defaults to :attr:`hv_penalty`, else
            the default for the resolved space.
        interpolation : {"step", "linear"}, optional
            Defaults to :attr:`hv_interpolation`.
        space : {"linear", "log10"}, optional
            Objective space. Defaults to :attr:`hv_space`. Pass ``"log10"`` to re-read the
            same front on a log loss axis — the usual reason being that the linear box
            compresses the region you care about. ``penalty`` is then a bound on
            ``|log10 loss|``, so it wants a much smaller number.
        details : bool
            Return the decomposition dict instead of a float.

        Warns
        -----
        UserWarning
            When the ceiling has swallowed the whole front (``hv`` pinned at 0, no gradient
            to stop on), or when it clips more than
            :data:`~foresight_gpu.domination.HV_CLIP_WARN_FRACTION` of front-0 — at which
            point improvements in the clipped part are invisible to the indicator.
        """
        metric = self._resolve_metric(metric)
        eta, loss, front = self.front_objectives(X, y, metric)

        # getattr throughout: ensembles pickled before these attributes existed still load.
        space = space or getattr(self, "hv_space", "linear")
        if penalty is None:
            penalty = getattr(self, "hv_penalty", None)
        if penalty is None:
            penalty = DEFAULT_HV_LOG_PENALTY if space == "log10" else DEFAULT_HV_PENALTY
        P = float(penalty)

        parts = double_pareto_hypervolume(
            np.column_stack([eta, loss]), P, front=front,
            interpolation=interpolation or getattr(self, "hv_interpolation", "step"),
            space=space, details=True,
        )

        # Two rungs, worst first. Individual tail particles above P are the normal regime --
        # the extreme-eta ones are *meant* to be biased -- which is why neither rung fires on
        # a healthy front. Both texts are constant across checks, so Python's warning dedup
        # emits each once per fit rather than once per generation.
        clipped = parts["clipped_fraction"]
        ceiling = f"P={P:g}" + (f" (raw loss {10 ** P:g})" if space == "log10" else "")
        if clipped >= 1.0:
            warnings.warn(
                f"Every front-0 loss exceeds the hypervolume ceiling {ceiling} under metric "
                f"{metric} in {space} space, so hv is pinned at 0 and early stopping cannot "
                f"discriminate. Raise hv_penalty above the plausible {metric} loss scale.",
                UserWarning,
            )
        elif clipped > HV_CLIP_WARN_FRACTION:
            warnings.warn(
                f"The hypervolume ceiling {ceiling} clips {clipped:.0%} of front-0 under "
                f"metric {metric} in {space} space, so the indicator cannot see improvements "
                f"in that part of the front. Raise hv_penalty, or use hv_space='log10' to "
                f"spread the loss axis. Tracked as 'clipped_fraction' in history_.",
                UserWarning,
            )

        return parts if details else parts["hv"]
