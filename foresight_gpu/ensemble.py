"""The fitted GPU artifact: turning a Pareto front into a predictive distribution.

After the optimiser converges, the retained front is a set of deterministic models, each
with an **observed exceedance** on the training data. :class:`ParetoEnsemble` implements the
GPU prediction rule (WRR draft, Eq. 4): to estimate the value at non-exceedance probability
``p``, aggregate (median of) the models whose observed exceedance lies within ``band_width``
of ``p``. Bands are then monotonically ordered and gaps interpolated in a custom log space so
the result is a valid, sharp inverse CDF.

This object is what :attr:`GPURegressor.ensemble_` holds; it is pickle-serialisable.
"""

import numpy as np

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
    x_scaler, y_scaler : sklearn scaler or None
        Input/output scalers (present only when the model scales inputs/outputs).
    quantiles : sequence of float
        Default non-exceedance-probability levels.
    band_width : float
        Half-width of the exceedance window used to select models per band.
    min_models : int
        Minimum models required to populate a band.
    force_positive : bool
        Clip predictions at zero.
    """

    def __init__(self, model, params, exceedances, x_scaler=None, y_scaler=None,
                 quantiles=DEFAULT_QUANTILES, band_width=0.025, min_models=1,
                 force_positive=False):
        self.model = model
        self.params = np.asarray(params, dtype=float)
        self.exceedances = np.asarray(exceedances, dtype=float)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.quantiles = list(quantiles)
        self.band_width = band_width
        self.min_models = min_models
        self.force_positive = force_positive
        self.band_bounds = band_bounds(self.quantiles, band_width)
        self.band_probabilities = self.band_bounds[1] - self.band_bounds[0]

    @property
    def n_models(self):
        return self.params.shape[0]

    def _simulate(self, X):
        """Run every retained model on ``X``; return ``(sims, valid_mask)``."""
        X = np.asarray(X, dtype=float)
        valid = np.isfinite(X).all(axis=1)
        Xv = X[valid]
        Xn = self.x_scaler.transform(Xv) if self.x_scaler is not None else Xv
        raw = self.model.forward(Xn, self.params)
        if self.y_scaler is not None:
            sims = raw * self.y_scaler.scale_[0] + self.y_scaler.mean_[0]
        else:
            sims = raw
        return sims, valid

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
