"""Feature-engineering transformers (helpers).

Small, Pipeline-compatible scikit-learn transformers for the kind of inputs GPU forecasts
often use — periodic day-of-year encodings, lagged observations, rolling sums, and an
Oudin PET estimate for the hydrological model. They operate on plain arrays (row order =
time order) so they compose in a :class:`~sklearn.pipeline.Pipeline`; rows they cannot fill
(early lags/windows) become NaN and are dropped by the estimator's screening step.

These are helpers, kept out of the core so ``GPURegressor`` stays domain-agnostic.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PeriodicFeatures(TransformerMixin, BaseEstimator):
    """Encode a cyclic column (e.g. day-of-year) as sin/cos pair (WRR Eq. 6).

    Parameters
    ----------
    column : int
        Index of the cyclic column.
    period : float
        Cycle length in the same units as the column (e.g. 365.25 for day-of-year).
    drop : bool
        Drop the original cyclic column from the output.
    """

    def __init__(self, column=0, period=365.25, drop=True):
        self.column = column
        self.period = period
        self.drop = drop

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        v = X[:, self.column]
        pair = np.column_stack(
            [np.sin(2 * np.pi * v / self.period), np.cos(2 * np.pi * v / self.period)]
        )
        if self.drop:
            keep = [i for i in range(X.shape[1]) if i != self.column]
            base = X[:, keep] if keep else np.empty((X.shape[0], 0))
            return np.column_stack([base, pair])
        return np.column_stack([X, pair])


class LagFeatures(TransformerMixin, BaseEstimator):
    """Append lagged copies of a column (row order assumed chronological).

    Parameters
    ----------
    column : int
        Column to lag.
    lags : sequence of int
        Positive lags to append (``lag`` rows back). Early rows become NaN.
    """

    def __init__(self, column=0, lags=(1,)):
        self.column = column
        self.lags = lags

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        col = X[:, self.column]
        extra = []
        for lag in self.lags:
            shifted = np.full(col.shape[0], np.nan)
            if lag < col.shape[0]:
                shifted[lag:] = col[:-lag] if lag > 0 else col
            extra.append(shifted)
        return np.column_stack([X] + extra)


class RollingSum(TransformerMixin, BaseEstimator):
    """Append a trailing rolling sum of a column.

    Parameters
    ----------
    column : int
        Column to accumulate.
    window : int
        Window length; the first ``window - 1`` rows become NaN.
    """

    def __init__(self, column=0, window=10):
        self.column = column
        self.window = window

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        col = X[:, self.column]
        rolled = np.convolve(col, np.ones(self.window), mode="full")[: col.shape[0]]
        rolled[: self.window - 1] = np.nan
        return np.column_stack([X, rolled])


class OudinPET(TransformerMixin, BaseEstimator):
    """Append potential evapotranspiration via the Oudin (2005) temperature model.

    Parameters
    ----------
    temp_column : int
        Column of mean air temperature [degC].
    doy_column : int
        Column of day-of-year (1-365/366).
    latitude : float
        Catchment latitude [degrees].
    append : bool
        Append PET (``True``) or return only the PET column.

    Notes
    -----
    Extraterrestrial radiation follows the standard FAO-56 equations; PET
    ``= 0.408 * Ra * (T + 5) / 100`` for ``T > -5`` else 0.
    """

    def __init__(self, temp_column=0, doy_column=1, latitude=0.0, append=True):
        self.temp_column = temp_column
        self.doy_column = doy_column
        self.latitude = latitude
        self.append = append

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        temp = X[:, self.temp_column]
        doy = X[:, self.doy_column]
        phi = np.radians(self.latitude)

        angle = 2 * np.pi * doy / 365.0
        dr = 1 + 0.033 * np.cos(angle)
        decl = 0.409 * np.sin(angle - 1.39)
        ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))
        gsc = 0.0820  # MJ / m^2 / min
        ra = (24 * 60 / np.pi) * gsc * dr * (
            ws * np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.sin(ws)
        )
        ra_mm = 0.408 * ra  # water-equivalent [mm/day]
        pet = np.where(temp > -5.0, ra_mm * (temp + 5.0) / 100.0, 0.0)
        pet = np.maximum(pet, 0.0)
        return np.column_stack([X, pet]) if self.append else pet[:, None]
