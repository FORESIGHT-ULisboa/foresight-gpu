"""Feature-engineering transformer tests + Pipeline integration."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from foresight_gpu import GPURegressor
from foresight_gpu.utils import LagFeatures, OudinPET, PeriodicFeatures, RollingSum


class TestPeriodicFeatures:
    def test_sin_cos_values(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        out = PeriodicFeatures(column=0, period=4, drop=True).fit_transform(X)
        assert out.shape == (4, 2)
        np.testing.assert_allclose(out[:, 0], [0, 1, 0, -1], atol=1e-12)
        np.testing.assert_allclose(out[:, 1], [1, 0, -1, 0], atol=1e-12)

    def test_keep_original(self):
        X = np.array([[0.0, 9.0], [1.0, 8.0]])
        out = PeriodicFeatures(column=0, period=4, drop=False).fit_transform(X)
        assert out.shape == (2, 4)  # original 2 + sin + cos


class TestLagFeatures:
    def test_lagging(self):
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        out = LagFeatures(column=0, lags=(1, 2)).fit_transform(X)
        assert out.shape == (4, 3)
        assert np.isnan(out[0, 1]) and np.isnan(out[0, 2]) and np.isnan(out[1, 2])
        np.testing.assert_allclose(out[2:, 1], [20.0, 30.0])  # lag 1
        np.testing.assert_allclose(out[2:, 2], [10.0, 20.0])  # lag 2


class TestRollingSum:
    def test_window(self):
        X = np.ones((5, 1))
        out = RollingSum(column=0, window=3).fit_transform(X)
        assert np.isnan(out[0, 1]) and np.isnan(out[1, 1])
        np.testing.assert_allclose(out[2:, 1], [3.0, 3.0, 3.0])


class TestOudinPET:
    def test_nonnegative_and_cold_is_zero(self):
        doy = np.arange(1, 13) * 30.0
        temp = np.linspace(-10, 25, 12)
        X = np.column_stack([temp, doy])
        out = OudinPET(temp_column=0, doy_column=1, latitude=-15.0).fit_transform(X)
        pet = out[:, -1]
        assert out.shape == (12, 3)
        assert np.all(pet >= 0)
        assert pet[0] == 0.0  # T = -10 <= -5


class TestPipeline:
    def test_periodic_then_regressor(self, rng):
        doy = rng.integers(1, 366, size=250).astype(float)
        drivers = rng.uniform(-1, 1, size=(250, 2))
        X = np.column_stack([doy, drivers])
        y = np.sin(2 * np.pi * doy / 365.25) + 0.3 * drivers[:, 0] + 0.2 * rng.standard_normal(250)
        pipe = Pipeline(
            [("periodic", PeriodicFeatures(column=0, period=365.25, drop=True)),
             ("gpu", GPURegressor(population=100, n_iter=20, random_state=0))]
        )
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (250,)
        assert np.all(np.isfinite(pred))
