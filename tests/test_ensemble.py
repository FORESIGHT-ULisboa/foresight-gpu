"""ParetoEnsemble: band aggregation (Eq. 4), monotonicity, inverse-CDF, pickling."""

import pickle

import numpy as np
import pytest

from foresight_gpu import GPURegressor
from foresight_gpu.ensemble import ParetoEnsemble, band_bounds
from foresight_gpu.models import BaseForwardModel


class _ConstModel(BaseForwardModel):
    """Each model outputs a constant equal to its single parameter."""

    scales_inputs = False
    scales_outputs = False

    def n_parameters(self, n_features):
        return 1

    def forward(self, X, params):
        params = np.atleast_2d(params)
        return np.repeat(params[:, 0][None, :], X.shape[0], axis=0)


class TestAggregation:
    def test_band_selection_and_median(self):
        # models with values 1,2,3,4 at exceedances 0.05, 0.5, 0.5, 0.95
        ens = ParetoEnsemble(
            _ConstModel(),
            params=np.array([[1.0], [2.0], [3.0], [4.0]]),
            exceedances=np.array([0.05, 0.5, 0.5, 0.95]),
            quantiles=[0.05, 0.5, 0.95],
            band_width=0.1,
            min_models=1,
        )
        agg = ens.predict_quantiles(np.zeros((1, 1)), post_process=False)
        np.testing.assert_allclose(agg[0], [1.0, 2.5, 4.0])  # median of {2,3} = 2.5

    def test_min_models_leaves_nan(self):
        ens = ParetoEnsemble(
            _ConstModel(),
            params=np.array([[1.0], [4.0]]),
            exceedances=np.array([0.05, 0.95]),
            quantiles=[0.05, 0.5, 0.95],
            band_width=0.02,
            min_models=1,
        )
        agg = ens.predict_quantiles(np.zeros((1, 1)), post_process=False)
        assert np.isnan(agg[0, 1])  # nothing near 0.5

    def test_band_bounds_shapes(self):
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        bounds = band_bounds(q, 0.025)
        assert bounds.shape == (2, len(q))
        assert np.all(bounds[0] <= q)
        assert np.all(bounds[1] >= q)


@pytest.fixture(scope="module")
def fitted_ensemble():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.4 * rng.standard_normal(300)
    gpu = GPURegressor(population=200, n_iter=40, random_state=0).fit(X, y)
    return gpu, X, y


class TestFittedEnsemble:
    def test_quantiles_monotone(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        bands = gpu.predict_quantiles(X)
        diffs = np.diff(bands, axis=1)
        assert np.nanmin(diffs) >= -1e-8  # non-decreasing across quantiles

    def test_inverse_cdf_finite_and_increasing_on_average(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        cols = np.column_stack([gpu.ensemble_.inverse_cdf(X, p) for p in levels])
        assert np.all(np.isfinite(cols))
        # per-level medians need not be monotone sample-by-sample, but the mean band
        # rises with the probability level (strict monotonicity is post_process's job).
        assert np.all(np.diff(cols.mean(axis=0)) >= -1e-6)

    def test_pickle_roundtrip(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        restored = pickle.loads(pickle.dumps(gpu.ensemble_))
        np.testing.assert_allclose(restored.predict(X), gpu.ensemble_.predict(X))

    def test_n_models(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        assert gpu.ensemble_.n_models == 200
