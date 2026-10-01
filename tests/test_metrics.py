"""Metric tests: vectorisation, loss orientation, and equivalence with forecast_performance."""

import numpy as np
import pandas as pd
import pytest

from foresight_gpu.metrics import (
    Metric,
    get_metric,
    kge,
    kge_prime,
    mae,
    mse,
    nse,
    rmse,
)
from foresight_gpu.metrics.exceedance import non_exceedance
from foresight_gpu.metrics.regularization import lp_penalty


class TestVectorisation:
    def test_shapes(self, population_simulations):
        sim, obs = population_simulations
        for metric in (nse, kge, kge_prime, mae, mse, rmse):
            out = metric(sim, obs)
            assert out.shape == (sim.shape[1],)

    def test_vectorised_matches_columnwise(self, population_simulations):
        sim, obs = population_simulations
        for metric in (nse, kge, kge_prime, mae, mse, rmse):
            vec = metric(sim, obs)
            col = np.array([metric(sim[:, j], obs) for j in range(sim.shape[1])])
            np.testing.assert_allclose(vec, col, rtol=1e-10)

    def test_scalar_when_1d(self, population_simulations):
        sim, obs = population_simulations
        assert np.isscalar(nse(sim[:, 0], obs))


class TestForecastPerformanceEquivalence:
    """Our fast routines must agree numerically with the reference package."""

    def test_columnwise_equivalence(self, population_simulations):
        import forecast_performance.metrics.deterministic as fp
        pairs = [(nse, fp.nse), (kge, fp.kge), (kge_prime, fp.kge_prime),
                 (mae, fp.mae), (mse, fp.mse), (rmse, fp.rmse)]
        sim, obs = population_simulations
        obs_s = pd.Series(obs)
        for ours, theirs in pairs:
            for j in range(0, sim.shape[1], 5):
                mine = ours(sim[:, j], obs)
                ref = theirs(pd.Series(sim[:, j]), obs_s)
                np.testing.assert_allclose(mine, ref, rtol=1e-9, atol=1e-12)


class TestLossOrientation:
    def test_efficiency_metrics_flip(self, population_simulations):
        sim, obs = population_simulations
        for metric in (nse, kge, kge_prime):
            np.testing.assert_allclose(metric.loss(sim, obs), 1.0 - metric(sim, obs))

    def test_error_metrics_pass_through(self, population_simulations):
        sim, obs = population_simulations
        for metric in (mae, mse, rmse):
            np.testing.assert_allclose(metric.loss(sim, obs), metric(sim, obs))

    def test_perfect_forecast_zero_loss(self, population_simulations):
        _, obs = population_simulations
        perfect = obs[:, None]
        for metric in (nse, kge, kge_prime, mae, mse, rmse):
            np.testing.assert_allclose(metric.loss(perfect, obs), 0.0, atol=1e-9)


class TestMetricObject:
    def test_stringifies_to_name(self):
        assert str(nse) == "nse" == nse
        assert nse.__name__ == "nse"

    def test_registry_resolution(self):
        assert get_metric("NSE") is nse
        assert get_metric("kgeprime") is kge_prime
        assert get_metric(rmse) is rmse

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            get_metric("not_a_metric")

    def test_callable_and_greater_is_better(self):
        assert nse.greater_is_better is True
        assert mae.greater_is_better is False
        assert isinstance(nse, Metric)


class TestNonExceedance:
    def test_all_above(self, population_simulations):
        _, obs = population_simulations
        sim = obs[:, None] + 1.0
        assert np.isclose(non_exceedance(sim, obs), 1.0)

    def test_all_below(self, population_simulations):
        _, obs = population_simulations
        sim = obs[:, None] - 1.0
        assert np.isclose(non_exceedance(sim, obs), 0.0)

    def test_vectorised(self, population_simulations):
        sim, obs = population_simulations
        eta = non_exceedance(sim, obs)
        assert eta.shape == (sim.shape[1],)
        assert np.all((eta >= 0) & (eta <= 1))


class TestRegularization:
    def test_l1_l2(self, rng):
        params = rng.normal(size=(10, 6))
        mask = np.array([True, True, False, False, True, True])
        l1 = lp_penalty(params, mask, lam=0.5, p=1)
        l2 = lp_penalty(params, mask, lam=0.5, p=2)
        np.testing.assert_allclose(l1, 0.5 * np.sum(np.abs(params[:, mask]), axis=1))
        np.testing.assert_allclose(l2, 0.5 * np.sum(params[:, mask] ** 2, axis=1))

    def test_disabled(self, rng):
        params = rng.normal(size=(4, 6))
        assert lp_penalty(params, None, lam=0.0) == 0.0
        assert lp_penalty(params, None, lam=None) == 0.0


class TestEdgeCases:
    def test_nse_zero_variance_reference_is_nan(self):
        obs = np.full(50, 3.0)
        sim = np.linspace(1, 5, 50)
        assert np.isnan(nse(sim, obs))
