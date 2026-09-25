"""ParetoEnsemble: band aggregation (Eq. 4), monotonicity, inverse-CDF, pickling."""

import pickle

import numpy as np
import pytest

from foresight_gpu import GPURegressor
from foresight_gpu.domination import HV_CLIP_WARN_FRACTION
from foresight_gpu.ensemble import ParetoEnsemble, band_bounds
from foresight_gpu.models import BaseForwardModel
from foresight_gpu.scoring import score_ensemble


class _ConstModel(BaseForwardModel):
    """Each model outputs a constant equal to its single parameter."""

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


class TestFrontHypervolume:
    """The front indicator lives on the ensemble, so any fitted front can be re-read."""

    def test_stored_metric_is_a_default_not_a_lock(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        ens = gpu.ensemble_
        assert ens.metric == "nse"
        assert ens.score_hypervolume(X, y) == pytest.approx(
            ens.score_hypervolume(X, y, "nse")
        )
        # the point of the parameter: the same front, read under another metric
        under_mae = ens.score_hypervolume(X, y, "mae")
        assert 0.0 <= under_mae <= 1.0
        assert under_mae != pytest.approx(ens.score_hypervolume(X, y))

    def test_post_hoc_metrics_all_score(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        scores = {
            m: gpu.ensemble_.score_hypervolume(X, y, m) for m in ("nse", "kge", "mae")
        }
        assert all(np.isfinite(v) and 0.0 <= v <= 1.0 for v in scores.values())
        assert len(set(np.round(list(scores.values()), 9))) == len(scores)

    def test_missing_metric_raises(self):
        ens = ParetoEnsemble(
            _ConstModel(), np.array([[1.0], [2.0]]), np.array([0.1, 0.9])
        )
        with pytest.raises(ValueError, match="no metric"):
            ens.score_hypervolume(np.zeros((5, 1)), np.zeros(5))

    def test_front_objectives_is_what_the_score_integrates(self, fitted_ensemble):
        """What the notebook plots must be the arithmetic the indicator used."""
        from foresight_gpu import double_pareto_hypervolume

        gpu, X, y = fitted_ensemble
        ens = gpu.ensemble_
        eta, loss, front = ens.front_objectives(X, y)
        objectives = np.column_stack([eta, loss])
        direct = double_pareto_hypervolume(objectives, ens.hv_penalty, front=front)
        assert direct == pytest.approx(ens.score_hypervolume(X, y))
        # the precomputed front is the one the indicator would have found itself
        assert double_pareto_hypervolume(
            objectives, ens.hv_penalty
        ) == pytest.approx(direct)

    def test_front_objectives_drops_nan_y(self, fitted_ensemble):
        """_simulate masks X only; a NaN in y would otherwise pin hv at 0 silently."""
        gpu, X, y = fitted_ensemble
        holed = y.copy()
        holed[3] = np.nan
        eta, loss, _ = gpu.ensemble_.front_objectives(X, holed)
        assert np.isfinite(loss).any() and np.all((eta >= 0.0) & (eta <= 1.0))
        assert np.isfinite(gpu.ensemble_.score_hypervolume(X, holed))

    def test_warns_only_when_the_whole_front_is_clipped(self, fitted_ensemble):
        """P=100 suits NSE-scale losses and is far too small for thousand-scale MAE."""
        import warnings as _w

        gpu, X, y = fitted_ensemble
        with _w.catch_warnings():  # healthy run: tail particles above P are normal
            _w.simplefilter("error")
            gpu.ensemble_.score_hypervolume(X, y)

        ens = ParetoEnsemble(
            _ConstModel(), np.array([[0.0], [1e6]]), np.array([0.05, 0.95]),
            metric="mae", hv_penalty=100.0,
        )
        big = np.arange(50, dtype=float) * 1000.0
        with pytest.warns(UserWarning, match="pinned at 0"):
            assert ens.score_hypervolume(np.zeros((50, 1)), big) == pytest.approx(0.0)
        with _w.catch_warnings():
            _w.simplefilter("error")
            assert ens.score_hypervolume(np.zeros((50, 1)), big, penalty=1e7) > 0.0

    def test_warns_when_the_ceiling_clips_much_of_the_front(self, fitted_ensemble):
        """The diagnostic rung: a too-tight P is otherwise invisible.

        Clipping is correct behaviour -- there is no gradient among models you would never
        use -- so the point is not to change the indicator but to say when it has stopped
        discriminating. Measured at the NSE climatology ceiling P=1: 0.79 of front-0.
        """
        import warnings as _w

        gpu, X, y = fitted_ensemble
        with pytest.warns(UserWarning, match="clips .* of front-0"):
            parts = gpu.ensemble_.score_hypervolume(X, y, penalty=1.0, details=True)
        assert parts["clipped_fraction"] > HV_CLIP_WARN_FRACTION

        with _w.catch_warnings():  # the generous default clips nothing here
            _w.simplefilter("error")
            loose = gpu.ensemble_.score_hypervolume(X, y, details=True)
        assert loose["clipped_fraction"] <= HV_CLIP_WARN_FRACTION
        assert loose["hv"] > parts["hv"]

    def test_clipped_fraction_is_reported(self, fitted_ensemble):
        gpu, X, y = fitted_ensemble
        parts = gpu.ensemble_.score_hypervolume(X, y, details=True)
        assert 0.0 <= parts["clipped_fraction"] <= 1.0
        assert parts["space"] == "linear"

    def test_log10_space_round_trips_through_the_ensemble(self, fitted_ensemble):
        """hv_space is carried like metric and hv_penalty, and is overridable per call."""
        gpu, X, y = fitted_ensemble
        ens = gpu.ensemble_
        assert ens.hv_space == "linear"

        direct = ens.score_hypervolume(X, y, penalty=2.0, space="log10", details=True)
        assert direct["space"] == "log10"
        assert 0.0 <= direct["hv"] <= 1.0

        # a copy: the fixture is module-scoped, so never mutate it in place
        carried = pickle.loads(pickle.dumps(ens))
        carried.hv_space, carried.hv_penalty = "log10", 2.0
        assert carried.score_hypervolume(X, y) == pytest.approx(direct["hv"])
        assert score_ensemble(carried, X, y, "hypervolume") == pytest.approx(direct["hv"])

    def test_old_pickles_without_hv_space_still_score(self, fitted_ensemble):
        """getattr fallbacks: an ensemble pickled before hv_space existed must still work."""
        gpu, X, y = fitted_ensemble
        ens = pickle.loads(pickle.dumps(gpu.ensemble_))
        del ens.hv_space
        assert ens.score_hypervolume(X, y) == pytest.approx(
            gpu.ensemble_.score_hypervolume(X, y)
        )

    def test_metric_survives_pickling(self, fitted_ensemble):
        """Metric reduces to a plain str, re-resolved with get_metric on use."""
        gpu, X, y = fitted_ensemble
        restored = pickle.loads(pickle.dumps(gpu.ensemble_))
        assert restored.metric == gpu.ensemble_.metric
        assert restored.score_hypervolume(X, y) == pytest.approx(
            gpu.ensemble_.score_hypervolume(X, y)
        )
