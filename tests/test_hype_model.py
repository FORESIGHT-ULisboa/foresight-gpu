"""HYPEModel: the forward-model contract, X-as-dates, caching, sklearn plumbing."""

import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone

from foresight_gpu.models import BaseForwardModel
from foresight_gpu.models.hype import HYPEModel, as_X, freeze, load_observations
from foresight_gpu.models.hype.cache import SeriesCache


def _params(model, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return model.search_transform(rng.uniform(0, 1, (n, model.n_parameters(1))))


class TestContract:
    def test_is_a_forward_model_with_no_fit(self, hype_model):
        assert isinstance(hype_model, BaseForwardModel)
        assert not hasattr(hype_model, "fit")

    def test_parameter_count_is_independent_of_n_features(self, hype_model):
        assert hype_model.n_parameters(1) == hype_model.n_parameters(7)

    def test_bounds_shape_and_ordering(self, hype_model):
        low, high = hype_model.parameter_bounds(1)
        assert low.shape == high.shape == (hype_model.n_parameters(1),)
        assert np.all(high > low)

    def test_no_regularisation_on_physical_parameters(self, hype_model):
        mask = hype_model.regularizable_mask(1)
        assert mask.dtype == bool and not mask.any()

    def test_forward_shape(self, hype_model):
        X = as_X(hype_model.dates_[:40])
        assert hype_model.forward(X, _params(hype_model, 4)).shape == (40, 4)

    def test_single_particle_gives_one_column(self, hype_model):
        X = as_X(hype_model.dates_[:10])
        row = _params(hype_model, 1)[0]
        assert hype_model.forward(X, row).shape == (10, 1)

    def test_wrong_parameter_count_raises(self, hype_model):
        X = as_X(hype_model.dates_[:5])
        with pytest.raises(ValueError, match="expected"):
            hype_model.forward(X, np.zeros((2, hype_model.n_parameters(1) + 1)))

    def test_parameters_change_the_output(self, hype_model):
        X = as_X(hype_model.dates_[:60])
        out = hype_model.forward(X, _params(hype_model, 3))
        assert not np.allclose(out[:, 0], out[:, 1])

    def test_output_is_finite_and_physical(self, hype_model):
        X = as_X(hype_model.dates_[:60])
        out = hype_model.forward(X, _params(hype_model, 3))
        assert np.isfinite(out).all() and (out >= 0).all()


class TestDatesAsX:
    def test_rows_follow_the_requested_dates(self, hype_model):
        dates = hype_model.dates_
        full = hype_model.forward(as_X(dates[:100]), _params(hype_model, 1))
        picked = np.array([90, 3, 3, 41])  # out of order, with a duplicate
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*chronological order.*")
            subset = hype_model.forward(as_X(dates[picked]), _params(hype_model, 1))
        np.testing.assert_array_equal(subset[:, 0], full[picked, 0])

    def test_gaps_are_fine(self, hype_model):
        """Missing observations are dropped by the engine; the dates keep the alignment."""
        dates = hype_model.dates_[:200:7]
        out = hype_model.forward(as_X(dates), _params(hype_model, 2))
        assert out.shape == (dates.size, 2)

    def test_extra_feature_columns_are_ignored(self, hype_model):
        X = as_X(hype_model.dates_[:20])
        padded = np.hstack([X, np.random.default_rng(0).normal(size=(20, 3))])
        np.testing.assert_array_equal(
            hype_model.forward(X, _params(hype_model, 1)),
            hype_model.forward(padded, _params(hype_model, 1)),
        )

    def test_date_column_can_be_moved(self, hype_model, hype_template, stub_command,
                                      tmp_path):
        X = as_X(hype_model.dates_[:20])
        shifted = np.hstack([np.zeros((20, 1)), X])
        other = HYPEModel(
            template_dir=hype_template, subbasin=1234, executable=stub_command,
            parameters=["wcfc", "rrcs1", "cmlt"], date_column=1,
            work_root=tmp_path / "moved", warn_unrequested=False,
        )
        try:
            np.testing.assert_array_equal(
                hype_model.forward(X, _params(hype_model, 1)),
                other.forward(shifted, _params(other, 1)),
            )
        finally:
            other.close()

    def test_scaled_dates_raise_rather_than_returning_nonsense(self, hype_model):
        from sklearn.preprocessing import StandardScaler

        X = as_X(hype_model.dates_[:50])
        scaled = StandardScaler().fit_transform(X)
        with pytest.raises(ValueError, match="StandardScaler"):
            hype_model.forward(scaled, _params(hype_model, 1))

    def test_out_of_window_dates_raise(self, hype_model):
        X = as_X(hype_model.dates_[:5]) - 5000
        with pytest.raises(ValueError, match="outside the simulated window"):
            hype_model.forward(X, _params(hype_model, 1))

    def test_unordered_dates_warn_once_about_shuffle(self, hype_model):
        X = as_X(hype_model.dates_[:40][::-1])
        with pytest.warns(UserWarning, match="chronological order"):
            hype_model.forward(X, _params(hype_model, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a second call must not warn again
            hype_model.forward(X, _params(hype_model, 1))

    def test_align_trims_to_the_window(self, hype_model):
        X, y = load_observations(hype_model.template_dir / "Qobs.txt", column="1234")
        Xa, ya = hype_model.align(X, y)
        assert Xa.shape[0] == ya.size == hype_model.output_window_[1]
        assert Xa.shape[0] < X.shape[0]  # the file spans the warmup year too

    def test_dates_exclude_the_warmup_span(self, hype_model):
        bdate, cdate, edate = hype_model.simulation_window_
        assert str(hype_model.dates_[0]) == cdate
        assert str(hype_model.dates_[-1]) == edate
        assert np.datetime64(bdate) < np.datetime64(cdate)


class TestCache:
    def test_repeat_evaluation_is_free(self, hype_model):
        X = as_X(hype_model.dates_[:30])
        params = _params(hype_model, 4)
        first = hype_model.forward(X, params)
        runs = hype_model.n_runs_
        second = hype_model.forward(X, params)
        assert hype_model.n_runs_ == runs
        np.testing.assert_array_equal(first, second)

    def test_a_different_window_costs_nothing(self, hype_model):
        """The whole window is simulated at once, so re-slicing is free."""
        params = _params(hype_model, 3)
        hype_model.forward(as_X(hype_model.dates_[:30]), params)
        runs = hype_model.n_runs_
        hype_model.forward(as_X(hype_model.dates_[100:200]), params)
        assert hype_model.n_runs_ == runs

    def test_duplicate_rows_are_deduplicated(self, hype_model):
        X = as_X(hype_model.dates_[:20])
        one = _params(hype_model, 1)
        params = np.repeat(one, 5, axis=0)
        out = hype_model.forward(X, params)
        assert hype_model.n_runs_ == 1
        assert out.shape == (20, 5)
        np.testing.assert_array_equal(out[:, 0], out[:, 4])

    def test_disabled_cache_gives_identical_values(self, hype_template, stub_command,
                                                   tmp_path):
        common = dict(template_dir=hype_template, subbasin=1234, executable=stub_command,
                      parameters=["wcfc", "rrcs1", "cmlt"], warn_unrequested=False)
        cached = HYPEModel(work_root=tmp_path / "a", cache_size=64, **common)
        uncached = HYPEModel(work_root=tmp_path / "b", cache_size=0, **common)
        try:
            X = as_X(cached.dates_[:30])
            params = _params(cached, 3)
            np.testing.assert_array_equal(
                cached.forward(X, params), uncached.forward(X, params)
            )
            cached.forward(X, params)
            uncached.forward(X, params)
            assert cached.n_runs_ == 3 and uncached.n_runs_ == 6
        finally:
            cached.close()
            uncached.close()

    def test_fingerprint_isolates_configurations(self, hype_template, stub_command,
                                                 tmp_path):
        a = SeriesCache(8, "fingerprint-a")
        b = SeriesCache(8, "fingerprint-b")
        row = np.array([1.0, 2.0])
        a.put(a.key(row), np.arange(4.0))
        assert b.get(b.key(row)) is None

    def test_eviction_respects_capacity(self):
        cache = SeriesCache(2, "f")
        for i in range(5):
            cache.put(cache.key(np.array([float(i)])), np.zeros(3))
        assert len(cache) <= 2

    def test_pinning_protects_a_validation_batch(self):
        cache = SeriesCache(4, "f")
        keys = [cache.key(np.array([float(i)])) for i in range(3)]
        for key in keys:
            cache.put(key, np.zeros(3))
        cache.pin(keys)
        for i in range(3, 12):  # candidate churn
            cache.put(cache.key(np.array([float(i)])), np.zeros(3))
        assert all(cache.get(key) is not None for key in keys)

    def test_minus_zero_does_not_split_a_key(self):
        cache = SeriesCache(4, "f")
        assert cache.key(np.array([0.0])) == cache.key(np.array([-0.0]))


class TestSklearnPlumbing:
    def test_get_set_params(self, hype_model):
        assert hype_model.get_params()["subbasin"] == 1234
        hype_model.set_params(verbose=1)
        assert hype_model.verbose == 1

    def test_constructor_does_not_transform_its_arguments(self, hype_model):
        """The invariant ``clone`` enforces: rebuilding from get_params keeps identity.

        ``clone`` itself deep-copies, so it is the *constructor* that must not normalise -
        a ``list(parameters)`` or ``dict(model_options)`` in ``__init__`` would make clone
        raise "constructor either does not set or modifies parameter".
        """
        params = hype_model.get_params(deep=False)
        rebuilt = type(hype_model)(**params).get_params(deep=False)
        for name, value in params.items():
            assert rebuilt[name] is value, f"__init__ transformed {name!r}"

    def test_clone_succeeds_and_preserves_values(self, hype_model):
        copied = clone(hype_model)
        assert copied.parameters == hype_model.parameters
        assert str(copied.template_dir) == str(hype_model.template_dir)

    def test_clone_with_dict_options(self, hype_template, stub_command):
        model = HYPEModel(template_dir=hype_template, executable=stub_command,
                          model_options={"snowmeltmodel": 2}, parameters=["cmlt"],
                          warn_unrequested=False)
        assert clone(model).model_options == {"snowmeltmodel": 2}

    def test_a_clone_owns_no_workspace(self, hype_model):
        hype_model.forward(as_X(hype_model.dates_[:5]), _params(hype_model, 1))
        copied = clone(hype_model)
        assert getattr(copied, "_runner", None) is None
        assert copied.n_runs_ == 0

    def test_nested_params_route_through_the_estimator(self, hype_model):
        from foresight_gpu import GPURegressor

        gpu = GPURegressor(model=hype_model)
        assert gpu.get_params()["model__subbasin"] == 1234
        gpu.set_params(model__output_variable="crun")
        assert gpu.model.output_variable == "crun"

    def test_pickle_drops_the_pool_and_still_predicts(self, hype_model):
        X = as_X(hype_model.dates_[:20])
        params = _params(hype_model, 2)
        expected = hype_model.forward(X, params)
        blob = pickle.dumps(hype_model)
        assert b"multiprocessing" not in blob
        restored = pickle.loads(blob)
        try:
            assert restored._runner is None and restored._cache is None
            np.testing.assert_array_equal(restored.forward(X, params), expected)
        finally:
            restored.close()

    def test_context_manager_closes(self, hype_template, stub_command, tmp_path):
        with HYPEModel(template_dir=hype_template, subbasin=1234,
                       executable=stub_command, parameters=["cmlt"],
                       work_root=tmp_path / "ctx", warn_unrequested=False) as model:
            model.forward(as_X(model.dates_[:5]), _params(model, 1))
            root = model._runner.spec.work_root
        from pathlib import Path

        assert not Path(root).exists()


class TestIntrospection:
    def test_names_match_the_dimension_count(self, hype_model):
        assert len(hype_model.parameter_names_) == hype_model.n_parameters(1)

    def test_active_and_dropped_are_reported(self, hype_template, stub_command):
        with pytest.warns(UserWarning):
            model = HYPEModel(template_dir=hype_template, subbasin=1234,
                              executable=stub_command,
                              parameters=["cmlt", "snalbmin"], warn_unrequested=False)
            assert model.active_parameters_ == ("cmlt",)
            assert "snalbmin" in model.dropped_parameters_

    def test_model_options_reflect_overrides(self, hype_template, stub_command):
        model = HYPEModel(template_dir=hype_template, executable=stub_command,
                          parameters=["cmlt"], model_options={"snowmeltmodel": 2},
                          warn_unrequested=False)
        assert model.model_options_["snowmeltmodel"] == 2
        assert model.model_options_["petmodel"] == 2  # inherited from the template

    def test_routine_toggle_changes_the_parameter_set(self, hype_template, stub_command):
        common = dict(template_dir=hype_template, executable=stub_command,
                      parameters=["cmlt", "snalbmin", "cmrad"], warn_unrequested=False)
        with pytest.warns(UserWarning, match="needs snowmeltmodel=2"):
            off = HYPEModel(**common).active_parameters_
        on = HYPEModel(model_options={"snowmeltmodel": 2}, **common).active_parameters_
        assert off == ("cmlt",)
        assert on == ("cmlt", "snalbmin", "cmrad")

    def test_describe_lists_dimensions(self, hype_model):
        pytest.importorskip("pandas")
        table = hype_model.describe()
        assert list(table["label"]) == hype_model.parameter_names_

    def test_describe_with_a_population_gives_quantiles(self, hype_model):
        pytest.importorskip("pandas")
        table = hype_model.describe(_params(hype_model, 8))
        assert {"p10", "median", "p90"} <= set(table.columns)

    def test_write_par_emits_a_usable_file(self, hype_model, tmp_path):
        from foresight_gpu.models.hype.files import ParFile

        out = hype_model.write_par(tmp_path / "par.txt", _params(hype_model, 1)[0])
        written = ParFile.read(out)
        assert set(hype_model.active_parameters_) <= set(written.values)


class TestFreeze:
    def test_frozen_ensemble_predicts_without_the_template(self, hype_model,
                                                           hype_observations, tmp_path):
        from foresight_gpu import GPURegressor

        X, y = hype_observations
        gpu = GPURegressor(model=hype_model, metric="mae", population=4, n_iter=1,
                           random_state=0).fit(X, y)
        expected = gpu.predict_quantiles(X)
        frozen = freeze(gpu.ensemble_)

        moved = hype_model.template_dir.rename(
            hype_model.template_dir.parent / "template_moved"
        )
        try:
            np.testing.assert_allclose(frozen.predict_quantiles(X), expected,
                                       equal_nan=True)
            blob = pickle.dumps(frozen)
            np.testing.assert_allclose(
                pickle.loads(blob).predict_quantiles(X), expected, equal_nan=True
            )
        finally:
            moved.rename(hype_model.template_dir)

    def test_freeze_rejects_a_non_hype_ensemble(self):
        from foresight_gpu.ensemble import ParetoEnsemble
        from foresight_gpu.models import MLPModel

        ensemble = ParetoEnsemble(MLPModel(), np.zeros((2, 25)), np.array([0.2, 0.8]))
        with pytest.raises(TypeError, match="HYPEModel"):
            freeze(ensemble)
