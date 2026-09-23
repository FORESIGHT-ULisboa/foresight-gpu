"""HYPEModel driven by the engine: fit, early stopping, cross-validation, run accounting.

The stub stands in for HYPE, so these run in seconds while exercising the real code path.
The tests marked ``real_hype_folder`` need ``FORESIGHT_HYPE_FOLDER`` set and are skipped
otherwise.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

from foresight_gpu import GPURegressor
from foresight_gpu.models.hype import HYPEModel, as_X, load_observations


#: A budget small enough to keep the stub suite quick, yet well posed enough to populate
#: bands. ``screen=True`` matters: the search box is wide and log-scaled, so a purely random
#: initial swarm clusters at one exceedance extreme and no band would be filled. Wide bands
#: and few quantiles then let a handful of particles cover the distribution.
SMALL_FIT = dict(metric="mae", population=8, n_iter=2, screen=True,
                 screen_oversample=3, band_width=0.15, quantiles=[0.1, 0.5, 0.9],
                 force_positive=True, random_state=0)


def _fit(model, X, y, **kw):
    options = dict(SMALL_FIT)
    options.update(kw)
    return GPURegressor(model=model, **options).fit(X, y)


def _inner(gpu):
    """The model that actually ran: ``fit`` works on a clone, not the instance passed in."""
    return gpu.model_


class TestFit:
    def test_fit_predict_and_score(self, hype_model, hype_observations):
        X, y = hype_observations
        gpu = _fit(hype_model, X, y)
        assert gpu.predict(X).shape == (X.shape[0],)
        assert np.isfinite(gpu.predict(X)).all()
        assert np.isfinite(gpu.score(X, y))

    def test_bands_are_produced(self, hype_model, hype_observations):
        X, y = hype_observations
        gpu = _fit(hype_model, X, y)
        bands = gpu.predict_quantiles(X)
        assert bands.shape == (X.shape[0], 3)
        assert not np.isnan(bands).all(axis=0).any(), "every band should be populated"
        # and they must be ordered: an inverse CDF is monotone across quantiles
        assert np.all(np.diff(bands, axis=1) >= -1e-9)

    def test_population_spreads_across_the_exceedance_axis(self, hype_model,
                                                           hype_observations):
        """The point of the method: survivors should span exceedance, not cluster."""
        X, y = hype_observations
        gpu = _fit(hype_model, X, y)
        eta = gpu.ensemble_.exceedances
        assert eta.min() < 0.25 and eta.max() > 0.75

    def test_ensemble_parameters_are_physical(self, hype_model, hype_observations):
        """The engine searches a unit box; the stored parameters must be real values."""
        X, y = hype_observations
        gpu = _fit(hype_model, X, y)
        low, high = gpu.model_.parameter_bounds(1)
        params = gpu.ensemble_.params
        assert params.shape[1] == gpu.model_.n_parameters(1)
        assert np.all(params >= low - 1e-9) and np.all(params <= high + 1e-9)
        # ...and not confined to [0, 1], which is what a leaked search space would look like
        assert params.max() > 1.0 or high.max() <= 1.0

    def test_observations_with_gaps(self, hype_model):
        """Gaps are the normal case for a discharge record.

        ``GPURegressor.fit`` cannot take NaN in ``y`` - scikit-learn validates the target
        strictly, whatever ``ensure_all_finite`` does for ``X`` - so ``align`` drops those
        rows. That is only safe because the date travels with the row: the survivors are
        still matched to the right simulated days, which a positional design could not
        guarantee.
        """
        X, y = load_observations(hype_model.template_dir / "Qobs.txt", column="1234")
        y = y.copy()
        y[::5] = np.nan
        Xa, ya = hype_model.align(X, y)
        assert np.isfinite(ya).all()
        assert Xa.shape[0] == ya.size < X.shape[0]

        gpu = _fit(hype_model, Xa, ya)
        assert np.isfinite(gpu.predict(Xa)).all()
        # every surviving date is still inside the simulated window
        t0, n_steps = hype_model.output_window_
        ordinals = Xa[:, 0].astype(int)
        assert ordinals.min() >= t0 and ordinals.max() < t0 + n_steps

    def test_align_keeps_gaps_when_asked(self, hype_model):
        X, y = load_observations(hype_model.template_dir / "Qobs.txt", column="1234")
        y = y.copy()
        y[::5] = np.nan
        _, ya = hype_model.align(X, y, drop_missing=False)
        assert np.isnan(ya).any()

    def test_warm_start_continues(self, hype_model, hype_observations):
        X, y = hype_observations
        gpu = GPURegressor(model=hype_model,
                           **{**SMALL_FIT, "n_iter": 1, "warm_start": True})
        gpu.fit(X, y)
        shape = gpu._population.shape
        gpu.fit(X, y)
        assert gpu._population.shape == shape

    def test_reproducible_with_a_fixed_seed(self, hype_template, stub_command, tmp_path):
        def once(tag):
            model = HYPEModel(template_dir=hype_template, subbasin=1234,
                              executable=stub_command,
                              parameters=["wcfc", "rrcs1", "cmlt"],
                              work_root=tmp_path / tag, warn_unrequested=False)
            try:
                X, y = load_observations(hype_template / "Qobs.txt", column="1234")
                X, y = model.align(X, y)
                return _fit(model, X, y).predict(X)
            finally:
                model.close()

        np.testing.assert_allclose(once("r1"), once("r2"))


class TestRunAccounting:
    def test_plain_fit_costs_one_batch_per_generation(self, hype_model,
                                                     hype_observations):
        X, y = hype_observations
        population, n_iter = 5, 3
        gpu = _fit(hype_model, X, y, population=population, n_iter=n_iter, screen=False)
        # The initial evaluation plus one batch of candidates per generation. Duplicate
        # parameter rows are deduplicated, so this is an upper bound.
        assert _inner(gpu).n_runs_ <= (n_iter + 1) * population
        assert _inner(gpu).n_runs_ >= population

    def test_screening_reuses_its_own_evaluations(self, hype_model, hype_observations):
        """screen evaluates S*P candidates, then fit re-evaluates P of them.

        Without the cache those P re-runs are pure waste; with it the total is exactly
        ``(S + n_iter) * P``.
        """
        X, y = hype_observations
        population, oversample, n_iter = 8, 3, 2
        gpu = _fit(hype_model, X, y, population=population, n_iter=n_iter,
                   screen=True, screen_oversample=oversample)
        assert _inner(gpu).n_runs_ == (oversample + n_iter) * population

    def test_early_stopping_checks_cost_no_runs(self, hype_template, stub_command,
                                                tmp_path):
        """The check re-evaluates the whole population; the cache must absorb it."""
        def run(tag, **kw):
            model = HYPEModel(template_dir=hype_template, subbasin=1234,
                              executable=stub_command,
                              parameters=["wcfc", "rrcs1", "cmlt"],
                              work_root=tmp_path / f"es_{tag}",
                              warn_unrequested=False)
            try:
                X, y = load_observations(hype_template / "Qobs.txt", column="1234")
                X, y = model.align(X, y)
                gpu = _fit(model, X, y, population=8, n_iter=3, **kw)
                return _inner(gpu).n_runs_
            finally:
                model.close()

        plain = run("plain")
        # Both monitors re-evaluate the whole population: the ensemble scorers via
        # predict_quantiles, the hypervolume via a second forward pass. The cached window
        # must absorb either.
        assert run("reliability", validation_fraction=0.2, check_every=1,
                   scoring="reliability", n_iter_no_change=99) == plain
        assert run("hypervolume", validation_fraction=0.2, check_every=1,
                   n_iter_no_change=99) == plain

    def test_cache_hits_are_counted(self, hype_model, hype_observations):
        X, y = hype_observations
        gpu = _fit(hype_model, X, y)
        runs = _inner(gpu).n_runs_
        gpu.predict_quantiles(X)
        assert _inner(gpu).n_cache_hits_ > 0
        assert _inner(gpu).n_runs_ == runs, "predicting on fitted dates must be free"


class TestCrossValidation:
    def test_time_series_split(self, hype_model, hype_observations):
        X, y = hype_observations
        scores = cross_val_score(
            GPURegressor(model=hype_model, **{**SMALL_FIT, "population": 6, "n_iter": 1}),
            X, y, cv=TimeSeriesSplit(2),
        )
        assert scores.shape == (2,) and np.isfinite(scores).all()

    def test_each_fold_gets_its_own_clone(self, hype_model, hype_observations):
        """The user's model instance must not accumulate state across folds."""
        X, y = hype_observations
        cross_val_score(
            GPURegressor(model=hype_model, **{**SMALL_FIT, "population": 6, "n_iter": 1}),
            X, y, cv=TimeSeriesSplit(2),
        )
        assert hype_model.n_runs_ == 0, "fit must not touch the instance the user passed"
        assert getattr(hype_model, "_runner", None) is None

    def test_block_kfold(self, hype_model, hype_observations):
        """Contiguous-block k-fold: every block is tested once, nothing is shuffled."""
        X, y = hype_observations
        scores = cross_val_score(
            GPURegressor(model=hype_model, **{**SMALL_FIT, "population": 6, "n_iter": 1}),
            X, y, cv=KFold(n_splits=3, shuffle=False),
        )
        assert scores.shape == (3,) and np.isfinite(scores).all()

    def test_block_folds_stay_chronological(self, hype_model, hype_observations):
        """Blocks must not shuffle: `HYPEModel` warns when dates arrive out of order."""
        X, y = hype_observations
        for train, _ in KFold(n_splits=3, shuffle=False).split(X):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # would raise if _warn_if_unordered fired
                hype_model.align(X[train], y[train])

    def test_kfold_run_accounting(self, hype_template, stub_command, tmp_path):
        """k folds cost k x a single fit: each clone gets its own workspace and cache."""
        from sklearn.base import clone

        X, y = load_observations(hype_template / "Qobs.txt", column="1234")
        population, n_iter, oversample, k = 4, 1, 2, 3
        total = 0
        for fold, (train, _) in enumerate(KFold(n_splits=k, shuffle=False).split(X)):
            model = HYPEModel(
                template_dir=hype_template, subbasin=1234, executable=stub_command,
                parameters=["wcfc", "rrcs1", "cmlt"],
                work_root=tmp_path / f"fold{fold}", warn_unrequested=False,
            )
            try:
                Xf, yf = model.align(X[train], y[train])
                gpu = GPURegressor(
                    model=model, **{**SMALL_FIT, "population": population,
                                    "n_iter": n_iter, "screen_oversample": oversample},
                ).fit(Xf, yf)
                inner = _inner(gpu)
                assert inner.n_failed_runs_ == 0
                total += inner.n_runs_
                # scoring the same estimator on a *different* block is free: the whole
                # window is already simulated and cached.
                before = inner.n_runs_
                gpu.predict_quantiles(model.align(X)[:50])
                assert inner.n_runs_ == before
                inner.close()
            finally:
                model.close()
        assert total == k * (oversample + n_iter) * population


class TestFailedRunsOnTheFront:
    def test_nan_run_reaches_front_zero_while_eta_is_narrow(self):
        """Locks the measured behaviour that motivates ``max_failure_fraction``.

        A NaN column scores non-exceedance exactly 0.0, and the double-Pareto sorter ranks
        by exceedance *coverage* rather than loss dominance - so while the population is
        still clustered near 0.5 the failure extends front 0 and survives selection. Once
        the population has spread across the exceedance axis a healthy particle already
        occupies 0.0, and the failure is properly dominated.
        """
        from foresight_gpu.domination import DoubleParetoSorter
        from foresight_gpu.metrics import get_metric
        from foresight_gpu.metrics.exceedance import non_exceedance

        rng = np.random.default_rng(0)
        obs = rng.gamma(2.0, 3.0, 300) + 0.5
        nse = get_metric("nse")

        def front_of_last(sims):
            eta = non_exceedance(sims, obs)
            loss = np.log10(np.maximum(nse.loss(sims, obs), np.finfo(float).tiny))
            loss = np.where(np.isnan(loss) | (loss == np.inf), 1e3, loss)
            levels = DoubleParetoSorter().front_levels(np.column_stack([eta, loss]))
            return levels[-1]

        narrow = obs[:, None] * rng.uniform(0.7, 1.3, (300, 20))
        narrow[:, -1] = np.nan
        assert front_of_last(narrow) == 0

        spread = obs[:, None] * np.linspace(0.35, 2.2, 20)[None, :]
        spread[:, -1] = np.nan
        assert front_of_last(spread) > 0

    def test_a_fit_with_failures_still_completes_and_reports(self, hype_template,
                                                             stub_command, tmp_path,
                                                             monkeypatch):
        from hype_stub import MODE_ENV

        monkeypatch.setenv(MODE_ENV, "crash")
        model = HYPEModel(template_dir=hype_template, subbasin=1234,
                          executable=stub_command, parameters=["cmlt"],
                          work_root=tmp_path / "bad", max_failure_fraction=1.0,
                          warn_unrequested=False)
        try:
            X, y = load_observations(hype_template / "Qobs.txt", column="1234")
            X, y = model.align(X, y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpu = _fit(model, X, y, population=4, n_iter=1, screen=False)
            assert _inner(gpu).n_failed_runs_ > 0
        finally:
            model.close()

    def test_a_wholly_broken_setup_aborts(self, hype_template, stub_command, tmp_path,
                                          monkeypatch):
        from hype_stub import MODE_ENV

        monkeypatch.setenv(MODE_ENV, "nonzero")
        model = HYPEModel(template_dir=hype_template, subbasin=1234,
                          executable=stub_command, parameters=["cmlt"],
                          work_root=tmp_path / "abort", warn_unrequested=False)
        try:
            X, y = load_observations(hype_template / "Qobs.txt", column="1234")
            X, y = model.align(X, y)
            with pytest.raises(RuntimeError, match="max_failure_fraction"):
                _fit(model, X, y, population=4, n_iter=1, screen=False)
        finally:
            model.close()


class TestAutocalibrate:
    def test_writes_optpar_and_parses_results(self, hype_model, tmp_path, monkeypatch):
        """The stub does not implement DE-MC, so this checks the file plumbing only."""
        from foresight_gpu.models.hype import autocalibrate
        from foresight_gpu.models.hype.files import write_optpar

        # The stub writes no respar.txt; assert the message says so usefully.
        with pytest.raises(RuntimeError, match="no respar.txt"):
            autocalibrate(hype_model, ngen=2, npop=2, work_dir=tmp_path / "ac")

        optpar = tmp_path / "ac" / "optpar.txt"
        assert optpar.exists()
        # read_bytes, not read_text: universal newlines would erase the CRLF layout
        lines = optpar.read_bytes().decode("latin-1").split("\r\n")
        assert lines[0] == "Info optimization"
        assert lines[21].split("\t")[0] in hype_model.active_parameters_
        info = (tmp_path / "ac" / "info.txt").read_bytes().decode("latin-1")
        assert "calibration\ty" in info and "crit 1 criterion" in info

    def test_requires_observations(self, hype_model, tmp_path):
        from foresight_gpu.models.hype import autocalibrate

        (hype_model.template_dir / "Qobs.txt").unlink()
        with pytest.raises(FileNotFoundError, match="no Qobs.txt"):
            autocalibrate(hype_model, ngen=2, npop=2, work_dir=tmp_path / "ac2")

    def test_optpar_arity_matches_par_txt(self, hype_model, tmp_path):
        from foresight_gpu.models.hype.files import ParFile, write_optpar
        from foresight_gpu.models.hype.parameters import optpar_entries

        par = ParFile.read(hype_model.template_dir / "par.txt")
        entries = optpar_entries(hype_model._ensure_layout())
        for name, low, high, step in entries:
            assert low.size == par.arity(name)
        write_optpar(tmp_path / "o.txt", entries)


# -- real HYPE ---------------------------------------------------------------------------


@pytest.fixture
def real_model(real_hype_folder, tmp_path):
    """A HYPEModel over a real folder, with a deliberately small parameter set."""
    model = HYPEModel(
        template_dir=real_hype_folder,
        subbasin=int(os.environ.get("FORESIGHT_HYPE_SUBBASIN", 50675)),
        parameters=["wcfc", "rrcs1", "cmlt"],
        n_workers=int(os.environ.get("FORESIGHT_HYPE_WORKERS", 2)),
        work_root=tmp_path / "real",
        warn_unrequested=False,
    )
    yield model
    model.close()


class TestRealHype:
    def test_layout_from_a_real_template(self, real_model):
        assert real_model.n_parameters(1) >= 3
        low, high = real_model.parameter_bounds(1)
        assert np.all(high > low)

    def test_par_txt_values_survive_a_real_round_trip(self, real_model, tmp_path):
        """Fortran free-format output must read back bit-identically."""
        from foresight_gpu.models.hype.files import ParFile

        rng = np.random.default_rng(0)
        params = real_model.search_transform(
            rng.uniform(0, 1, real_model.n_parameters(1))
        )
        wanted = real_model._ensure_layout().to_par_values(params)
        out = real_model.write_par(tmp_path / "par.txt", params)
        back = ParFile.read(out).values
        for name, values in wanted.items():
            np.testing.assert_array_equal(back[name], values)

    @pytest.mark.slow
    def test_forward_runs_the_real_executable(self, real_model):
        X = as_X(real_model.dates_[:120])
        rng = np.random.default_rng(0)
        out = real_model.forward(
            X, real_model.search_transform(rng.uniform(0, 1, (2, real_model.n_parameters(1))))
        )
        assert out.shape == (120, 2)
        assert real_model.n_failed_runs_ == 0, "real HYPE run failed"
        assert np.isfinite(out).all()

    @pytest.mark.slow
    def test_small_fit_against_real_hype(self, real_model):
        """A real fit, tiny by design.

        Observations often live outside the HYPE folder (HYPE only needs ``Qobs.txt`` for
        its own calibration), so ``FORESIGHT_HYPE_QOBS`` can point at them.
        """
        qobs = Path(os.environ.get(
            "FORESIGHT_HYPE_QOBS", real_model.template_dir / "Qobs.txt"
        ))
        if not qobs.exists():
            pytest.skip(f"no observations: set FORESIGHT_HYPE_QOBS (tried {qobs})")
        X, y = load_observations(qobs, column=str(real_model.subbasin))
        X, y = real_model.align(X, y)

        population, n_iter, oversample = 6, 1, 3
        gpu = _fit(real_model, X, y, population=population, n_iter=n_iter,
                   screen_oversample=oversample)
        inner = _inner(gpu)

        assert inner.n_failed_runs_ == 0, "real HYPE runs failed"
        assert inner.n_runs_ == (oversample + n_iter) * population
        assert gpu.ensemble_.n_models == population
        low, high = inner.parameter_bounds(1)
        assert np.all(gpu.ensemble_.params >= low - 1e-9)
        assert np.all(gpu.ensemble_.params <= high + 1e-9)
        assert gpu.predict_quantiles(X).shape == (X.shape[0], len(SMALL_FIT["quantiles"]))
        # Deliberately no assertion that the bands are populated: at this budget the swarm
        # cannot spread across the exceedance axis, so every band may be empty and predict
        # returns NaN. That is a budget limit, not a defect - a realistic run (population
        # 40, n_iter 25 over ~10 years) fills all 16 default bands.
