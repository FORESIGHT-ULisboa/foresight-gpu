"""Supplying HYPE's forcing from Python: coercion, validation, staging, cache identity."""

import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import clone

pd = pytest.importorskip("pandas")

from foresight_gpu.models.hype import HYPEModel, as_X, read_hype_table  # noqa: E402
from foresight_gpu.models.hype import forcing as fc  # noqa: E402


def make_frame(start="1991-06-01", end="1993-05-31", ids=("1234",), seed=0, scale=1.0):
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy()
    season = 1 + 0.8 * np.sin(2 * np.pi * (doy - 30) / 365.25)
    data = {i: np.round(rng.gamma(1.2, 2.0, len(dates)) * season * scale, 3) for i in ids}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def forcing():
    return {"P": make_frame(seed=1), "T": make_frame(seed=2)}


@pytest.fixture
def model_factory(hype_template, stub_command, tmp_path):
    made = []

    def build(**kw):
        options = dict(template_dir=hype_template, subbasin=1234,
                       executable=stub_command, parameters=["wcfc", "cmlt"],
                       work_root=tmp_path / f"w{len(made)}", warn_unrequested=False)
        options.update(kw)
        model = HYPEModel(**options)
        made.append(model)
        return model

    yield build
    for model in made:
        model.close()


def _one(model):
    return model.search_transform(np.full(model.n_parameters(1), 0.5))


class TestNames:
    @pytest.mark.parametrize("key,expected", [
        ("P", "Pobs.txt"), ("precipitation", "Pobs.txt"), ("Pobs", "Pobs.txt"),
        ("Pobs.txt", "Pobs.txt"), ("T", "Tobs.txt"), ("tmin", "TMINobs.txt"),
        ("Qobs", "Qobs.txt"), ("discharge", "Qobs.txt"),
    ])
    def test_aliases_resolve(self, key, expected):
        assert fc.canonical_name(key) == expected

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(ValueError, match="Unknown forcing"):
            fc.canonical_name("rainfall_mm")

    def test_duplicate_entries_rejected(self):
        with pytest.raises(ValueError, match="two entries"):
            fc.resolve({"P": make_frame(), "Pobs.txt": make_frame()})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            fc.resolve([make_frame()])


class TestCoercion:
    def test_dataframe(self, forcing):
        frames = fc.resolve(forcing)
        assert set(frames) == {"Pobs.txt", "Tobs.txt"}
        assert list(frames["Pobs.txt"].columns) == ["1234"]

    def test_series_becomes_one_column(self):
        series = make_frame()["1234"]
        frames = fc.resolve({"P": series})
        assert list(frames["Pobs.txt"].columns) == ["1234"]

    def test_path(self, hype_template):
        frames = fc.resolve({"P": hype_template / "Pobs.txt"})
        assert frames["Pobs.txt"].shape[1] == 1

    def test_string_index_is_parsed(self):
        frame = make_frame()
        frame.index = frame.index.astype(str)
        assert isinstance(fc.resolve({"P": frame})["Pobs.txt"].index, pd.DatetimeIndex)

    def test_positional_index_is_refused(self):
        """pd.to_datetime would read it as nanoseconds and hand back 1970 dates."""
        frame = make_frame().reset_index(drop=True)
        with pytest.raises(TypeError, match="numeric index"):
            fc.resolve({"P": frame})

    def test_unparseable_index_raises(self):
        frame = make_frame()
        frame.index = [f"not-a-date-{i}" for i in range(len(frame))]
        with pytest.raises(TypeError, match="date index"):
            fc.resolve({"P": frame})

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="DataFrame, a Series or a path"):
            fc.resolve({"P": 42})

    def test_duplicate_dates_collapse_and_sort(self):
        frame = make_frame(end="1991-06-05")
        doubled = pd.concat([frame, frame.iloc[[0]]])
        # Silently discarding a duplicated day is not acceptable, so it warns.
        with pytest.warns(UserWarning, match="duplicated date"):
            out = fc.resolve({"P": doubled})["Pobs.txt"]
        assert out.index.is_monotonic_increasing
        assert not out.index.duplicated().any()
        # last wins
        assert out.loc[frame.index[0], "1234"] == frame.iloc[0]["1234"]

    def test_tz_aware_index_is_refused(self):
        """Day ordinals floor in UTC, so a +09:00 index would move every date back a day."""
        frame = make_frame(end="1991-06-05")
        frame.index = frame.index.tz_localize("Asia/Tokyo")
        with pytest.raises(TypeError, match="tz_localize"):
            fc.resolve({"P": frame})

    def test_columns_become_strings(self):
        frame = make_frame(ids=(1234,))
        assert list(fc.resolve({"P": frame})["Pobs.txt"].columns) == ["1234"]


class TestFileFormat:
    def test_round_trip(self, tmp_path, forcing):
        frames = fc.resolve(forcing)
        path = tmp_path / "Pobs.txt"
        fc.write(frames["Pobs.txt"], path, allow_gaps=False)
        back = read_hype_table(path)
        np.testing.assert_allclose(back.to_numpy(), frames["Pobs.txt"].to_numpy())
        assert list(back.columns) == ["1234"]
        assert back.index.equals(frames["Pobs.txt"].index)

    def test_written_in_hype_conventions(self, tmp_path, forcing):
        path = tmp_path / "Pobs.txt"
        fc.write(fc.resolve(forcing)["Pobs.txt"], path, allow_gaps=False)
        raw = path.read_bytes().decode("latin-1")
        assert raw.startswith("DATE\t1234\r\n")
        assert "\n" not in raw.replace("\r\n", "")  # CRLF only
        assert raw.split("\r\n")[1].startswith("1991-06-01\t")  # ISO dates

    def test_gaps_written_as_missing_code(self, tmp_path):
        frame = make_frame(end="1991-06-05")
        frame.iloc[1, 0] = np.nan
        path = tmp_path / "Qobs.txt"
        fc.write(frame, path, allow_gaps=True)
        assert "-9999" in path.read_bytes().decode("latin-1")
        assert np.isnan(read_hype_table(path).to_numpy()[1, 0])

    def test_expected_ids_from_pobs_header(self, hype_template):
        assert fc.expected_ids(hype_template, 1234) == ["1234"]

    def test_expected_ids_falls_back_to_geodata(self, hype_template):
        (hype_template / "Pobs.txt").unlink()
        assert fc.expected_ids(hype_template, None) == ["1234"]


class TestValidation:
    def test_short_series_raises_naming_the_gap(self, forcing):
        frames = fc.resolve(forcing)
        with pytest.raises(ValueError, match="does not cover the simulation window"):
            fc.validate(frames, "1991-01-01", "1993-05-31")

    def test_nan_in_forcing_raises(self):
        frame = make_frame(end="1991-06-10")
        frame.iloc[3, 0] = np.nan
        frames = fc.resolve({"P": frame})
        with pytest.raises(ValueError, match="cannot integrate through a gap"):
            fc.validate(frames, "1991-06-01", "1991-06-10")

    def test_nan_in_observations_is_allowed(self):
        frame = make_frame(end="1991-06-10")
        frame.iloc[3, 0] = np.nan
        fc.validate(fc.resolve({"Qobs": frame}), "1991-06-01", "1991-06-10")

    def test_missing_column_raises(self):
        frames = fc.resolve({"P": make_frame(ids=("999",), end="1991-06-10")})
        with pytest.raises(ValueError, match="missing column"):
            fc.validate(frames, "1991-06-01", "1991-06-10", expected_ids=["1234"])

    def test_extra_column_warns_but_passes(self):
        frames = fc.resolve({"P": make_frame(ids=("1234", "999"), end="1991-06-10")})
        with pytest.warns(UserWarning, match="extra column"):
            fc.validate(frames, "1991-06-01", "1991-06-10", expected_ids=["1234"])

    def test_coverage_is_the_intersection(self):
        frames = fc.resolve({
            "P": make_frame("1990-01-01", "1995-01-01"),
            "T": make_frame("1991-01-01", "1994-01-01"),
        })
        start, end = fc.coverage(frames)
        assert str(start) == "1991-01-01" and str(end) == "1994-01-01"

    def test_switches_follow_the_series(self, forcing):
        assert fc.switches(fc.resolve(forcing)) == {}
        with_tmin = dict(forcing, tmin=make_frame(seed=3))
        assert fc.switches(fc.resolve(with_tmin)) == {"readtminobs": "y"}

    def test_digest_tracks_the_data(self, forcing):
        base = fc.digest(fc.resolve(forcing))
        assert base == fc.digest(fc.resolve(forcing))
        changed = dict(forcing, P=make_frame(seed=1, scale=3.0))
        assert fc.digest(fc.resolve(changed)) != base
        assert fc.digest({}) == "no-forcing"


class TestModelIntegration:
    def test_window_follows_the_data(self, model_factory, forcing):
        model = model_factory(forcing=forcing)
        bdate, cdate, edate = model.simulation_window_
        assert (bdate, edate) == ("1991-06-01", "1993-05-31")
        assert model.output_window_[1] == 731

    def test_explicit_dates_win_over_the_data(self, model_factory, forcing):
        model = model_factory(forcing=forcing, bdate="1991-06-01",
                              cdate="1992-01-01", edate="1993-01-01")
        assert model.simulation_window_ == ("1991-06-01", "1992-01-01", "1993-01-01")

    def test_explicit_window_outside_the_data_raises(self, model_factory, forcing):
        model = model_factory(forcing=forcing, bdate="1985-01-01", edate="1993-05-31")
        with pytest.raises(ValueError, match="does not cover"):
            model.n_parameters(1)

    def test_no_forcing_falls_back_to_the_folder(self, model_factory):
        model = model_factory()
        assert model.forcing_ == {}
        # the template info.txt window, untouched
        assert model.simulation_window_ == ("1990-01-01", "1991-01-01", "1994-12-31")
        assert np.isfinite(model.forward(as_X(model.dates_[:5]), _one(model))).all()

    def test_supplying_the_folders_own_files_equals_the_fallback(self, model_factory,
                                                                 hype_template):
        """The writer must round-trip, not merely produce something plausible.

        Handing back exactly what the template already holds has to give the identical
        simulation - otherwise the forcing writer is subtly reformatting the data and every
        `forcing=` run differs from the fallback for reasons nobody asked for.
        """
        fallback = model_factory()
        explicit = model_factory(forcing={
            "P": read_hype_table(hype_template / "Pobs.txt"),
            "T": read_hype_table(hype_template / "Tobs.txt"),
        })
        # same window, so the two are directly comparable
        assert explicit.simulation_window_ == fallback.simulation_window_
        X = as_X(fallback.dates_[:120])
        np.testing.assert_array_equal(
            fallback.forward(X, _one(fallback)), explicit.forward(X, _one(explicit))
        )

    def test_supplied_forcing_is_written_into_the_worker_folder(self, model_factory,
                                                                forcing):
        model = model_factory(forcing=forcing)
        model.forward(as_X(model.dates_[:5]), _one(model))
        from pathlib import Path

        root = Path(model._runner.spec.work_root)
        worker = next(p for p in root.iterdir() if p.is_dir() and p.name.startswith("w"))
        written = read_hype_table(worker / "Pobs.txt")
        expected = fc.resolve(forcing)["Pobs.txt"]
        np.testing.assert_allclose(written.to_numpy(), expected.to_numpy())

    def test_forcing_changes_the_simulation(self, model_factory, forcing):
        """The whole point: the data must actually reach HYPE."""
        wet = dict(forcing, P=make_frame(seed=1, scale=3.0))
        a = model_factory(forcing=forcing)
        b = model_factory(forcing=wet)
        X = as_X(a.dates_[:60])
        assert not np.allclose(a.forward(X, _one(a)), b.forward(X, _one(b)))

    def test_forcing_is_part_of_the_cache_identity(self, model_factory, forcing):
        """Same parameters + different weather = different answer, so different key."""
        wet = dict(forcing, P=make_frame(seed=1, scale=3.0))
        a = model_factory(forcing=forcing)
        b = model_factory(forcing=wet)
        assert a.fingerprint_ != b.fingerprint_

    def test_read_switch_is_set_in_info_txt(self, model_factory, forcing):
        model = model_factory(forcing=dict(forcing, tmin=make_frame(seed=3)))
        assert model.forcing_switches_ == {"readtminobs": "y"}
        model.forward(as_X(model.dates_[:3]), _one(model))
        from pathlib import Path

        root = Path(model._runner.spec.work_root)
        worker = next(p for p in root.iterdir() if p.is_dir() and p.name.startswith("w"))
        active = [
            line for line in (worker / "info.txt").read_bytes().decode("latin-1").split("\r\n")
            if line.strip() and not line.lstrip().startswith("!!")
        ]
        assert "readtminobs\ty" in active

    @pytest.mark.parametrize("route", ["path", "dataframe", "series", "string_index"])
    def test_every_forcing_route_gives_the_same_simulation(self, model_factory,
                                                           hype_template, route):
        """Same data, four ways in: identical simulation *and* identical fingerprint.

        The fingerprint matters as much as the values - if it differed, the cache would
        treat one route's weather as different weather from another's.
        """
        pobs, tobs = hype_template / "Pobs.txt", hype_template / "Tobs.txt"
        frames = {"P": read_hype_table(pobs), "T": read_hype_table(tobs)}

        def string_index(frame):
            out = frame.copy()
            out.index = out.index.strftime("%Y-%m-%d")
            return out

        routes = {
            "path": {"P": pobs, "T": tobs},
            "dataframe": frames,
            "series": {k: v.iloc[:, 0] for k, v in frames.items()},
            "string_index": {k: string_index(v) for k, v in frames.items()},
        }
        reference = model_factory(forcing=routes["path"])
        candidate = model_factory(forcing=routes[route])
        X = as_X(reference.dates_[:120])
        np.testing.assert_array_equal(
            reference.forward(X, _one(reference)), candidate.forward(X, _one(candidate))
        )
        assert candidate.fingerprint_ == reference.fingerprint_

    def test_template_is_still_never_written_to(self, model_factory, forcing,
                                                hype_template):
        before = (hype_template / "Pobs.txt").read_bytes()
        model = model_factory(forcing=forcing)
        model.forward(as_X(model.dates_[:5]), _one(model))
        assert (hype_template / "Pobs.txt").read_bytes() == before

    def test_clone_and_pickle_survive_dataframes(self, model_factory, forcing):
        model = model_factory(forcing=forcing)
        # the constructor must not transform the dict, or clone raises
        params = model.get_params(deep=False)
        assert type(model)(**params).get_params(deep=False)["forcing"] is forcing
        assert set(clone(model).forcing) == set(forcing)

        X = as_X(model.dates_[:5])
        expected = model.forward(X, _one(model))
        restored = pickle.loads(pickle.dumps(model))
        try:
            np.testing.assert_array_equal(restored.forward(X, _one(restored)), expected)
        finally:
            restored.close()

    def test_fit_end_to_end_on_supplied_forcing(self, model_factory, forcing):
        from foresight_gpu import GPURegressor

        model = model_factory(forcing=forcing)
        # observations over the same window, so align keeps everything
        qobs = make_frame(seed=9)["1234"] * 2.0
        X, y = as_X(model.dates_), qobs.reindex(pd.DatetimeIndex(model.dates_)).to_numpy()
        X, y = model.align(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpu = GPURegressor(model=model, metric="mae", population=6, n_iter=1,
                               screen=True, screen_oversample=2, band_width=0.15,
                               quantiles=[0.1, 0.5, 0.9], force_positive=True,
                               random_state=0).fit(X, y)
        assert gpu.model_.n_failed_runs_ == 0
        bands = gpu.predict_quantiles(X)
        assert bands.shape == (X.shape[0], 3)

        # the downstream artifacts must work on a forcing-driven fit too
        from foresight_gpu.models.hype import freeze

        inner = gpu.model_
        assert len(inner.describe(gpu.ensemble_.params)) == inner.n_parameters(1)
        frozen = freeze(gpu.ensemble_)
        np.testing.assert_allclose(frozen.predict_quantiles(X), bands, equal_nan=True)
