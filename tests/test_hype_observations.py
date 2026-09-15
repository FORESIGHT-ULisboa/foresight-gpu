"""Observation import routes and date-aware alignment.

Two properties carry most of the weight here:

* every supported way of handing the *same* observations to the model produces the *same*
  ``(X, y)`` - otherwise the routes are not interchangeable and a notebook that switches
  from a path to a DataFrame silently calibrates against something else;
* ``align`` joins on dates when it can and refuses to guess when it cannot, because the
  alternative is a plausible-looking fit against the wrong rows.
"""

import warnings

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

pd = pytest.importorskip("pandas")

from foresight_gpu.models.hype import (  # noqa: E402
    HYPEModel,
    as_X,
    load_observations,
    read_hype_table,
)


@pytest.fixture
def qobs_path(hype_template):
    return hype_template / "Qobs.txt"


@pytest.fixture
def qobs_frame(qobs_path):
    """The template observations as a date-indexed DataFrame."""
    return read_hype_table(qobs_path)


@pytest.fixture
def model(hype_template, stub_command, tmp_path):
    m = HYPEModel(template_dir=hype_template, subbasin=1234, executable=stub_command,
                  parameters=["wcfc", "cmlt"], work_root=tmp_path / "w",
                  warn_unrequested=False)
    yield m
    m.close()


# -- import routes ---------------------------------------------------------------------


class TestObservationRoutes:
    """The same data, five ways in, must give identical ``(X, y)``."""

    def test_path_route_shape(self, qobs_path):
        X, y = load_observations(qobs_path, column="1234")
        assert X.shape == (y.size, 1)
        assert np.isfinite(y).all()

    @pytest.mark.parametrize("route", [
        "path_with_column", "path_without_column", "dataframe", "series", "date_column",
    ])
    def test_every_route_agrees(self, qobs_path, qobs_frame, route):
        expected = load_observations(qobs_path, column="1234")
        sources = {
            "path_with_column": lambda: load_observations(qobs_path, column="1234"),
            "path_without_column": lambda: load_observations(qobs_path),
            "dataframe": lambda: load_observations(qobs_frame, column="1234"),
            "series": lambda: load_observations(qobs_frame["1234"]),
            # a frame that still carries the date as a column rather than an index
            "date_column": lambda: load_observations(
                qobs_frame.rename_axis("DATE").reset_index()
            ),
        }
        X, y = sources[route]()
        np.testing.assert_array_equal(X, expected[0])
        np.testing.assert_array_equal(y, expected[1])

    def test_semicolon_and_dayfirst_path(self, tmp_path):
        """Real observation records are not always in HYPE's own format."""
        dates = pd.date_range("1991-01-01", periods=5, freq="D")
        text = "Date;9022\n" + "\n".join(
            f"{d.strftime('%d/%m/%Y')};{v}" for d, v in zip(dates, [1.0, 2.0, 3.0, 4.0, 5.0])
        )
        path = tmp_path / "Qobs.csv"
        path.write_text(text)
        X, y = load_observations(path, sep=";", dayfirst=True)
        assert y.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_array_equal(X, as_X(dates.to_numpy()))

    def test_missing_code_becomes_nan_from_a_file(self, tmp_path):
        path = tmp_path / "Qobs.txt"
        path.write_text("DATE\t1234\r\n1991-01-01\t1.5\r\n1991-01-02\t-9999\r\n",
                        newline="")
        _, y = load_observations(path)
        assert y[0] == 1.5 and np.isnan(y[1])

    def test_missing_code_becomes_nan_from_a_frame(self, qobs_frame):
        frame = qobs_frame.copy()
        frame.iloc[3, 0] = -9999.0
        _, y = load_observations(frame)
        assert np.isnan(y[3])

    def test_missing_column_lists_what_is_available(self, qobs_path):
        with pytest.raises(KeyError, match=r"Available: \['1234'\]"):
            load_observations(qobs_path, column="9022")

    def test_multi_column_refuses_to_guess(self, qobs_frame):
        """Picking the first of several gauges silently is the wrong kind of convenient."""
        with pytest.raises(ValueError, match="pass column="):
            load_observations(qobs_frame.assign(other=1.0))

    def test_numeric_index_refused(self, qobs_frame):
        with pytest.raises(TypeError, match="numeric index"):
            load_observations(qobs_frame.reset_index(drop=True))

    def test_tz_aware_index_refused(self, qobs_frame):
        frame = qobs_frame.copy()
        frame.index = frame.index.tz_localize("Asia/Tokyo")
        with pytest.raises(TypeError, match="tz_localize"):
            load_observations(frame)

    def test_duplicate_dates_warn_and_last_wins(self, qobs_frame):
        doubled = pd.concat([qobs_frame, qobs_frame.iloc[[0]] * 99])
        with pytest.warns(UserWarning, match="duplicated date"):
            _, y = load_observations(doubled)
        assert y[0] == qobs_frame.iloc[0, 0] * 99


class TestObservationsMethod:
    def test_matches_load_plus_align(self, model, qobs_path):
        got = model.observations(qobs_path, column="1234")
        want = model.align(*load_observations(qobs_path, column="1234"))
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])

    def test_restricted_to_the_output_window(self, model, qobs_path):
        X, y = model.observations(qobs_path)
        t0, n_steps = model.output_window_
        ordinals = X[:, 0]
        assert ordinals.min() >= t0 and ordinals.max() < t0 + n_steps
        assert np.isfinite(y).all()

    def test_honours_a_moved_date_column(self, hype_template, stub_command, tmp_path,
                                         qobs_path):
        """``load_observations`` cannot know where the date belongs; the model does."""
        other = HYPEModel(template_dir=hype_template, subbasin=1234,
                          executable=stub_command, parameters=["wcfc"], date_column=2,
                          work_root=tmp_path / "moved", warn_unrequested=False)
        try:
            X, y = other.observations(qobs_path)
            assert X.shape[1] == 3
            assert (X[:, 0] == 0).all() and (X[:, 1] == 0).all()
            assert X[:, 2].min() >= other.output_window_[0]
            # and it round-trips through the model
            sims = other.forward(X[:20], other.search_transform(
                np.full(other.n_parameters(1), 0.5)))
            assert sims.shape == (20, 1)
        finally:
            other.close()


# -- align -----------------------------------------------------------------------------


class TestAlignPositional:
    def test_x_only_returns_one_array(self, model, qobs_path):
        X, _ = load_observations(qobs_path)
        out = model.align(X)
        assert isinstance(out, np.ndarray) and out.ndim == 2

    def test_equal_length_arrays_unchanged(self, model, qobs_path):
        X, y = load_observations(qobs_path, column="1234")
        Xa, ya = model.align(X, y)
        t0, n_steps = model.output_window_
        expected = (X[:, 0] >= t0) & (X[:, 0] < t0 + n_steps) & np.isfinite(y)
        np.testing.assert_array_equal(Xa, X[expected])
        np.testing.assert_array_equal(ya, y[expected])

    def test_unequal_length_arrays_raise_clearly(self, model, qobs_path):
        X, _ = load_observations(qobs_path)
        with pytest.raises(ValueError, match="carries no dates"):
            model.align(X, np.arange(7.0))

    def test_range_indexed_series_is_positional(self, model, qobs_path):
        X, y = load_observations(qobs_path, column="1234")
        bare = pd.Series(y)  # RangeIndex: just an array
        Xa, ya = model.align(X, bare)
        np.testing.assert_array_equal(ya, model.align(X, y)[1])

    def test_drop_missing_false_keeps_gaps(self, model, qobs_path):
        X, y = load_observations(qobs_path, column="1234")
        y = y.copy()
        # Blank days *inside* the window: a hole at the head of the record would land in
        # the warmup span and be trimmed out regardless of drop_missing.
        t0, _ = model.output_window_
        inside = np.flatnonzero(X[:, 0] >= t0)[:50]
        y[inside] = np.nan
        _, ya = model.align(X, y, drop_missing=False)
        assert np.isnan(ya).sum() == 50

    def test_plain_window_trim_is_silent(self, model, qobs_path):
        """The hype_observations fixture relies on this; a chatty align would pollute it."""
        X, y = load_observations(qobs_path, column="1234")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.align(X, y)

    def test_scaled_x_raises_instead_of_returning_empty(self, model, qobs_path):
        X, y = load_observations(qobs_path, column="1234")
        with pytest.raises(ValueError, match="whole-day ordinal"):
            model.align(StandardScaler().fit_transform(X), y)


class TestAlignDateJoin:
    def test_dated_series_from_a_different_source(self, model, qobs_frame):
        """The case that used to raise an opaque broadcast error."""
        X = as_X(model.dates_)
        series = qobs_frame["1234"]
        assert len(series) != X.shape[0]
        Xa, ya = model.align(X, series)
        lookup = dict(zip(qobs_frame.index.to_numpy().astype("datetime64[D]"),
                          qobs_frame["1234"].to_numpy()))
        from foresight_gpu.models.hype import from_ordinals

        expected = [lookup[d] for d in from_ordinals(Xa[:, 0])]
        np.testing.assert_allclose(ya, expected)

    def test_one_column_frame_equals_series(self, model, qobs_frame):
        X = as_X(model.dates_)
        a = model.align(X, qobs_frame)
        b = model.align(X, qobs_frame["1234"])
        np.testing.assert_array_equal(a[1], b[1])

    def test_multi_column_frame_needs_a_column(self, model, qobs_frame):
        X = as_X(model.dates_)
        with pytest.raises(ValueError, match="pass column="):
            model.align(X, qobs_frame.assign(other=1.0))
        Xa, ya = model.align(X, qobs_frame.assign(other=1.0), column="1234")
        np.testing.assert_array_equal(ya, model.align(X, qobs_frame)[1])

    def test_partial_overlap_warns_and_uses_the_common_dates(self, model, qobs_frame):
        X = as_X(model.dates_)
        half = qobs_frame["1234"].iloc[: len(qobs_frame) // 2]
        with pytest.warns(UserWarning, match="have no observation"):
            Xa, ya = model.align(X, half)
        assert 0 < Xa.shape[0] < X.shape[0]
        assert np.isfinite(ya).all()

    def test_a_subset_of_x_does_not_warn(self, model, qobs_frame):
        """Asking for a few dates out of many is ordinary use, not a mismatch."""
        X = as_X(model.dates_[:10])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.align(X, qobs_frame["1234"])

    def test_order_and_multiplicity_follow_x(self, model, qobs_frame):
        """A sorted-intersection join would reorder and deduplicate; that is not a filter.

        Duplicated rows change the effective weighting of the loss and of the exceedance
        objective, and any parallel array the caller holds would desynchronise.
        """
        picked = np.array([90, 3, 3, 41])
        X = as_X(model.dates_[picked])
        Xa, ya = model.align(X, qobs_frame["1234"])
        np.testing.assert_array_equal(Xa, X)
        assert ya[1] == ya[2]

    def test_object_index_refuses_to_guess(self, model, qobs_frame):
        X = as_X(model.dates_)
        series = qobs_frame["1234"].copy()
        series.index = [f"row-{i}" for i in range(len(series))]
        with pytest.raises(TypeError, match="dates or row labels"):
            model.align(X, series)

    def test_tz_aware_series_refused(self, model, qobs_frame):
        X = as_X(model.dates_)
        series = qobs_frame["1234"].copy()
        series.index = series.index.tz_localize("Asia/Tokyo")
        with pytest.raises(TypeError, match="tz_localize"):
            model.align(X, series)

    def test_empty_intersection_reports_both_spans(self, model, qobs_frame):
        X = as_X(model.dates_)
        shifted = qobs_frame["1234"].copy()
        shifted.index = shifted.index + pd.DateOffset(years=20)
        with pytest.raises(ValueError, match="no date shared"):
            model.align(X, shifted)

    def test_drop_missing_false_with_a_dated_y(self, model, qobs_frame):
        X = as_X(model.dates_)
        # Punch the hole inside the window, not in the warmup year the frame also covers.
        window_dates = pd.DatetimeIndex(model.dates_)
        holed = qobs_frame["1234"].drop(window_dates[100:200])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ya = model.align(X, holed, drop_missing=False)
        assert ya.size == X.shape[0]
        assert np.isnan(ya).sum() == 100
