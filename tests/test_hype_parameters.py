"""HYPE parameter layout: modes, routine gating, bounds validation, unit-box transforms."""

import warnings

import numpy as np
import pytest

from foresight_gpu.models.hype.dates import (
    as_X,
    as_ordinals,
    from_ordinals,
    resolve_indices,
    window_ordinals,
)
from foresight_gpu.models.hype.files import InfoFile, ParFile
from foresight_gpu.models.hype.parameters import (
    CATALOGUE,
    HypeParameter,
    build_layout,
    effective_options,
    is_enabled,
    multiplier_bounds,
    optpar_entries,
)


@pytest.fixture
def par(hype_template):
    return ParFile.read(hype_template / "par.txt")


@pytest.fixture
def options(hype_template):
    return effective_options(InfoFile.read(hype_template / "info.txt").options)


def layout_for(par, options, requested, **kw):
    kw.setdefault("warn_unrequested", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_layout(par, options, requested, **kw)


class TestSpec:
    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError, match="high > low"):
            HypeParameter(name="x", low=1.0, high=1.0)

    def test_log_with_non_positive_low_raises(self):
        # log10(0) = -inf would make every particle in that dimension NaN, silently.
        with pytest.raises(ValueError, match="log search needs low > 0"):
            HypeParameter(name="x", low=0.0, high=1.0, log=True)

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            HypeParameter(name="x", low=0.1, high=1.0, mode="scale")

    def test_catalogue_specs_are_self_consistent(self):
        for name, spec in CATALOGUE.items():
            assert spec.name == name
            assert spec.high > spec.low
            if spec.log:
                assert spec.low > 0


class TestModes:
    def test_multiply_costs_one_dimension(self, par, options):
        layout = layout_for(par, options, ["cmlt"])  # 3 landuse values in the template
        assert layout.n_parameters == 1
        assert layout.names == ["cmlt"]

    def test_substitute_costs_one_dimension_per_value(self, par, options):
        layout = layout_for(par, options, ["ttmp"])  # 3 landuse values
        assert layout.n_parameters == 3
        assert layout.names == ["ttmp[0]", "ttmp[1]", "ttmp[2]"]

    def test_single_valued_substitute_keeps_a_bare_label(self, par, options):
        assert layout_for(par, options, ["gratk"]).names == ["gratk"]

    def test_multiply_scales_the_whole_vector(self, par, options):
        layout = layout_for(par, options, ["cmlt"])
        base = par.values["cmlt"]
        values = layout.to_par_values([2.0])
        np.testing.assert_allclose(values["cmlt"], base * 2.0)

    def test_substitute_replaces_one_value(self, par, options):
        layout = layout_for(par, options, ["ttmp"])
        base = par.values["ttmp"]
        values = layout.to_par_values([1.0, base[1], base[2]])
        assert values["ttmp"][0] == 1.0
        np.testing.assert_allclose(values["ttmp"][1:], base[1:])

    def test_multiplier_bounds_keep_every_class_in_range(self, par):
        spec = CATALOGUE["wcfc"]
        base = par.values["wcfc"]
        low, high = multiplier_bounds(spec, base)
        assert (base * low).min() >= spec.low - 1e-12
        assert (base * high).max() <= spec.high + 1e-12

    def test_all_zero_multiply_row_is_dropped(self, par, options):
        # A multiplier cannot move a row of zeros: the dimension would be inert.
        with pytest.warns(UserWarning, match="positive template value"):
            layout = build_layout(par, options, ["cmlt", "deadpar"],
                                  warn_unrequested=False)
        assert "deadpar" not in layout.active
        assert "deadpar" in layout.dropped

    def test_clip_is_applied_in_physical_space(self, par, options):
        layout = layout_for(par, options, ["wcfc"])
        values = layout.to_par_values(layout.bounds()[1])  # push to the top
        assert values["wcfc"].max() <= CATALOGUE["wcfc"].clip[1] + 1e-12


class TestRoutineGating:
    def test_parameter_dropped_when_routine_is_off(self, par, options):
        assert options["snowmeltmodel"] == 0
        with pytest.warns(UserWarning, match="needs snowmeltmodel=2"):
            layout = build_layout(par, options, ["cmlt", "snalbmin"],
                                  warn_unrequested=False)
        assert layout.active == ("cmlt",)
        assert "snalbmin" in layout.dropped

    def test_parameter_active_when_routine_is_on(self, par, options):
        on = effective_options(options, {"snowmeltmodel": 2})
        layout = layout_for(par, on, ["cmlt", "snalbmin", "cmrad"])
        assert layout.active == ("cmlt", "snalbmin", "cmrad")

    def test_is_enabled_direct(self):
        spec = CATALOGUE["snalbmin"]
        assert not is_enabled(spec, {"snowmeltmodel": 0})
        assert is_enabled(spec, {"snowmeltmodel": 2})
        assert is_enabled(CATALOGUE["cmlt"], {})  # ungated

    def test_absent_from_par_txt_is_dropped(self, par, options):
        # ``srrate`` is commented out in the template, so HYPE would ignore it.
        with pytest.warns(UserWarning, match="absent from par.txt"):
            layout = build_layout(par, options, ["cmlt", "srrate"],
                                  warn_unrequested=False)
        assert "srrate" in layout.dropped

    def test_warns_about_enabled_but_unrequested(self, par, options):
        with pytest.warns(UserWarning, match="not requested"):
            build_layout(par, options, ["cmlt"], warn_unrequested=True)

    def test_uncatalogued_name_warns_and_uses_fallback(self, par, options):
        with pytest.warns(UserWarning, match="not in the HYPE catalogue"):
            layout = build_layout(par, options, ["sswcorr"], warn_unrequested=False)
        assert layout.n_parameters == 1
        assert layout.active == ("sswcorr",)

    def test_explicit_spec_overrides_the_catalogue(self, par, options):
        spec = HypeParameter(name="cmlt", low=2.0, high=3.0, mode="substitute")
        layout = layout_for(par, options, ["cmlt"], specs={"cmlt": spec})
        assert layout.n_parameters == 3  # substitute now, not multiply
        np.testing.assert_allclose(layout.bounds()[0], 2.0)

    def test_nothing_left_raises(self, par, options):
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="No HYPE parameters left"):
                build_layout(par, options, ["snalbmin"], warn_unrequested=False)


class TestTransforms:
    @pytest.fixture
    def layout(self, par, options):
        # A mix of log/linear and multiply/substitute dimensions.
        return layout_for(par, options, ["wcfc", "cmlt", "ttmp", "preccorr"])

    def test_bounds_map_to_the_unit_box(self, layout):
        low, high = layout.bounds()
        np.testing.assert_allclose(layout.to_search(low), 0.0, atol=1e-12)
        np.testing.assert_allclose(layout.to_search(high), 1.0, atol=1e-12)

    def test_search_bounds_min_max_is_a_no_op(self, layout):
        # The estimator does np.minimum/np.maximum on the transformed ends; a monotone
        # transform must make that a no-op.
        low, high = layout.bounds()
        lo_s, hi_s = layout.to_search(low), layout.to_search(high)
        np.testing.assert_allclose(np.minimum(lo_s, hi_s), lo_s)
        np.testing.assert_allclose(np.maximum(lo_s, hi_s), hi_s)

    @pytest.mark.parametrize("shape", ["1d", "2d"])
    def test_round_trip(self, layout, shape):
        # The estimator calls to_search on 1-D bounds and to_model on 2-D populations, so
        # both shapes must work - this is the ``[:, mask]`` versus ``[..., mask]`` detector.
        rng = np.random.default_rng(0)
        u = rng.uniform(0, 1, (5, layout.n_parameters))
        u = u[0] if shape == "1d" else u
        np.testing.assert_allclose(layout.to_search(layout.to_model(u)), u, atol=1e-12)

    def test_transform_does_not_mutate_or_alias_input(self, layout):
        u = np.full((3, layout.n_parameters), 0.5)
        original = u.copy()
        out = layout.to_model(u)
        out[:] = -1.0
        np.testing.assert_array_equal(u, original)

    def test_to_model_clips_to_physical_bounds(self, layout):
        low, high = layout.bounds()
        out = layout.to_model(np.array([[-0.5] * layout.n_parameters,
                                        [1.5] * layout.n_parameters]))
        assert np.all(out[0] >= low - 1e-12)
        assert np.all(out[1] <= high + 1e-12)

    def test_log_dimensions_are_searched_logarithmically(self, layout):
        # Midpoint of a log dimension is the geometric mean, not the arithmetic one.
        idx = layout.names.index("cmlt")
        low, high = layout.bounds()
        mid = layout.to_model(np.full(layout.n_parameters, 0.5))[idx]
        np.testing.assert_allclose(mid, np.sqrt(low[idx] * high[idx]), rtol=1e-9)

    def test_linear_dimensions_stay_linear(self, layout):
        idx = layout.names.index("ttmp[0]")
        low, high = layout.bounds()
        mid = layout.to_model(np.full(layout.n_parameters, 0.5))[idx]
        np.testing.assert_allclose(mid, 0.5 * (low[idx] + high[idx]), rtol=1e-9)


class TestFingerprint:
    def test_changes_with_options(self, par, options):
        a = layout_for(par, options, ["cmlt"])
        b = layout_for(par, effective_options(options, {"snowmeltmodel": 2}), ["cmlt"])
        assert a.fingerprint != b.fingerprint

    def test_changes_with_parameter_set(self, par, options):
        a = layout_for(par, options, ["cmlt"])
        b = layout_for(par, options, ["cmlt", "wcfc"])
        assert a.fingerprint != b.fingerprint

    def test_changes_with_template_values(self, par, options):
        a = layout_for(par, options, ["cmlt"])
        b = layout_for(par.with_values({"cmlt": [9.0, 9.0, 9.0]}), options, ["cmlt"])
        assert a.fingerprint != b.fingerprint

    def test_stable_across_rebuilds(self, par, options):
        assert (layout_for(par, options, ["cmlt", "wcfc"]).fingerprint
                == layout_for(par, options, ["cmlt", "wcfc"]).fingerprint)


class TestOptparEntries:
    def test_arity_matches_par_txt(self, par, options):
        layout = layout_for(par, options, ["wcfc", "preccorr", "gratk"])
        for name, low, high, step in optpar_entries(layout):
            assert low.size == par.arity(name)
            assert high.size == step.size == low.size
            assert np.all(high > low)

    def test_substitute_uses_the_spec_bounds(self, par, options):
        layout = layout_for(par, options, ["preccorr"])
        (_, low, high, _), = optpar_entries(layout)
        np.testing.assert_allclose(low, CATALOGUE["preccorr"].low)
        np.testing.assert_allclose(high, CATALOGUE["preccorr"].high)

    def test_multiply_expands_to_per_class_physical_ranges(self, par, options):
        layout = layout_for(par, options, ["cmlt"])
        (_, low, high, _), = optpar_entries(layout)
        assert low.size == 3 and np.all(high > low)

    def test_clip_of_one_parameter_does_not_leak_into_another(self, par, options):
        # preccorr has no clip; wcfc does. A leaked loop variable would clamp preccorr
        # to wcfc's range.
        layout = layout_for(par, options, ["wcfc", "preccorr"])
        entries = dict((name, (low, high)) for name, low, high, _ in optpar_entries(layout))
        low, high = entries["preccorr"]
        np.testing.assert_allclose(low, CATALOGUE["preccorr"].low)
        np.testing.assert_allclose(high, CATALOGUE["preccorr"].high)


class TestDates:
    def test_ordinals_round_trip(self):
        stamps = np.array(["1980-01-01", "1999-12-31"], dtype="datetime64[D]")
        np.testing.assert_array_equal(from_ordinals(as_ordinals(stamps)), stamps)

    def test_as_X_is_a_column(self):
        X = as_X(np.array(["1990-01-01", "1990-01-02"], dtype="datetime64[D]"))
        assert X.shape == (2, 1)

    def test_numeric_input_rejected(self):
        with pytest.raises(TypeError, match="expects dates"):
            as_ordinals(np.array([1.0, 2.0]))

    def test_window_ordinals_counts_inclusively(self):
        t0, n = window_ordinals("1991-01-01", "1991-01-31")
        assert n == 31

    def test_inverted_window_raises(self):
        with pytest.raises(ValueError, match="precedes"):
            window_ordinals("1991-02-01", "1991-01-01")

    def test_resolves_gaps_duplicates_and_arbitrary_order(self):
        t0, n = window_ordinals("1991-01-01", "1991-12-31")
        wanted = np.array([t0 + 100, t0 + 5, t0 + 5, t0], dtype=float)
        np.testing.assert_array_equal(resolve_indices(wanted, t0, n), [100, 5, 5, 0])

    def test_scaled_column_raises_with_a_hint(self):
        t0, n = window_ordinals("1991-01-01", "1991-12-31")
        with pytest.raises(ValueError, match="StandardScaler"):
            resolve_indices(np.array([-1.73, 0.42]), t0, n)

    def test_nanosecond_ordinals_raise(self):
        t0, n = window_ordinals("1991-01-01", "1991-12-31")
        with pytest.raises(ValueError, match="implausibly large"):
            resolve_indices(np.array([6.6e16]), t0, n)

    def test_out_of_window_raises_naming_the_window(self):
        t0, n = window_ordinals("1991-01-01", "1991-12-31")
        with pytest.raises(ValueError, match="outside the simulated window"):
            resolve_indices(np.array([float(t0 - 1)]), t0, n)

    def test_empty_input(self):
        assert resolve_indices(np.array([]), 0, 10).size == 0
