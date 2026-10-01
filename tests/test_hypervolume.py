"""Hypervolume indicators: analytic values, the staircase reduction, and monotonicity.

Pure NumPy — no fitting, so the whole file is fast.
"""

import numpy as np
import pytest

from foresight_gpu.domination import (
    HV_CLIP_WARN_FRACTION,
    DoubleParetoSorter,
    default_hv_reference,
    double_pareto_hypervolume,
    hypervolume,
    non_dominated_mask,
)
from foresight_gpu.metrics import kge, kge_prime, mae, mse, nse, rmse


def _front(objectives):
    """Front-0 rows of a 2-column objective array, eta-ordered."""
    return objectives[DoubleParetoSorter().sort(objectives)[0]]


def _random_front(seed, n_max=60):
    """A random front-0. Losses are non-negative, as every registry metric's loss is."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, n_max))
    return _front(np.column_stack([rng.random(n), rng.random(n) * 1.6]))


# --- the indicator, analytic ---------------------------------------------------------

def test_worked_example():
    """Front (0.1, 0.8), (0.4, 0.2), (0.9, 0.5) with R = 1 integrates to D = 0.525."""
    obj = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]])
    # Cells beside the minimum split at 0.25 and 0.65:
    # D = 0.1*1 (uncovered) + 0.15*0.8 + 0.15*0.2 + 0.25*0.2 + 0.25*0.5 + 0.1*1 (uncovered)
    assert double_pareto_hypervolume(obj, 1.0) == pytest.approx(0.475)


def test_full_coverage_zero_loss_is_one():
    obj = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert double_pareto_hypervolume(obj, 1.0) == pytest.approx(1.0)


def test_no_coverage_is_zero():
    """A front spanning no eta covers nothing, however good its loss."""
    assert double_pareto_hypervolume(np.array([[0.5, 0.0]]), 1.0) == 0.0


def test_uncovered_span_is_penalised():
    """Halving the covered span at zero loss halves the indicator."""
    wide = double_pareto_hypervolume(np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0)
    narrow = double_pareto_hypervolume(np.array([[0.25, 0.0], [0.75, 0.0]]), 1.0)
    assert narrow == pytest.approx(0.5 * wide)


def test_losses_clipped_at_reference():
    """A particle worse than P is worth exactly as much as one sitting at P: nothing."""
    at_p = np.array([[0.1, 1.0], [0.4, 0.2], [0.9, 1.0]])
    beyond = np.array([[0.1, 10.0], [0.4, 0.2], [0.9, 10.0]])
    assert double_pareto_hypervolume(beyond, 1.0) == pytest.approx(
        double_pareto_hypervolume(at_p, 1.0)
    )


def test_failed_particle_contributes_zero():
    """A non-finite loss lengthens the covered span at zero height."""
    base = np.array([[0.3, 0.2], [0.7, 0.4]])
    failed = np.vstack([base, [0.0, np.nan]])
    assert double_pareto_hypervolume(failed, 1.0) == pytest.approx(
        double_pareto_hypervolume(base, 1.0)
    )


def test_lower_loss_raises_the_indicator():
    better = np.array([[0.1, 0.4], [0.5, 0.1], [0.9, 0.4]])
    worse = np.array([[0.1, 0.6], [0.5, 0.3], [0.9, 0.6]])
    assert double_pareto_hypervolume(better, 1.0) > double_pareto_hypervolume(worse, 1.0)


def test_indicator_is_bounded():
    for seed in range(50):
        hv = double_pareto_hypervolume(_random_front(seed), 1.0)
        assert 0.0 <= hv <= 1.0


def test_reference_scales_out_for_dimensionless_losses():
    """Doubling both P and every loss leaves the normalised indicator unchanged."""
    obj = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]])
    doubled = obj.copy()
    doubled[:, 1] *= 2.0
    assert double_pareto_hypervolume(doubled, 2.0) == pytest.approx(
        double_pareto_hypervolume(obj, 1.0)
    )


def test_rejects_bad_reference():
    obj = np.array([[0.1, 0.5], [0.9, 0.5]])
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            double_pareto_hypervolume(obj, bad)


def test_rejects_bad_interpolation():
    with pytest.raises(ValueError):
        double_pareto_hypervolume(
            np.array([[0.1, 0.5], [0.9, 0.5]]), 1.0, interpolation="cubic"
        )


def test_details_decomposition():
    obj = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]])
    d = double_pareto_hypervolume(obj, 1.0, details=True)
    assert d["hv"] == pytest.approx(0.475)
    assert d["dispersion"] == pytest.approx(0.525)
    assert d["integral"] == pytest.approx(0.525)
    assert d["coverage"] == pytest.approx(0.8)
    assert (d["eta_min"], d["eta_max"], d["n_front"]) == (0.1, 0.9, 3)
    assert d["front_min_loss"] == pytest.approx(0.2)
    assert d["reference"] == pytest.approx(1.0)


# --- step vs linear ------------------------------------------------------------------

def test_linear_never_below_step():
    for seed in range(100):
        obj = _random_front(seed)
        step = double_pareto_hypervolume(obj, 1.0, interpolation="step")
        linear = double_pareto_hypervolume(obj, 1.0, interpolation="linear")
        assert linear >= step - 1e-12


def test_linear_equals_step_on_a_flat_front():
    obj = np.array([[0.2, 0.3], [0.5, 0.3], [0.8, 0.3]])
    assert double_pareto_hypervolume(obj, 1.0, interpolation="linear") == pytest.approx(
        double_pareto_hypervolume(obj, 1.0, interpolation="step")
    )


def test_linear_rejected_for_multiple_loss_axes():
    obj = np.array([[0.2, 0.3, 0.4], [0.8, 0.4, 0.3]])
    with pytest.raises(NotImplementedError):
        double_pareto_hypervolume(obj, 1.0, interpolation="linear")


# --- the reduction: split-half hypervolume plus the two midpoint half-cells ----------

def _split_reference(objectives, reference):
    """Two standard minimisation hypervolumes, split at the minimum, summed."""
    eta, loss = objectives[:, 0], np.clip(objectives[:, 1], 0.0, reference)
    anchor = eta[int(np.argmin(loss))]
    left, right = eta <= anchor, eta >= anchor
    total = hypervolume(
        np.column_stack([eta[left], loss[left]]), np.array([anchor, reference])
    )
    total += hypervolume(
        np.column_stack([-eta[right], loss[right]]), np.array([-anchor, reference])
    )
    return total


def _midpoint_cells(objectives, reference):
    """Area the step rule adds over the standard HV: half of each cell beside the minimum,
    lowered from the neighbour's loss to the minimum loss."""
    e = objectives[:, 0]
    l = np.clip(objectives[:, 1], 0.0, reference)
    a = int(np.argmin(l))
    extra = 0.0
    if a > 0:
        extra += 0.5 * (e[a] - e[a - 1]) * (l[a - 1] - l[a])
    if a < e.size - 1:
        extra += 0.5 * (e[a + 1] - e[a]) * (l[a + 1] - l[a])
    return extra


def test_step_equals_split_hypervolume_plus_midpoint_cells():
    """The O(m) closed form is exactly the standard hypervolume plus the two half-cells
    that give the minimum positive width. Keep it green."""
    for seed in range(300):
        obj = _random_front(seed)
        if obj.shape[0] < 2:
            continue
        closed = double_pareto_hypervolume(obj, 1.0)  # box = R * span = 1
        expected = _split_reference(obj, 1.0) + _midpoint_cells(obj, 1.0)
        assert closed == pytest.approx(expected, abs=1e-12)


def test_improving_only_the_minimum_raises_the_indicator():
    """The case the midpoint cells exist for: under plain max(l_i, l_i+1) it was flat."""
    obj = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]])
    better = obj.copy()
    better[1, 1] = 0.05
    assert double_pareto_hypervolume(better, 1.0) > double_pareto_hypervolume(obj, 1.0)
    assert _split_reference(better, 1.0) == pytest.approx(_split_reference(obj, 1.0))


def test_minimum_at_the_front_edge_splits_one_cell():
    obj = np.array([[0.2, 0.1], [0.6, 0.4], [1.0, 0.7]])
    # 0.2*1 (uncovered) + 0.2*0.1 + 0.2*0.4 + 0.4*0.7
    assert double_pareto_hypervolume(obj, 1.0) == pytest.approx(1.0 - 0.58)


def test_two_point_front():
    obj = np.array([[0.0, 0.6], [1.0, 0.2]])
    assert double_pareto_hypervolume(obj, 1.0) == pytest.approx(1.0 - (0.5 * 0.6 + 0.5 * 0.2))


# --- the general N-dimensional hypervolume -------------------------------------------

def test_hypervolume_single_point_box():
    for d in range(1, 5):
        point = np.zeros((1, d))
        ref = np.full(d, 2.0)
        assert hypervolume(point, ref) == pytest.approx(2.0**d)


def test_hypervolume_ignores_points_beyond_reference():
    pts = np.array([[0.5, 0.5], [3.0, 0.1], [0.1, 3.0]])
    assert hypervolume(pts, np.array([1.0, 1.0])) == pytest.approx(0.25)


def test_hypervolume_two_points_2d():
    pts = np.array([[0.0, 1.0], [1.0, 0.0]])
    # (0,1) covers x in [0,1] at height 1; (1,0) covers x in [1,2] at height 2.
    assert hypervolume(pts, np.array([2.0, 2.0])) == pytest.approx(3.0)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_hypervolume_matches_monte_carlo(d):
    rng = np.random.default_rng(d)
    ref = np.ones(d)
    pts = rng.random((6, d)) * 0.9
    pts = pts[non_dominated_mask(pts)]
    sample = rng.random((400_000, d))
    dominated = (sample[:, None, :] >= pts[None, :, :]).all(-1).any(-1)
    assert hypervolume(pts, ref) == pytest.approx(dominated.mean(), abs=0.01)


def test_hypervolume_monotone():
    rng = np.random.default_rng(0)
    ref = np.ones(3)
    pts = rng.random((8, 3)) * 0.8
    base = hypervolume(pts, ref)
    assert hypervolume(np.vstack([pts, pts[0] + 0.05]), ref) == pytest.approx(base)
    assert hypervolume(np.vstack([pts, pts[0] * 0.5]), ref) >= base - 1e-12


def test_non_dominated_mask_keeps_duplicates():
    pts = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    assert non_dominated_mask(pts).tolist() == [True, True, False]


# --- the k > 1 seam -------------------------------------------------------------------

def test_two_loss_axes_is_monotone():
    """The rejected per-cell framing collapses 100x here; the split framing must not.

    Front A=(0.2,(0,10)), B=(0.8,(10,0)) with P=(20,20); inserting the mediocre
    D=(0.21,(19,19)) must not reduce the indicator.
    """
    base = np.array([[0.2, 0.0, 10.0], [0.8, 10.0, 0.0]])
    with_mediocre = np.vstack([base, [0.21, 19.0, 19.0]])
    P = np.array([20.0, 20.0])
    assert double_pareto_hypervolume(with_mediocre, P) >= (
        double_pareto_hypervolume(base, P) - 1e-12
    )


def test_two_loss_axes_stays_bounded():
    obj = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]])
    doubled = np.column_stack([obj[:, 0], obj[:, 1], obj[:, 1]])
    assert 0.0 <= double_pareto_hypervolume(doubled, 1.0) <= 1.0


# --- the default reference ---------------------------------------------------------------

@pytest.mark.parametrize("metric", [nse, kge, kge_prime])
def test_default_reference_is_one_for_efficiency_metrics(metric, rng):
    """loss = 1 - value is dimensionless, so P = 1 is the metric-value-0 no-skill line."""
    y = rng.gamma(2.0, 1.0, 400)
    assert default_hv_reference(metric, y) == 1.0


def test_default_reference_for_error_metrics(rng):
    y = rng.gamma(2.0, 1.0, 400)
    assert default_hv_reference(mse, y) == pytest.approx(np.var(y))
    assert default_hv_reference(rmse, y) == pytest.approx(np.std(y))
    assert default_hv_reference(mae, y) == pytest.approx(np.mean(np.abs(y - y.mean())))


def test_default_reference_is_nses_denominator(rng):
    """nse.loss == mse.loss / default_hv_reference(mse, y) — the two branches are one rule."""
    y = rng.gamma(2.0, 1.0, 300)
    sim = y[:, None] + rng.standard_normal((300, 5))
    assert nse.loss(sim, y) == pytest.approx(
        mse.loss(sim, y) / default_hv_reference(mse, y)
    )


def test_default_reference_never_returns_the_degenerate_kge_value():
    """The trap the analytic branch avoids.

    A constant simulation has zero variance, so Pearson r is 0/0 and KGE' gamma is x/0.
    Whether that surfaces as ``nan`` or as ~1e15 depends on whether ``mean()`` of the
    constant array leaves a 2e-16 residue, i.e. on the data — so the value is not even
    deterministic in kind. Both appear across these seeds.
    """
    seen = set()
    for seed in range(6):
        gen = np.random.default_rng(seed)
        y = gen.gamma(2.0, 1.0, 400)
        const = np.full_like(y, y.mean())
        for metric in (kge, kge_prime):
            value = metric.loss(const, y)
            assert np.isnan(value) or value > 1e10 or value == pytest.approx(np.sqrt(2))
            seen.add("nan" if np.isnan(value) else "huge" if value > 1e10 else "sqrt2")
            assert default_hv_reference(metric, y) == 1.0
    assert {"nan", "huge"} <= seen  # both failure modes really do occur


def test_default_reference_honours_an_explicit_predictor(rng):
    y = rng.gamma(2.0, 1.0, 200)
    assert default_hv_reference(mse, y, predictor=0.0) == pytest.approx(np.mean(y**2))


# --- clipped_fraction: making a too-tight ceiling visible ----------------------------

def test_clipped_fraction_counts_the_front_at_the_ceiling():
    # V-shaped, so all three points really are on front 0 (see the module docstring).
    front = np.array([[0.1, 5.0], [0.5, 0.2], [0.9, 0.3]])
    assert double_pareto_hypervolume(front, 10.0, details=True)["clipped_fraction"] == 0.0
    assert double_pareto_hypervolume(front, 1.0, details=True)["clipped_fraction"] == 1 / 3
    assert double_pareto_hypervolume(front, 0.1, details=True)["clipped_fraction"] == 1.0


def test_clipped_fraction_counts_failed_particles():
    """Non-finite losses are +inf, so they sit at the ceiling and must be counted."""
    front = np.array([[0.0, np.nan], [0.5, 0.2], [1.0, 0.3]])
    parts = double_pareto_hypervolume(front, 1.0, details=True)
    assert parts["clipped_fraction"] == pytest.approx(1 / 3)


def test_clipped_fraction_matches_the_documented_climatology_regime(rng):
    """The measured case this diagnostic exists for: P=1 clips most of a real front."""
    losses = np.concatenate([rng.uniform(0.8, 1.0, 5), rng.uniform(1.2, 6.0, 20)])
    front = _front(np.column_stack([np.linspace(0.02, 0.98, losses.size), losses]))
    parts = double_pareto_hypervolume(front, 1.0, details=True)
    assert parts["clipped_fraction"] > HV_CLIP_WARN_FRACTION


# --- space="log10": the symmetric box ------------------------------------------------

def test_log10_no_skill_front_scores_one_half():
    """Constant loss 1 over full coverage sits exactly mid-box: log10(1) = 0."""
    flat = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]])
    assert double_pareto_hypervolume(flat, 2.0, space="log10") == pytest.approx(0.5)


def test_log10_perfect_front_clips_to_the_floor_not_minus_inf():
    """loss == 0 is reachable for every registry metric; log10 -> -inf must clip to -P."""
    perfect = np.array([[0.0, 0.0], [1.0, 0.0]])
    hv = double_pareto_hypervolume(perfect, 2.0, space="log10")
    assert np.isfinite(hv) and hv == pytest.approx(1.0)


def test_log10_failed_front_is_zero_not_nan():
    bad = np.array([[0.0, np.nan], [1.0, np.inf]])
    assert double_pareto_hypervolume(bad, 2.0, space="log10") == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(40))
def test_log10_indicator_is_bounded(seed):
    front = _random_front(seed)
    hv = double_pareto_hypervolume(front, 2.0, space="log10")
    assert 0.0 <= hv <= 1.0


@pytest.mark.parametrize("seed", range(20))
def test_front_zero_is_identical_in_both_spaces(seed):
    """log10 is monotone, so it cannot reorder the front — a precomputed front= stays valid."""
    rng_ = np.random.default_rng(seed)
    n = int(rng_.integers(6, 40))
    obj = np.column_stack([rng_.random(n), rng_.random(n) * 3.0])
    linear = double_pareto_hypervolume(obj, 2.0, details=True)
    log10 = double_pareto_hypervolume(obj, 2.0, space="log10", details=True)
    assert linear["n_front"] == log10["n_front"]
    assert (linear["eta_min"], linear["eta_max"]) == (log10["eta_min"], log10["eta_max"])


def test_log10_spreads_the_usable_range():
    """The reason the space exists.

    A wide linear ceiling spends almost the whole box on losses nobody cares about, so
    fronts that differ a lot where it matters score almost the same. Measured on a real run
    the usable range was 1.2e-3 linear against 8.5e-3 log10; this reproduces the effect on
    a family of progressively better synthetic fronts.
    """
    eta = np.linspace(0.02, 0.98, 30)
    fronts = [_front(np.column_stack([eta, np.full(30, level)]))
              for level in (3.0, 2.0, 1.5, 1.0, 0.7, 0.5)]
    linear = [double_pareto_hypervolume(f, 100.0) for f in fronts]
    log10 = [double_pareto_hypervolume(f, 2.0, space="log10") for f in fronts]

    assert all(np.diff(linear) > 0) and all(np.diff(log10) > 0)   # both see the improvement
    assert (max(log10) - min(log10)) > 5 * (max(linear) - min(linear))


def test_space_is_validated():
    obj = np.array([[0.1, 0.8], [0.9, 0.5]])
    with pytest.raises(ValueError, match="space must be"):
        double_pareto_hypervolume(obj, 1.0, space="ln")


def test_linear_space_is_the_default_and_unchanged():
    """Guard: the space switch must not perturb the existing numbers."""
    obj = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]])
    assert double_pareto_hypervolume(obj, 1.0) == pytest.approx(0.475)
    assert double_pareto_hypervolume(obj, 1.0, space="linear") == pytest.approx(0.475)
    assert double_pareto_hypervolume(obj, 1.0, details=True)["space"] == "linear"
