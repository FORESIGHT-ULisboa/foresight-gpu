"""Adaptive hypervolume reference, patience replay, pruning, diagnostics_ and verbose."""

import numpy as np
import pytest

from foresight_gpu import (
    HV_REFERENCE_MARGIN,
    GPURegressor,
    MLPModel,
    double_pareto_hypervolume,
    reference_nadir,
)
from foresight_gpu import estimator as est


@pytest.fixture
def split(rng):
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.standard_normal(300)
    return X[:240], y[:240], X[240:], y[240:]


def _front(seed, n=12, worst=3.0):
    """A V-shaped front: eta-ordered, loss falling to an anchor then rising."""
    r = np.random.default_rng(seed)
    eta = np.sort(r.uniform(0.0, 1.0, n))
    a = n // 2
    loss = np.abs(np.arange(n) - a) / a * worst + r.uniform(0.0, 0.1, n)
    return eta, loss


# --- reference_nadir -----------------------------------------------------------------------

def test_reference_nadir_linear_ignores_non_finite():
    assert reference_nadir([0.5, 2.0, np.inf, np.nan], "linear") == 2.0


def test_reference_nadir_log10_is_symmetric_and_ignores_zero():
    # |log10| of 1e-3 (3) beats 50 (1.7); 0 and inf clip whatever R is
    assert reference_nadir([1e-3, 50.0, 0.0, np.inf], "log10") == pytest.approx(3.0)
    assert reference_nadir([np.inf, np.nan], "log10") == 0.0


def test_reference_nadir_rejects_unknown_space():
    with pytest.raises(ValueError, match="space"):
        reference_nadir([1.0], "sqrt")


# --- rescoring ------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["linear", "log10"])
def test_a_front_that_never_changes_never_improves(space):
    """The fatal case of an un-rescored adaptive R: identical fronts must score identically
    after the reference moves, and equal a direct evaluation at the new R."""
    gpu = GPURegressor(hv_space=space)
    eta, loss = _front(0)
    fixed, reference = {"val": None}, {"val": 0.0}
    assert gpu._raise_reference(reference, fixed, "val", loss, []) is None  # first check
    R0 = reference["val"]
    assert R0 == pytest.approx(HV_REFERENCE_MARGIN * reference_nadir(loss, space))
    before = gpu._hv(eta, loss, R0)["hv"]
    history = [{"iteration": i, "val_eta": eta, "val_loss": loss, "val_hv": before}
               for i in range(4)]

    outlier = np.append(loss, 40.0)               # a much worse particle appears
    assert gpu._raise_reference(reference, fixed, "val", outlier, history) == R0
    R1 = reference["val"]
    assert R1 > R0

    direct = double_pareto_hypervolume(np.column_stack([eta, loss]), R1, space=space,
                                       front=np.arange(eta.size))
    assert all(h["val_hv"] == pytest.approx(direct) for h in history)
    assert all(h["val_clipped_fraction"] == 0.0 for h in history)
    assert history[0]["val_hv"] != pytest.approx(before)    # it was rescored


def test_a_fixed_reference_never_moves():
    gpu = GPURegressor()
    eta, loss = _front(0)
    reference = {"val": 1.0}
    assert gpu._raise_reference(reference, {"val": 1.0}, "val", loss * 100, [{}]) is None
    assert reference["val"] == 1.0


@pytest.mark.parametrize("space", ["linear", "log10"])
@pytest.mark.parametrize("interpolation", ["step", "linear"])
def test_intercept_from_two_references_is_the_limit(space, interpolation):
    """hv is linear in 1/R while nothing clips, so 2 hv(2R) - hv(R) is hv at R -> inf."""
    eta, loss = _front(1)
    R = HV_REFERENCE_MARGIN * reference_nadir(loss, space)
    gpu = GPURegressor(hv_space=space, hv_interpolation=interpolation)
    hv = lambda r: gpu._hv(eta, loss, r)["hv"]  # noqa: E731
    assert 2.0 * hv(2.0 * R) - hv(R) == pytest.approx(hv(1e12), abs=1e-9)


# --- patience replay and pruning ------------------------------------------------------------

def _incremental(scores, tol):
    """The pre-0.7.0 loop, verbatim: the reference the replay must reproduce."""
    best, best_score, no_improve = None, -np.inf, 0
    for i, s in enumerate(scores):
        if np.isfinite(s) and s > best_score + tol:
            best, best_score, no_improve = i, s, 0
        elif best_score <= 0.0:
            pass
        else:
            no_improve += 1
    return best, no_improve


@pytest.mark.parametrize("scores", [
    [0.1, 0.2, 0.2, 0.19, 0.3, 0.3],
    [0.0, 0.0, 0.0, 0.0],                      # zero guard: patience held
    [np.nan, 0.5, 0.4, 0.6000001],             # within tol
])
def test_replay_matches_the_incremental_rule(scores):
    assert est._early_stopping_state(scores, 1e-4) == _incremental(scores, 1e-4)


def test_pruning_never_drops_a_check_the_replay_could_still_pick():
    """Random lines s_k(u) = a_k - b_k u under a non-increasing u, as an adaptive R makes.

    Checked against the replay at every later u, not only at the end."""
    rng = np.random.default_rng(0)
    tol = 1e-3
    for _ in range(300):
        n = 12
        a = rng.uniform(0.5, 1.0, n)
        b = rng.uniform(0.0, 1.0, n)
        u = np.sort(rng.uniform(0.0, 1.0, n))[::-1]
        for t in np.flatnonzero(rng.uniform(size=n) < 0.5)[1:]:
            u[t] = u[t - 1]                            # R did not move at this check
        kept = {}
        for t in range(n):
            scores = a[: t + 1] - b[: t + 1] * u[t]
            best, _ = est._early_stopping_state(scores, tol)
            kept[t] = t
            est._prune_candidates(kept, scores, a[: t + 1], best, tol)
            assert best in kept
            for later_u in np.linspace(0.0, u[t], 7):   # any future reference
                future_best, _ = est._early_stopping_state(
                    a[: t + 1] - b[: t + 1] * later_u, tol
                )
                assert future_best in kept


def test_pruning_does_not_change_the_fit(split, monkeypatch):
    X_tr, y_tr, X_val, y_val = split
    kw = dict(population=80, n_iter=40, check_every=2, n_iter_no_change=6,
              random_state=0)
    pruned = GPURegressor(**kw).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    monkeypatch.setattr(est, "_prune_candidates", lambda *a, **k: None)
    full = GPURegressor(**kw).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert pruned.best_iteration_ == full.best_iteration_
    np.testing.assert_array_equal(pruned.predict(X_val), full.predict(X_val))


def test_best_iteration_is_the_replay_over_the_rescored_history(split):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=80, n_iter=40, check_every=2, n_iter_no_change=5,
                       tol=1e-4, random_state=1).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    best, _ = _incremental([h["val_hv"] for h in gpu.history_], 1e-4)
    assert gpu.history_[best]["iteration"] == gpu.best_iteration_
    # every entry sits on the final reference
    assert all(
        h["val_hv"] == pytest.approx(double_pareto_hypervolume(
            np.column_stack([h["val_eta"], h["val_loss"]]), gpu.hv_reference_,
            front=np.arange(h["val_eta"].size)))
        for h in gpu.history_
    )


# --- one forward per check --------------------------------------------------------------

class _CountingMLP(MLPModel):
    rows = []

    def forward(self, X, params):
        _CountingMLP.rows.append(X.shape[0])
        return super().forward(X, params)


@pytest.mark.parametrize("every", [None, 6, 4])
def test_one_validation_forward_per_generation_and_none_for_training(split, every):
    """Checks and diagnostics share a generation's validation forward; training reuses the
    cached simulations. The retained-model row costs one forward per side, once."""
    X_tr, y_tr, X_val, y_val = split
    _CountingMLP.rows = []
    gpu = GPURegressor(model=_CountingMLP(), population=40, n_iter=13, check_every=3,
                       diagnostics_every=every, n_iter_no_change=99, random_state=0)
    gpu.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    rows = np.asarray(_CountingMLP.rows)
    checked = {h["iteration"] for h in gpu.history_}
    computed = {it for it in range(13) if every and (it % every == 0 or it == 12)}
    retained_extra = int(gpu.best_iteration_ not in computed)
    assert np.sum(rows == len(X_val)) == len(checked | computed) + retained_extra
    # initial + one per generation, + the retained row when it was not computed live
    assert np.sum(rows == len(X_tr)) == gpu.n_iter_ + 1 + retained_extra


# --- diagnostics_ ---------------------------------------------------------------------

def test_diagnostics_follow_their_own_cadence(split):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=13, check_every=3, diagnostics_every=5,
                       n_iter_no_change=99, random_state=0).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert [h["iteration"] for h in gpu.history_] == [0, 3, 6, 9, 12]
    its = [d["iteration"] for d in gpu.diagnostics_]
    assert set(its) >= {0, 5, 10, 12} and its == sorted(its)
    keys = {f"{side}_{k}" for side in ("train", "val")
            for k in ("reliability", "resolution", "crps")}
    assert all(keys <= set(d) for d in gpu.diagnostics_)
    # history_ is the hv record only; diagnostics_ is reference-free
    assert not any("reliability" in k for h in gpu.history_ for k in h)
    assert not any(k.endswith("_hv") for d in gpu.diagnostics_ for k in d)


def test_the_retained_model_always_gets_a_row(split):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=60, n_iter=40, check_every=2, n_iter_no_change=3,
                       random_state=0).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert gpu.diagnostics_ == [d for d in gpu.diagnostics_ if d["retained"]]
    assert gpu.diagnostics_[0]["iteration"] == gpu.best_iteration_


def test_retained_row_recomputed_at_the_end_equals_the_live_row(split):
    """The end-of-fit row re-simulates the kept ensemble; it must match what the check
    would have produced live, so the plotted star is the model predict uses."""
    X_tr, y_tr, X_val, y_val = split
    kw = dict(population=60, n_iter=40, check_every=2, n_iter_no_change=3,
              random_state=0)
    live = GPURegressor(diagnostics_every=2, **kw).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    late = GPURegressor(**kw).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert live.best_iteration_ == late.best_iteration_
    a = next(d for d in live.diagnostics_ if d["retained"])
    b = late.diagnostics_[0]
    for k in a:
        assert a[k] == pytest.approx(b[k]), k


def test_diagnostics_without_validation_are_training_only(split):
    X_tr, y_tr, _, _ = split
    gpu = GPURegressor(population=40, n_iter=8, check_every=2, diagnostics_every=4,
                       random_state=0).fit(X_tr, y_tr)
    assert [d["iteration"] for d in gpu.diagnostics_] == [0, 4, 7]
    assert gpu.diagnostics_[-1]["retained"]       # the last generation is the kept one
    assert not any(k.startswith("val_") for d in gpu.diagnostics_ for k in d)


def test_records_are_plain_tables(split):
    """Both records must give a DataFrame joinable on iteration with no helper."""
    pd = pytest.importorskip("pandas")
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=12, check_every=2, diagnostics_every=4,
                       n_iter_no_change=99, random_state=0).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val)
    merged = pd.merge(pd.DataFrame(gpu.history_), pd.DataFrame(gpu.diagnostics_),
                      on="iteration", how="outer")
    assert {"train_hv", "val_hv", "val_reliability", "train_crps"} <= set(merged)
    assert merged["iteration"].is_monotonic_increasing


# --- verbose ----------------------------------------------------------------------------

def test_verbose_prints_at_the_diagnostics_cadence(split, capsys):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=10, check_every=3, diagnostics_every=4,
                       n_iter_no_change=99, random_state=0, verbose=1).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val)
    lines = capsys.readouterr().out.splitlines()
    header, rows = lines[0], [ln for ln in lines if ln[:6].strip().isdigit()]
    for col in ("Iter", "Train HV", "Val HV", "Train alpha", "Val alpha", "Train pi",
                "Val pi", "Train CRPS", "Val CRPS", "Front", "Elapsed"):
        assert col in header
    assert [int(r.split()[0]) for r in rows] == [0, 4, 8, 9]   # every 4 + the last
    assert any(ln.startswith("Finished 10 iterations; restored iteration") for ln in lines)
    assert lines[-1].startswith("Retained model, val: alpha")


def test_verbose_without_diagnostics_prints_hv_at_each_check(split, capsys):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=10, check_every=3, n_iter_no_change=99,
                       random_state=0, verbose=True).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val)
    lines = capsys.readouterr().out.splitlines()
    assert "Val HV" in lines[0] and "alpha" not in lines[0]
    rows = [ln for ln in lines if ln[:6].strip().isdigit()]
    assert [int(r.split()[0]) for r in rows] == [h["iteration"] for h in gpu.history_]
    # on the check grid the printed hv is the stored one (up to later rescoring)
    assert float(rows[0].split()[1]) == pytest.approx(gpu.history_[0]["train_hv"], abs=1e-4)


def test_verbose_without_validation_has_train_columns_only(split, capsys):
    X_tr, y_tr, _, _ = split
    GPURegressor(population=40, n_iter=4, check_every=2, diagnostics_every=2,
                 random_state=0, verbose=True).fit(X_tr, y_tr)
    out = capsys.readouterr().out
    assert "Train HV" in out and "Train alpha" in out and "Val" not in out


def test_quiet_by_default(split, capsys):
    X_tr, y_tr, X_val, y_val = split
    GPURegressor(population=40, n_iter=4, check_every=2, diagnostics_every=2,
                 random_state=0).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert capsys.readouterr().out == ""


def test_verbose_does_not_change_what_is_stored(split):
    X_tr, y_tr, X_val, y_val = split
    kw = dict(population=40, n_iter=10, check_every=3, diagnostics_every=4,
              n_iter_no_change=99, random_state=0)
    quiet = GPURegressor(**kw).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    loud = GPURegressor(verbose=1, **kw).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert [h["val_hv"] for h in quiet.history_] == [h["val_hv"] for h in loud.history_]
    assert quiet.diagnostics_ == loud.diagnostics_
