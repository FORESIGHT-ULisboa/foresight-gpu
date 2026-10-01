"""Early stopping: inferred from the data, stops before n_iter, records history.

There is no ``early_stopping`` flag — it runs whenever validation data is available, either
from ``validation_fraction`` or from ``fit(X, y, X_val=, y_val=)``.
"""

import numpy as np
import pytest

from foresight_gpu import GPURegressor


@pytest.fixture
def data(rng):
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.standard_normal(300)
    return X, y


@pytest.fixture
def split(data):
    X, y = data
    return X[:240], y[:240], X[240:], y[240:]


# --- when early stopping runs ---------------------------------------------------------

def test_off_without_validation_data(data):
    """No early stopping, but the training side is still checked every check_every."""
    X, y = data
    gpu = GPURegressor(population=80, n_iter=20, random_state=0).fit(X, y)
    assert gpu.early_stopping_ is False
    assert gpu.n_iter_ == 20
    assert [h["iteration"] for h in gpu.history_] == [0, 5, 10, 15, 19]
    assert all("train_hv" in h and not any(k.startswith("val_") for k in h)
               for h in gpu.history_)
    # The ensemble carries the final training reference, so it reproduces the last check.
    assert gpu.score_hypervolume(X, y) == pytest.approx(gpu.history_[-1]["train_hv"])


def test_on_with_validation_fraction(data):
    X, y = data
    gpu = GPURegressor(
        population=150, n_iter=300, n_iter_no_change=3, check_every=2, tol=1e-3,
        validation_fraction=0.2, random_state=0,
    ).fit(X, y)
    assert gpu.early_stopping_ is True
    assert gpu.n_iter_ < 300
    assert len(gpu.history_) > 0
    assert 0 <= gpu.best_iteration_ <= gpu.n_iter_


def test_on_with_explicit_validation_data(split):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(
        population=120, n_iter=40, check_every=3, n_iter_no_change=4, random_state=0,
    ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert gpu.early_stopping_ is True
    assert len(gpu.history_) > 0
    # no tail was carved: the whole of X_tr trained
    assert gpu._simulations.shape[0] == len(X_tr)


def test_explicit_validation_beats_the_fraction(split):
    X_tr, y_tr, X_val, y_val = split
    with pytest.warns(UserWarning, match="validation_fraction"):
        gpu = GPURegressor(
            population=80, n_iter=8, check_every=4, validation_fraction=0.3,
            random_state=0,
        ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert gpu._simulations.shape[0] == len(X_tr)


# --- validation-data validation --------------------------------------------------------

def test_mismatched_validation_arity_raises(split):
    X_tr, y_tr, X_val, _ = split
    with pytest.raises(ValueError, match="both X_val and y_val"):
        GPURegressor(population=40, n_iter=2).fit(X_tr, y_tr, X_val=X_val)


def test_validation_feature_mismatch_raises(split):
    X_tr, y_tr, X_val, y_val = split
    with pytest.raises(ValueError, match="features"):
        GPURegressor(population=40, n_iter=2).fit(
            X_tr, y_tr, X_val=X_val[:, :1], y_val=y_val
        )


def test_validation_data_follows_the_training_nan_policy(split):
    """X_val rows with NaN are dropped; a NaN in y_val raises, exactly as for training."""
    X_tr, y_tr, X_val, y_val = split
    holed = X_val.copy()
    holed[0, 0] = np.nan
    gpu = GPURegressor(population=60, n_iter=6, check_every=3, random_state=0).fit(
        X_tr, y_tr, X_val=holed, y_val=y_val
    )
    assert all(np.isfinite(h["val_hv"]) for h in gpu.history_)

    bad_y = y_val.copy()
    bad_y[1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        GPURegressor(population=40, n_iter=2).fit(X_tr, y_tr, X_val=X_val, y_val=bad_y)


def test_all_validation_rows_dropped_raises(split):
    X_tr, y_tr, X_val, y_val = split
    with pytest.raises(ValueError, match="no finite rows"):
        GPURegressor(population=40, n_iter=2).fit(
            X_tr, y_tr, X_val=np.full_like(X_val, np.nan), y_val=y_val
        )


# --- what the history carries -----------------------------------------------------------

def test_history_records_hypervolume(split):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(
        population=120, n_iter=20, check_every=4, n_iter_no_change=10, random_state=0,
    ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    per_side = {"hv", "coverage", "clipped_fraction", "n_front", "reference", "eta",
                "loss"}
    for h in gpu.history_:
        assert {"iteration", "min_loss", "space"} <= set(h)
        for side in ("train", "val"):
            assert {f"{side}_{k}" for k in per_side} <= set(h)
            assert 0.0 <= h[f"{side}_hv"] <= 1.0
            assert h[f"{side}_n_front"] == h[f"{side}_eta"].size == h[f"{side}_loss"].size
        assert h["space"] == "linear"


def test_best_iteration_not_last_when_plateaued(data):
    X, y = data
    gpu = GPURegressor(
        population=150, n_iter=300, n_iter_no_change=4, check_every=2, tol=1e-3,
        validation_fraction=0.1, random_state=0,
    ).fit(X, y)
    assert gpu.best_iteration_ <= gpu.n_iter_ - 1
    assert np.all(np.isfinite(gpu.predict(X)))


def test_fit_does_not_mutate_params(split):
    """sklearn's check_estimators_overwrite_params contract."""
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=4, check_every=2, random_state=0)
    before = gpu.get_params()
    gpu.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert gpu.get_params() == before


# --- the scorer -------------------------------------------------------------------------

def test_score_hypervolume_matches_the_kept_generation(split):
    """The scorer and the loop must be the same arithmetic on the same front.

    Holds even when patience fired: both read ``ensemble_``, the front ``predict`` uses.
    """
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(
        population=100, n_iter=24, check_every=4, n_iter_no_change=2, random_state=0,
    ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    kept = next(h for h in gpu.history_ if h["iteration"] == gpu.best_iteration_)
    assert gpu.score_hypervolume(X_val, y_val) == pytest.approx(kept["val_hv"])


def test_patience_is_held_while_the_hypervolume_is_zero(split):
    """A swarm that has not beaten the ceiling anywhere has no gradient to stop on."""
    X_tr, y_tr, X_val, y_val = split
    with pytest.warns(UserWarning, match="pinned at 0"):
        gpu = GPURegressor(
            population=60, n_iter=12, check_every=1, n_iter_no_change=2,
            hv_reference=1e-9, random_state=0,    # every loss clips -> hv == 0
        ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert all(h["val_hv"] == 0.0 for h in gpu.history_)
    assert gpu.n_iter_ == 12       # 12 flat checks, patience 2, never stopped


def test_hv_reference_climatology_and_bad_value(split):
    from foresight_gpu import default_hv_reference, mae

    X_tr, y_tr, X_val, y_val = split
    with pytest.warns(UserWarning, match="pinned at 0"):
        gpu = GPURegressor(
            population=40, n_iter=4, check_every=2, metric="mae",
            hv_reference="climatology", random_state=0,
        ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert gpu.hv_reference_ == pytest.approx(
        default_hv_reference(mae, y_val, float(np.mean(y_tr)))
    )
    assert gpu.ensemble_.hv_reference == gpu.hv_reference_

    with pytest.raises(ValueError, match="'climatology'"):
        GPURegressor(population=40, n_iter=2, hv_reference="cheese").fit(
            X_tr, y_tr, X_val=X_val, y_val=y_val
        )


def test_default_reference_is_adaptive(split):
    """R = margin x the worst front-0 validation loss over every check."""
    from foresight_gpu import HV_REFERENCE_MARGIN

    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=12, check_every=2, random_state=0).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val
    )
    nadir = max(np.max(h["val_loss"][np.isfinite(h["val_loss"])]) for h in gpu.history_)
    assert gpu.hv_reference_ == pytest.approx(HV_REFERENCE_MARGIN * nadir)
    assert gpu.ensemble_.hv_reference == gpu.hv_reference_
    assert all(h["val_clipped_fraction"] == 0.0 for h in gpu.history_)


def test_fixed_reference_is_kept(split):
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(population=40, n_iter=4, check_every=2, hv_reference=100.0,
                       random_state=0).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    assert gpu.hv_reference_ == 100.0
    assert all(h["val_reference"] == 100.0 for h in gpu.history_)


# --- the log10 objective space -----------------------------------------------------------

def test_log10_space_drives_early_stopping(split):
    """hv_space reaches the monitor through the ensemble, and history_ records it."""
    X_tr, y_tr, X_val, y_val = split
    gpu = GPURegressor(
        population=120, n_iter=20, check_every=4, n_iter_no_change=10,
        hv_space="log10", random_state=0,
    ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

    from foresight_gpu import HV_REFERENCE_MARGIN, reference_nadir

    # adaptive in log units: margin x the largest |log10 loss| seen
    nadir = max(reference_nadir(h["val_loss"], "log10") for h in gpu.history_)
    assert gpu.hv_reference_ == pytest.approx(HV_REFERENCE_MARGIN * nadir)
    assert gpu.ensemble_.hv_space == "log10"
    assert all(h["space"] == "log10" and 0.0 <= h["val_hv"] <= 1.0 for h in gpu.history_)
    assert gpu.score_hypervolume(X_val, y_val) == pytest.approx(
        next(h["val_hv"] for h in gpu.history_ if h["iteration"] == gpu.best_iteration_)
    )


def test_log10_climatology_falls_back_rather_than_giving_a_zero_ceiling(split):
    """log10 of an NSE climatology is exactly 0, which is not a usable ceiling."""
    X_tr, y_tr, X_val, y_val = split
    with pytest.warns(UserWarning, match="falling back"):
        gpu = GPURegressor(
            population=40, n_iter=4, check_every=2, metric="nse",
            hv_reference="climatology", hv_space="log10", random_state=0,
        ).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    from foresight_gpu.domination import DEFAULT_HV_LOG_REFERENCE

    assert gpu.hv_reference_ == DEFAULT_HV_LOG_REFERENCE
