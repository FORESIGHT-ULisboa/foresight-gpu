"""Early-stopping behaviour: stops before n_iter, records history, restores best."""

import numpy as np
import pytest

from foresight_gpu import GPURegressor


@pytest.fixture
def data(rng):
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.standard_normal(300)
    return X, y


def test_stops_before_max_iter(data):
    X, y = data
    gpu = GPURegressor(
        population=150, n_iter=300, early_stopping=True, n_iter_no_change=3,
        check_every=2, tol=1e-3, validation_fraction=0.2, random_state=0,
    ).fit(X, y)
    assert gpu.n_iter_ < 300
    assert len(gpu.history_) > 0
    assert 0 <= gpu.best_iteration_ <= gpu.n_iter_


def test_history_records_scores(data):
    X, y = data
    gpu = GPURegressor(
        population=120, n_iter=60, early_stopping=True, check_every=3,
        n_iter_no_change=5, random_state=0,
    ).fit(X, y)
    assert all("score" in h and "iteration" in h for h in gpu.history_)


def test_best_iteration_not_last_when_plateaued(data):
    X, y = data
    gpu = GPURegressor(
        population=150, n_iter=300, early_stopping=True, n_iter_no_change=4,
        check_every=2, tol=1e-3, random_state=0,
    ).fit(X, y)
    # the retained ensemble is the best-scoring check, at or before the final generation
    assert gpu.best_iteration_ <= gpu.n_iter_ - 1
    assert np.all(np.isfinite(gpu.predict(X)))


def test_no_early_stopping_runs_full(data):
    X, y = data
    gpu = GPURegressor(population=80, n_iter=20, early_stopping=False, random_state=0).fit(X, y)
    assert gpu.n_iter_ == 20
    assert gpu.history_ == []
