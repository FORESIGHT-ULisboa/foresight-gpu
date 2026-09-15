"""Cross-validation and hyperparameter search with probabilistic scorers."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

from foresight_gpu import GPURegressor
from foresight_gpu.models import MLPModel
from foresight_gpu.scoring import make_gpu_scorer, reliability_scorer

EXPECTED_RELIABILITY_SCORE = np.array([0.96733412, 0.90483457, 0.93794496])  # Expected reliability score for the synthetic data
EXPECTED_R2_SCORE = np.array([0.06734914, 0.05555201, 0.11575378]) # Expected R2 score for the synthetic data


@pytest.fixture
def data(rng):
    X = rng.uniform(-1, 1, size=(360, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * X[:, 1] + 0.2 * rng.standard_normal(360)
    return X, y


def test_cross_val_score_reliability(data):
    X, y = data
    est = GPURegressor(population=80, n_iter=15, random_state=0)
    scores = cross_val_score(
        est, X, y, cv=TimeSeriesSplit(3), scoring=reliability_scorer
    )
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores <= 1.0 + 1e-9)
    np.testing.assert_allclose(scores, EXPECTED_RELIABILITY_SCORE, rtol=1e-4, atol=1e-6)


def test_cross_val_score_default_r2(data):
    X, y = data
    est = GPURegressor(population=80, n_iter=15, random_state=0)
    scores = cross_val_score(est, X, y, cv=TimeSeriesSplit(3))
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, EXPECTED_R2_SCORE, rtol=1e-4, atol=1e-6)


def test_gridsearch_nested_params(data):
    X, y = data
    search = GridSearchCV(
        GPURegressor(model=MLPModel(), population=70, n_iter=12, random_state=0),
        param_grid={"model__n_hidden": [4, 8], "reg_lambda": [0.0, 1e-3]},
        cv=TimeSeriesSplit(3),
        scoring=make_gpu_scorer("reliability"),
    )
    search.fit(X, y)
    assert "model__n_hidden" in search.best_params_
    assert "reg_lambda" in search.best_params_
    assert np.all(np.isfinite(search.predict(X)))
