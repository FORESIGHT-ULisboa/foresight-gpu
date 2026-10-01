"""Shared pytest fixtures.

Synthetic, seeded fixtures back the analytic-exact assertions. Plotting runs headless.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture
def rng():
    """Deterministic random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def regression_data(rng):
    """A small, well-posed regression problem (heteroscedastic noise)."""
    n = 400
    X = rng.uniform(-1.0, 1.0, size=(n, 3))
    signal = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1]
    noise = (0.2 + 0.2 * np.abs(X[:, 0])) * rng.standard_normal(n)
    y = signal + noise
    return X, y


@pytest.fixture
def population_simulations(rng):
    """A simulation matrix ``[n_samples, n_particles]`` and observations."""
    n_samples, n_particles = 120, 25
    obs = rng.normal(5.0, 1.5, size=n_samples)
    sim = obs[:, None] + rng.normal(0.0, 1.0, size=(n_samples, n_particles))
    return sim, obs

