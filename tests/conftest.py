"""Shared pytest fixtures.

Synthetic, seeded fixtures back the analytic-exact assertions; the example series under
``examples/data/`` back optional integration tests (skipped if absent). Plotting runs
headless.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

EXAMPLE_DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


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


@pytest.fixture(scope="session")
def zambezi():
    """Load the bundled example series, or skip if they are not present."""
    p = EXAMPLE_DATA / "Pobs.txt"
    t = EXAMPLE_DATA / "Tobs.txt"
    q = EXAMPLE_DATA / "Qobs.txt"
    if not (p.exists() and t.exists() and q.exists()):
        pytest.skip("example data not available")
    read = dict(sep="\t", header=0, index_col=0, parse_dates=True)
    precip = pd.read_csv(p, **read).aggregate(["sum"], axis=1)
    precip.columns = ["tp"]
    temp = pd.read_csv(t, **read).aggregate(["mean"], axis=1)
    temp.columns = ["t2m"]
    flow = pd.read_csv(q, sep="\t", header=0, names=["date", "Qobs"],
                       index_col=0, parse_dates=True)
    return precip, temp, flow
