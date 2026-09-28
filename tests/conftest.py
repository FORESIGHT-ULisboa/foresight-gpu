"""Shared pytest fixtures.

Synthetic, seeded fixtures back the analytic-exact assertions. Plotting runs headless.
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

#: Synthetic HYPE folder, and the Python stub that stands in for the executable.
HYPE_TEMPLATE = Path(__file__).resolve().parent / "data" / "hype_template"
HYPE_STUB = Path(__file__).resolve().parent / "hype_stub.py"

#: Point this at a real HYPE folder to enable the integration tests.
HYPE_FOLDER_ENV = "FORESIGHT_HYPE_FOLDER"


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


# -- HYPE ------------------------------------------------------------------------------
#
# The stub is a real script invoked as ``[sys.executable, hype_stub.py]``, so the whole
# pipeline - file writing, subprocess launch, multiprocessing, output parsing - is exercised
# without HYPE installed. ``HYPEModel`` takes its command as a parameter precisely so this
# substitution is possible.


@pytest.fixture
def hype_template(tmp_path):
    """A private copy of the synthetic HYPE folder, so a test may modify it freely."""
    import shutil

    if not HYPE_TEMPLATE.is_dir():
        pytest.skip("synthetic HYPE template missing")
    target = tmp_path / "template"
    shutil.copytree(HYPE_TEMPLATE, target)
    return target


@pytest.fixture
def stub_command():
    """Command that makes ``HYPEModel`` run the stub instead of HYPE."""
    return [sys.executable, str(HYPE_STUB)]


@pytest.fixture
def hype_model(hype_template, stub_command, tmp_path):
    """A small, closed-over-teardown ``HYPEModel`` backed by the stub."""
    from foresight_gpu.models.hype import HYPEModel

    model = HYPEModel(
        template_dir=hype_template,
        subbasin=1234,
        executable=stub_command,
        parameters=["wcfc", "rrcs1", "cmlt"],
        work_root=tmp_path / "work",
        warn_unrequested=False,
    )
    yield model
    model.close()


@pytest.fixture
def hype_observations(hype_model):
    """``(X, y)`` from the template observations, restricted to the output window."""
    from foresight_gpu.models.hype import load_observations

    X, y = load_observations(hype_model.template_dir / "Qobs.txt", column="1234")
    return hype_model.align(X, y)


@pytest.fixture(scope="session")
def real_hype_folder():
    """A real HYPE folder from the environment, or skip."""
    raw = os.environ.get(HYPE_FOLDER_ENV)
    if not raw:
        pytest.skip(f"set {HYPE_FOLDER_ENV} to a HYPE folder to run this")
    folder = Path(raw)
    if not folder.is_dir():
        pytest.skip(f"{HYPE_FOLDER_ENV}={raw} is not a directory")
    return folder
