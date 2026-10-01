"""Headless plotting smoke tests (matplotlib Agg set in conftest)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from foresight_gpu import GPURegressor
from foresight_gpu import double_pareto_hypervolume
from foresight_gpu.utils import (
    plot_double_pareto_front,
    plot_history,
    plot_hypervolume,
    plot_qq,
    plot_timeseries,
)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(250, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.standard_normal(250)
    gpu = GPURegressor(population=150, n_iter=30, random_state=0).fit(X, y)
    return gpu, X, y


def test_plot_timeseries(fitted):
    gpu, X, y = fitted
    bands = gpu.predict_quantiles(X[:120])
    ax = plot_timeseries(bands, gpu.ensemble_.quantiles, observed=y[:120])
    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0  # fill_between bands
    assert len(ax.get_lines()) >= 1
    plt.close("all")


def test_plot_qq(fitted):
    gpu, X, y = fitted
    pv = gpu.predictive_pvalues(X, y)
    ax = plot_qq(pv)
    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) >= 2  # observations + diagonal
    plt.close("all")


def test_plot_double_pareto_front(fitted):
    gpu, X, y = fitted
    fit = gpu._fit
    ax = plot_double_pareto_front(fit[:, 0], fit[:, 1])
    assert isinstance(ax, Axes)
    assert ax.get_xlim() == (0.0, 1.0)
    plt.close("all")


def test_plot_double_pareto_front_with_masks():
    rng = np.random.default_rng(1)
    exc = rng.uniform(0, 1, 100)
    loss = rng.uniform(0, 3, 100)
    kept = rng.random(100) > 0.5
    ax = plot_double_pareto_front(exc, loss, kept=kept, rejected=~kept)
    assert isinstance(ax, Axes)
    labels = ax.get_legend_handles_labels()[1]
    assert "kept" in labels and "rejected" in labels
    plt.close("all")


@pytest.fixture(scope="module")
def front(fitted):
    gpu, X, y = fitted
    return gpu.ensemble_.front_objectives(X, y)


def test_plot_hypervolume(front):
    eta, loss, idx = front
    ax = plot_hypervolume(eta, loss, 10.0, front=idx)
    assert isinstance(ax, Axes)
    assert len(ax.collections) > 0  # the two fill_between regions
    assert ax.get_xlim() == (0.0, 1.0)
    plt.close("all")


def test_plot_hypervolume_interpolations_differ(front):
    """step and linear must draw different boundaries -- reporting one while drawing the
    other is the bug this helper replaces."""
    eta, loss, idx = front
    data = {}
    for interpolation in ("step", "linear"):
        ax = plot_hypervolume(eta, loss, 10.0, front=idx, interpolation=interpolation)
        data[interpolation] = ax.get_lines()[0].get_xydata().shape
        plt.close("all")
    assert data["step"] != data["linear"]


def test_plot_hypervolume_log_space(front):
    eta, loss, idx = front
    ax = plot_hypervolume(eta, loss, 2.0, front=idx, space="log10")
    assert ax.get_yscale() == "log"
    plt.close("all")


@pytest.mark.parametrize("space,reference", [("linear", 10.0), ("log10", 2.0)])
@pytest.mark.parametrize("interpolation", ["step", "linear"])
def test_plot_hypervolume_title_matches_the_indicator(front, space, reference,
                                                      interpolation):
    """The figure and its number come from one call, and must stay that way."""
    eta, loss, idx = front
    ax = plot_hypervolume(eta, loss, reference, front=idx, space=space,
                          interpolation=interpolation)
    expected = double_pareto_hypervolume(
        np.column_stack([eta, loss]), reference, space=space,
        interpolation=interpolation, front=idx,
    )
    assert float(ax.get_title().split("hv = ")[1]) == pytest.approx(expected, abs=5e-5)
    plt.close("all")


def test_plot_hypervolume_step_reaches_the_minimum_between_midpoints():
    eta, loss = np.array([0.1, 0.4, 0.9]), np.array([0.8, 0.2, 0.5])
    ax = plot_hypervolume(eta, loss, 1.0, front=[0, 1, 2])
    xs, ys = ax.get_lines()[0].get_xydata().T
    # minimum loss held from midpoint 0.25 to midpoint 0.65
    np.testing.assert_allclose(xs[ys == 0.2], [0.25, 0.4, 0.4, 0.65])
    plt.close("all")


def test_plot_hypervolume_degenerate_front():
    """A single front point spans no eta; it must draw rather than raise."""
    ax = plot_hypervolume(np.array([0.5]), np.array([0.2]), 1.0, front=[0])
    assert isinstance(ax, Axes)
    plt.close("all")


def test_shared_axes_compose(fitted, front):
    gpu, X, y = fitted
    eta, loss, idx = front
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    plot_double_pareto_front(gpu._fit[:, 0], gpu._fit[:, 1], ax=axes[0])
    plot_qq(gpu.predictive_pvalues(X, y), ax=axes[1])
    plot_timeseries(gpu.predict_quantiles(X[:100]), gpu.ensemble_.quantiles,
                    observed=y[:100], ax=axes[2])
    plot_hypervolume(eta, loss, 10.0, front=idx, ax=axes[3])
    assert all(isinstance(a, Axes) for a in axes)
    plt.close("all")


def test_plot_history_draws_each_panel_from_its_record():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(250, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.standard_normal(250)
    gpu = GPURegressor(population=60, n_iter=12, check_every=2, diagnostics_every=4,
                       n_iter_no_change=99, validation_fraction=0.2,
                       random_state=0).fit(X, y)
    axes = plot_history(gpu.history_, gpu.diagnostics_, gpu.best_iteration_)
    assert len(axes) == 4
    hv_train = axes[0].get_lines()[0]
    np.testing.assert_array_equal(hv_train.get_xdata(),
                                  [h["iteration"] for h in gpu.history_])
    np.testing.assert_allclose(hv_train.get_ydata(), [h["train_hv"] for h in gpu.history_])
    alpha_val = [ln for ln in axes[1].get_lines() if ln.get_label() == "val"][0]
    np.testing.assert_allclose(alpha_val.get_ydata(),
                               [d["val_reliability"] for d in gpu.diagnostics_])
    plt.close("all")

    # no diagnostics rows -> the hypervolume panel alone
    assert len(plot_history(gpu.history_)) == 1
    plt.close("all")
