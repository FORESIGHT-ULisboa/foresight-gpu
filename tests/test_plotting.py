"""Headless plotting smoke tests (matplotlib Agg set in conftest)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from foresight_gpu import GPURegressor
from foresight_gpu.utils import plot_double_pareto_front, plot_qq, plot_timeseries


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


def test_shared_axes_compose(fitted):
    gpu, X, y = fitted
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    plot_double_pareto_front(gpu._fit[:, 0], gpu._fit[:, 1], ax=axes[0])
    plot_qq(gpu.predictive_pvalues(X, y), ax=axes[1])
    plot_timeseries(gpu.predict_quantiles(X[:100]), gpu.ensemble_.quantiles,
                    observed=y[:100], ax=axes[2])
    assert all(isinstance(a, Axes) for a in axes)
    plt.close("all")
