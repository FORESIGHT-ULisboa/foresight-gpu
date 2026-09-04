"""Plotting helpers for GPU diagnostics (matplotlib).

Three views: probabilistic time-series bands, the predictive QQ plot, and the double-Pareto
front. Each accepts an optional ``ax`` and returns it, so they compose into subplots and are
testable headlessly (``matplotlib.use("Agg")``).
"""

import numpy as np

from ..domination import DoubleParetoSorter


def _get_ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def plot_timeseries(bands, quantiles, observed=None, index=None, ax=None, color="C0"):
    """Plot nested prediction bands (and observations) over time.

    Parameters
    ----------
    bands : ndarray
        Band values, shape ``[n_samples, n_quantiles]``.
    quantiles : sequence of float
        Non-exceedance-probability of each column (ascending).
    observed : array-like, optional
        Observations to overlay.
    index : array-like, optional
        X coordinates (defaults to a range).
    ax : matplotlib axis, optional
    color : str
        Band colour.
    """
    ax = _get_ax(ax)
    bands = np.asarray(bands, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    n, nq = bands.shape
    x = np.arange(n) if index is None else np.asarray(index)

    for i in range(nq // 2):
        ax.fill_between(x, bands[:, i], bands[:, nq - 1 - i], color=color,
                        alpha=0.15, linewidth=0.0)
    mid = int(np.argmin(np.abs(quantiles - 0.5)))
    ax.plot(x, bands[:, mid], color=color, lw=1.0, label="median")
    if observed is not None:
        ax.plot(x, np.asarray(observed), color="k", lw=0.8, alpha=0.8, label="observed")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_qq(pvalues, ax=None, color="C1"):
    """Predictive QQ plot: sorted p-values against the Uniform[0, 1] diagonal."""
    ax = _get_ax(ax)
    pv = np.sort(np.asarray(pvalues, dtype=float))
    pv = pv[np.isfinite(pv)]
    uniform = np.linspace(0.0, 1.0, pv.size) if pv.size else np.array([])
    ax.plot(uniform, pv, color=color, lw=1.5, label="observations")
    ax.plot([0, 1], [0, 1], "--k", lw=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("theoretical quantile of U[0, 1]")
    ax.set_ylabel("observed p-value quantile")
    ax.legend(frameon=False, fontsize=9)
    return ax


def plot_double_pareto_front(exceedance, loss, fronts=None, kept=None, rejected=None,
                             ax=None, max_fronts=None):
    """Scatter the objective space and draw the double-Pareto front(s).

    Parameters
    ----------
    exceedance, loss : ndarray
        Objective coordinates (x = exceedance, y = loss).
    fronts : list of list of int, optional
        Precomputed fronts; if ``None`` they are computed with
        :class:`~foresight_gpu.domination.DoubleParetoSorter`.
    kept, rejected : ndarray of bool, optional
        Masks to colour retained vs rejected solutions.
    ax : matplotlib axis, optional
    max_fronts : int, optional
        Only draw the first ``max_fronts`` front lines.
    """
    ax = _get_ax(ax)
    exceedance = np.asarray(exceedance, dtype=float)
    loss = np.asarray(loss, dtype=float)

    if kept is not None or rejected is not None:
        if rejected is not None:
            ax.plot(exceedance[rejected], loss[rejected], "o", color="0.6",
                    ms=3, label="rejected")
        if kept is not None:
            ax.plot(exceedance[kept], loss[kept], "o", color="C3", ms=3, label="kept")
    else:
        ax.plot(exceedance, loss, "o", color="0.6", ms=3)

    if fronts is None:
        fronts = DoubleParetoSorter().sort(np.column_stack([exceedance, loss]))
    for level, front in enumerate(fronts):
        if max_fronts is not None and level >= max_fronts:
            break
        idx = np.asarray(front, dtype=int)
        order = np.argsort(exceedance[idx])
        ax.plot(exceedance[idx][order], loss[idx][order], "-", lw=0.8)

    ax.set_xlim(0, 1)
    ax.set_xlabel("exceedance")
    ax.set_ylabel("loss (log10 error metric)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(frameon=False, fontsize=9)
    return ax
