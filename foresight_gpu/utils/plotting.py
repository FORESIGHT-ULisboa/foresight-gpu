"""Plotting helpers for GPU diagnostics (matplotlib).

Five views: probabilistic time-series bands, the predictive QQ plot, the double-Pareto front,
the hypervolume decomposition, and the fit history. Each accepts an optional ``ax`` (or
``axes``) and returns it, so they compose into subplots and are testable headlessly
(``matplotlib.use("Agg")``).
"""

import numpy as np

from ..domination import DoubleParetoSorter, double_pareto_hypervolume
from ..domination.hypervolume import _step_cells


def _get_ax(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def _pct(q):
    return f"{100 * q:.1f}".rstrip("0").rstrip(".")


def plot_timeseries(bands, quantiles, observed=None, index=None, ax=None, color="k",
                    observed_color="C3"):
    """Plot non-exceedance-probability bands (and observations) over time.

    Each interval between consecutive quantiles is filled and labelled (e.g. ``"5-25%"``),
    shaded from ``color`` at the median to near-white in the tails. The legend sits in a row
    above the axis.

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
        Colour of the central band; outer bands fade towards white.
    observed_color : str
        Colour of the observed line.
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle

    ax = _get_ax(ax)
    bands = np.asarray(bands, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    n, nq = bands.shape
    x = np.arange(n) if index is None else np.asarray(index)
    base = np.asarray(to_rgb(color))

    # shade by rank away from the median: full colour (0.6 if a median line is drawn over
    # it) for the central interval, 0.25 for the outer tails
    has_median = bool(np.any(np.isclose(quantiles, 0.5)))
    top = 0.6 if has_median else 1.0
    centre = (quantiles[:-1] + quantiles[1:]) / 2.0
    rank = np.unique(np.round(np.abs(centre - 0.5), 9), return_inverse=True)[1]
    steps = max(rank.max(), 1)
    for i in range(nq - 1):
        shade = 1.0 - (1.0 - base) * (top - (top - 0.25) * rank[i] / steps)
        # thin same-colour edge hides antialiasing seams between adjacent fills
        ax.fill_between(x, bands[:, i], bands[:, i + 1], color=shade, edgecolor=shade,
                        linewidth=0.3, label=f"{_pct(quantiles[i])}-{_pct(quantiles[i + 1])}%")
    if has_median:
        mid = int(np.argmin(np.abs(quantiles - 0.5)))
        ax.plot(x, bands[:, mid], color=color, lw=1.0, label="median")
    if observed is not None:
        ax.plot(x, np.asarray(observed), color=observed_color, lw=0.9, alpha=0.9,
                label="observed")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    # legend in rows of <= 8 above the axis; a title set afterwards needs pad= to clear it
    handles, labels = ax.get_legend_handles_labels()
    nrow = -(-len(handles) // 8)
    ncol = -(-len(handles) // nrow)
    blank = nrow * ncol - len(handles)
    handles += [Rectangle((0, 0), 0, 0, visible=False)] * blank
    labels += [""] * blank
    order = [r * ncol + c for c in range(ncol) for r in range(nrow)]  # mpl fills by column
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left",
              bbox_to_anchor=(0.0, 1.01, 1.0, 0.1), mode="expand", ncol=ncol,
              frameon=False, fontsize=8, handlelength=1.5)
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


def _hv_bounds(reference, space):
    """Floor and ceiling in **raw loss** units for the given integration space."""
    if space == "log10":
        return 10.0 ** -reference, 10.0 ** reference
    return 0.0, reference


def _hv_boundary(e, l, finite, interpolation):
    """The attainment boundary actually integrated, as ``(x, y)`` polyline coordinates.

    ``step`` draws the indicator's own cells (``_step_cells``); ``max`` and the minimum are
    monotone, so the raw-loss cells are the log-space ones too. ``linear`` is the polyline
    through the points; on a log-scaled axis that is the geometric mean per cell, which is
    exactly what ``space="log10"`` integrates, so one construction serves both spaces.
    """
    if interpolation == "step":
        edges, heights = _step_cells(e, l, finite)
        return np.repeat(edges, 2)[1:-1], np.repeat(heights, 2)
    return e, l


def plot_hypervolume(eta, loss, reference, *, front=None, interpolation="step",
                     space="linear", eta_range=(0.0, 1.0), ylabel=None, ax=None):
    """Draw the double-Pareto hypervolume decomposition of a front.

    The shaded regions are the two halves of ``hv = 1 - D / box``: the area between the
    attainment boundary and the ceiling is ``HV``, everything below it (plus the full-height
    uncovered stretches of eta) is ``D``.

    ``hv`` is recomputed here from
    :func:`~foresight_gpu.domination.double_pareto_hypervolume` with the same arguments the
    figure is drawn from, so the picture and the number in the title cannot disagree.

    Parameters
    ----------
    eta, loss : ndarray
        Per-particle non-exceedance and **raw** loss, as returned by
        :meth:`~foresight_gpu.ensemble.ParetoEnsemble.front_objectives`.
    reference : float
        Reference (ceiling) ``R``, in the units of ``space`` -- e.g. ``gpu.hv_reference_``.
    front : sequence of int, optional
        Front-0 indices. Computed if omitted.
    interpolation : {"step", "linear"}
        Boundary between consecutive front points.
    space : {"linear", "log10"}
        Integration space. ``"log10"`` draws raw loss on a log axis, which makes the plotted
        areas faithful to the log-space integral.
    eta_range : tuple of float
        Span the indicator normalises over.
    ylabel : str, optional
        Y-axis label (defaults to ``"loss"``).
    ax : matplotlib axis, optional
    """
    ax = _get_ax(ax)
    eta = np.asarray(eta, dtype=float).ravel()
    loss = np.asarray(loss, dtype=float).ravel()
    objectives = np.column_stack([eta, loss])

    parts = double_pareto_hypervolume(
        objectives, reference, interpolation=interpolation, space=space,
        eta_range=eta_range, front=front, details=True,
    )
    if front is None:
        front = DoubleParetoSorter().sort(objectives)[0]
    idx = np.asarray(front, dtype=int).ravel()
    idx = idx[np.argsort(eta[idx], kind="stable")]

    floor, ceiling = _hv_bounds(float(reference), space)
    e = eta[idx]
    l = np.clip(loss[idx], floor, ceiling)

    ax.scatter(eta, np.clip(loss, floor, ceiling), s=6, color="0.75", label="population")

    if e.size >= 2:
        xs, ys = _hv_boundary(e, l, np.isfinite(loss[idx]), interpolation)
        ax.fill_between(xs, ys, ceiling, color="tab:blue", alpha=0.18,
                        label="hypervolume  HV")
        ax.fill_between(xs, floor, ys, color="tab:orange", alpha=0.25,
                        label="integral  D")
        ax.plot(xs, ys, color="tab:blue", lw=1.6)
        for lo, hi in ([eta_range[0], e[0]], [e[-1], eta_range[1]]):
            if hi > lo:
                ax.axvspan(lo, hi, color="tab:red", alpha=0.15)

    ax.scatter(e, l, s=18, color="tab:blue", zorder=5, label="front 0")
    ax.axhline(ceiling, color="k", ls="--", lw=1, label=f"ceiling  {ceiling:g}")
    if space == "log10":
        ax.set_yscale("log")
        ax.axhline(floor, color="k", ls=":", lw=1, label=f"floor  {floor:g}")
        ax.set_ylim(floor * 0.9, ceiling * 1.1)
    else:
        ax.set_ylim(0.0, ceiling * 1.05)

    ax.set_xlim(*eta_range)
    ax.set_xlabel(r"non-exceedance  $\eta$")
    ax.set_ylabel(ylabel or "loss")
    ax.set_title(f"{space}, {interpolation}:  hv = {parts['hv']:.4f}")
    ax.legend(frameon=False, fontsize=8)
    return ax


#: Panels of :func:`plot_history`: (key suffix, y label). ``hv`` comes from ``history_``,
#: the rest from ``diagnostics_``.
_HISTORY_PANELS = (
    ("hv", "hypervolume"),
    ("reliability", r"reliability $\alpha$"),
    ("resolution", r"resolution $\pi$"),
    ("crps", "CRPS"),
)


def plot_history(history, diagnostics=(), best_iteration=None, *, axes=None):
    """Hypervolume and probabilistic diagnostics across generations, train vs validation.

    Each panel is drawn from the record that owns it, at its own cadence: hypervolume from
    ``history_`` (every check -- the curve early stopping read), reliability, resolution and
    CRPS from ``diagnostics_`` (every ``diagnostics_every``, with markers so the sparser
    sampling shows). Both are plain lists of flat dicts keyed by ``iteration``, so the same
    figure is a few lines of ``pd.DataFrame(...).plot`` without this helper.

    Parameters
    ----------
    history : list of dict
        ``GPURegressor.history_``.
    diagnostics : list of dict, optional
        ``GPURegressor.diagnostics_``. Empty -> the hypervolume panel only. The row flagged
        ``retained`` (the model ``predict`` uses) is drawn as a star.
    best_iteration : int, optional
        ``GPURegressor.best_iteration_``, marked on every panel.
    axes : sequence of matplotlib axes, optional
        One per panel drawn.

    Returns
    -------
    ndarray of axes
    """
    panels = _HISTORY_PANELS if len(diagnostics) else _HISTORY_PANELS[:1]
    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                               figsize=(7, 2.2 * len(panels)))
    axes = np.asarray(axes).ravel()

    for ax, (key, label) in zip(axes, panels):
        rows = history if key == "hv" else diagnostics
        it = np.array([r["iteration"] for r in rows])
        for side, color in (("train", "darkcyan"), ("val", "firebrick")):
            col = f"{side}_{key}"
            if not rows or col not in rows[0]:
                continue
            values = np.array([r[col] for r in rows], dtype=float)
            ax.plot(it, values, "-o", ms=3, color=color,
                    label=side)
            done = [r.get("retained", False) for r in rows]
            if any(done):
                ax.plot(it[done], values[done], "*", ms=11, color=color, zorder=5)
        if best_iteration is not None:
            ax.axvline(best_iteration, color="k", ls=":", lw=1)
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8)
    axes[len(panels) - 1].set_xlabel("generation")
    return axes
