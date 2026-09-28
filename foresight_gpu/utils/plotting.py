"""Plotting helpers for GPU diagnostics (matplotlib).

Four views: probabilistic time-series bands, the predictive QQ plot, the double-Pareto front,
and the hypervolume decomposition. Each accepts an optional ``ax`` and returns it, so they
compose into subplots and are testable headlessly (``matplotlib.use("Agg")``).
"""

import numpy as np

from ..domination import DoubleParetoSorter, double_pareto_hypervolume


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


#: Front sizes above which the anchor connector is drawn without its label.
_ANCHOR_LABEL_MAX_FRONT = 20


def _hv_bounds(penalty, space):
    """Floor and ceiling in **raw loss** units for the given integration space."""
    if space == "log10":
        return 10.0 ** -penalty, 10.0 ** penalty
    return 0.0, penalty


def _hv_boundary(e, l, interpolation):
    """The attainment boundary actually integrated, as ``(x, y)`` polyline coordinates.

    ``step`` takes ``max(l_i, l_i+1)`` per cell -- the standard hypervolume staircase, which
    on a V-shaped front is always the *outer* endpoint. ``linear`` is the polyline through the
    points; on a log-scaled axis that is the geometric mean per cell, which is exactly what
    ``space="log10"`` integrates, so one construction serves both spaces.
    """
    if interpolation == "step":
        heights = np.maximum(l[:-1], l[1:])
        return np.repeat(e, 2)[1:-1], np.repeat(heights, 2)
    return e, l


def plot_hypervolume(eta, loss, penalty, *, front=None, interpolation="step",
                     space="linear", eta_range=(0.0, 1.0), show_anchor=True,
                     ylabel=None, ax=None):
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
    penalty : float
        Ceiling ``P``, in the units of ``space``.
    front : sequence of int, optional
        Front-0 indices. Computed if omitted.
    interpolation : {"step", "linear"}
        Boundary between consecutive front points.
    space : {"linear", "log10"}
        Integration space. ``"log10"`` draws raw loss on a log axis, which makes the plotted
        areas faithful to the log-space integral.
    eta_range : tuple of float
        Span the indicator normalises over.
    show_anchor : bool
        In ``step`` mode, mark the anchor and connect it to the boundary. The anchor is the
        one front point the staircase never touches: its cell has zero width, so the single
        best model on the front contributes no area at all.
    ylabel : str, optional
        Y-axis label (defaults to ``"loss"``).
    ax : matplotlib axis, optional
    """
    ax = _get_ax(ax)
    eta = np.asarray(eta, dtype=float).ravel()
    loss = np.asarray(loss, dtype=float).ravel()
    objectives = np.column_stack([eta, loss])

    parts = double_pareto_hypervolume(
        objectives, penalty, interpolation=interpolation, space=space,
        eta_range=eta_range, front=front, details=True,
    )
    if front is None:
        front = DoubleParetoSorter().sort(objectives)[0]
    idx = np.asarray(front, dtype=int).ravel()
    idx = idx[np.argsort(eta[idx], kind="stable")]

    floor, ceiling = _hv_bounds(float(penalty), space)
    e = eta[idx]
    l = np.clip(loss[idx], floor, ceiling)

    ax.scatter(eta, np.clip(loss, floor, ceiling), s=6, color="0.75", label="population")

    if e.size >= 2:
        xs, ys = _hv_boundary(e, l, interpolation)
        ax.fill_between(xs, ys, ceiling, color="tab:blue", alpha=0.18,
                        label="hypervolume  HV")
        ax.fill_between(xs, floor, ys, color="tab:orange", alpha=0.25,
                        label="integral  D")
        ax.plot(xs, ys, color="tab:blue", lw=1.6)
        for lo, hi in ([eta_range[0], e[0]], [e[-1], eta_range[1]]):
            if hi > lo:
                ax.axvspan(lo, hi, color="tab:red", alpha=0.15)

        if show_anchor and interpolation == "step":
            a = int(np.argmin(l))
            reached = np.max(np.maximum(l[:-1], l[1:])[max(a - 1, 0):a + 1])
            ax.plot([e[a], e[a]], [l[a], reached], ls=":", color="0.35", lw=1.2,
                    zorder=6)
            if e.size <= _ANCHOR_LABEL_MAX_FRONT:
                # On a dense front the risers are sub-pixel and the label lands on the
                # boundary, so the connector carries the point on its own.
                ax.annotate("anchor: zero cell width,\ncontributes no area",
                            xy=(e[a], l[a]), xytext=(6, -14),
                            textcoords="offset points", fontsize=7, color="0.35")

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
