"""Hypervolume indicators over the double-Pareto objective space.

The indicator answers one question with one number: *how much of the exceedance axis does
the front cover, and how low is its loss there?* Both improvements push it up, so accuracy
and coverage cannot trade off invisibly — which is why it replaces the single-scalar
(reliability / CRPS) early-stopping criterion.

Two equivalent readings of the same quantity, both worth knowing:

* **Penalised integral** (minimise) — ``D = integral of L(eta) over the covered span
  + P * (uncovered span)``. ``P`` is the price of a unit of uncovered exceedance.
* **Hypervolume** (maximise) — the area between the front's staircase and a ceiling at
  ``P``. ``HV = P - D``.

They are algebraically identical. We report the normalised complement ``hv = 1 - D / P`` in
``[0, 1]``, higher = better, to match the ``scoring`` contract in ``scoring.py``.

Conventions
-----------
``objectives[:, 0]`` is the exceedance axis eta (mirrored around 0.5, bounded ``[0, 1]``);
``objectives[:, 1:]`` are **raw** losses, minimised, floor 0. Always pass the raw loss --
``space=`` selects the axis the indicator integrates on, and doing the transform yourself
would apply it twice.

Objective space (``space=``)
----------------------------
The caller never passes a log loss; the module transforms it. Two spaces, one box shape:

===========  ==================  ==========  =============  ====================
``space``    axis                floor ``F``  ceiling        box height
===========  ==================  ==========  =============  ====================
``linear``   ``L``               ``0``        ``P``          ``P``
``log10``    ``log10(L)``        ``-P``       ``P``          ``2P``
===========  ==================  ==========  =============  ====================

``log10`` is **symmetric**: ``P`` is read as an absolute bound on ``|log10 L|``, so ``P=2``
means raw loss in ``[1e-2, 1e2]`` and needs no second parameter. A front sitting at loss 1
everywhere -- the no-skill line for NSE/KGE -- scores exactly ``hv = 0.5``, which is the
anchor that makes the number readable.

**Why it is offered.** Linear space spends almost the whole box on losses nobody cares
about: at the default ``P=100`` the interesting region (losses of order 1) is a hundredth of
the axis, so fronts that differ where it matters score almost the same. Measured on this
repo's synthetic problem the usable ``hv`` range was **2-7x wider** in log10 (2.3x over one
run's early-stopping trace, 6.9x over converged fronts at 5..160 generations); the factor
depends on which fronts are compared, the direction does not.

**Why the estimator's log10 is still not this one.** The sorter ranks on ``log10(loss)``
floored at ``tiny`` and carries the ``_BAD_LOSS`` sentinel, which is unbounded below and
meaningless to integrate. The bound here is the symmetric clip, not a sentinel. Every metric
in the registry has ``Metric.loss >= 0`` with optimum exactly 0, so ``log10`` is defined
everywhere except at 0 -- and a loss of exactly 0 clips to the floor ``-P``, taking maximum
credit, which is correct.

Because ``log10`` is monotone, **front 0 is identical in both spaces**: the transform is
applied after front selection and a precomputed ``front=`` stays valid.

Load-bearing property of the sorter
-----------------------------------
``_double_pareto`` consumes points in ascending loss and only ever *extends* the span, so
**front 0 comes back eta-ordered and V-shaped**: loss falls monotonically to the anchor,
then rises. Verified on 500/500 random populations. This is what makes the 2-D integral a
single ``np.diff`` and what forces the staircase height to be ``max(l_i, l_i+1)`` — see
:func:`double_pareto_hypervolume`.

Two rejected alternatives, recorded so they are not re-derived
--------------------------------------------------------------
* **Joining consecutive front points per eta cell** (the obvious k-dimensional
  generalisation) is **non-monotone**. With k=2, ``P=(20, 20)`` and front
  ``A=(0.2, (0,10))``, ``B=(0.8, (10,0))`` the indicator is 60; inserting one mediocre
  point ``D=(0.21, (19,19))`` collapses it to 0.6. Use the anchor split instead.
* **Choosing the split point by maximising HV_left + HV_right** over-credits: on front
  ``(0,1), (0.3,0), (1,1)`` with ``P=1`` the true value is 0, but splitting at ``s=1.0``
  lets the anchor's low loss leak across cells and reports 0.7. The split must be the
  anchor.
"""

import numpy as np

from .double_pareto import DoubleParetoSorter

__all__ = [
    "hypervolume",
    "double_pareto_hypervolume",
    "default_hv_penalty",
    "non_dominated_mask",
    "DEFAULT_HV_PENALTY",
    "DEFAULT_HV_LOG_PENALTY",
    "HV_CLIP_WARN_FRACTION",
]

#: Default hypervolume ceiling ``P`` in ``space="linear"``. A deliberate constant, **not** a
#: climatology: GPU's extreme-eta particles are *meant* to be biased (a particle at eta~0.95
#: has to systematically over-predict), so they score below the no-skill line by construction
#: -- 80% of front-0 measured worse than climatology, loss quartiles [0.81, 1.10, 1.64, 2.21,
#: 6.36]. A ceiling at 1.0 clips away exactly the particles that give the distribution its
#: width. Measured usable hv range: 5.4e-2 at P=100 against 2.7e-2 at P=1.
DEFAULT_HV_PENALTY = 100.0

#: Default ceiling in ``space="log10"``, where ``P`` is read in log units: the box is raw
#: loss in ``[1e-2, 1e2]``. A separate constant because the two spaces need numbers orders of
#: magnitude apart -- reusing 100.0 here would mean a box of ``[1e-100, 1e100]``, inside which
#: every real front is flat and the indicator has no gradient at all.
DEFAULT_HV_LOG_PENALTY = 2.0

#: Fraction of front-0 at the ceiling above which the indicator is reported as too tight.
#: Clipping is correct behaviour (no gradient among models you would never use), but past
#: this share of the front it stops responding to real improvements, so it is worth saying
#: out loud. Measured at the NSE climatology ceiling P=1: ~0.8-0.9 of front-0 clipped.
HV_CLIP_WARN_FRACTION = 0.25


def non_dominated_mask(points):
    """Boolean mask of the Pareto-optimal rows of ``points`` ``[m, d]`` (all minimised).

    Duplicated points all survive: domination requires a strict improvement somewhere.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    le = (points[:, None, :] <= points[None, :, :]).all(axis=-1)
    lt = (points[:, None, :] < points[None, :, :]).any(axis=-1)
    return ~(le & lt).any(axis=0)


def _hv_2d(points, reference):
    """O(m log m) sweep over the staircase corners."""
    p = points[np.all(points < reference, axis=1)]
    if p.shape[0] == 0:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    running = np.minimum.accumulate(p[:, 1])
    corners = np.concatenate(([True], running[1:] < running[:-1]))
    p = p[corners]
    x_next = np.concatenate((p[1:, 0], reference[:1]))
    return float(np.sum((x_next - p[:, 0]) * (reference[1] - p[:, 1])))


def hypervolume(points, reference):
    """Volume dominated by ``points`` and bounded by ``reference``, any dimensionality.

    Parameters
    ----------
    points : ndarray
        ``[m, d]``; **every** column is minimised.
    reference : ndarray
        ``[d]``, the worst corner. Points not strictly better than it in every column
        contribute nothing.

    Returns
    -------
    float

    Notes
    -----
    ``d == 2`` uses a sweep; ``d >= 3`` recurses by slicing the last axis (HSO). The
    k-objective space does not exist yet, so the recursion is kept simple rather than
    WFG-fast — the 2-D path is the only hot one.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    reference = np.asarray(reference, dtype=float).ravel()
    if points.shape[1] != reference.size:
        raise ValueError(
            f"points has {points.shape[1]} columns but reference has {reference.size}."
        )
    d = points.shape[1]

    if d == 1:
        inside = points[points[:, 0] < reference[0], 0]
        return 0.0 if inside.size == 0 else float(reference[0] - inside.min())
    if d == 2:
        return _hv_2d(points, reference)

    points = points[np.all(points < reference, axis=1)]
    if points.shape[0] == 0:
        return 0.0
    points = points[np.argsort(points[:, -1], kind="stable")]
    total = 0.0
    for i in range(points.shape[0]):
        upper = points[i + 1, -1] if i + 1 < points.shape[0] else reference[-1]
        depth = upper - points[i, -1]
        if depth <= 0.0:
            continue
        slab = points[: i + 1, :-1]
        total += depth * hypervolume(slab[non_dominated_mask(slab)], reference[:-1])
    return float(total)


def default_hv_penalty(metric, obs, predictor=None):
    """Default penalty ``P`` — the cost of a unit of uncovered exceedance.

    ``P`` is a cost *in the units of the loss*, so it can only be a constant where the loss
    is dimensionless:

    * ``greater_is_better`` metrics (NSE, KGE, KGE') have ``loss = 1 - value``, so ``P = 1``
      is exactly the metric-value-0 no-skill line.
    * error metrics (MAE, MSE, RMSE) carry the units of ``y``, so ``P`` is the loss of the
      constant forecast: ``var(obs)`` for MSE, ``std(obs)`` for RMSE, ``mean|obs - mu|`` for
      MAE.

    The two branches are the same rule. ``NSE = 1 - MSE / var(y)``, so
    ``nse.loss == mse.loss / default_hv_penalty(mse, y)`` — the constant-predictor loss *is*
    NSE's denominator, and this simply gives the error metrics the normalisation NSE already
    has built in.

    Evaluating ``metric.loss`` against a constant series is avoided for the KGE family on
    purpose: a constant has zero variance, so Pearson r is 0/0 and KGE' gamma is x/0. In
    practice ``kge_prime.loss(const, y)`` returns 3e15-6e15 *varying with the seed* (the
    float noise in ``_std1`` of a constant array), which is why that branch is analytic.

    Parameters
    ----------
    metric : Metric
        Only the generic contract is used (``greater_is_better``, ``loss``); no metric is
        named here, so ``domination/`` stays free of metric knowledge.
    obs : ndarray
        Observations the indicator is computed on.
    predictor : float, optional
        Constant to forecast. Defaults to ``mean(obs)``; pass the **training** mean when
        scoring a held-out window so nothing leaks from ``obs``.
    """
    if getattr(metric, "greater_is_better", False):
        return 1.0
    obs = np.asarray(obs, dtype=float).ravel()
    mu = float(np.mean(obs)) if predictor is None else float(predictor)
    return float(metric.loss(np.full(obs.shape, mu), obs))


def _front_zero(eta, losses):
    """Indices of the non-dominated front over the mirrored surface."""
    if losses.shape[1] == 1:
        return DoubleParetoSorter().sort(np.column_stack([eta, losses[:, 0]]))[0]
    # k > 1: mirror each half independently, since eta is a coverage axis, not an objective.
    anchor = eta[int(np.argmin(losses.sum(axis=1)))]
    keep = np.zeros(eta.shape[0], dtype=bool)
    for sign, half in ((1.0, eta <= anchor), (-1.0, eta >= anchor)):
        idx = np.flatnonzero(half)
        if idx.size:
            pts = np.column_stack([sign * eta[idx], losses[idx]])
            keep[idx[non_dominated_mask(pts)]] = True
    return np.flatnonzero(keep)


def _split_hypervolume(eta, losses, penalty, eta_split):
    """Framing B: two standard minimisation hypervolumes, disjoint in eta, summed."""
    if eta_split is None:
        # The anchor: the front point with the largest single-point box. Reduces to
        # argmin(loss) when k == 1. Choosing the split to maximise the total instead
        # over-credits (see the module docstring).
        eta_split = float(eta[int(np.argmax(np.prod(penalty - losses, axis=1)))])
    total = 0.0
    for sign, half in ((1.0, eta <= eta_split), (-1.0, eta >= eta_split)):
        idx = np.flatnonzero(half)
        if idx.size == 0:
            continue
        pts = np.column_stack([sign * eta[idx], losses[idx]])
        total += hypervolume(pts, np.concatenate(([sign * eta_split], penalty)))
    return total


def double_pareto_hypervolume(objectives, penalty, *, interpolation="step",
                              space="linear", eta_range=(0.0, 1.0), front=None,
                              eta_split=None, details=False):
    """Mirrored hypervolume indicator of a population, normalised to ``[0, 1]``.

    Parameters
    ----------
    objectives : ndarray
        ``[m, 1 + k]``; column 0 is eta, columns ``1:`` are **raw** losses (not log).
    penalty : float or sequence of float
        ``P`` per loss axis — the ceiling, equivalently the price of a unit of uncovered
        exceedance, **in the units of** ``space``. See :func:`default_hv_penalty`. With
        ``space="log10"`` it is a bound on ``|log10 L|``, so ``P=2`` is a raw loss of 100;
        the two spaces therefore want values orders of magnitude apart
        (:data:`DEFAULT_HV_PENALTY` against :data:`DEFAULT_HV_LOG_PENALTY`).
    space : {"linear", "log10"}
        Axis the loss is integrated on. ``"log10"`` uses a **symmetric** box, clipping
        ``log10(L)`` to ``[-P, P]``, which spreads the low-loss region the linear axis
        compresses (measured 6.9x more usable range). Pass the **raw** loss either way.
        See the module docstring.
    interpolation : {"step", "linear"}
        ``"step"`` is the standard hypervolume: the staircase height over a cell is
        ``max(l_i, l_i+1)``. That is **forced, not a style choice** — it is what makes the
        closed form equal the split-half hypervolume (matched to 2.2e-16 over 2000 random
        fronts; the ``min`` rule is off by up to 0.84). ``"linear"`` averages the two
        endpoints instead: an optimistic smoothing with no hypervolume interpretation, less
        sensitive to how many particles happen to land on the front. Always
        ``hv(linear) >= hv(step)``. 1 loss axis only.
    eta_range : tuple of float
        Span the indicator normalises over. eta outside it is clipped.
    front : sequence of int, optional
        Precomputed front-0 indices, to skip a second sort.
    eta_split : float, optional
        Override the anchor (``k > 1`` only).
    details : bool
        Return the decomposition dict instead of a float.

    Returns
    -------
    float or dict
        ``hv`` in ``[0, 1]``, higher = better. With ``details``, also ``dispersion``
        (``1 - hv``), ``integral`` (``D``), ``penalty``, ``space``, ``coverage``,
        ``clipped_fraction``, ``eta_min``, ``eta_max``, ``n_front`` and ``front_min_loss``.

    Notes
    -----
    Losses are clipped to ``[F, P]`` (``F = 0`` linear, ``-P`` log10): a particle worse than
    the penalty rate is no better than a gap, and a (non-registry) negative loss would
    otherwise push ``hv`` above 1. Non-finite losses become ``+inf``, so failed particles
    clip to ``P`` and contribute exactly zero area — they lengthen the covered span at zero
    height.

    ``clipped_fraction`` reports the share of front-0 sitting at the ceiling, failed
    particles included. Clipping is deliberate — there is no gradient to be had among models
    you would never use — but once it covers much of the front the indicator stops
    responding to real improvements, and that is otherwise invisible. Callers should warn
    above :data:`HV_CLIP_WARN_FRACTION`; :meth:`ParetoEnsemble.score_hypervolume` does.
    """
    objectives = np.atleast_2d(np.asarray(objectives, dtype=float))
    if objectives.shape[1] < 2:
        raise ValueError("objectives must have >= 2 columns (eta + at least one loss).")
    if interpolation not in ("step", "linear"):
        raise ValueError(f"interpolation must be 'step' or 'linear', got {interpolation!r}.")
    if space not in ("linear", "log10"):
        raise ValueError(f"space must be 'linear' or 'log10', got {space!r}.")

    n_loss = objectives.shape[1] - 1
    P = np.broadcast_to(
        np.atleast_1d(np.asarray(penalty, dtype=float)), (n_loss,)
    ).astype(float)
    if not np.all(np.isfinite(P)) or np.any(P <= 0.0):
        raise ValueError(f"penalty must be finite and > 0 on every loss axis, got {P}.")

    lo, hi = float(eta_range[0]), float(eta_range[1])
    span = hi - lo
    if span <= 0.0:
        raise ValueError(f"eta_range must be increasing, got {eta_range!r}.")

    eta = np.clip(objectives[:, 0], lo, hi)
    raw = np.where(np.isfinite(objectives[:, 1:]), objectives[:, 1:], np.inf)

    if front is None:
        front = _front_zero(eta, raw)
    idx = np.asarray(front, dtype=int).ravel()
    e = eta[idx]
    order = np.argsort(e, kind="stable")  # a no-op on the sorter's output, cheap insurance
    e = e[order]

    # log10 is monotone, so transforming here -- after front selection -- gives the same
    # front as transforming first, and a precomputed ``front=`` stays valid.
    if space == "log10":
        with np.errstate(divide="ignore"):
            axis = np.log10(raw[idx][order])   # loss 0 -> -inf, clipped to the floor
        floor = -P
    else:
        axis = raw[idx][order]
        floor = np.zeros_like(P)
    ell = np.clip(axis, floor, P)
    clipped_fraction = float(np.mean(axis >= P)) if axis.size else 0.0

    box = float(np.prod(P - floor)) * span
    if e.size < 2:
        hv_abs = 0.0
    elif n_loss == 1:
        width = np.diff(e)
        col = ell[:, 0]
        if interpolation == "step":
            height = np.maximum(col[:-1], col[1:])
        else:
            height = 0.5 * (col[:-1] + col[1:])
        hv_abs = float(np.sum(width * (P[0] - height)))
    else:
        if interpolation != "step":
            raise NotImplementedError(
                "interpolation='linear' is defined for a single loss axis only."
            )
        hv_abs = _split_hypervolume(e, ell, P, eta_split)

    hv = hv_abs / box
    if not details:
        return hv
    return {
        "hv": hv,
        "dispersion": 1.0 - hv,
        "integral": (1.0 - hv) * box,
        "penalty": float(P[0]) if n_loss == 1 else tuple(float(v) for v in P),
        "space": space,
        "coverage": float(e[-1] - e[0]) / span if e.size else 0.0,
        "clipped_fraction": clipped_fraction,
        "eta_min": float(e[0]) if e.size else float("nan"),
        "eta_max": float(e[-1]) if e.size else float("nan"),
        "n_front": int(e.size),
        "front_min_loss": float(np.min(ell)) if ell.size else float("nan"),
    }
