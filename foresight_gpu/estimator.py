"""``GPURegressor`` — the scikit-learn-facing Generalized Pareto Uncertainty estimator.

The estimator owns the optimisation loop (so it can early-stop between generations), wiring
together a forward model, an error metric, a MOPSO optimiser, the double-Pareto sorter and,
after convergence, a :class:`~foresight_gpu.ensemble.ParetoEnsemble`.

``metric`` is the per-particle **training** loss that drives the Pareto front. Early
stopping always monitors the **validation hypervolume** of the front; the probabilistic
scorers in :mod:`foresight_gpu.scoring` are for ``cross_val_score`` / ``GridSearchCV``.
"""

import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .domination import (
    DEFAULT_HV_LOG_REFERENCE,
    DEFAULT_HV_REFERENCE,
    HV_REFERENCE_MARGIN,
    DoubleParetoSorter,
    default_hv_reference,
    double_pareto_hypervolume,
    reference_nadir,
)
from .ensemble import DEFAULT_QUANTILES, ParetoEnsemble, _warn_if_clipped
from .metrics import get_metric
from .metrics.exceedance import non_exceedance
from .models import MLPModel
from .optimizers import MOPSO, evolve
from .scoring import probabilistic_diagnostics
from .utils.screening import prepare_arrays, screen_initial_population

_BAD_LOSS = 1e3  # log10-loss assigned to non-finite (failed) evaluations


def _check_X_y_allow_nan(X, y):
    try:  # sklearn >= 1.6
        return check_X_y(X, y, dtype=float, y_numeric=True, ensure_all_finite=False)
    except TypeError:  # sklearn < 1.6
        return check_X_y(X, y, dtype=float, y_numeric=True, force_all_finite=False)


def _check_array_allow_nan(X):
    try:
        return check_array(X, dtype=float, ensure_all_finite=False)
    except TypeError:
        return check_array(X, dtype=float, force_all_finite=False)


def _warn_if_model_requests_scaling(model):
    """Flag forward models still carrying the removed ``scales_inputs``/``scales_outputs``.

    Dropped in 0.5.0: ``X`` is the user's job (a ``Pipeline``), and the MLP's output scaling
    became :class:`MLPModel` hyperparameters. The flags are now inert, so a model still
    setting them would silently lose its normalisation.
    """
    stale = [f for f in ("scales_inputs", "scales_outputs") if getattr(model, f, False)]
    if stale:
        warnings.warn(
            f"{type(model).__name__} sets {' and '.join(stale)}, which GPURegressor no "
            f"longer reads. Scale X with a Pipeline, e.g. "
            f"make_pipeline(StandardScaler(), GPURegressor(...)); for the MLP's output "
            f"range use MLPModel(output_scale=y.std(), output_offset=y.mean()).",
            UserWarning,
        )


def _warn_if_exceedance_collapsed(eta, model):
    """Warn when no particle brackets ``y``, which yields an all-NaN ``predict``.

    Every model sitting at the same exceedance means the swarm is entirely on one side of
    the observations, so no band can be populated and the aggregation returns NaN
    throughout -- the failure mode of a model whose output range cannot reach the target
    (the MLP's weights are bounded, hence ``MLPModel(output_scale=...)``). Guarded on an
    exactly-zero span so it cannot fire on a merely narrow, healthy front.
    """
    if eta.size and float(np.ptp(eta)) == 0.0:
        warnings.warn(
            f"Every particle has exceedance {float(eta[0]):g}: {type(model).__name__} never "
            f"brackets y, so no band can be populated and predict() will be all-NaN. The "
            f"usual cause is a model whose output range cannot reach the target -- for the "
            f"MLP, set MLPModel(output_scale=y.std(), output_offset=y.mean()).",
            UserWarning,
        )


def _log_loss(loss):
    """The sorter's ranking axis: log10 of the raw loss, with failures sent to ``_BAD_LOSS``.

    Unbounded below (loss -> 0 gives -inf), which is why the hypervolume integrates the raw
    loss instead.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(np.maximum(loss, np.finfo(float).tiny))
    return np.where(np.isnan(out) | (out == np.inf), _BAD_LOSS, out)


def _early_stopping_state(scores, tol):
    """Replay the patience rule over a whole score series -> ``(best_idx, no_improve)``.

    Run on the full series at every check because an adaptive reference rescores earlier
    checks, which can move the best one. ``best_idx`` is ``None`` while no score is finite.
    While the best score is still ``<= 0`` the counter is held: a front that has not beaten
    the reference anywhere has no gradient to stop on.
    """
    best, best_score, no_improve = None, -np.inf, 0
    for i, s in enumerate(scores):
        if np.isfinite(s) and s > best_score + tol:
            best, best_score, no_improve = i, s, 0
        elif best_score > 0.0:
            no_improve += 1
    return best, no_improve


def _prune_candidates(kept, scores, intercepts, best, tol):
    """Drop kept ensembles that :func:`_early_stopping_state` can never select again.

    Valid under an adaptive reference only. With ``u = 1/R`` (non-increasing) and nothing
    clipped, check k scores ``s_k(u) = a_k - b_k u``; ``a_k`` is ``intercepts[k]``. Two lines
    that order the same way at ``u = 0`` and at the current ``u`` order that way on the whole
    remaining interval, so k is dead if

    * an earlier i has ``a_i >= a_k`` and ``s_i >= s_k``: the replay's best is always
      ``>= s_i - tol``, so k never clears it by ``tol``;
    * a later j has ``a_j > a_k + tol`` and ``s_j > s_k + tol``: if k were best when j
      arrives, j replaces it, and the best never moves back.
    """
    s, a = np.asarray(scores, dtype=float), np.asarray(intercepts, dtype=float)
    for k in list(kept):
        if k == best:
            continue
        earlier = (a[:k] >= a[k]) & (s[:k] >= s[k])
        later = (a[k + 1:] > a[k] + tol) & (s[k + 1:] > s[k] + tol)
        if not np.isfinite(s[k]) or earlier.any() or later.any():
            del kept[k]


def _front_points(ensemble, sims, y, metric):
    """Front-0 ``(eta, loss)`` of ``ensemble`` on simulations already in hand, eta-ordered."""
    eta, loss, front = ensemble._front_from_sims(sims, y, metric)
    front = front[np.argsort(eta[front], kind="stable")]
    return eta[front], loss[front]


def _diagnose(ensemble, sims, iteration):
    """One ``diagnostics_`` row: alpha, pi and CRPS per side (``sims``: side -> (sims, y))."""
    row = {"iteration": iteration, "retained": False}
    for side, (s, y) in sims.items():
        diag = probabilistic_diagnostics(ensemble, ensemble._bands_from_sims(s), y)
        row.update({f"{side}_{k}": v for k, v in diag.items()})
    return row


_HV_COLUMN = ("HV", "hv", "{:10.4f}")
_DIAG_COLUMNS = (
    ("alpha", "reliability", "{:12.3f}"),
    ("pi", "resolution", "{:11.4g}"),
    ("CRPS", "crps", "{:12.4g}"),
)


def _format_header(sides, columns):
    cols = [f"{side.capitalize()} {label}" for label, _, _ in columns for side in sides]
    widths = [len(fmt.format(0.0)) for _, _, fmt in columns for _ in sides]
    body = "".join(f"{c:>{w + 1}}" for c, w in zip(cols, widths))
    return f"{'Iter':>6}{body}{'Front':>7}{'Elapsed':>9}"


def _format_row(row, sides, columns, new_best, elapsed):
    cells = []
    for _, key, fmt in columns:
        for side in sides:
            mark = "*" if new_best and key == "hv" and side == "val" else " "
            cells.append(fmt.format(row[f"{side}_{key}"]) + mark)
    front = row[f"{sides[-1]}_n_front"]
    return f"{row['iteration']:>6}{''.join(cells)}{front:>7}{elapsed:>8.1f}s"


class GPURegressor(RegressorMixin, BaseEstimator):
    """Generalized Pareto Uncertainty regressor.

    Parameters
    ----------
    model : BaseForwardModel or None
        Deterministic forward model. ``None`` -> :class:`MLPModel`.
    metric : str or Metric
        Per-particle training loss driving the front (``"nse"``, ``"kge"``, ``"mae"``, ...).
    optimizer : BaseOptimizer or None
        Multi-objective optimiser. ``None`` -> :class:`MOPSO`.
    population : int
        Number of particles (models).
    n_iter : int
        Maximum number of generations.
    quantiles : sequence of float or None
        Non-exceedance-probability bands. ``None`` -> :data:`DEFAULT_QUANTILES`.
    force_positive : bool
        Clip predictions at zero.
    force_non_exceedance : float or None
        Slope of the exceedance-spreading penalty applied during ranking.
    band_width : float
        Half-width of the exceedance window per band.
    min_models : int
        Minimum models required to populate a band.
    screen : bool
        Pre-select the initial population for exceedance spread.
    screen_oversample : int
        Candidate-pool multiplier for screening.
    warm_start : bool
        Continue the population from a previous ``fit`` instead of reinitialising.
    validation_fraction : float or None
        Held-out fraction for the built-in split (chronological tail unless ``shuffle``).
        ``None`` (default) means no internal split. See the note on early stopping below.
    n_iter_no_change : int
        Number of checks without improvement of the validation hypervolume before stopping.
    tol : float
        Minimum improvement in the validation hypervolume to count as progress.
    hv_reference : "adaptive" or float or "climatology"
        Hypervolume reference point ``R`` (the ceiling; the price of a unit of uncovered
        exceedance), in the units of the loss **as seen in** ``hv_space``.

        * ``"adaptive"`` (default): ``R = HV_REFERENCE_MARGIN * nadir``, the nadir being the
          worst front-0 loss over every check so far (training and validation separately).
          ``R`` only moves up, and whenever it does every earlier check is **rescored**
          under it before the patience rule is replayed, so the stopping decision always
          compares fronts in one box. Nothing is ever clipped.
        * float: a fixed reference. A warning fires when it clips the whole validation
          front or more than :data:`~foresight_gpu.domination.HV_CLIP_WARN_FRACTION` of it.
        * ``"climatology"``: fixed, from
          :func:`~foresight_gpu.domination.default_hv_reference`.
    hv_reference_scale : float
        Multiplier on ``R`` in every mode (on top of the adaptive margin).
    hv_interpolation : {"step", "linear"}
        How the front is integrated between consecutive particles.
    hv_space : {"linear", "log10"}
        Objective space the front is integrated on. ``"log10"`` integrates ``log10(loss)``
        in a symmetric ``[-R, R]`` box, which spreads the low-loss region that the linear
        axis compresses — measured 6.9x more usable ``hv`` range on the same run.
    check_every : int
        Generations between checks: one ``history_`` entry (hypervolume) and one
        early-stopping decision per check. Checks are cheap (one validation forward), so
        1-5 is reasonable; keep the patience window ``check_every * n_iter_no_change`` in
        generations roughly constant when changing it.
    diagnostics_every : int or None
        Generations between ``diagnostics_`` rows (reliability alpha, resolution pi and CRPS
        on training and validation). ``None`` (default) computes them only for the retained
        model. Each row costs a band aggregation over the training rows, several times a
        check, so ~``n_iter / 40`` (10 at the default ``n_iter``) gives smooth curves. A
        multiple of ``check_every`` shares the check's validation forward.
    shuffle : bool
        Shuffle before the validation split (default ``False`` — correct for time series).
    random_state : int or None
        Seed for reproducibility.
    n_jobs : int
        Reserved for parallel model evaluation (currently unused by the NumPy backend).
    verbose : int or bool
        Print progress. Rows follow ``diagnostics_every`` (hypervolume plus alpha, pi and
        CRPS), or every check when it is ``None`` (hypervolume only). Printing never changes
        what is computed or stored. A row off the check grid shows the hypervolume as seen
        live; ``history_`` holds the rescored check values.

    Attributes
    ----------
    ensemble_ : ParetoEnsemble
        The fitted predictive artifact.
    n_iter_ : int
        Generations actually run.
    best_iteration_ : int
        Generation whose ensemble was retained (early stopping).
    history_ : list of dict
        The early-stopping record: one entry per check (every ``check_every`` generations
        and the last), with or without validation data. Keys: ``iteration``, ``min_loss``,
        ``space`` and, per side (``train_*`` always, ``val_*`` with early stopping), ``hv``,
        ``coverage``, ``clipped_fraction``, ``n_front``, ``reference`` (in force at that
        check), ``eta`` and ``loss`` (the front-0 points scored). ``*_hv`` is rescored
        whenever the adaptive reference moves, so it is always on the final reference.
    diagnostics_ : list of dict
        One row per ``diagnostics_every`` generation (and the last), plus the retained
        model's iteration, sorted by ``iteration``. Keys: ``iteration``, ``retained``
        (``True`` on the row of ``best_iteration_``) and ``{train,val}_{reliability,
        resolution,crps}``. Reference-free, so never rescored. Join with ``history_`` on
        ``iteration``: ``pd.DataFrame(gpu.history_)`` and ``pd.DataFrame(gpu.diagnostics_)``
        are both plain tables (see :func:`~foresight_gpu.utils.plotting.plot_history`).
    early_stopping_ : bool
        Whether early stopping ran, resolved from the data (see :meth:`fit`).
    hv_reference_ : float or None
        The final reference ``R`` of the monitored side (validation, else training), in
        ``hv_space`` units. Also stored on ``ensemble_``, so
        ``score_hypervolume(X_val, y_val)`` equals ``history_[best]["val_hv"]``.

    Notes
    -----
    **Early stopping is inferred from the data, never flagged.** There is no
    ``early_stopping`` parameter: it runs whenever validation data is available.

    ==========================================  ==============================
    call                                        early stopping
    ==========================================  ==============================
    ``fit(X, y)``                               off — full ``n_iter``
    ``GPURegressor(validation_fraction=0.2)``   on, internal chronological tail
    ``fit(X, y, X_val=Xv, y_val=yv)``           on, the caller's own split
    ==========================================  ==============================
    """

    def __init__(self, model=None, metric="nse", optimizer=None, population=1000,
                 n_iter=400, quantiles=None, force_positive=False,
                 force_non_exceedance=None, band_width=0.025,
                 min_models=1, screen=False, screen_oversample=3, warm_start=False,
                 validation_fraction=None, n_iter_no_change=10, tol=1e-4,
                 hv_reference="adaptive", hv_reference_scale=1.0,
                 hv_interpolation="step", hv_space="linear", check_every=5,
                 diagnostics_every=None, shuffle=False, random_state=None, n_jobs=1,
                 verbose=0):
        self.model = model
        self.metric = metric
        self.optimizer = optimizer
        self.population = population
        self.n_iter = n_iter
        self.quantiles = quantiles
        self.force_positive = force_positive
        self.force_non_exceedance = force_non_exceedance
        self.band_width = band_width
        self.min_models = min_models
        self.screen = screen
        self.screen_oversample = screen_oversample
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.hv_reference = hv_reference
        self.hv_reference_scale = hv_reference_scale
        self.hv_interpolation = hv_interpolation
        self.hv_space = hv_space
        self.check_every = check_every
        self.diagnostics_every = diagnostics_every
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -- fit ---------------------------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the population, early-stopping whenever validation data is available.

        Parameters
        ----------
        X, y : ndarray
            Training data. When ``X_val``/``y_val`` are given, **all** of it trains.
        X_val, y_val : ndarray, optional
            Caller-supplied validation set — pass both or neither. Takes precedence over
            ``validation_fraction``, which is then ignored with a warning. Splitting
            outside the estimator is what lets a domain-aware holdout (a water year, a
            gauge, a ``TimeSeriesSplit`` fold) drive early stopping.

        Notes
        -----
        ``cross_val_score`` / ``GridSearchCV`` do **not** slice ``X_val`` per fold: passing
        it through ``fit_params`` gives every fold the same block, which for a time series
        means later folds train on data the "validation" block precedes. Inside CV, set
        ``validation_fraction`` so each fold carves its own tail. Under
        ``sklearn.set_config(enable_metadata_routing=True)`` these also become routable
        params that must be requested explicitly.
        """
        t0 = time.perf_counter()
        X, y = _check_X_y_allow_nan(X, y)
        self.n_features_in_ = X.shape[1]
        # the rng generates reproducible random numbers for the training (For operational use it should be None)
        rng = np.random.default_rng(self.random_state)
        quantiles = list(self.quantiles) if self.quantiles is not None else DEFAULT_QUANTILES

        metric = get_metric(self.metric)
        warm = self.warm_start and getattr(self, "is_fitted_", False)
        model = self.model_ if warm else (
            clone(self.model) if self.model is not None else MLPModel()
        )
        optimizer = self.optimizer_ if warm else (
            clone(self.optimizer) if self.optimizer is not None else MOPSO()
        )
        sorter = DoubleParetoSorter()

        Xc, yc, _ = prepare_arrays(X, y)
        X_tr, y_tr, X_val, y_val = self._resolve_split(Xc, yc, X_val, y_val, rng)
        enabled = X_val is not None

        _warn_if_model_requests_scaling(model)

        evaluate = self._make_evaluate(model, metric, X_tr, y_tr)
        penalize = self._make_penalize()

        if warm:
            population, fit, simulations = self._population, self._fit, self._simulations
        else:
            low, high = self._search_bounds(model)
            population = optimizer.initialise(low, high, self.population, rng)
            if self.screen:
                population = screen_initial_population(
                    evaluate, low, high, self.population, rng, self.screen_oversample
                )
            simulations, fit = evaluate(population)

        # Per side: ``fixed`` is a float, or None for adaptive; ``reference`` is the R in
        # force (adaptive starts at 0 and is raised by the first check).
        sides = ("train", "val") if enabled else ("train",)
        mean_tr = float(np.mean(y_tr))
        fixed = {"train": self._resolve_reference(metric, y_tr, mean_tr)}
        if enabled:
            fixed["val"] = self._resolve_reference(metric, y_val, mean_tr)
        reference = {side: fixed[side] or 0.0 for side in sides}

        make_ensemble = lambda pop, ft: ParetoEnsemble(  # noqa: E731
            model, model.search_transform(pop), ft[:, 0],
            quantiles, self.band_width, self.min_models, self.force_positive,
            metric, None, self.hv_interpolation, self.hv_space,
        )

        every = self.diagnostics_every
        columns = (_HV_COLUMN,) + (_DIAG_COLUMNS if every else ())
        if self.verbose:
            print(_format_header(sides, columns), flush=True)

        # kept: check index -> candidate ensemble that could still be restored.
        # intercepts: per check, val hv as R -> inf (see _prune_candidates).
        history, diagnostics, kept, intercepts = [], [], {}, []
        best, no_improve, last_iter, stopped = None, 0, -1, False
        for it in range(self.n_iter):
            population, fit, simulations, _ = evolve(
                optimizer, sorter, population, fit, simulations, evaluate, penalize
            )
            last_iter = it
            last = it == self.n_iter - 1
            is_check = it % self.check_every == 0 or last
            is_diag = bool(every) and (it % every == 0 or last)
            show = bool(self.verbose) and (is_diag or (is_check and not every))
            if not (is_check or is_diag):
                continue

            candidate = make_ensemble(population, fit)
            # At most one forward per generation: training reuses the population's cached
            # simulations; validation is simulated once for fronts and bands alike.
            sims = {"train": (simulations, y_tr)}
            if enabled:
                sims["val"] = (candidate._simulate(X_val)[0], y_val)
            if is_check or show:
                fronts = {side: _front_points(candidate, *sims[side], metric)
                          for side in sides}

            previous, new_best, moved = best, False, {}
            if is_check:
                entry = {"iteration": it, "min_loss": float(np.nanmin(fit[:, 1])),
                         "space": self.hv_space}
                for side in sides:
                    eta, loss = fronts[side]
                    old = self._raise_reference(reference, fixed, side, loss, history)
                    if old is not None:
                        moved[side] = old
                    parts = self._hv(eta, loss, reference[side])
                    entry.update({f"{side}_{k}": parts[k] for k in
                                  ("hv", "coverage", "clipped_fraction", "n_front")})
                    entry.update({f"{side}_reference": reference[side],
                                  f"{side}_eta": eta, f"{side}_loss": loss})
                    if side == "val" and fixed["val"] is not None:
                        _warn_if_clipped(parts, metric)
                history.append(entry)

                if enabled:
                    eta, loss = fronts["val"]
                    # Exact while nothing is clipped: hv is then linear in 1/R.
                    hv2 = self._hv(eta, loss, 2.0 * reference["val"])["hv"]
                    intercepts.append(2.0 * hv2 - entry["val_hv"])
                    scores = [h["val_hv"] for h in history]
                    best, no_improve = _early_stopping_state(scores, self.tol)
                    new_best = best == len(history) - 1
                    kept[len(history) - 1] = candidate
                    if fixed["val"] is None:
                        _prune_candidates(kept, scores, intercepts, best, self.tol)
                    else:
                        kept = {best: kept[best]} if best is not None else {}
                    if best is not None and best not in kept:
                        raise RuntimeError(f"Best check {best} was pruned (bug).")
                    stopped = no_improve >= self.n_iter_no_change

            if is_diag:
                diagnostics.append(_diagnose(candidate, sims, it))

            if self.verbose:
                for side, old in moved.items():
                    note = ""
                    if side == "val" and best != previous and best is not None:
                        note = f"; best is now iteration {history[best]['iteration']}"
                    print(f"  -- {side} reference {old:.4g} -> {reference[side]:.4g}: "
                          f"earlier checks rescored{note}", flush=True)
                if show:
                    row = {"iteration": it}
                    for side in sides:     # live hv off the check grid; same values on it
                        parts = self._hv(*fronts[side], reference[side])
                        row.update({f"{side}_hv": parts["hv"],
                                    f"{side}_n_front": parts["n_front"]})
                    if is_diag:
                        row.update(diagnostics[-1])
                    print(_format_row(row, sides, columns, new_best,
                                      time.perf_counter() - t0), flush=True)
            if stopped:
                break

        _warn_if_exceedance_collapsed(fit[:, 0], model)

        restored = enabled and best is not None
        if restored:
            self.ensemble_ = kept[best]
            self.best_iteration_ = history[best]["iteration"]
        else:
            self.ensemble_ = make_ensemble(population, fit)
            self.best_iteration_ = last_iter
        monitored = "val" if enabled else "train"
        self.hv_reference_ = reference[monitored] if history else fixed[monitored]
        self.ensemble_.hv_reference = self.hv_reference_
        self.n_iter_ = last_iter + 1
        self.history_ = history
        self.early_stopping_ = enabled

        # The retained model always gets a diagnostics row. A restored ensemble needs its
        # own forward; the final population's simulations are already cached.
        retained = next((d for d in diagnostics if d["iteration"] == self.best_iteration_),
                        None)
        if retained is None and last_iter >= 0:
            sims = {"train": (self.ensemble_._simulate(X_tr)[0] if restored
                              else simulations, y_tr)}
            if enabled:
                sims["val"] = (self.ensemble_._simulate(X_val)[0], y_val)
            retained = _diagnose(self.ensemble_, sims, self.best_iteration_)
            diagnostics.append(retained)
            diagnostics.sort(key=lambda d: d["iteration"])
        if retained is not None:
            retained["retained"] = True
        self.diagnostics_ = diagnostics

        if self.verbose:
            if not enabled:
                print(f"Finished {self.n_iter_} iterations (no validation data, no early "
                      f"stopping).", flush=True)
            elif best is not None:
                why = (f"Early stopping at iteration {last_iter}: no val hv improvement "
                       f"> {self.tol:g} in {no_improve} checks" if stopped
                       else f"Finished {self.n_iter_} iterations")
                print(f"{why}; restored iteration {self.best_iteration_} "
                      f"(val hv {history[best]['val_hv']:.4f}).", flush=True)
            if retained is not None:
                side = sides[-1]
                print(f"Retained model, {side}: alpha {retained[f'{side}_reliability']:.3f}"
                      f", pi {retained[f'{side}_resolution']:.4g}, CRPS "
                      f"{retained[f'{side}_crps']:.4g}.", flush=True)

        # State retained for warm_start / introspection.
        self.model_, self.optimizer_ = model, optimizer
        self._population, self._fit, self._simulations = population, fit, simulations
        self.is_fitted_ = True
        return self

    # -- prediction --------------------------------------------------------------------

    def predict(self, X):
        """Point estimate (median band), shape ``(n_samples,)``."""
        check_is_fitted(self)
        return self.ensemble_.predict(self._as_array(X))

    def predict_quantiles(self, X, quantiles=None):
        """Probabilistic bands, shape ``(n_samples, n_quantiles)`` (DataFrame if X is one)."""
        check_is_fitted(self)
        agg = self.ensemble_.predict_quantiles(self._as_array(X), quantiles=quantiles)
        try:
            import pandas as pd

            if isinstance(X, pd.DataFrame):
                cols = quantiles if quantiles is not None else self.ensemble_.quantiles
                return pd.DataFrame(agg, index=X.index, columns=cols)
        except ImportError:  # pragma: no cover
            pass
        return agg

    def predictive_pvalues(self, X, y):
        """PIT p-values of ``y`` within the predictive distribution."""
        check_is_fitted(self)
        y = np.asarray(y, dtype=float).ravel()
        return self.ensemble_.predictive_pvalues(self._as_array(X), y)

    def score_hypervolume(self, X, y, metric=None, *, reference=None,
                          interpolation=None, space=None, details=False):
        """Double-Pareto hypervolume of the **fitted ensemble's** front on ``(X, y)``.

        In ``[0, 1]``, higher = better. Unlike the band-based scorers this measures the
        *front*, so it re-simulates the retained models rather than calling
        ``predict_quantiles``. Defaults come from the ensemble: the calibration ``metric``,
        the resolved :attr:`hv_reference_`, ``hv_interpolation`` and ``hv_space``. Pass
        ``metric=`` to re-read the same front under any other metric, or ``space="log10"``
        (with a matching ``reference=``) to re-read it on a log loss axis.

        .. versionchanged:: 0.4.0
           Scores ``ensemble_`` — the front kept after any early-stopping rollback, i.e. the
           one ``predict`` uses — rather than the final generation's population. Numbers
           recorded before 0.4.0 will differ whenever a rollback happened.
        """
        check_is_fitted(self)
        return self.ensemble_.score_hypervolume(
            self._as_array(X), y, metric, reference=reference,
            interpolation=interpolation, space=space, details=details,
        )

    # -- helpers -----------------------------------------------------------------------

    def _as_array(self, X):
        X = _check_array_allow_nan(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}."
            )
        return X

    def _resolve_split(self, Xc, yc, X_val, y_val, rng):
        """Return ``(X_tr, y_tr, X_val, y_val)``; the last two are ``None`` when there is
        no validation data, which is how early stopping is switched off."""
        if (X_val is None) != (y_val is None):
            raise ValueError("Pass both X_val and y_val, or neither.")

        if X_val is not None:
            if self.validation_fraction:
                warnings.warn(
                    f"X_val/y_val were supplied; validation_fraction="
                    f"{self.validation_fraction!r} is ignored and all of X trains.",
                    UserWarning,
                )
            Xv, yv = _check_X_y_allow_nan(X_val, y_val)
            if Xv.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X_val has {Xv.shape[1]} features; expected {self.n_features_in_}."
                )
            Xv, yv, _ = prepare_arrays(Xv, yv)
            if Xv.shape[0] == 0:
                raise ValueError("X_val/y_val contain no finite rows.")
            return Xc, yc, Xv, yv

        if not self.validation_fraction:
            return Xc, yc, None, None

        n = Xc.shape[0]
        n_val = max(1, int(round(self.validation_fraction * n)))
        if self.shuffle:
            perm = rng.permutation(n)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
        else:
            tr_idx, val_idx = np.arange(n - n_val), np.arange(n - n_val, n)
        return Xc[tr_idx], yc[tr_idx], Xc[val_idx], yc[val_idx]

    def _resolve_reference(self, metric, y_eval, y_train_mean):
        """Fixed hypervolume reference ``R`` for one side, or ``None`` for ``"adaptive"``.

        ``"climatology"`` is evaluated on the window the side scores, with the **training**
        mean as the constant predictor, so nothing leaks from held-out observations.

        ``R`` carries the units of :attr:`hv_space`, so ``"climatology"`` is taken into that
        space. Under ``"log10"`` a climatology of exactly 1.0 (every ``greater_is_better``
        metric) gives ``log10 -> 0``, which is not a valid ceiling; that falls through the
        non-positive guard below rather than needing its own.
        """
        base = self.hv_reference
        if isinstance(base, str) and base == "adaptive":
            return None
        log_space = self.hv_space == "log10"
        fallback = DEFAULT_HV_LOG_REFERENCE if log_space else DEFAULT_HV_REFERENCE
        if base is None or isinstance(base, str):
            if base != "climatology":
                raise ValueError(
                    f"hv_reference must be 'adaptive', a float or 'climatology', "
                    f"got {base!r}."
                )
            base = default_hv_reference(metric, y_eval, y_train_mean)
            if log_space:
                with np.errstate(divide="ignore", invalid="ignore"):
                    base = np.log10(base)
        reference = float(self.hv_reference_scale) * float(base)
        if not np.isfinite(reference) or reference <= 0.0:
            warnings.warn(
                f"hypervolume reference resolved to {reference!r} for metric {metric} in "
                f"{self.hv_space} space; falling back to {fallback}. Set hv_reference= "
                f"explicitly.",
                UserWarning,
            )
            reference = fallback
        return reference

    def _hv(self, eta, loss, reference):
        """Hypervolume details of a stored front-0 (all points on it) under ``reference``."""
        return double_pareto_hypervolume(
            np.column_stack([eta, loss]), reference, front=np.arange(eta.size),
            interpolation=self.hv_interpolation, space=self.hv_space, details=True,
        )

    def _raise_reference(self, reference, fixed, side, loss, history):
        """Raise an adaptive reference to cover a new front's ``loss``.

        When it moves, every earlier ``history`` entry of that side is rescored under it and
        the old value is returned; otherwise ``None``. The rescoring is what makes an
        adaptive R safe: hv of a front that never changes still grows with R, so checks are
        only comparable on one reference.
        """
        if fixed[side] is not None:
            return None
        old = reference[side]
        scale = float(self.hv_reference_scale) * HV_REFERENCE_MARGIN
        nadir = reference_nadir(loss, self.hv_space)
        reference[side] = max(old, scale * nadir, np.finfo(float).tiny)
        if reference[side] <= old or not history:
            return None
        for h in history:
            parts = self._hv(h[f"{side}_eta"], h[f"{side}_loss"], reference[side])
            h[f"{side}_hv"] = parts["hv"]
            h[f"{side}_clipped_fraction"] = parts["clipped_fraction"]
        return old

    def _search_bounds(self, model):
        low_m, high_m = model.parameter_bounds(self.n_features_in_)
        low_s = model.inverse_search_transform(low_m)
        high_s = model.inverse_search_transform(high_m)
        return np.minimum(low_s, high_s), np.maximum(low_s, high_s)

    def _make_loss_fn(self, model, metric, X, y):
        """Return ``core(params_search) -> (sims, eta, raw loss)``.

        The **raw** loss (metric + ``model.regularization``) is what the hypervolume
        integrates; ``_make_evaluate`` wraps this to produce the sorter's log10 ranking axis.
        """
        n_features = self.n_features_in_

        def core(params_search):
            params_model = model.search_transform(params_search)
            sims = model.forward(X, params_model)
            loss = metric.loss(sims, y) + model.regularization(params_model, n_features)
            return sims, non_exceedance(sims, y), loss

        return core

    def _make_evaluate(self, model, metric, X_tr, y_tr):
        core = self._make_loss_fn(model, metric, X_tr, y_tr)

        def evaluate(params_search):
            sims, eta, loss = core(params_search)
            return sims, np.column_stack([eta, _log_loss(loss)])

        return evaluate

    def _make_penalize(self):
        slope = self.force_non_exceedance
        if slope is None:
            return None

        def penalize(joint_fit):
            anchor = joint_fit[np.argmin(joint_fit[:, 1]), 0]
            joint_fit[:, 1] = joint_fit[:, 1] + slope * np.abs(joint_fit[:, 0] - anchor)
            return joint_fit

        return penalize