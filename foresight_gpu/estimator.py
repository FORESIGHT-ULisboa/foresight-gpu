"""``GPURegressor`` — the scikit-learn-facing Generalized Pareto Uncertainty estimator.

The estimator owns the optimisation loop (so it can early-stop between generations), wiring
together a forward model, an error metric, a MOPSO optimiser, the double-Pareto sorter and,
after convergence, a :class:`~foresight_gpu.ensemble.ParetoEnsemble`.

Two knobs that are easy to conflate:

* ``metric``  — the per-particle **training** loss that drives the Pareto front.
* ``scoring`` — the held-out **probabilistic** criterion for early stopping / model selection.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .domination import (
    DEFAULT_HV_LOG_PENALTY,
    DEFAULT_HV_PENALTY,
    DoubleParetoSorter,
    default_hv_penalty,
)
from .ensemble import DEFAULT_QUANTILES, ParetoEnsemble
from .metrics import get_metric
from .metrics.exceedance import non_exceedance
from .metrics.regularization import lp_penalty
from .models import MLPModel
from .optimizers import MOPSO, evolve
from .scoring import score_ensemble
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
    reg_lambda, reg_p : float, int
        L-p regularisation coefficient and norm order (WRR Eq. 2).
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
        Number of checks without improvement before stopping.
    tol : float
        Minimum improvement in the monitored score to count as progress.
    scoring : str or callable
        Early-stopping criterion (``"hypervolume"``, ``"reliability"``, ``"resolution"``,
        ``"crps"`` or a callable ``scoring(ensemble, X, y)``).
    hv_penalty : float or "climatology" or None
        Hypervolume ceiling ``P`` — the price of a unit of uncovered exceedance, in the
        units of the loss **as seen in** ``hv_space``. ``None`` (default) takes the default
        for that space: :data:`~foresight_gpu.domination.DEFAULT_HV_PENALTY` (100.0) for
        ``"linear"``, :data:`~foresight_gpu.domination.DEFAULT_HV_LOG_PENALTY` (2.0) for
        ``"log10"``. The linear default is a deliberate constant rather than a climatology:
        GPU's extreme-η particles are *meant* to be biased, so they score below the no-skill
        line by construction, and a tight ceiling clips away exactly the particles that give
        the distribution its width. ``"climatology"`` restores the metric-derived ceiling
        (see :func:`~foresight_gpu.domination.default_hv_penalty`). Raise it when the loss
        carries large units — MAE on flows in the thousands sits above 100, which pins
        ``hv`` at 0. The ensemble warns both when the whole front is clipped and when more
        than :data:`~foresight_gpu.domination.HV_CLIP_WARN_FRACTION` of it is.
    hv_penalty_scale : float
        Multiplier on ``P``; raise it to penalise uncovered exceedance harder. It multiplies
        whichever default the space selects, not 1.0.
    hv_interpolation : {"step", "linear"}
        How the front is integrated between consecutive particles.
    hv_space : {"linear", "log10"}
        Objective space the front is integrated on. ``"log10"`` clips ``log10(loss)`` to a
        symmetric ``[-P, P]`` box, which spreads the low-loss region that the linear axis
        compresses — measured 6.9x more usable ``hv`` range on the same run — and lets ``P``
        be a small readable number (``P=2`` is a raw loss of 100).
    check_every : int
        Generations between early-stopping checks.
    shuffle : bool
        Shuffle before the validation split (default ``False`` — correct for time series).
    random_state : int or None
        Seed for reproducibility.
    n_jobs : int
        Reserved for parallel model evaluation (currently unused by the NumPy backend).
    verbose : int
        Verbosity.

    Attributes
    ----------
    ensemble_ : ParetoEnsemble
        The fitted predictive artifact.
    n_iter_ : int
        Generations actually run.
    best_iteration_ : int
        Generation whose ensemble was retained (early stopping).
    history_ : list of dict
        Per-check diagnostics.
    early_stopping_ : bool
        Whether early stopping ran, resolved from the data (see :meth:`fit`).
    hv_penalty_ : float or None
        The ceiling ``P`` actually used, when ``scoring="hypervolume"``, in ``hv_space``
        units.

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
                 n_iter=400, quantiles=None, reg_lambda=0.0, reg_p=1,
                 force_positive=False, force_non_exceedance=None, band_width=0.025,
                 min_models=1, screen=False, screen_oversample=3, warm_start=False,
                 validation_fraction=None, n_iter_no_change=10,
                 tol=1e-4, scoring="hypervolume", hv_penalty=None, hv_penalty_scale=1.0,
                 hv_interpolation="step", hv_space="linear", check_every=5, shuffle=False,
                 random_state=None, n_jobs=1, verbose=0):
        self.model = model
        self.metric = metric
        self.optimizer = optimizer
        self.population = population
        self.n_iter = n_iter
        self.quantiles = quantiles
        self.reg_lambda = reg_lambda
        self.reg_p = reg_p
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
        self.scoring = scoring
        self.hv_penalty = hv_penalty
        self.hv_penalty_scale = hv_penalty_scale
        self.hv_interpolation = hv_interpolation
        self.hv_space = hv_space
        self.check_every = check_every
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

        # Resolved once and frozen: hv = 1 - D/P is only comparable between checks if P is
        # the same box both times. A P that tracked the worst particle would report
        # improvement whenever the swarm found something worse.
        penalty = self._resolve_penalty(
            metric, y_val if y_val is not None else y_tr, float(np.mean(y_tr))
        )

        make_ensemble = lambda pop, ft: ParetoEnsemble(  # noqa: E731
            model, model.search_transform(pop), ft[:, 0],
            quantiles, self.band_width, self.min_models, self.force_positive,
            metric, penalty, self.hv_interpolation, self.hv_space,
        )

        # The scorer contract returns a float, so the hypervolume path asks the ensemble for
        # the decomposition instead — purely to enrich history_, not to compute it differently.
        use_hv = not callable(self.scoring) and str(self.scoring) == "hypervolume"

        best_ensemble, best_score, best_iter, no_improve = None, -np.inf, 0, 0
        history, last_iter = [], -1
        for it in range(self.n_iter):
            population, fit, simulations, _ = evolve(
                optimizer, sorter, population, fit, simulations, evaluate, penalize
            )
            last_iter = it
            if enabled and (it % self.check_every == 0 or it == self.n_iter - 1):
                candidate = make_ensemble(population, fit)
                if use_hv:
                    parts = candidate.score_hypervolume(X_val, y_val, details=True)
                    score = parts["hv"]
                else:
                    score = score_ensemble(candidate, X_val, y_val, self.scoring)
                    parts = {}
                history.append(
                    {"iteration": it, "score": float(score),
                     "min_loss": float(np.nanmin(fit[:, 1])), **parts}
                )
                if np.isfinite(score) and score > best_score + self.tol:
                    best_ensemble, best_score, best_iter = candidate, score, it
                    no_improve = 0
                elif use_hv and best_score <= 0.0:
                    # Nothing has beaten the ceiling anywhere yet, so the indicator is
                    # pinned at its floor and offers no gradient to stop on. Gated on
                    # use_hv because the other scorers are not floored at zero -- negated
                    # CRPS is always < 0, and an ungated guard would never let it stop.
                    pass
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        break

        _warn_if_exceedance_collapsed(fit[:, 0], model)

        if enabled and best_ensemble is not None:
            self.ensemble_ = best_ensemble
            self.best_iteration_ = best_iter
        else:
            self.ensemble_ = make_ensemble(population, fit)
            self.best_iteration_ = last_iter
        self.n_iter_ = last_iter + 1
        self.history_ = history
        self.early_stopping_ = enabled
        self.hv_penalty_ = penalty

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

    def score_hypervolume(self, X, y, metric=None, *, penalty=None,
                          interpolation=None, space=None, details=False):
        """Double-Pareto hypervolume of the **fitted ensemble's** front on ``(X, y)``.

        In ``[0, 1]``, higher = better. Unlike the band-based scorers this measures the
        *front*, so it re-simulates the retained models rather than calling
        ``predict_quantiles``. Defaults come from the ensemble: the calibration ``metric``,
        the resolved :attr:`hv_penalty_`, ``hv_interpolation`` and ``hv_space``. Pass
        ``metric=`` to re-read the same front under any other metric, or ``space="log10"``
        (with a matching ``penalty=``) to re-read it on a log loss axis.

        .. versionchanged:: 0.4.0
           Scores ``ensemble_`` — the front kept after any early-stopping rollback, i.e. the
           one ``predict`` uses — rather than the final generation's population. Numbers
           recorded before 0.4.0 will differ whenever a rollback happened.
        """
        check_is_fitted(self)
        return self.ensemble_.score_hypervolume(
            self._as_array(X), y, metric, penalty=penalty,
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

    def _resolve_penalty(self, metric, y_eval, y_train_mean):
        """Hypervolume ceiling ``P``, resolved once at fit time and stored on the ensemble.

        ``"climatology"`` is evaluated on the window the indicator scores, with the
        **training** mean as the constant predictor, so nothing leaks from held-out
        observations. That is also why it is resolved here rather than on the ensemble: an
        ensemble carries no training mean, so a self-referential climatology would leak.

        ``P`` carries the units of :attr:`hv_space`, so the default and the ``"climatology"``
        value are both taken into that space. Under ``"log10"`` a climatology of exactly 1.0
        (every ``greater_is_better`` metric) gives ``log10 -> 0``, which is not a valid
        ceiling; that falls through the non-positive guard below rather than needing its own.
        """
        log_space = self.hv_space == "log10"
        fallback = DEFAULT_HV_LOG_PENALTY if log_space else DEFAULT_HV_PENALTY
        base = self.hv_penalty
        if base is None:
            base = fallback
        elif isinstance(base, str):
            if base != "climatology":
                raise ValueError(
                    f"hv_penalty must be a float or 'climatology', got {base!r}."
                )
            base = default_hv_penalty(metric, y_eval, y_train_mean)
            if log_space:
                with np.errstate(divide="ignore", invalid="ignore"):
                    base = np.log10(base)
        penalty = float(self.hv_penalty_scale) * float(base)
        if not np.isfinite(penalty) or penalty <= 0.0:
            warnings.warn(
                f"hypervolume penalty resolved to {penalty!r} for metric {metric} in "
                f"{self.hv_space} space; falling back to {fallback}. Set hv_penalty= "
                f"explicitly.",
                UserWarning,
            )
            penalty = fallback
        return penalty

    def _search_bounds(self, model):
        low_m, high_m = model.parameter_bounds(self.n_features_in_)
        low_s = model.inverse_search_transform(low_m)
        high_s = model.inverse_search_transform(high_m)
        return np.minimum(low_s, high_s), np.maximum(low_s, high_s)

    def _make_loss_fn(self, model, metric, X, y, *, regularize=True):
        """Return ``core(params_search) -> (sims, eta, raw loss)``.

        The **raw** loss is what the hypervolume integrates; ``_make_evaluate`` wraps this
        to produce the sorter's log10 ranking axis.
        """
        reg_lambda = self.reg_lambda if regularize else 0.0
        reg_p = self.reg_p
        reg_mask = model.regularizable_mask(self.n_features_in_) if reg_lambda > 0 else None

        def core(params_search):
            params_model = model.search_transform(params_search)
            sims = model.forward(X, params_model)
            loss = metric.loss(sims, y)
            if reg_lambda > 0:
                loss = loss + lp_penalty(params_model, reg_mask, reg_lambda, reg_p)
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