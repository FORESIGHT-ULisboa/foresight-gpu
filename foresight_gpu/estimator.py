"""``GPURegressor`` — the scikit-learn-facing Generalized Pareto Uncertainty estimator.

The estimator owns the optimisation loop (so it can early-stop between generations), wiring
together a forward model, an error metric, a MOPSO optimiser, the double-Pareto sorter and,
after convergence, a :class:`~foresight_gpu.ensemble.ParetoEnsemble`.

Two knobs that are easy to conflate:

* ``metric``  — the per-particle **training** loss that drives the Pareto front.
* ``scoring`` — the held-out **probabilistic** criterion for early stopping / model selection.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .domination import DoubleParetoSorter
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
    early_stopping : bool
        Monitor a held-out probabilistic score and stop when it stalls.
    validation_fraction : float
        Held-out fraction (chronological tail unless ``shuffle``).
    n_iter_no_change : int
        Number of checks without improvement before stopping.
    tol : float
        Minimum improvement in the monitored score to count as progress.
    scoring : str or callable
        Early-stopping criterion (``"reliability"``, ``"resolution"``, ``"crps"`` or a
        callable ``scoring(ensemble, X, y)``).
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
    """

    def __init__(self, model=None, metric="nse", optimizer=None, population=1000,
                 n_iter=400, quantiles=None, reg_lambda=0.0, reg_p=1,
                 force_positive=False, force_non_exceedance=None, band_width=0.025,
                 min_models=1, screen=False, screen_oversample=3, warm_start=False,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=10,
                 tol=1e-4, scoring="crps", check_every=5, shuffle=False,
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
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.scoring = scoring
        self.check_every = check_every
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -- fit ---------------------------------------------------------------------------

    def fit(self, X, y):
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
        X_tr, y_tr, X_val, y_val = self._split(Xc, yc, rng)

        if warm:
            x_scaler, y_scaler = self.x_scaler_, self.y_scaler_
        else:
            x_scaler = StandardScaler().fit(X_tr) if model.scales_inputs else None
            y_scaler = (
                StandardScaler().fit(y_tr.reshape(-1, 1)) if model.scales_outputs else None
            )
        Xn_tr = x_scaler.transform(X_tr) if x_scaler is not None else X_tr

        evaluate = self._make_evaluate(model, metric, Xn_tr, y_tr, y_scaler)
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

        make_ensemble = lambda pop, ft: ParetoEnsemble(  # noqa: E731
            model, model.search_transform(pop), ft[:, 0], x_scaler, y_scaler,
            quantiles, self.band_width, self.min_models, self.force_positive,
        )

        best_ensemble, best_score, best_iter, no_improve = None, -np.inf, 0, 0
        history, last_iter = [], -1
        for it in range(self.n_iter):
            population, fit, simulations, _ = evolve(
                optimizer, sorter, population, fit, simulations, evaluate, penalize
            )
            last_iter = it
            if self.early_stopping and (it % self.check_every == 0 or it == self.n_iter - 1):
                candidate = make_ensemble(population, fit)
                score = score_ensemble(candidate, X_val, y_val, self.scoring)
                history.append(
                    {"iteration": it, "score": float(score),
                     "min_loss": float(np.nanmin(fit[:, 1]))}
                )
                if np.isfinite(score) and score > best_score + self.tol:
                    best_ensemble, best_score, best_iter = candidate, score, it
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        break

        if self.early_stopping and best_ensemble is not None:
            self.ensemble_ = best_ensemble
            self.best_iteration_ = best_iter
        else:
            self.ensemble_ = make_ensemble(population, fit)
            self.best_iteration_ = last_iter
        self.n_iter_ = last_iter + 1
        self.history_ = history

        # State retained for warm_start / introspection.
        self.model_, self.optimizer_ = model, optimizer
        self.x_scaler_, self.y_scaler_ = x_scaler, y_scaler
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

    # -- helpers -----------------------------------------------------------------------

    def _as_array(self, X):
        X = _check_array_allow_nan(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features; expected {self.n_features_in_}."
            )
        return X

    def _split(self, Xc, yc, rng):
        if not self.early_stopping:
            return Xc, yc, None, None
        n = Xc.shape[0]
        n_val = max(1, int(round(self.validation_fraction * n)))
        if self.shuffle:
            perm = rng.permutation(n)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
        else:
            tr_idx, val_idx = np.arange(n - n_val), np.arange(n - n_val, n)
        return Xc[tr_idx], yc[tr_idx], Xc[val_idx], yc[val_idx]

    def _search_bounds(self, model):
        low_m, high_m = model.parameter_bounds(self.n_features_in_)
        low_s = model.inverse_search_transform(low_m)
        high_s = model.inverse_search_transform(high_m)
        return np.minimum(low_s, high_s), np.maximum(low_s, high_s)

    def _make_evaluate(self, model, metric, Xn_tr, y_tr, y_scaler):
        reg_lambda, reg_p = self.reg_lambda, self.reg_p
        reg_mask = model.regularizable_mask(self.n_features_in_) if reg_lambda > 0 else None

        def evaluate(params_search):
            params_model = model.search_transform(params_search)
            raw = model.forward(Xn_tr, params_model)
            if y_scaler is not None:
                sims = raw * y_scaler.scale_[0] + y_scaler.mean_[0]
            else:
                sims = raw
            loss = metric.loss(sims, y_tr)
            if reg_lambda > 0:
                loss = loss + lp_penalty(params_model, reg_mask, reg_lambda, reg_p)
            with np.errstate(divide="ignore", invalid="ignore"):
                logloss = np.log10(np.maximum(loss, np.finfo(float).tiny))
            logloss = np.where(np.isnan(logloss) | (logloss == np.inf), _BAD_LOSS, logloss)
            return sims, np.column_stack([non_exceedance(sims, y_tr), logloss])

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
