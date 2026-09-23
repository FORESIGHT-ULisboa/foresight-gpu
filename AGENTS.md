# AGENTS.md — conventions for AI coding agents

This file is the canonical instruction set for AI agents (Claude Code, GitHub Copilot,
etc.) working in this repository. `CLAUDE.md` and `.github/copilot-instructions.md` are
thin pointers to this file.

## What this project is

`foresight_gpu` implements the **Generalized Pareto Uncertainty (GPU)** methodology: a
data-driven, largely non-parametric way to turn a family of deterministic models into a
**reliable probabilistic forecast**. A multi-objective optimiser (MOPSO) evolves a
population of parameter sets for a deterministic *forward model*, ranks them on a
**double / mirrored Pareto front** over two objectives — *non-exceedance* (η) versus an
*error metric* (ε) — and aggregates the surviving models by non-exceedance band into an
estimate of the inverse conditional CDF (see the WRR draft, Eq. 4). Reliability and
resolution are scored with the Renard et al. (2010) diagnostics.

"GPU" is a double meaning: *Generalized Pareto Uncertainty* (the method) and *Graphics
Processing Units* (an optional compute backend). **The core runs on NumPy.** OpenCL is a
future overload point (see below).

## Guiding principle

**The core package is a general scikit-learn regressor.** Anything hydrology-specific or
convenience-only (feature engineering, screening heuristics, plotting) is a *helper* and
lives under `foresight_gpu/utils/`. `GPURegressor` must be usable on plain arrays with no
domain code.

**The estimator scales nothing.** Since 0.5.0 there is no `StandardScaler` inside `fit`:
normalise `X` with a `Pipeline`, and if a model's parameter bounds cannot reach the target's
range, say so on the model (`MLPModel(output_scale=, output_offset=)`). See the forward-model
contract below.

## Architecture (who owns what)

The one idea that shapes everything: **the optimiser owns the model parameters; the model
is a stateless parametric function evaluated for the whole population at once.** So a
"model" is a `BaseForwardModel`, *not* a scikit-learn estimator. Only `GPURegressor` is the
estimator.

```
GPURegressor (estimator)                 foresight_gpu/estimator.py
  ├─ model: BaseForwardModel             foresight_gpu/models/      (MLPModel, GR4JModel, byo)
  ├─ optimizer: BaseOptimizer            foresight_gpu/optimizers/  (MOPSO)
  ├─ metric: Metric (training loss)      foresight_gpu/metrics/
  ├─ DominanceSorter + hypervolume       foresight_gpu/domination/  (ranking & indicators)
  ├─ crowding (NSGA-II)                  foresight_gpu/crowding.py
  ├─ scoring (early stop / CV)           foresight_gpu/scoring.py
  └─ ensemble_: ParetoEnsemble (fitted)  foresight_gpu/ensemble.py
utils/ : features, screening, plotting   foresight_gpu/utils/       (helpers, not core)
```

`fit` drives the loop itself (calls `optimizer.step()` per generation) so it can early-stop
between generations. After convergence it builds a `ParetoEnsemble` — the fitted artifact
that does the band aggregation and inverse-CDF prediction.

## The forward-model contract (`models/base.py`)

`BaseForwardModel` subclasses `sklearn.base.BaseEstimator` **only for `get_params` /
`set_params`** (so searches can reach `model__*`). It has no `fit`. Implement:

- `n_parameters(n_features) -> int` — length of one parameter vector.
- `forward(X, params) -> ndarray` — `X: [n_samples, n_features]`,
  `params: [n_particles, n_params]` → `[n_samples, n_particles]`.
  **Must be vectorised over the particle axis.**
- `parameter_bounds(n_features) -> (low, high)` — per-parameter search bounds. These are
  **authoritative** (MLP and GR4J differ, which is why bounds live on the model).
- `regularizable_mask(n_features) -> bool[]` — which parameters the Lp term penalises.
- Optional `search_transform` / `inverse_search_transform` — a per-model search-space
  warp (the MLP uses `(w/4)**5`; GR4J uses identity). Do **not** put this on the engine.

**The model owns its units.** `forward` returns simulations directly comparable with `y`;
nothing rescales either side. `scales_inputs` / `scales_outputs` were **removed in 0.5.0** —
they were policy living in the estimator, and only ever one model used them. `fit` warns if a
model still carries them so the change cannot be silent.

- **`X`**: the caller's job, via `Pipeline([StandardScaler(), GPURegressor(...)])`. (Not for
  HYPE — `X` holds dates there; see that section.)
- **`y`**: out of `Pipeline`'s reach, and `TransformedTargetRegressor` is not a substitute —
  it exposes no `predict_quantiles`, and its `fit(X, y, **fit_params)` would forward
  `X_val`/`y_val` untransformed while the regressor trains on scaled `y`. So the *model*
  carries it: `MLPModel(output_scale=y.std(), output_offset=y.mean())`.
- **Why the MLP needs it at all.** Its weights are bounded at ±30 in model space, so with
  `n_hidden` nodes the output tops out near `30·(n_hidden+1)` — about 270 by default —
  whatever the data looks like. Measured: NSE 0.28 with the affine map, **−12403** at a
  target mean of 100, and at 1000 **no prediction at all** (every particle on one side of
  every observation → exceedance axis collapses → no band populated → `predict` all-NaN).
  Because the output layer is linear this is exactly a reparameterisation of it
  (`σ·(w·h + b) + μ`), so it is a plain hyperparameter — no fitted state, reachable as
  `model__output_scale` in a search.
- `fit` warns when the final population's exceedance span is **exactly zero**, which is that
  all-NaN failure. Guarded on an exactly-zero span so it cannot fire on a merely narrow front.

## Metrics (`metrics/`)

- Every metric is a `Metric` (see `metrics/base.py`) — a `str` subclass that is also
  callable and **equals its own name** (`str(nse) == nse == "nse"`), mirroring the
  `forecast_performance` convention. Never write `metric.__name__` where the object works.
- Two additions over the sister repo: `greater_is_better: bool` and `loss(sim, obs)`
  returning the **minimisation** form (MAE/MSE/RMSE → value; NSE/KGE/KGE′ → `1 − value`).
  **The engine only ever calls `.loss`.**
- Deterministic metrics are **vectorised over the population**: `sim` is
  `[n_samples, n_particles]`, `obs` is `[n_samples]`, result is `[n_particles]`, reducing
  along `axis=0`. These are the fast, GPU-dedicated routines — deliberately distinct from
  `forecast_performance`'s pandas-per-series metrics. Tests cross-check the two agree per
  column.
- Resolve a metric from a name/alias via the registry in `metrics/__init__.py`
  (case-insensitive). `metric="nse"` and `metric=nse` are equivalent everywhere.

## Domination (`domination/`)

`objectives[:, 0]` is the exceedance (probabilistic) axis; `objectives[:, 1:]` are one or
more loss axes. `DoubleParetoSorter` handles the current **1 exceedance + 1 loss** case
(mirrored front around η = 0.5). More than one loss axis raises `NotImplementedError` — the
array shape is the seam for a future N-objective sorter. The sorter runs every generation
on ~2× the population, so it is a hot path; keep it fast and, if you optimise it, prove
exact equivalence to the reference implementation with a test before removing the old one.

**Front 0 is η-ordered and V-shaped.** `_double_pareto` consumes points in ascending loss and
only ever *extends* the span, so front 0 comes back sorted along η with the loss falling
monotonically to the anchor and rising after it (verified on 500/500 random populations).
Everything in `hypervolume.py` rests on this.

### Hypervolume (`domination/hypervolume.py`)

The front-quality indicator that drives early stopping. Two readings of one quantity:
minimise `D = ∫_covered L(η)dη + P·(uncovered span)`, or maximise the area between the
staircase and a ceiling at `P`. They are algebraically identical (`HV = P − D`), so **`P` is
the price of a unit of uncovered exceedance**. Reported as `hv = 1 − D/P ∈ [0,1]`, higher =
better, to match the `scoring` contract.

- **The computation lives on `ParetoEnsemble`, not the estimator.** `score_hypervolume(X, y,
  metric=None, *, penalty, interpolation, details)` and
  `front_objectives(X, y, metric=None) -> (eta, loss, front)` are the public surface. The
  ensemble already carries the whole population and the model, so the only
  thing it was ever missing was the metric — and that is a **parameter**, with the
  calibration metric stored as an overridable default. A front calibrated on NSE can
  therefore be re-read under KGE or MAE with no refit, and `"hypervolume"` is a normal
  `score_ensemble` option rather than a special case. Plotting should use
  `front_objectives`; nothing needs privates.
- **Always pass the raw loss; `hv_space` picks the axis.** The loss is computed from the
  simulations, so the Lp term (which penalises parameters, not held-out fit) cannot reach it
  by construction.
  - `"linear"` (default): clip to `[0, P]`, box `P`. Unchanged from 0.4.0.
  - `"log10"`: clip `log10(loss)` to `[−P, +P]`, box `2P`. **Symmetric**, so `P` is read as a
    bound on `|log10 L|` — `P=2` is raw loss in `[1e-2, 1e2]` — and no second parameter is
    needed. A front at loss 1 everywhere (the no-skill line) scores exactly `hv = 0.5`, which
    is what makes the number readable.
  - **Why offer it.** Linear space spends almost the whole box on losses nobody cares about:
    at `P=100` the region of interest (losses of order 1) is a hundredth of the axis. Measured
    on the synthetic problem in `02_diagnostics`, the usable `hv` range is **2–7× wider** in
    log10 (2.3× over one run's early-stopping trace, 6.9× over converged fronts at 5…160
    generations). The factor depends on which fronts are compared; the direction does not.
  - **This is not the sorter's `log10`.** That one is floored at `tiny`, unbounded below and
    carries the `_BAD_LOSS` sentinel; integrating it is meaningless. The bound here is the
    symmetric clip. Every registry metric has `Metric.loss ≥ 0` with optimum exactly 0, so
    `log10` is defined everywhere except at 0 — and loss 0 clips to the floor `−P`, taking
    maximum credit, which is correct.
  - `log10` is monotone, so **front 0 is identical in both spaces**; the transform is applied
    after front selection and a precomputed `front=` stays valid.
  - The two spaces need `P` orders of magnitude apart, so `hv_penalty` now defaults to `None`
    and resolves per space: `DEFAULT_HV_PENALTY` (100.0) or `DEFAULT_HV_LOG_PENALTY` (2.0).
    Under `log10`, `"climatology"` is taken into log units — which for every
    `greater_is_better` metric is `log10(1.0) = 0`, not a usable ceiling, so it falls through
    the existing non-positive guard and warns.
- **`front_objectives` drops rows where *either* `X` or `y` is non-finite.** `_simulate`
  masks only `X`; one NaN in `y` makes `metric.loss` NaN for every particle, clipping them
  all to `P` and reporting `hv == 0` silently. The estimator path got this free from
  `prepare_arrays`; the `cross_val_score` path does not.
- **The staircase height is `max(lᵢ, lᵢ₊₁)`, and that is forced, not chosen.** It is what
  makes the O(m) closed form equal the standard split-half hypervolume (matched to 2.2e-16
  over 2000 random fronts; the `min` rule is off by up to 0.84). `test_step_equals_split_hypervolume`
  is that proof as an executable test — **keep it green**, same discipline as the
  domination-equivalence test. `interpolation="linear"` is an optimistic trapezoidal
  smoothing with no hypervolume interpretation, 1 loss axis only.
- Losses clip to `[F, P]` (`F = 0` linear, `−P` log10). A particle worse than `P` is no
  better than a gap, and failed particles (non-finite loss, η exactly 0.0) clip to `P` and
  contribute **zero area** — they lengthen the covered span at zero height. That partly
  answers the failed-run follow-up below, though the *ranking* path is still open.
- **`clipped_fraction` makes a too-tight `P` visible.** Clipping is the intended semantics —
  no gradient among models you would never use — but past some share of the front the
  indicator simply stops responding, and that was previously invisible. The share of front-0
  at the ceiling (failed particles included) is reported in `details` and therefore lands in
  `history_` next to `coverage`; `score_hypervolume` warns above `HV_CLIP_WARN_FRACTION`
  (0.25). Measured at the NSE climatology ceiling `P=1`: **~0.8–0.9 of front-0 clipped**, `hv`
  ~0.01 against ~0.98 at `P=100` — the same worse-than-climatology figure that motivated the
  constant default, now reported instead of inferred. The clean demonstration is a
  low-variance validation window (`02_diagnostics`): it inflates every loss at once while
  `coverage` stays at 1.00, so `clipped_fraction` is the only thing that moves. A plain bias
  moves `coverage` instead — the front keeps some high-η particle that tracks the shift.
- **`P` defaults to the constant `DEFAULT_HV_PENALTY = 100.0`, deliberately not a
  climatology.** GPU's extreme-η particles are *meant* to be biased — a particle at η≈0.95
  has to systematically over-predict to get there, so its NSE is necessarily poor. Measured
  on a real front, **80% of front-0 scores worse than climatology** (loss quartiles
  `[0.81, 1.10, 1.64, 2.21, 6.36]`). A ceiling at 1.0 clips exactly those to zero
  contribution, making the indicator blind to the particles that give the distribution its
  width; P=100 measured a *larger* usable range (5.4e-2 vs 2.7e-2). `hv_penalty` takes a
  float or `"climatology"`; `hv_penalty_scale` now multiplies 100, not 1.
  - **`P` must be resolved once and frozen** — `hv = 1 − D/P` is only comparable between
    checks if it is the same box both times. A `P` tracking the worst particle was
    considered and is *fatal*: on a front that never changes, hv goes 0.50 → 0.75 → 0.875 →
    0.975 as P grows 1 → 2 → 4 → 20, so the monitor would report improvement whenever the
    swarm found something worse, and reset patience each time.
  - `ParetoEnsemble.score_hypervolume` warns on two rungs: **the whole front above `P`**
    (`hv` pinned at 0, no gradient to stop on), else **more than `HV_CLIP_WARN_FRACTION` of
    it**. Neither fires on individual tail particles above `P` — that is the normal regime
    and would fire on healthy runs. Both texts are constant across checks so Python's dedup
    emits each once per fit, not once per generation.
- **`default_hv_penalty(metric, obs, predictor)`** is what `hv_penalty="climatology"` reaches
  for: `1.0` for `greater_is_better` metrics (`loss = 1 − value` is dimensionless, so this is
  the metric-value-0 no-skill line), otherwise the constant-predictor loss (`var` for MSE,
  `std` for RMSE, `mean|y−μ|` for MAE). The two branches are one rule — `NSE = 1 − MSE/var(y)`,
  so `nse.loss == mse.loss / default_hv_penalty(mse, y)`; the constant-predictor loss *is*
  NSE's denominator. It touches only the generic `Metric` contract and names no metric, so
  `domination/` stays free of metric knowledge. **Do not** just call `metric.loss(const, y)`:
  a constant series has zero variance, so Pearson r is 0/0 and KGE′ γ is x/0, and the result
  is `nan` or ~1e15 *depending on whether `mean()` leaves a 2e-16 residue* — not even
  deterministic in kind.
- **Drawing it: `utils/plotting.py::plot_hypervolume(eta, loss, penalty, ...)`.** Takes the
  `(eta, loss, front)` tuple `front_objectives` returns and covers all four cases
  (`space` x `interpolation`). It **recomputes `hv` from the same call it draws from** and puts
  it in the title, so the picture and the number cannot drift apart — which they had, in
  notebook 02: the figure reported `interpolation="linear"` while shading the step staircase.
  `test_plot_hypervolume_title_matches_the_indicator` is the lock; keep it green.
  - `space="log10"` draws raw loss on a log axis. That is not cosmetic: a straight polyline
    between two points on a log axis has the geometric mean as its per-cell average, which is
    exactly what `interpolation="linear"` integrates in log space, and `max` is monotone — so
    one construction is faithful in both spaces.
  - The figure **marks the anchor** rather than hiding it. In `step` mode every cell takes
    `max(l_i, l_i+1)`, which on a V-shaped front is always the *outer* endpoint, so the
    lowest-loss particle is the one point the boundary never touches. Its cell has zero width:
    the single best model on the front contributes no area at all. That is correct geometry,
    and it is why `hv` is a tails-and-shoulders measure — "hv improved" never means "my best
    model got better".

- **N objectives.** `double_pareto_hypervolume` splits at the anchor into η ≤ s and η ≥ s,
  negates η on the right half and sums two standard `hypervolume()` calls; the halves are
  disjoint in η so the sum is exact for any number of loss axes. Two alternatives are
  recorded as rejected in the module docstring, with counterexamples: joining consecutive
  front points per η cell is **non-monotone** (one mediocre point collapses it 100×), and
  choosing the split by maximising the total **over-credits**. The split must be the anchor.

## scikit-learn conventions

- `__init__` **only stores parameters** as same-named attributes — no validation, no work.
  Defaults like `model=None` are resolved inside `fit` (never mutate params in `__init__`).
- Fitted state uses trailing-underscore attributes (`ensemble_`, `n_iter_`,
  `best_iteration_`, `history_`, `is_fitted_`). `predict` calls `check_is_fitted`.
- `predict(X)` returns a point estimate shaped `(n_samples,)` (median band) so
  `RegressorMixin.score` and `Pipeline` keep working. Probabilistic output is the separate
  `predict_quantiles(X)` → `[n_samples, n_quantiles]`.
- `random_state` → `np.random.default_rng`, threaded everywhere for reproducibility. Do not
  use the global `np.random`.

## Cross-validation & early stopping

- **`metric` vs `scoring` are different roles.** `metric` is the per-particle *training*
  loss driving the Pareto front. `scoring` is the held-out criterion for early stopping and
  model selection. Keep both.
- **Early stopping is inferred from the data, never flagged.** There is no `early_stopping`
  parameter — it runs whenever validation data exists, which leaves no contradictory state
  to warn about:

  | call | early stopping |
  |---|---|
  | `fit(X, y)` | off — full `n_iter`, whole record trains |
  | `GPURegressor(validation_fraction=0.2).fit(X, y)` | on, internal chronological tail |
  | `fit(X, y, X_val=Xv, y_val=yv)` | on, the caller's split; **all** of `X` trains |

  `validation_fraction` defaults to `None` for exactly this reason: at `0.1` it would switch
  early stopping on for every caller. Passing both is the one ambiguity — explicit data wins
  with a `UserWarning`. The resolved state is the fitted `early_stopping_`. Splitting outside
  the estimator is the point: a water year, a gauge or a `TimeSeriesSplit` fold can now drive
  it. `X_val` follows the training NaN policy exactly (rows with non-finite `X` dropped by
  `prepare_arrays`; a non-finite `y` raises in `check_X_y`, for training and validation
  alike).
- **`cross_val_score` / `GridSearchCV` do not slice `X_val` per fold.** Passing it through
  `fit_params` hands every fold the same block, which for a time series means later folds
  train on data the "validation" block precedes. Inside CV, set `validation_fraction` so each
  fold carves its own tail. Under `sklearn.set_config(enable_metadata_routing=True)` these
  also become routable params that must be requested.
- **The default `scoring` is `"hypervolume"`** (see the Domination section). It scores the
  population *front* re-evaluated on held-out data, not the aggregated ensemble — the
  training front only ever improves, so it cannot detect overfitting. It is bounded in
  `[0, 1]`, and it removes the optional `forecast_performance` dependency from the default
  path (`"crps"` raises `ImportError` without it). `"reliability"` alone was the thing this
  replaces: a population can cut CRPS sharply while alpha sits still, and vice versa.
- Mechanics: monitored every `check_every` generations, stopping after `n_iter_no_change`
  checks without improvement > `tol`, restoring the best ensemble (`shuffle=False` by
  default — correct for time series). Two deviations:
  - `fit` builds the ensemble at **every** check (it needs `score_hypervolume(details=True)`
    for `history_`, since the scorer contract returns a float). Benchmarked cost-neutral:
    108.23 → 108.56 ms per check at population 1000 / n_val 600, of which `model.forward` is
    103 ms and `make_ensemble` 0.42 ms. (Measured at 0.4.0, when a further 0.11 ms went on
    `x_scaler.transform`; 0.5.0 removed that step.)
  - While the hypervolume is still exactly `0.0` the patience counter is **held**: a swarm
    that has not beaten the ceiling anywhere has no gradient to stop on, and
    `n_iter_no_change` flat checks would otherwise stop it at generation
    `check_every × n_iter_no_change`. The guard is gated on the hypervolume path —
    `"crps"` is a negated score, measured −2.18 … −0.41, so an ungated guard would mean a
    CRPS fit never stops (`test_negative_scorers_still_stop`).
- Probabilistic scorers live in `scoring.py` with the `scorer(estimator, X, y)` signature
  (higher = better) so they plug into `cross_val_score(..., scoring=...)` and `GridSearchCV`;
  they call `predict_quantiles` / `predictive_pvalues` (the default `make_scorer` only sees
  `predict`). The same callables back the early-stopping monitor. **All four go through
  `score_ensemble`**; `"hypervolume"` branches *before* `predict_quantiles`, since it needs
  neither the bands nor the p-values, which makes it the cheapest of the four. The ensemble
  carries `metric`, the resolved `hv_penalty` and `hv_interpolation`, so a scorer reaching it
  through `GridSearchCV` honours the configuration instead of silently defaulting.
  `GPURegressor.score_hypervolume` is a thin delegate to `ensemble_` — which means it scores
  the front kept after a rollback, i.e. the one `predict` uses, not the final generation.
- Never shuffle internally. That is about the engine's own default (`shuffle=False`), not
  about the user's choice of `cv`: two splitters answer two different questions, and both
  are legitimate to document.
  - `cv=TimeSeriesSplit(...)` estimates **forecast skill** — it never trains on the future,
    but the earliest years are never tested and each fold trains on a growing prefix.
  - `cv=KFold(k, shuffle=False)` gives k **contiguous blocks**: every part of the record is
    tested once and every fold trains on ~(k−1)/k of it, which conditions the exceedance
    axis better. It measures **record-wide consistency**, not forecast skill, and it is
    mildly optimistic because the blocks touch (the day before a held-out block is a
    training day, and recession memory spans days to weeks; `KFold` has no `gap=`). Carve
    out a separate test block first if the headline number matters.
  - `KFold(shuffle=True)` is the one to avoid for a daily series: each held-out day then
    sits between two training days, and autocorrelation makes the score flattering.
- `warm_start=True` continues the population across `fit` calls.

## The HYPE model (`models/hype/`)

HYPE is an external executable driven by a folder of text files, so it bends the contract in
one documented way and is worth reading before touching it.

- **The meteorology arrives via `forcing=`, not `X`.** `{"P": df, "T": df}` (DataFrame,
  Series or path; aliases in `forcing.py::ALIASES`) is written into every worker folder in
  HYPE's format, overlaying the template's own files, and the matching `read*obs` switch is
  set in `info.txt`. Omitting it falls back to the folder's existing files — supported, but
  then the data is invisible to the caller. `forcing.py` owns coercion, coverage/gap
  validation (forcing must be complete; `Qobs`/`Xobs` may have gaps → `-9999`), and the
  content digest. **The digest is part of the cache fingerprint** — the same parameters under
  different weather give a different simulation, so `model.fingerprint_` mixes layout,
  forcing, window and output selection.
- **Observations arrive the same way.** `load_observations(source)` and
  `HYPEModel.observations(source)` take a DataFrame, a Series or a path, so the observation
  and forcing sides are symmetric. `observations` = load + window-trim in one call and knows
  the model's `date_column`; `load_observations` cannot, so it always returns a width-1 `X`.
  `-9999` becomes NaN. A column is **never guessed** when a frame has two or more.
- **`align` keeps two arities; it is a row filter, not a loader.** `align(X)` returns one
  array, `align(X, y)` two. A `y` that carries a `DatetimeIndex` is **joined on dates** (so
  the two sides may come from different sources); anything else is positional and a length
  mismatch raises. It is deliberately *not* overloaded to accept a lone DataFrame: the return
  arity would then depend on the argument's runtime type, and an `X` that keeps its dates on
  the index would be read as observations, calibrating against day ordinals. Use
  `observations()` for the one-call route.
  - The join preserves `X`'s **order and multiplicity** — sorting or de-duplicating would
    change the effective weighting of the loss and of `eta`, and desynchronise any parallel
    array the caller holds. Implemented with the same integer-offset lookup as
    `resolve_indices`, so it is O(n) and safe on both sides.
  - It warns **only** when in-window `X` rows have no observation. Observations outside `X`
    are not a mismatch (`X` is routinely a deliberate subset), and a plain window trim is
    this method's documented purpose — warning there would fire on every call and pollute
    the `hype_observations` fixture.
- **Window precedence**: explicit `bdate`/`edate` argument > span of the supplied
  forcing > template `info.txt`. `cdate` is different: it is the *warmup boundary*, not a
  period choice, so the template's value is kept whenever it still lies inside the window.
  Supplying data must not silently discard the spin-up the template asked for — that is what
  makes "no `forcing=`" and "`forcing=` holding the folder's own files" give identical
  simulations, which is a test.
- **Two silent-date traps, both guarded.** A positional (numeric) DataFrame index is refused:
  pandas reads it as nanoseconds since 1970 and hands back a plausible-looking 1970 window. A
  **tz-aware** index is refused too: day ordinals floor in UTC, so an `Asia/Tokyo` index
  turns 2020-01-01 into 2019-12-31 with nothing to show it happened. `check_ordinals`
  (`dates.py`) is called by both `resolve_indices` and `align`, so a scaled `X` raises instead
  of silently filtering to zero rows.
- **`X` carries dates, not features.** `X[:, date_column]` holds the `datetime64[D]` day
  ordinal and `forward` returns the simulated value for exactly those dates. Day resolution
  is required: float64 is exact to 2⁵³, so day ordinals (~2e4) round-trip but nanosecond ones
  (~1.8e18) do not. Consequences: never place a scaler in front of the estimator (the dates
  are destroyed outside it, where nothing can recover them — `forward` raises instead), and
  keep `shuffle=False`.
- **One continuous run per parameter set, cached.** `forward` slices rows out of a cached
  full-window simulation. Sizing matters: `evolve` only re-evaluates new candidates, but an
  early-stopping check forwards the *whole* population, mostly long-lived survivors, so
  `cache_size` must exceed `(check_every + 1) × population` or checks stop being free.
  `cache_size=0` disables it; a fresh run is quantised to the cache's float32 first so cached
  and uncached results are bit-identical.
- **Search space is the unit box**, via `search_transform` / `inverse_search_transform`.
  `parameter_bounds` stays honest (physical values), but MOPSO's `c3` perturbation is
  *absolute* (`mopso.py:90`), so a raw physical box would switch exploration off in the wide
  dimensions. Use broadcasting (`np.where`), never boolean indexing: `inverse_search_transform`
  is called on 1-D bounds and `search_transform` on 2-D populations.
- **Run budget** — the whole cost model. Screening costs `screen_oversample × population`,
  each generation `population`, each early-stopping check zero — but that last one is
  conditional, not automatic. Every monitor re-evaluates the whole population on the
  validation window: the ensemble scorers through `predict_quantiles`, `"hypervolume"`
  through `ensemble._simulate`, which is the same second `forward` pass. It stays free only while that window sits inside the
  cached run, so keep `cache_size > (check_every + 1) × population` **and** keep `X_val`
  inside the model window. `test_early_stopping_checks_cost_no_runs` pins both monitors at
  zero. Keep test budgets tiny.
  **Cross-validation costs k× a full fit** (each fold `clone`s the model, so each gets its
  own workspace *and* its own cache); but *scoring* a fitted model on any other window costs
  **zero**, because the whole window is already simulated. So a held-out test block is free
  and the folds are not.
- **To plot or compare per-fold results, keep the arrays, not the models.** Each fold's
  `gpu.model_` owns a temp workspace and `n_workers` processes, so it must be `close()`d as
  soon as the fold is scored. Stash `predict_quantiles` / `predictive_pvalues` output inside
  the fold loop instead — scoring a held-out block there is already free (the window is
  cached), and the plots then neither cost a run nor depend on a live workspace.
- **An under-resourced swarm degrades silently.** Measured on Tomar (8 parameters): at
  population 24 × 8 generations the bands collapse onto one value, so `pi = 1/std` is `inf`,
  a fifth of observations get no p-value, and `alpha` is computed from what is left — 0.23
  with `xi` 0.03. At 40 × 15 the same setup gives `alpha` 0.75, `xi` 0.86, everything scored.
  Nothing raises in the first case. Report `xi` and the finite-p-value fraction, not `alpha`
  alone.
- **`fit` runs a `clone`**, so the model instance the user passes never executes: read
  counters and workspace off `estimator.model_`.
- **Failed runs.** A NaN column scores non-exceedance *exactly* 0.0, and `DoubleParetoSorter`
  ranks by exceedance coverage rather than loss dominance, so while the population is still
  narrow a failure extends front 0 and survives selection with infinite crowding; once the
  population has spread across the axis it is properly dominated. Hence `max_failure_fraction`
  and the loud warning. **Open follow-up (core, not the model):** `_make_evaluate` should
  exclude particles with non-finite simulations from ranking. The *hypervolume* already
  neutralises them — η exactly 0.0 plus a non-finite loss clips to `P` and contributes zero
  area — so the monitor is not fooled; the ranking path still is.
- **Testing needs no HYPE.** `executable` accepts a full command, so `tests/hype_stub.py`
  (invoked as `[sys.executable, stub]`) stands in for the exe and exercises the entire
  pipeline including multiprocessing. Integration tests against a real folder read
  `FORESIGHT_HYPE_FOLDER` and skip when unset. The stub and `tests/data/hype_template/` are a
  matched pair: the stub reads `par_reference.txt` to measure each parameter as a ratio to
  its default, reads `Pobs.txt` so a `forcing=` swap is observable (scaled by a *fixed*
  nominal rainfall — normalising by the series' own mean would make it blind to the
  magnitude of the forcing), and its level response saturates so a swarm can span the
  exceedance axis.

## OpenCL overload

NumPy is the default and the only tested backend. `models/mlp_opencl.py` documents how a
kernel-backed `forward` would override the NumPy one; it raises a clear error until the
`[opencl]` extra ships kernels. Keep OpenCL confined to per-model overrides — never let it
leak into the engine.

## Environment

- Develop in the conda environment named **`foresight_gpu`** (`conda env create -f
  environment.yml`), used for the package, tests and notebooks (register it as a Jupyter
  kernel named `foresight_gpu`).
- A machine with the full stack (incl. the `forecast_performance` / `performance` package)
  has been seen at `C:\Users\<user>\.conda\envs\analise_desempenho\python.exe` — **path
  differs per machine; resolve the env yourself, don't hard-code it.**
- Setup: `pip install -e ".[dev]"` (also pulls the `forecast-performance` git dependency).
- Run tests: `pytest tests/ -v`.

## Style

- `snake_case` primary; PascalCase class names. `black`, line length 88.
- **NumPy-style docstrings** (Parameters / Returns). British spelling in prose
  ("visualise", "normalise", "behaviour").
- Bump the version in **both** `pyproject.toml` and `foresight_gpu.__version__` together.
- Don't break the public re-exports in `foresight_gpu/__init__.py`.

## Tests

- `pytest`, tests in `tests/`, config in `pyproject.toml`. Synthetic fixtures (seeded
  `np.random.default_rng`) for analytic-exact assertions; the example series under
  `examples/data/` back optional integration tests (skip if absent).
- Metrics validated against `forecast_performance`. Plotting tests run **headless**
  (`matplotlib.use("Agg")`), inspecting `fig`/axes/artists rather than rendering.
- Keep the sklearn estimator checks, the "three calling styles agree" metric tests, the CV
  and early-stopping tests, the domination-equivalence test, and the hypervolume reduction
  test (`test_step_equals_split_hypervolume`) green.
- **Known pre-existing failure:** `tests/test_cross_validation.py::test_cross_val_score_reliability`
  pins `EXPECTED_RELIABILITY_SCORE` to values ~1e-3 away from what this environment produces.
  It fails identically on `74d5d4b`, byte for byte, so it is an environment-pinning issue,
  not a regression. It was useful as a control when refactoring the training path; note that
  since 0.5.0 it now differs for **two** reasons — the environment, and the removal of the
  default MLP's scaling — so it is no longer a clean control. `EXPECTED_R2_SCORE` in the same
  file was re-pinned in 0.5.0 for the second reason alone.


## Important Notes

- The legacy code that this project was based on was validated for a long time and is considered reliable. Being so when you are using it you should not assume that any errors should be solved with wrapping the legacy code with a lot of fall backs.

- The same is applicable to the new code, you should not invent if you are unsure of how to do things, and you instead ask for help.
