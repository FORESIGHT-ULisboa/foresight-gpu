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
estimate of the inverse conditional CDF (manuscript in preparation, Eq. 4). Reliability
and resolution are scored with the Renard et al. (2010) diagnostics.

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
  ├─ scoring (CV, diagnostics)           foresight_gpu/scoring.py
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
- Optional `regularization(params, n_features) -> [n_particles] | 0.0` — term added to the
  training loss on model-space `params` (default `0.0`). **Regularisation is the model's
  job**, not the estimator's: what is meaningful depends on the parameterisation (the MLP
  takes `reg_lambda`/`reg_p` and L-p-penalises its weights; GR4J has none). The
  estimator's `reg_lambda`/`reg_p` were removed; search `model__reg_lambda` instead.
- Optional `search_transform` / `inverse_search_transform` — a per-model search-space
  warp (the MLP uses `(w/4)**5`; GR4J uses identity). Do **not** put this on the engine.

**The model owns its units.** `forward` returns simulations directly comparable with `y`;
nothing rescales either side. `scales_inputs` / `scales_outputs` were **removed in 0.5.0** —
they were policy living in the estimator, and only ever one model used them. `fit` warns if a
model still carries them so the change cannot be silent.

- **`X`**: the caller's job, via `Pipeline([StandardScaler(), GPURegressor(...)])`.
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

The front-quality indicator that drives early stopping (the only criterion since 0.7.0).
`R` is the **reference point** of the hypervolume literature (`penalty` until 0.6.0). Two
readings of one quantity:
minimise `D = ∫_covered L(η)dη + R·(uncovered span)`, or maximise the area between the
staircase and a ceiling at `R`. They are algebraically identical (`HV = R − D`), so **`R` is
the price of a unit of uncovered exceedance**. Reported as `hv = 1 − D/R ∈ [0,1]`, higher =
better, to match the `scoring` contract.

- **The computation lives on `ParetoEnsemble`, not the estimator.** `score_hypervolume(X, y,
  metric=None, *, reference, interpolation, space, details)` and
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
  - `"linear"` (default): clip to `[0, R]`, box `R`. Unchanged from 0.4.0.
  - `"log10"`: clip `log10(loss)` to `[−R, +R]`, box `2R`. **Symmetric**, so `R` is read as a
    bound on `|log10 L|` — `R=2` is raw loss in `[1e-2, 1e2]` — and no second parameter is
    needed. A front at loss 1 everywhere (the no-skill line) scores exactly `hv = 0.5`, which
    is what makes the number readable.
  - **Why offer it.** Linear space spends almost the whole box on losses nobody cares about:
    at `R=100` the region of interest (losses of order 1) is a hundredth of the axis. Measured
    on a synthetic sine-plus-noise problem, the usable `hv` range is **2–7× wider** in
    log10 (2.3× over one run's early-stopping trace, 6.9× over converged fronts at 5…160
    generations). The factor depends on which fronts are compared; the direction does not.
  - **This is not the sorter's `log10`.** That one is floored at `tiny`, unbounded below and
    carries the `_BAD_LOSS` sentinel; integrating it is meaningless. The bound here is the
    symmetric clip. Every registry metric has `Metric.loss ≥ 0` with optimum exactly 0, so
    `log10` is defined everywhere except at 0 — and loss 0 clips to the floor `−R`, taking
    maximum credit, which is correct.
  - `log10` is monotone, so **front 0 is identical in both spaces**; the transform is applied
    after front selection and a precomputed `front=` stays valid.
  - The two spaces need `R` orders of magnitude apart, so the fixed defaults are per space:
    `DEFAULT_HV_REFERENCE` (100.0) or `DEFAULT_HV_LOG_REFERENCE` (2.0). Under `log10`,
    `"climatology"` is taken into log units — which for every
    `greater_is_better` metric is `log10(1.0) = 0`, not a usable ceiling, so it falls through
    the existing non-positive guard and warns.
- **`front_objectives` drops rows where *either* `X` or `y` is non-finite.** `_simulate`
  masks only `X`; one NaN in `y` makes `metric.loss` NaN for every particle, clipping them
  all to `R` and reporting `hv == 0` silently. The estimator path got this free from
  `prepare_arrays`; the `cross_val_score` path does not.
- **The step staircase (0.8.0).** Each cell takes `max(lᵢ, lᵢ₊₁)`, the *outer* point on the
  V, except the two cells beside the minimum. Those are split at their η-midpoint, and the
  inner half takes the minimum loss.
  - **Why.** With plain `max`, every point owns one cell except the minimum, which owns zero
    width. A front where only the best model improved scored the same hv.
  - **Identity.** The value is exactly the standard split-half hypervolume plus those two
    half-cells. `test_step_equals_split_hypervolume_plus_midpoint_cells` is that identity as
    an executable test. **Keep it green**, same discipline as the domination-equivalence test.
  - **Failed neighbours.** A cell is not split toward a failed (non-finite) neighbour,
    otherwise the gap to it would be credited at the minimum.
  - **Cost of the rule.** Inserting a point next to the minimum shrinks its half-cells, so
    monotonicity under insertion is lost in those two cells only.
  - **Shared code.** `_step_cells` serves both the indicator and the plot. `k > 1` is still
    the plain split HV.
  - `interpolation="linear"` is an optimistic trapezoidal smoothing with no hypervolume
    interpretation, 1 loss axis only. It is always `>=` step, and equal on the split cells.
- Losses clip to `[F, R]` (`F = 0` linear, `−R` log10). A particle worse than `R` is no
  better than a gap, and failed particles (non-finite loss, η exactly 0.0) clip to `R` and
  contribute **zero area** — they lengthen the covered span at zero height. That partly
  answers the failed-run follow-up below, though the *ranking* path is still open.
- **`clipped_fraction` makes a too-tight `R` visible.** Clipping is the intended semantics —
  no gradient among models you would never use — but past some share of the front the
  indicator simply stops responding, and that was previously invisible. The share of front-0
  at the ceiling (failed particles included) is reported in `details` and therefore lands in
  `history_` next to `coverage`; `score_hypervolume` warns above `HV_CLIP_WARN_FRACTION`
  (0.25). Measured at the NSE climatology ceiling `R=1`: **~0.8–0.9 of front-0 clipped**, `hv`
  ~0.01 against ~0.98 at `R=100` — the same worse-than-climatology figure that motivated the
  constant default, now reported instead of inferred. The clean demonstration is a
  low-variance validation window: it inflates every loss at once while
  `coverage` stays at 1.00, so `clipped_fraction` is the only thing that moves. A plain bias
  moves `coverage` instead — the front keeps some high-η particle that tracks the shift.
- **`R` is adaptive by default (`hv_reference="adaptive"`, 0.7.0).** `R =
  HV_REFERENCE_MARGIN (1.1) × nadir`, the nadir being the worst finite front-0 loss over
  *every* check so far (`reference_nadir`; `|log10 L|` in log10), tracked separately for the
  train and val sides. `R` only moves up, and **whenever it moves every stored front is
  rescored** under it.
  - **Why the rescoring is load-bearing.** A reference tracking the worst particle *without*
    it is fatal: on a front that never changes, hv goes 0.50 → 0.75 → 0.875 → 0.975 as R
    grows 1 → 2 → 4 → 20, so the monitor would report improvement whenever the swarm found
    something worse, and reset patience each time. That is why 0.4–0.6 froze `R` instead.
    With rescoring every check shares one box; `estimator._early_stopping_state` replays the
    patience rule over the whole rescored series at each check, so the best check can move
    *backwards*. `test_a_front_that_never_changes_never_improves` is the lock.
  - **The margin** (Ishibuchi et al., 2018): with `R` exactly at the nadir the worst point
    sits on the ceiling and, under `max(lᵢ, lᵢ₊₁)`, its whole outer cell is zero — hiding the
    extreme-η tail the method exists for.
  - **Nothing ever clips**, so each check's hv is exactly linear in `1/R`: `s_k = a_k − b_k/R`,
    with `a_k = 2·hv(2R) − hv(R)`. `_prune_candidates` uses this to drop kept ensembles the
    replay can never select again (two-rule proof in its docstring); without it every check's
    population would have to be kept. Locked by a 300-trial property test and by
    `test_pruning_does_not_change_the_fit`.
  - **Observed, not yet a problem but worth knowing:** the nadir is set by the first checks,
    when the swarm is widest and worst, and rarely moves afterwards (sine test problem: R=26
    at check 0, never moved). A large nadir in `linear` space makes hv ≈ coverage (val hv
    ~0.93 there), diluting loss resolution — the same compression `log10` was introduced for.
  - hv from *different* fits sits on different references. Compare fronts with
    `ensemble_.score_hypervolume(reference=<fixed>)`.
- **Fixed references remain** (`hv_reference=` a float or `"climatology"`); `hv_reference_scale`
  multiplies `R` in every mode. `DEFAULT_HV_REFERENCE = 100.0` is the fixed fallback and is
  deliberately not a climatology: GPU's extreme-η particles are *meant* to be biased — a
  particle at η≈0.95 has to systematically over-predict to get there, so its NSE is
  necessarily poor. Measured on a real front, **80% of front-0 scores worse than
  climatology** (loss quartiles `[0.81, 1.10, 1.64, 2.21, 6.36]`); a ceiling at 1.0 clips
  exactly those, and R=100 measured a *larger* usable range (5.4e-2 vs 2.7e-2).
  - `ParetoEnsemble.score_hypervolume` (and the fit loop, in fixed mode on the val side) warns
    on two rungs via `ensemble._warn_if_clipped`: **the whole front above `R`** (`hv` pinned
    at 0), else **more than `HV_CLIP_WARN_FRACTION` of it**. Neither fires on individual tail
    particles above `R`. Both texts are constant across checks so Python's dedup emits each
    once per fit. Not called in adaptive mode, where failed (∞-loss) particles are the only
    thing at the ceiling and would trip it misleadingly.
- **The ensemble's stored `hv_reference` applies only to the calibration reading** (same
  metric *and* space) — it carries that metric's loss units. Any other metric or space falls
  back to the fixed default for the space. Found the hard way: an NSE-adaptive R=36.5 reused
  under KGE clipped 51% of the front. After a fit it is the final R of the monitored side
  (val, else train), so `ensemble_.score_hypervolume(X_val, y_val) ==
  history_[best]["val_hv"]` exactly. Pre-0.7.0 pickles keep their `hv_penalty`.
- **`default_hv_reference(metric, obs, predictor)`** is what `hv_reference="climatology"` reaches
  for: `1.0` for `greater_is_better` metrics (`loss = 1 − value` is dimensionless, so this is
  the metric-value-0 no-skill line), otherwise the constant-predictor loss (`var` for MSE,
  `std` for RMSE, `mean|y−μ|` for MAE). The two branches are one rule — `NSE = 1 − MSE/var(y)`,
  so `nse.loss == mse.loss / default_hv_reference(mse, y)`; the constant-predictor loss *is*
  NSE's denominator. It touches only the generic `Metric` contract and names no metric, so
  `domination/` stays free of metric knowledge. **Do not** just call `metric.loss(const, y)`:
  a constant series has zero variance, so Pearson r is 0/0 and KGE′ γ is x/0, and the result
  is `nan` or ~1e15 *depending on whether `mean()` leaves a 2e-16 residue* — not even
  deterministic in kind.
- **Drawing it: `utils/plotting.py::plot_hypervolume(eta, loss, reference, ...)`.** Takes the
  `(eta, loss, front)` tuple `front_objectives` returns and covers all four cases
  (`space` x `interpolation`). It **recomputes `hv` from the same call it draws from** and puts
  it in the title, so the picture and the number cannot drift apart — which they had in an
  earlier notebook: the figure reported `interpolation="linear"` while shading the step staircase.
  `test_plot_hypervolume_title_matches_the_indicator` is the lock; keep it green.
  - `space="log10"` draws raw loss on a log axis. That is not cosmetic: a straight polyline
    between two points on a log axis has the geometric mean as its per-cell average, which is
    exactly what `interpolation="linear"` integrates in log space, and `max` is monotone — so
    one construction is faithful in both spaces.

- **N objectives.** `double_pareto_hypervolume` splits at the anchor into η ≤ s and η ≥ s,
  negates η on the right half and sums two standard `hypervolume()` calls; the halves are
  disjoint in η so the sum is exact for any number of loss axes. Two alternatives are
  recorded as rejected in the module docstring, with counterexamples: joining consecutive
  front points per η cell is **non-monotone** (one mediocre point collapses it 100×), and
  choosing the split by maximising the total **over-credits**. The split must be the anchor.
- **Failed runs (open follow-up).** A NaN simulation column scores non-exceedance exactly
  0.0, so while the population is narrow it can extend front 0 and survive selection with
  infinite crowding. `_make_evaluate` should exclude non-finite particles from ranking. The
  hypervolume already neutralises them (non-finite loss clips to the reference, zero area).

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

- **`metric` is the training loss; early stopping always monitors the validation
  hypervolume.** `GPURegressor` has had no `scoring` parameter since 0.7.0. `scoring` lives
  only on the sklearn side (`cross_val_score(..., scoring=make_gpu_scorer(...))`), for model
  selection.
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
- **The monitor is the hypervolume** (see the Domination section). It scores the
  population *front* re-evaluated on held-out data, not the aggregated ensemble — the
  training front only ever improves, so it cannot detect overfitting. `"reliability"` alone
  was the thing it replaced: a population can cut CRPS sharply while alpha sits still, and
  vice versa. The reliability/resolution/CRPS monitors were removed in 0.7.0; the three are
  now *recorded* in `diagnostics_`, on their own cadence (below).
- Mechanics: checked every `check_every` generations (and at the last), stopping after
  `n_iter_no_change` checks without improvement > `tol`, restoring the best ensemble
  (`shuffle=False` by default — correct for time series).
  - **Two records, two cadences, one owner per number.**
    - `history_` (every `check_every`, with or without validation data) is the
      early-stopping record: per side (`train_*` always, `val_*` with early stopping)
      `hv`, `coverage`, `clipped_fraction`, `n_front`, `reference` (in force at that check)
      and the front-0 `eta`/`loss` arrays. `*_hv` is rewritten whenever its side's reference
      moves (`GPURegressor._raise_reference`).
    - `diagnostics_` (every `diagnostics_every`, default `None`) holds only `reliability`,
      `resolution` and `crps` per side. They are reference-free, so never rescored — which
      is why hv is deliberately *not* duplicated there: a copy would go stale the moment the
      adaptive reference moved.
    - Both are lists of flat dicts keyed by `iteration`: `pd.merge(pd.DataFrame(history_),
      pd.DataFrame(diagnostics_), on="iteration", how="outer")` is the whole join.
      `utils/plotting.py::plot_history` draws each panel from its owner.
    - **The retained model always gets a `diagnostics_` row** (`retained=True`), whatever
      the cadence: a fixed grid can miss `best_iteration_`, and the best can move backwards
      after a rescore, so it is computed at the end from `ensemble_` (one forward per side
      when restored; free otherwise). It equals what the check would have produced live
      (`test_retained_row_recomputed_at_the_end_equals_the_live_row`).
  - **`verbose` only prints.** Rows follow `diagnostics_every`, or every check when that is
    `None` (hv only). It never changes what is computed or stored
    (`test_verbose_does_not_change_what_is_stored`). A row off the check grid shows hv as
    seen live under the current reference. Plain `print` inside `fit`, no reporter class.
  - **At most one validation `forward` per generation**, shared by a check and a diagnostics
    row landing on it. The training side never forwards: it reuses the population's cached
    `simulations` (front via `ParetoEnsemble._front_from_sims`, bands via `_bands_from_sims`).
    Pinned by `test_one_validation_forward_per_generation_and_none_for_training`.
  - **Cost** (0.7.0, population 1000, MLP, 4 features, n_train 5000, n_val 600; one
    generation's training forward ~730 ms). A check: validation forward 86 ms (paid before
    0.7.0 too) + fronts ~36 ms, i.e. ~3% at `check_every=5`. A diagnostics row: ~555 ms,
    almost all training-side bands — **`post_process_bands` is 376 ms of it**, a per-row
    Python loop (ported, validated behaviour) and the obvious lever if it matters. Hence
    `check_every` 1–5 is cheap, `diagnostics_every` ~`n_iter/40` (10 at the default). For
    GR4J the forward dwarfs both.
  - While the hypervolume is still exactly `0.0` the patience counter is **held**: a swarm
    that has not beaten the ceiling anywhere has no gradient to stop on, and
    `n_iter_no_change` flat checks would otherwise stop it at generation
    `check_every × n_iter_no_change`. Only reachable with a fixed reference. (It used to be
    gated on the hypervolume path, because negated CRPS is always < 0; with HV the only
    monitor the gate went.)
- Probabilistic scorers live in `scoring.py` with the `scorer(estimator, X, y)` signature
  (higher = better) so they plug into `cross_val_score(..., scoring=...)` and `GridSearchCV`;
  they call `predict_quantiles` / `predictive_pvalues` (the default `make_scorer` only sees
  `predict`). They do not drive early stopping. **All four go through
  `score_ensemble`**; `"hypervolume"` branches *before* `predict_quantiles`, since it needs
  neither the bands nor the p-values, which makes it the cheapest of the four. The ensemble
  carries `metric`, the resolved `hv_reference`, `hv_interpolation` and `hv_space`, so a scorer reaching it
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

## OpenCL overload

NumPy is the default and the only tested backend. `models/mlp_opencl.py` documents how a
kernel-backed `forward` would override the NumPy one; it raises a clear error until the
`[opencl]` extra ships kernels. Keep OpenCL confined to per-model overrides — never let it
leak into the engine.

## Environment

- Develop in the conda environment named **`foresight_gpu`** (`conda env create -f
  environment.yml`), used for the package, tests and notebooks (register it as a Jupyter
  kernel named `foresight_gpu`).
- Setup: `pip install -e ".[dev]"` (also pulls `forecast-performance` from PyPI). It is a
  **core** dependency and imports as `forecast_performance` (≥1.0.0; the old `performance`
  module name is gone). CRPS is in every `diagnostics_` row, so nothing guards the import.
  It requires Python ≥3.11, which sets this package's floor.
- Run tests: `pytest tests/ -v`.

## Style

- `snake_case` primary; PascalCase class names. `black`, line length 88.
- **NumPy-style docstrings** (Parameters / Returns). British spelling in prose
  ("visualise", "normalise", "behaviour").
- Bump the version in **both** `pyproject.toml` and `foresight_gpu.__version__` together.
- Don't break the public re-exports in `foresight_gpu/__init__.py`.

## Tests

- `pytest`, tests in `tests/`, config in `pyproject.toml`. Synthetic fixtures (seeded
  `np.random.default_rng`) for analytic-exact assertions.
- Metrics validated against `forecast_performance`. Plotting tests run **headless**
  (`matplotlib.use("Agg")`), inspecting `fig`/axes/artists rather than rendering.
- Keep the sklearn estimator checks, the "three calling styles agree" metric tests, the CV
  and early-stopping tests, the domination-equivalence test, and the hypervolume reduction
  test (`test_step_equals_split_hypervolume_plus_midpoint_cells`) green.
- The whole suite is expected to pass. Avoid pinning exact scores from stochastic fits:
  they drift ~1e-3 across platforms and BLAS builds. Assert shape, finiteness and bounds,
  or exact identities between code paths, instead.


## Important Notes

- Do not invent: if you are unsure how to do something, ask for help instead of wrapping
  the code in fallbacks.
