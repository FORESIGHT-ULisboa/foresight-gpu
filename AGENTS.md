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
  ├─ DominanceSorter (Pareto ranking)    foresight_gpu/domination/
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
  loss driving the Pareto front. `scoring` is the held-out *probabilistic* criterion for
  early stopping and model selection. Keep both.
- Early stopping mirrors `MLPRegressor`/`HistGradientBoosting`: a chronological validation
  tail (`shuffle=False` by default — correct for time series), monitored every
  `check_every` generations with `scoring`, stopping after `n_iter_no_change` checks without
  improvement > `tol`, restoring the best ensemble.
- Probabilistic scorers live in `scoring.py` with the `scorer(estimator, X, y)` signature
  (higher = better) so they plug into `cross_val_score(..., scoring=...)` and `GridSearchCV`;
  they call `predict_quantiles` / `predictive_pvalues` (the default `make_scorer` only sees
  `predict`). The same callables back the early-stopping monitor.
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
  each generation `population`, each early-stopping check zero. Keep test budgets tiny.
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
  exclude particles with non-finite simulations from ranking.
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
  and early-stopping tests, and the domination-equivalence test green.
