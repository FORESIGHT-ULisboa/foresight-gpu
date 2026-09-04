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
- Never shuffle internally; document `cv=TimeSeriesSplit(...)` for users.
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
