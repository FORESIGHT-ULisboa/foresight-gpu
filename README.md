# foresight_gpu — Generalized Pareto Uncertainty

`foresight_gpu` turns a family of deterministic models into a **reliable probabilistic
forecast**. It implements the **Generalized Pareto Uncertainty (GPU)** methodology: a
multi-objective optimiser evolves a population of parameter sets for a deterministic
*forward model*, ranks them on a **double / mirrored Pareto front** over two objectives —
*non-exceedance* versus an *error metric* — and aggregates the survivors by non-exceedance
band into an estimate of the conditional distribution. Reliability and resolution are
scored with the Renard et al. (2010) diagnostics.

The public interface is a **scikit-learn regressor**, so it drops into `Pipeline`,
`cross_val_score`, `GridSearchCV` and friends.

> "GPU" is a double meaning — *Generalized Pareto Uncertainty* (the method) and *Graphics
> Processing Units* (an optional compute backend). **The core runs on NumPy;** OpenCL is a
> documented overload point.

> **New here?** Start with [`notebooks/00_quickstart.ipynb`](notebooks/00_quickstart.ipynb),
> then the other notebooks in order (see [Notebooks](#notebooks)).

## Features

| Area | What you get |
|---|---|
| Estimator | `GPURegressor` — `fit` / `predict` (point) / `predict_quantiles` (bands), sklearn-compatible |
| Forward models | tiny `MLPModel`, hydrological `GR4JModel`, external `HYPEModel` (SMHI HYPE executable, driven by your own forcing DataFrames), or **bring your own** `BaseForwardModel` |
| Error metrics | `nse`, `kge`, `kge_prime`, `mae`, `mse`, `rmse` — fast, vectorised over the population, + Lp regularisation |
| Uncertainty | `ParetoEnsemble` post-convergence estimator (inverse-CDF, band aggregation) |
| Model selection | cross-validation, `GridSearchCV`, and **early stopping** on the held-out double-Pareto hypervolume |
| Diagnostics | predictive QQ, time-series bands, double-Pareto front plots; Renard-2010 α/ξ/π |

## Installation

**A · From source (development)**

```bat
conda env create -f environment.yml
conda activate foresight_gpu
python -m ipykernel install --user --name foresight_gpu --display-name "foresight_gpu"
```

Or with an existing environment:

```bat
pip install -e ".[dev]"
```

The install pulls the companion package
[`forecast_performance`](https://github.com/FORESIGHT-ULisboa/forecast_performance) as a
git dependency (used for richer diagnostics and to cross-check the fast metrics).

> **Windows / BLAS note.** The `foresight_gpu` core is BLAS-free (it uses `einsum`), so
> fitting works everywhere. Plotting and scikit-learn helpers do use NumPy's LAPACK. If
> `numpy` matmul or `numpy.linalg` crashes with a delay-load error (`0xc06d007f`) — a
> known issue for some conda-forge OpenBLAS builds on Windows — install an MKL-backed
> NumPy (`conda install -c defaults numpy scipy`) in the environment.

## Quick start

```python
import numpy as np
from foresight_gpu import GPURegressor

rng = np.random.default_rng(0)
X = rng.uniform(size=(500, 3))
y = X[:, 0] + 0.3 * rng.standard_normal(500)

gpu = GPURegressor(population=500, n_iter=100, random_state=0).fit(X, y)

y_hat = gpu.predict(X)                                   # point estimate (median band)
bands = gpu.predict_quantiles(X, quantiles=[0.05, 0.5, 0.95])   # probabilistic bands
```

### Cross-validation & early stopping

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from foresight_gpu.models import MLPModel
from foresight_gpu.scoring import reliability_scorer

# Early stopping is inferred from the data — there is no early_stopping flag.
# It monitors the held-out double-Pareto hypervolume: one number that rises both when
# the front's loss drops and when it spreads further across the exceedance axis.

# (a) let the estimator carve a chronological validation tail
gpu = GPURegressor(validation_fraction=0.2, n_iter=500, random_state=0).fit(X, y)
print(gpu.n_iter_, gpu.best_iteration_, gpu.early_stopping_)

# (b) or split it yourself — any holdout you like, all of X_tr then trains
gpu = GPURegressor(n_iter=500, random_state=0).fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

# (c) or neither: plain fit(X, y) runs the full n_iter

# The same indicator is a normal scoring option, and takes the metric as a parameter,
# so a fitted front can be re-read under any metric with no refit:
gpu.ensemble_.score_hypervolume(X_test, y_test)           # calibration metric
gpu.ensemble_.score_hypervolume(X_test, y_test, "kge")    # any other

# time-series cross-validation on a probabilistic score
scores = cross_val_score(GPURegressor(random_state=0), X, y,
                         cv=TimeSeriesSplit(5), scoring=reliability_scorer)

# search nested model / regularisation hyper-parameters
search = GridSearchCV(GPURegressor(model=MLPModel(), random_state=0),
                      {"model__n_hidden": [4, 8], "reg_lambda": [0.0, 1e-3]},
                      cv=TimeSeriesSplit(3))
```

### Bring your own model

Implement two methods and you can drop any deterministic model into GPU:

```python
from foresight_gpu.models import BaseForwardModel

class MyModel(BaseForwardModel):
    def n_parameters(self, n_features):
        ...   # length of one parameter vector
    def forward(self, X, params):
        ...   # X:[n_samples, n_features], params:[n_particles, n_params]
              # -> [n_samples, n_particles]  (vectorised over particles)
```

See [`notebooks/01_custom_model.ipynb`](notebooks/01_custom_model.ipynb) and the contract in
[AGENTS.md](AGENTS.md).

## Notebooks

| notebook | what it shows |
|---|---|
| [`00_quickstart`](notebooks/00_quickstart.ipynb) | fit, predict and plot bands on a synthetic daily record |
| [`01_custom_model`](notebooks/01_custom_model.ipynb) | plug your own `BaseForwardModel` into GPU |
| [`02_gr4j`](notebooks/02_gr4j.ipynb) | a hydrological forward model (GR4J) on a synthetic catchment |
| [`03_extra_diagnostics`](notebooks/03_extra_diagnostics.ipynb) | hypervolume, early stopping, cross-validation, grid search, pipelines |
| [`04_uncertainty_experiments`](notebooks/04_uncertainty_experiments.ipynb) | nine synthetic tests with known error structure (T1–T9) |

## Project structure

```
foresight_gpu/     core package — a general sklearn regressor
  estimator.py     GPURegressor
  ensemble.py      ParetoEnsemble (fitted artifact)
  models/          BaseForwardModel, MLPModel, GR4JModel, MLPModelOpenCL (stub)
    hype/          HYPEModel: the SMHI HYPE executable as a forward model,
                   driven by forcing you pass in (DataFrames or file paths)
  metrics/         fast vectorised metrics + Lp regularisation + Renard diagnostics
  domination/      double-Pareto sorting + hypervolume indicator (seam for >2 objectives)
  optimizers/      MOPSO
  scoring.py       probabilistic sklearn scorers
  crowding.py      NSGA-II crowding
  utils/           helpers: features, screening, plotting
notebooks/         numbered, self-contained walkthroughs
tests/             pytest suite
```

## Running the tests

```bat
pytest tests/ -v
```

## License

MIT © 2026 FORESIGHT ULisboa. See [LICENSE](LICENSE).
