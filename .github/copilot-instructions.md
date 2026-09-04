See [AGENTS.md](../AGENTS.md) for this project's conventions, architecture, the
forward-model contract, the metric/domination system, cross-validation and early stopping,
the conda environment, and testing guidance. It is the single source of truth for AI coding
agents working in this repository.

Key reminders:
- The core package is a **general scikit-learn regressor**; hydrology-specific and
  convenience code lives in `foresight_gpu/utils/`.
- The optimiser owns the parameters; a model is a `BaseForwardModel` with
  `forward(X, params) -> [n_samples, n_particles]`, vectorised over particles — not a
  scikit-learn estimator.
- Metrics are `Metric` objects that stringify to their name; the engine only calls
  `metric.loss(sim, obs)` (lower = better). Deterministic metrics are vectorised over the
  population axis and cross-checked against `forecast_performance`.
- `predict` returns a point estimate `(n_samples,)`; the bands come from
  `predict_quantiles`. `__init__` stores params only; fitted attrs end with `_`.
- `metric` (training loss) and `scoring` (held-out probabilistic criterion for early
  stopping / CV) are different knobs. Never shuffle time series internally.
- NumPy is the default backend; OpenCL is a per-model overload only.
- Use the `foresight_gpu` conda environment. Run tests with `pytest tests/ -v`.
