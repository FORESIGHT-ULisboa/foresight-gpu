"""Seasonal streamflow forecast for the Upper Zambezi (WRR draft, Section 4.1).

A standalone rewrite of the original ``GPU_MLP_trainer`` using the packaged API. It uses
only the day-of-year (encoded as sin/cos) as a predictor, so it produces a *regime*
forecast — the same seasonal exceedance distribution every year — and evaluates it on an
independent validation period.

Run from the repository root:

    python examples/seasonal_forecast.py

Reads the (repo-only) example series in ``examples/data/`` and writes a diagnostics figure
to ``examples/seasonal_forecast.png``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from foresight_gpu import GPURegressor
from foresight_gpu.metrics.probabilistic import renard_metrics
from foresight_gpu.utils import plot_double_pareto_front, plot_qq, plot_timeseries

DATA = Path(__file__).resolve().parent / "data"
SPLIT_YEAR = 1997  # train < SPLIT_YEAR, validate >= SPLIT_YEAR


def load_discharge():
    """Observed daily discharge at Victoria Falls."""
    q = pd.read_csv(
        DATA / "Qobs.txt", sep="\t", header=0, names=["date", "Qobs"],
        index_col=0, parse_dates=True,
    )
    return q["Qobs"].dropna()


def day_of_year_features(index):
    """Encode the day-of-year as a sin/cos pair (avoids the year-end discontinuity)."""
    doy = index.dayofyear.to_numpy(dtype=float)
    return np.column_stack([np.sin(2 * np.pi * doy / 365.25),
                            np.cos(2 * np.pi * doy / 365.25)])


def main():
    import matplotlib.pyplot as plt

    flow = load_discharge()
    X = day_of_year_features(flow.index)
    y = flow.to_numpy(dtype=float)
    train = flow.index.year < SPLIT_YEAR

    print(f"training on {train.sum()} days, validating on {(~train).sum()} days")
    gpu = GPURegressor(
        metric="mae", population=500, n_iter=100, force_positive=True, random_state=0,
    ).fit(X[train], y[train])

    pvalues = gpu.predictive_pvalues(X[~train], y[~train])
    bands = gpu.predict_quantiles(X[~train])
    diag = renard_metrics(pvalues, bands, gpu.ensemble_.band_probabilities)
    print(f"validation  alpha={diag['alpha']:.3f}  xi={diag['xi']:.3f}  pi={diag['pi']:.3f}")

    # Diagnostics figure: one validation year of bands, the QQ plot, and the front.
    val_index = flow.index[~train]
    first_year = val_index.year[0]
    year_mask = val_index.year == first_year
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_timeseries(
        bands[year_mask], gpu.ensemble_.quantiles, observed=y[~train][year_mask],
        index=val_index[year_mask].dayofyear, ax=axes[0],
    )
    axes[0].set_title(f"seasonal bands, {first_year}")
    axes[0].set_xlabel("day of year")
    axes[0].set_ylabel("discharge [m3/s]")

    plot_qq(pvalues, ax=axes[1])
    axes[1].set_title(f"validation QQ (alpha={diag['alpha']:.3f})")

    plot_double_pareto_front(gpu._fit[:, 0], gpu._fit[:, 1], ax=axes[2], max_fronts=6)
    axes[2].set_title("double-Pareto front")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "seasonal_forecast.png"
    fig.savefig(out, dpi=150)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
