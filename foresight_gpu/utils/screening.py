"""Pre-processing / screening helpers used at the start of ``fit``.

* :func:`prepare_arrays` drops non-finite rows once and returns contiguous float arrays,
  so the per-generation hot loop stays allocation-light.
* :func:`screen_initial_population` optionally replaces a purely random initial swarm with
  one pre-selected for a good spread across the exceedance axis, reducing wasted early
  generations.

These are helpers (see AGENTS.md): the estimator calls them, but they carry no
hydrology-specific assumptions.
"""

import numpy as np


def prepare_arrays(X, y):
    """Drop non-finite rows and return contiguous float arrays.

    Parameters
    ----------
    X : array-like, shape ``[n_samples, n_features]``
    y : array-like, shape ``[n_samples]``

    Returns
    -------
    X_clean : ndarray
    y_clean : ndarray
    mask : ndarray of bool
        Which original rows were kept.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-D [n_samples, n_features].")
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    #Contiguous arrays are important for performance in the hot loop of the estimator, 
    # they ensure that the data is stored in a single block of memory, which can improve cache performance and reduce memory fragmentation. 
    return np.ascontiguousarray(X[mask]), np.ascontiguousarray(y[mask]), mask


def screen_initial_population(evaluate, low, high, population, rng, oversample=3):
    """Pre-select an initial population spread across the exceedance axis.

    Draws ``oversample * population`` random candidates, evaluates them once, then keeps the
    lowest-loss candidate in each of ``population`` exceedance bins, filling any spare slots
    with the next lowest-loss candidates overall.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(params) -> (simulations, fit)``; ``fit[:, 0]`` = exceedance,
        ``fit[:, 1]`` = loss.
    low, high : ndarray
        Search bounds.
    population : int
        Target population size.
    rng : numpy.random.Generator
    oversample : int
        Candidate pool multiplier.

    Returns
    -------
    ndarray
        Selected population, shape ``[population, n_params]``.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    pool = max(int(oversample), 1) * population
    candidates = rng.uniform(0.0, 1.0, (pool, low.size)) * (high - low) + low
    _, fit = evaluate(candidates)
    exceedance, loss = fit[:, 0], fit[:, 1]

    edges = np.linspace(0.0, 1.0, population + 1)
    bins = np.clip(np.digitize(exceedance, edges) - 1, 0, population - 1)

    chosen = []
    for b in range(population):
        idx = np.where(bins == b)[0]
        if idx.size:
            chosen.append(int(idx[np.argmin(loss[idx])]))

    if len(chosen) < population:
        taken = set(chosen)
        for i in np.argsort(loss):
            i = int(i)
            if i not in taken:
                chosen.append(i)
                taken.add(i)
            if len(chosen) >= population:
                break

    return candidates[np.asarray(chosen[:population], dtype=int)]
