"""NSGA-II phenotype crowding distance.

Crowding distance rewards solutions in sparsely populated regions of each front, keeping
the population diverse. Ported from the original GPU implementation; accepts front labels
either as a list of index lists or as an array of per-solution front levels.
"""

import numpy as np


def phenotype_crowding(*objectives, fronts=None):
    """Crowding distance per solution, summed across objective axes.

    Parameters
    ----------
    *objectives : ndarray
        One or more objective vectors, each shape ``[n_solutions]``.
    fronts : array-like or list of list of int, optional
        Front assignment. If an array, it holds each solution's front level; if a list of
        lists, each entry is a front's indices. Defaults to a single front spanning all
        solutions.

    Returns
    -------
    ndarray
        Crowding distance per solution (endpoints of each front are ``inf``).
    """
    first = np.asarray(objectives[0])
    if fronts is None:
        fronts = [list(range(first.shape[0]))]

    distance = np.zeros(first.shape[0], dtype=float)

    if isinstance(fronts, list):
        groups = [np.asarray(front, dtype=int) for front in fronts]
    else:
        fronts = np.asarray(fronts)
        groups = [np.where(fronts == level)[0] for level in np.sort(np.unique(fronts))]

    for values in objectives:
        values = np.asarray(values, dtype=float)
        for idx in groups:
            if idx.size == 0:
                continue
            order = np.argsort(values[idx])
            ordered = values[idx][order]
            span = ordered[-1] - ordered[0]
            if idx.size > 2:
                if span != 0:
                    distance[idx[order[1:-1]]] += (ordered[2:] - ordered[:-2]) / span
                else:
                    distance[idx[order[1:-1]]] = 0.0
            distance[idx[order[0]]] = np.inf
            distance[idx[order[-1]]] = np.inf

    return distance
