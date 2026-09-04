"""Dominance-sorting interface.

Objectives are passed as a 2-D array where **column 0 is the exceedance (probabilistic)
axis** and the remaining columns are one or more loss axes:

    objectives[:, 0]  = non-exceedance (eta)      # mirrored around 0.5
    objectives[:, 1:] = loss(es) (e.g. log10 error)

The current concrete sorter (:class:`~foresight_gpu.domination.double_pareto.DoubleParetoSorter`)
handles exactly one exceedance + one loss axis. The 2-D ``objectives`` contract is the
forward-compatible seam for a future N-objective sorter.
"""

from abc import ABC, abstractmethod

import numpy as np


class DominanceSorter(ABC):
    """Rank solutions into ordered non-domination fronts."""

    @abstractmethod
    def sort(self, objectives):
        """Return a list of fronts (lists of row indices), best front first.

        Parameters
        ----------
        objectives : ndarray
            Shape ``[n_solutions, n_objectives]``; column 0 is the exceedance axis.
        """

    def front_levels(self, objectives):
        """Return an array mapping each solution to its front index (0 = best)."""
        objectives = np.asarray(objectives, dtype=float)
        levels = np.full(objectives.shape[0], np.nan)
        for level, front in enumerate(self.sort(objectives)):
            levels[np.asarray(front, dtype=int)] = level
        return levels
