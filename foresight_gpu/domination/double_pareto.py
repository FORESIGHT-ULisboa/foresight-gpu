"""Double / mirrored Pareto sorting (manuscript in preparation, Sec. 2.6 & App. A).

The GPU objective space has two *mirrored* Pareto fronts: as exceedance moves away from
0.5 in either direction the achievable error rises, so the ideal front is a convex curve
minimal near eta = 0.5. This sorter ranks solutions into non-domination fronts over that
mirrored surface.

The original implementation exposed both ``doubleParetoSorting`` and a ``convexSorting``
wrapper. In the shipped code path the convex wrapper's reordering loop was unreachable
(guarded by a counter that never advanced), so it reduced to ``doubleParetoSorting`` plus
a defensive boundary-ordering assertion. We therefore implement the double-Pareto sort
directly; the equivalence test locks the output against the Appendix-A reference.

Efficiency note (see AGENTS.md / plan): this runs every generation on ~2x the population
and the point-placement loop is O(n_solutions x n_fronts) in the worst case. A faithful,
readable port lands first; a vectorised version (numpy boundaries + ``searchsorted``) can
replace it later *provided it proves identical front assignments* against the reference.
"""

import numpy as np

from .base import DominanceSorter


def _double_pareto(x0, x1):
    """Rank points by the mirrored double-Pareto criterion.

    Parameters
    ----------
    x0 : ndarray
        Exceedance values (the mirrored axis).
    x1 : ndarray
        Loss values (lower is better).

    Returns
    -------
    list of list of int
        Fronts of row indices, best (least-dominated) first. Within a front, indices are
        ordered left-to-right along the exceedance axis.
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    # Process by ascending loss; break ties by descending distance from eta = 0.5.
    order = np.lexsort((-((x0 - 0.5) ** 2), x1))

    fronts = [[int(order[0])]]
    left = [x0[order[0]]]
    right = [x0[order[0]]]

    for raw in order[1:]:
        i = int(raw)
        value = x0[i]
        if left[-1] <= value <= right[-1]:
            # Dominated by every existing front -> start a new one.
            fronts.append([i])
            left.append(value)
            right.append(value)
        else:
            # Belongs to the first front whose exceedance span it extends.
            for f in range(len(fronts)):
                if value < left[f]:
                    left[f] = value
                    fronts[f].insert(0, i)
                    break
                if value > right[f]:
                    right[f] = value
                    fronts[f].append(i)
                    break
    return fronts


class DoubleParetoSorter(DominanceSorter):
    """Non-domination sorting on the mirrored (exceedance, loss) surface."""

    def sort(self, objectives):
        objectives = np.asarray(objectives, dtype=float)
        if objectives.ndim != 2:
            raise ValueError("objectives must be 2-D [n_solutions, n_objectives].")
        n_objectives = objectives.shape[1]
        if n_objectives != 2:
            raise NotImplementedError(
                "DoubleParetoSorter handles exactly 2 objectives "
                "(1 exceedance + 1 loss). objectives[:, 0] = exceedance, "
                "objectives[:, 1:] = losses is the seam for a future N-objective sorter."
            )
        return _double_pareto(objectives[:, 0], objectives[:, 1])
