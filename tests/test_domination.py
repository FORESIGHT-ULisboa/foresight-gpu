"""Double-Pareto sorting: invariants + equivalence with the reference algorithm."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from foresight_gpu.domination import DoubleParetoSorter


def _reference_fronts(x0, x1):
    """Reference double-Pareto sort (WRR draft, Appendix A). Permanent regression anchor."""
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    idx = np.lexsort((-((x0 - 0.5) ** 2), x1))
    fronts = [[int(idx[0])]]
    left = [float(x0[idx[0]])]
    right = [float(x0[idx[0]])]
    for raw in idx[1:]:
        i0 = int(raw)
        v = float(x0[i0])
        if left[-1] <= v <= right[-1]:
            fronts.append([i0])
            left.append(v)
            right.append(v)
        else:
            for i1 in range(len(fronts)):
                if v < left[i1] or v > right[i1]:
                    if v < left[i1]:
                        left[i1] = v
                        fronts[i1].insert(0, i0)
                    else:
                        right[i1] = v
                        fronts[i1].append(i0)
                    break
    return fronts


def _objectives(rng, n):
    return np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 3, n)])


class TestEquivalence:
    def test_matches_reference_random(self, rng):
        sorter = DoubleParetoSorter()
        for n in (5, 50, 200, 1000):
            obj = _objectives(rng, n)
            assert sorter.sort(obj) == _reference_fronts(obj[:, 0], obj[:, 1])

    def test_matches_legacy_if_present(self, rng):
        legacy = Path(__file__).resolve().parent.parent / "_legacy" / "domination.py"
        if not legacy.exists():
            pytest.skip("legacy source removed")
        spec = importlib.util.spec_from_file_location("_legacy_domination", legacy)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # legacy code predates NumPy 2.0
            pytest.skip(f"legacy domination not runnable under this NumPy: {exc}")
        sorter = DoubleParetoSorter()
        for n in (10, 100, 500):
            obj = _objectives(rng, n)
            expected = mod.convexSorting(obj[:, 0].copy(), obj[:, 1].copy())
            assert sorter.sort(obj) == [list(f) for f in expected]


class TestStructure:
    def test_front_levels_cover_all(self, rng):
        sorter = DoubleParetoSorter()
        obj = _objectives(rng, 300)
        levels = sorter.front_levels(obj)
        assert not np.any(np.isnan(levels))
        assert levels.min() == 0

    def test_partition_is_complete_and_disjoint(self, rng):
        sorter = DoubleParetoSorter()
        obj = _objectives(rng, 300)
        flat = [i for front in sorter.sort(obj) for i in front]
        assert sorted(flat) == list(range(300))

    def test_best_front_holds_min_loss(self, rng):
        sorter = DoubleParetoSorter()
        obj = _objectives(rng, 300)
        assert int(np.argmin(obj[:, 1])) in sorter.sort(obj)[0]

    def test_more_than_two_objectives_not_implemented(self, rng):
        with pytest.raises(NotImplementedError):
            DoubleParetoSorter().sort(rng.uniform(size=(20, 3)))

    def test_requires_2d(self):
        with pytest.raises(ValueError):
            DoubleParetoSorter().sort(np.arange(10))
