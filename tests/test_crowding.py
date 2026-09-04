"""NSGA-II crowding-distance tests."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from foresight_gpu.crowding import phenotype_crowding


class TestCrowding:
    def test_endpoints_are_infinite(self):
        values = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        dist = phenotype_crowding(values, fronts=np.zeros(5, dtype=int))
        order = np.argsort(values)
        assert np.isinf(dist[order[0]])
        assert np.isinf(dist[order[-1]])
        assert np.all(np.isfinite(dist[order[1:-1]]))

    def test_known_values_single_objective(self):
        values = np.array([0.0, 1.0, 2.0, 3.0])  # evenly spaced
        dist = phenotype_crowding(values, fronts=np.zeros(4, dtype=int))
        # interior points: (x[i+1]-x[i-1]) / (max-min) = 2/3
        np.testing.assert_allclose(dist[1], 2.0 / 3.0)
        np.testing.assert_allclose(dist[2], 2.0 / 3.0)

    def test_separate_fronts_independent(self):
        values = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        fronts = np.array([0, 0, 0, 1, 1, 1])
        dist = phenotype_crowding(values, fronts=fronts)
        # each front has its own endpoints at inf
        assert np.isinf(dist[0]) and np.isinf(dist[2])
        assert np.isinf(dist[3]) and np.isinf(dist[5])
        assert np.isfinite(dist[1]) and np.isfinite(dist[4])

    def test_two_objectives_accumulate(self, rng):
        o0 = rng.uniform(size=20)
        o1 = rng.uniform(size=20)
        fronts = np.zeros(20, dtype=int)
        d0 = phenotype_crowding(o0, fronts=fronts)
        both = phenotype_crowding(o0, o1, fronts=fronts)
        interior = np.isfinite(d0) & np.isfinite(both)
        assert np.all(both[interior] >= d0[interior] - 1e-12)

    def test_matches_legacy_if_present(self, rng):
        legacy = Path(__file__).resolve().parent.parent / "_legacy" / "crowding.py"
        if not legacy.exists():
            pytest.skip("legacy source removed")
        spec = importlib.util.spec_from_file_location("_legacy_crowding", legacy)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            o0, o1 = rng.uniform(size=50), rng.uniform(size=50)
            fronts = rng.integers(0, 4, size=50)
            ref = mod.phenCrowdingNSGAII(o0, o1, fronts=fronts)
        except Exception as exc:  # legacy code predates NumPy 2.0 (e.g. np.Inf)
            pytest.skip(f"legacy crowding not runnable under this NumPy: {exc}")
        mine = phenotype_crowding(o0, o1, fronts=fronts)
        np.testing.assert_allclose(mine, ref, equal_nan=True)
