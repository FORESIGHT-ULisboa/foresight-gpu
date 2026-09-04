"""MOPSO mechanics + a toy convergence test driven through ``evolve``."""

import numpy as np
import pytest

from foresight_gpu.domination import DoubleParetoSorter
from foresight_gpu.metrics.deterministic import mae
from foresight_gpu.metrics.exceedance import non_exceedance
from foresight_gpu.optimizers import MOPSO, evolve
from foresight_gpu.utils.screening import screen_initial_population


@pytest.fixture
def toy_problem(rng):
    """A one-parameter constant-prediction problem: sim[:, k] = params[k, 0]."""
    obs = rng.normal(5.0, 2.0, size=150)
    low, high = np.array([-5.0]), np.array([15.0])

    def evaluate(params):
        params = np.atleast_2d(params)
        sim = np.repeat(params[:, 0][None, :], obs.size, axis=0)
        fit = np.column_stack(
            [non_exceedance(sim, obs), np.log10(mae.loss(sim, obs) + 1e-9)]
        )
        return sim, fit

    return obs, low, high, evaluate


class TestMechanics:
    def test_initialise_within_bounds(self, rng):
        opt = MOPSO()
        low, high = np.array([-2.0, 0.0]), np.array([2.0, 5.0])
        pop = opt.initialise(low, high, population=50, rng=rng)
        assert pop.shape == (50, 2)
        assert np.all(pop >= low) and np.all(pop <= high)

    def test_generate_shapes(self, toy_problem, rng):
        obs, low, high, evaluate = toy_problem
        opt = MOPSO(partial=0.5)
        pop = opt.initialise(low, high, population=40, rng=rng)
        _, fit = evaluate(pop)
        cand = opt.generate(fit, pop)
        assert cand.shape == pop.shape
        assert np.all(cand >= low) and np.all(cand <= high)
        assert opt._joint_velocities.shape == (80, 1)

    def test_select_keeps_population(self, toy_problem, rng):
        obs, low, high, evaluate = toy_problem
        opt = MOPSO()
        pop = opt.initialise(low, high, population=40, rng=rng)
        sims, fit = evaluate(pop)
        pop2, fit2, sims2, rejected = evolve(
            opt, DoubleParetoSorter(), pop, fit, sims, evaluate
        )
        assert pop2.shape == (40, 1)
        assert fit2.shape == (40, 2)
        assert sims2.shape[1] == 40
        assert opt._velocities.shape == (40, 1)

    def test_enforce_bounds(self, rng):
        opt = MOPSO()
        opt.initialise(np.array([0.0]), np.array([1.0]), population=5, rng=rng)
        bounded, changed = opt.enforce_bounds(np.array([[-1.0], [0.5], [2.0]]))
        np.testing.assert_array_equal(bounded.ravel(), [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(changed.ravel(), [True, False, True])

    def test_sklearn_params(self):
        opt = MOPSO(inertia=0.3)
        assert opt.get_params()["inertia"] == 0.3
        opt.set_params(c1=0.2)
        assert opt.c1 == 0.2


class TestConvergence:
    def test_spans_exceedance_and_reduces_loss(self, toy_problem, rng):
        obs, low, high, evaluate = toy_problem
        opt = MOPSO(partial=1.0)
        pop = opt.initialise(low, high, population=200, rng=rng)
        sims, fit = evaluate(pop)
        sorter = DoubleParetoSorter()
        for _ in range(80):
            pop, fit, sims, _ = evolve(opt, sorter, pop, fit, sims, evaluate)

        # The swarm should span the full exceedance axis...
        assert fit[:, 0].min() < 0.15
        assert fit[:, 0].max() > 0.85
        # ...and reach a low error near the median (mae well below the spread of obs).
        assert (10 ** fit[:, 1].min()) < 1.5


class TestScreening:
    def test_screen_initial_population_spread(self, toy_problem, rng):
        obs, low, high, evaluate = toy_problem
        pop = screen_initial_population(evaluate, low, high, 60, rng, oversample=5)
        assert pop.shape == (60, 1)
        _, fit = evaluate(pop)
        # screened swarm should already cover a broad exceedance range
        assert fit[:, 0].max() - fit[:, 0].min() > 0.7
