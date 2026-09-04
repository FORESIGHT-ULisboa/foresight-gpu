"""Forward-model tests: shapes, parameter counts/bounds, correctness, sklearn params."""

import numpy as np
import pytest

from foresight_gpu.models import BaseForwardModel, GR4JModel, MLPModel, MLPModelOpenCL


def _mlp_reference(X, params, n_hidden, activation):
    # einsum (not @) so the reference is independent of the BLAS matmul path.
    f, h = X.shape[1], n_hidden
    a = f * h
    w_hidden = params[:a].reshape(f, h)
    b_hidden = params[a : a + h]
    w_output = params[a + h : a + 2 * h]
    b_output = params[-1]
    hidden = np.einsum("nf,fh->nh", X, w_hidden) + b_hidden
    if activation == "tanh":
        hidden = np.tanh(hidden)
    return np.einsum("nh,h->n", hidden, w_output) + b_output


class TestMLPModel:
    def test_parameter_count(self):
        m = MLPModel(n_hidden=5)
        assert m.n_parameters(3) == (3 + 2) * 5 + 1

    def test_forward_shape(self, rng):
        m = MLPModel(n_hidden=4)
        X = rng.normal(size=(30, 3))
        params = rng.normal(size=(10, m.n_parameters(3)))
        assert m.forward(X, params).shape == (30, 10)

    def test_single_particle(self, rng):
        m = MLPModel(n_hidden=4)
        X = rng.normal(size=(20, 2))
        params = rng.normal(size=m.n_parameters(2))
        assert m.forward(X, params).shape == (20, 1)

    @pytest.mark.parametrize("activation", ["tanh", "linear"])
    def test_forward_matches_reference(self, rng, activation):
        m = MLPModel(n_hidden=6, activation=activation)
        X = rng.normal(size=(25, 4))
        p = rng.normal(size=m.n_parameters(4))
        got = m.forward(X, p[None, :])[:, 0]
        expected = _mlp_reference(X, p, 6, activation)
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_regularizable_mask(self):
        m = MLPModel(n_hidden=3)
        mask = m.regularizable_mask(2)
        assert mask.shape == (m.n_parameters(2),)
        assert mask[: 2 * 3].all()          # input->hidden
        assert mask[-4:-1].all()            # hidden->output
        assert not mask[2 * 3 : 2 * 3 + 3].any()  # hidden biases
        assert not mask[-1]                 # output bias

    def test_search_transform_roundtrip(self, rng):
        m = MLPModel()
        p = rng.normal(size=50)
        back = m.inverse_search_transform(m.search_transform(p))
        np.testing.assert_allclose(back, p, rtol=1e-9)

    def test_bounds_shape(self):
        m = MLPModel(n_hidden=4)
        low, high = m.parameter_bounds(3)
        assert low.shape == high.shape == (m.n_parameters(3),)
        assert np.all(low < high)

    def test_sklearn_params(self):
        m = MLPModel(n_hidden=8)
        assert m.get_params()["n_hidden"] == 8
        m.set_params(n_hidden=16)
        assert m.n_hidden == 16


class TestGR4JModel:
    def _forcing(self, rng, n=200):
        precip = np.clip(rng.gamma(0.6, 6.0, size=n), 0, None)
        pet = 2.0 + np.abs(rng.normal(0, 0.5, size=n))
        return np.column_stack([precip, pet])

    def test_four_parameters(self):
        assert GR4JModel().n_parameters(2) == 4
        assert GR4JModel().n_parameters(10) == 4

    def test_forward_shape_and_finiteness(self, rng):
        m = GR4JModel()
        X = self._forcing(rng)
        low, high = m.parameter_bounds(2)
        params = low + rng.uniform(size=(8, 4)) * (high - low)
        out = m.forward(X, params)
        assert out.shape == (X.shape[0], 8)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0)

    def test_single_particle(self, rng):
        m = GR4JModel()
        X = self._forcing(rng)
        params = np.array([350.0, 0.0, 90.0, 1.7])
        assert m.forward(X, params).shape == (X.shape[0], 1)

    def test_parameters_matter(self, rng):
        m = GR4JModel()
        X = self._forcing(rng)
        a = m.forward(X, np.array([200.0, 0.0, 50.0, 1.5]))
        b = m.forward(X, np.array([800.0, 2.0, 200.0, 3.0]))
        assert not np.allclose(a, b)

    def test_no_regularization(self):
        assert not GR4JModel().regularizable_mask(2).any()

    def test_more_rain_more_flow(self, rng):
        # Sanity: scaling precipitation up should not reduce total simulated flow.
        m = GR4JModel()
        X = self._forcing(rng)
        p = np.array([350.0, 0.0, 90.0, 1.7])
        base = m.forward(X, p).sum()
        wetter = X.copy()
        wetter[:, 0] *= 1.5
        assert m.forward(wetter, p).sum() >= base


class TestOpenCLStub:
    def test_raises(self, rng):
        with pytest.raises(NotImplementedError):
            MLPModelOpenCL().forward(rng.normal(size=(5, 2)), rng.normal(size=(3, 9)))


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseForwardModel()
