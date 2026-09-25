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


class TestMLPOutputRange:
    """``output_scale`` / ``output_offset``: how a bounded-weight MLP reaches its target.

    Weights are capped at +/-30 in model space, so the network's output tops out near
    ``30 * (n_hidden + 1)`` whatever the data looks like. The affine map is a
    reparameterisation of the linear output layer, which is why it is a plain hyperparameter
    and not fitted state.
    """

    def test_defaults_are_the_identity(self, rng):
        m = MLPModel(n_hidden=4)
        assert (m.output_scale, m.output_offset) == (1.0, 0.0)
        X = rng.uniform(-1, 1, size=(20, 3))
        params = rng.normal(size=(5, m.n_parameters(3)))
        plain = MLPModel(n_hidden=4).forward(X, params)
        np.testing.assert_allclose(m.forward(X, params), plain)

    def test_affine_map_is_applied_to_the_output(self, rng):
        X = rng.uniform(-1, 1, size=(20, 2))
        base = MLPModel(n_hidden=4)
        params = rng.normal(size=(5, base.n_parameters(2)))
        scaled = MLPModel(n_hidden=4, output_scale=7.0, output_offset=1000.0)
        np.testing.assert_allclose(
            scaled.forward(X, params), base.forward(X, params) * 7.0 + 1000.0
        )

    def test_reaches_a_target_the_bare_model_cannot(self, rng):
        """The measured failure: without it the swarm never brackets a large-mean target."""
        from foresight_gpu import GPURegressor

        X = rng.uniform(-1, 1, size=(300, 2))
        y = np.sin(2 * np.pi * X[:, 0]) + 0.25 * rng.standard_normal(300) + 1000.0

        with pytest.warns(UserWarning, match="never brackets y"):
            bare = GPURegressor(model=MLPModel(), population=150, n_iter=15,
                                random_state=0).fit(X, y)
        assert not np.isfinite(bare.predict(X)).any()

        scaled = GPURegressor(
            model=MLPModel(output_scale=y.std(), output_offset=y.mean()),
            population=150, n_iter=15, random_state=0,
        ).fit(X, y)
        assert np.isfinite(scaled.predict(X)).all()
        assert abs(np.mean(scaled.predict(X)) - y.mean()) < 5.0

    def test_reaches_sklearn_param_machinery(self):
        from sklearn.base import clone

        assert "output_scale" in MLPModel().get_params()
        assert clone(MLPModel(output_scale=3.0, output_offset=2.0)).output_scale == 3.0
        m = MLPModel()
        m.set_params(output_offset=9.0)
        assert m.output_offset == 9.0


def test_removed_scaling_flags_are_reported(rng):
    """Models written against the pre-0.5.0 contract must not lose scaling silently."""
    from foresight_gpu import GPURegressor

    class Stale(MLPModel):
        scales_inputs = True
        scales_outputs = True

    X = rng.uniform(-1, 1, size=(120, 2))
    y = np.sin(2 * np.pi * X[:, 0])
    with pytest.warns(UserWarning, match="no longer reads"):
        GPURegressor(model=Stale(), population=40, n_iter=4, random_state=0).fit(X, y)


def test_base_model_no_longer_declares_scaling_flags():
    assert not hasattr(BaseForwardModel, "scales_inputs")
    assert not hasattr(BaseForwardModel, "scales_outputs")
