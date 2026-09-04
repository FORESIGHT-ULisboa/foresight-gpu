"""A tiny single-hidden-layer perceptron forward model (NumPy).

Re-implements the original OpenCL MLP on NumPy, vectorised over the particle axis with
``einsum`` (no Python loop over particles). The optimiser supplies the weights; this class
only evaluates. The weight layout matches the original implementation:

    [ input->hidden (n_features * n_hidden) | hidden bias (n_hidden)
      | hidden->output (n_hidden) | output bias (1) ]

so ``n_parameters = (n_features + 2) * n_hidden + 1``.
"""

import numpy as np

from .base import BaseForwardModel

_ACTIVATIONS = {"tanh", "tan", "logistic", "log", "sigmoid", "linear", "lin",
                "identity", "relu", "leaky_relu"}


class MLPModel(BaseForwardModel):
    """Single-hidden-layer MLP with a linear output.

    Parameters
    ----------
    n_hidden : int, optional
        Number of hidden nodes.
    activation : str, optional
        Hidden activation: ``"tanh"`` (default), ``"logistic"``, ``"linear"`` or
        ``"leaky_relu"``.
    leaky_slope : float, optional
        Negative-side slope for ``leaky_relu``.
    """

    # The MLP searches weights in a normalised space, so scale inputs and outputs.
    scales_inputs = True
    scales_outputs = True

    def __init__(self, n_hidden=8, activation="tanh", leaky_slope=0.01):
        self.n_hidden = n_hidden
        self.activation = activation
        self.leaky_slope = leaky_slope

    def n_parameters(self, n_features):
        return (n_features + 2) * self.n_hidden + 1

    def parameter_bounds(self, n_features):
        k = self.n_parameters(n_features)
        return np.full(k, -30.0), np.full(k, 30.0)

    def regularizable_mask(self, n_features):
        # Regularise connection weights (input->hidden and hidden->output), not biases.
        h = self.n_hidden
        mask = np.zeros(self.n_parameters(n_features), dtype=bool)
        mask[: n_features * h] = True   # input -> hidden
        mask[-h - 1 : -1] = True        # hidden -> output
        return mask

    def search_transform(self, params):
        # Warp that spreads small weights (helps the swarm explore); matches the original.
        return np.power(params / 4.0, 5)

    def inverse_search_transform(self, params):
        params = np.asarray(params, dtype=float)
        return 4.0 * np.copysign(np.power(np.abs(params), 0.2), params)

    def _activate(self, a):
        act = self.activation
        if act in ("tanh", "tan"):
            return np.tanh(a)
        if act in ("logistic", "log", "sigmoid"):
            return 1.0 / (1.0 + np.exp(-a))
        if act in ("linear", "lin", "identity"):
            return a
        if act in ("relu", "leaky_relu"):
            return np.where(a > 0, a, self.leaky_slope * a)
        raise ValueError(f"Unknown activation {act!r}; choose from {sorted(_ACTIVATIONS)}")

    def _unpack(self, n_features, params):
        params = np.atleast_2d(np.asarray(params, dtype=float))
        n_particles = params.shape[0]
        h = self.n_hidden
        a = n_features * h
        w_hidden = params[:, :a].reshape(n_particles, n_features, h)
        b_hidden = params[:, a : a + h]                # [P, H]
        w_output = params[:, a + h : a + 2 * h]        # [P, H]
        b_output = params[:, -1]                       # [P]
        return w_hidden, b_hidden, w_output, b_output

    def forward(self, X, params):
        X = np.asarray(X, dtype=float)
        w_hidden, b_hidden, w_output, b_output = self._unpack(X.shape[1], params)
        # hidden pre-activation: [P, n_samples, H]
        hidden = np.einsum("nf,pfh->pnh", X, w_hidden) + b_hidden[:, None, :]
        activated = self._activate(hidden)
        # linear output: [P, n_samples]
        out = np.einsum("pnh,ph->pn", activated, w_output) + b_output[:, None]
        return out.T  # [n_samples, n_particles]
