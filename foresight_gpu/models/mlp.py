"""A tiny single-hidden-layer perceptron forward model (NumPy).

Re-implements the original OpenCL MLP on NumPy, vectorised over the particle axis with
``einsum`` (no Python loop over particles). The optimiser supplies the weights; this class
only evaluates. The weight layout matches the original implementation:

    [ input->hidden (n_features * n_hidden) | hidden bias (n_hidden)
      | hidden->output (n_hidden) | output bias (1) ]

so ``n_parameters = (n_features + 2) * n_hidden + 1``.
"""

import numpy as np

from ..metrics.regularization import lp_penalty
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
    output_scale, output_offset : float, optional
        Affine map applied to the network output: ``out * output_scale + output_offset``.

        **This is how the MLP reaches a target it could not otherwise address.** Weights are
        bounded at +/-30 in model space, so with ``n_hidden`` nodes and a bounded activation
        the output tops out near ``30 * (n_hidden + 1)`` — about 270 by default — *whatever
        the data looks like*. On a target near 1000 every particle then sits on the same side
        of every observation, the exceedance axis degenerates, no band is populated and
        ``predict`` returns all-NaN, silently. Measured: NSE 0.28 with the scaling, -12403 at
        a target mean of 100, no prediction at all at 1000.

        Because the output layer is linear, this is exactly a reparameterisation of it
        (``sigma * (w.h + b) + mu``), so it is a plain hyperparameter — no fitted state, and
        reachable as ``model__output_scale`` in a search. The usual setting is::

            MLPModel(output_scale=y.std(), output_offset=y.mean())

        Inputs are **not** scaled here; use a ``Pipeline`` for those.
    reg_lambda, reg_p : float, int, optional
        L-p regularisation coefficient and norm order (WRR Eq. 2) on the connection
        weights (biases are not penalised). ``reg_lambda=0`` disables it.
    """

    def __init__(self, n_hidden=8, activation="tanh", leaky_slope=0.01,
                 output_scale=1.0, output_offset=0.0, reg_lambda=0.0, reg_p=1):
        self.n_hidden = n_hidden
        self.activation = activation
        self.leaky_slope = leaky_slope
        self.output_scale = output_scale
        self.output_offset = output_offset
        self.reg_lambda = reg_lambda
        self.reg_p = reg_p

    def n_parameters(self, n_features):
        return (n_features + 2) * self.n_hidden + 1

    def parameter_bounds(self, n_features):
        k = self.n_parameters(n_features)
        return np.full(k, -30.0), np.full(k, 30.0)

    def regularization(self, params, n_features):
        if not self.reg_lambda:
            return 0.0
        return lp_penalty(params, self._weight_mask(n_features), self.reg_lambda, self.reg_p)

    def _weight_mask(self, n_features):
        # Connection weights (input->hidden and hidden->output), not biases.
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
        return out.T * self.output_scale + self.output_offset  # [n_samples, n_particles]
