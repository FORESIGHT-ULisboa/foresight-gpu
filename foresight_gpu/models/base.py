"""The forward-model contract.

A *forward model* is a stateless, parametric function that the optimiser evaluates for the
whole particle swarm at once. It is **not** a scikit-learn estimator — the optimiser owns
the parameters. It subclasses :class:`sklearn.base.BaseEstimator` only for
``get_params`` / ``set_params`` so hyperparameter search can reach ``model__*``.

To bring your own model, subclass :class:`BaseForwardModel` and implement
:meth:`n_parameters` and :meth:`forward`; override :meth:`parameter_bounds`,
:meth:`regularizable_mask` and the search transform as needed.

Shapes (the advertised contract):

* ``X``       : ``[n_samples, n_features]``
* ``params``  : ``[n_particles, n_params]`` (or ``[n_params]`` for a single particle)
* ``forward`` : returns ``[n_samples, n_particles]``  — **vectorised over particles**
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

#: Default per-parameter search bound magnitude (model space) when not overridden.
DEFAULT_BOUND = 30.0


class BaseForwardModel(BaseEstimator, ABC):
    """Abstract deterministic forward model evaluated over a particle swarm."""

    #: Whether the estimator should StandardScaler-normalise the inputs for this model.
    scales_inputs = False
    #: Whether model outputs are in a normalised space and must be inverse-transformed.
    scales_outputs = False

    @abstractmethod
    def n_parameters(self, n_features):
        """Length of one parameter vector for a problem with ``n_features`` inputs."""

    @abstractmethod
    def forward(self, X, params):
        """Evaluate the model.

        Parameters
        ----------
        X : ndarray
            Inputs, shape ``[n_samples, n_features]``.
        params : ndarray
            Parameter matrix ``[n_particles, n_params]`` (or ``[n_params]``).

        Returns
        -------
        ndarray
            Simulations, shape ``[n_samples, n_particles]``.
        """

    def parameter_bounds(self, n_features):
        """Per-parameter ``(low, high)`` search bounds in *model* space.

        Authoritative — overrides any global bound on the estimator. The default is a
        symmetric ``+/- DEFAULT_BOUND`` box; models with physical parameters (e.g. GR4J)
        should override this.
        """
        k = self.n_parameters(n_features)
        return np.full(k, -DEFAULT_BOUND), np.full(k, DEFAULT_BOUND)

    def regularizable_mask(self, n_features):
        """Boolean mask of parameters the L-p penalty applies to (default: all)."""
        return np.ones(self.n_parameters(n_features), dtype=bool)

    def search_transform(self, params):
        """Map parameters from *search* space to *model* space (default: identity)."""
        return params

    def inverse_search_transform(self, params):
        """Map parameters from *model* space to *search* space (default: identity)."""
        return params
