"""L-p regularisation of model parameters (WRR draft, Eq. 2).

Adds a penalty on the (subset of) parameters flagged by a model's
``regularizable_mask`` to the training loss, discouraging large weights. Vectorised over
the particle axis so it can be added to the per-particle loss each generation.
"""

import numpy as np


def lp_penalty(params, mask, lam, p=1):
    """Compute the L-p regularisation term per particle.

    Parameters
    ----------
    params : ndarray
        Parameter matrix, shape ``[n_particles, n_params]`` (or ``[n_params]``).
    mask : ndarray of bool or None
        Which parameters to penalise. ``None`` penalises all.
    lam : float
        Regularisation coefficient ``lambda``. ``0`` / ``None`` disables the term.
    p : int or float, optional
        Norm order. ``1`` = LASSO (sum of absolute values), ``2`` = Tikhonov (sum of
        squares), otherwise the generic ``(sum |w|^p)^(1/p)``.

    Returns
    -------
    float or ndarray
        Penalty per particle (shape ``[n_particles]``), or ``0.0`` when disabled.
    """
    if lam is None or lam <= 0:
        return 0.0

    params = np.atleast_2d(np.asarray(params, dtype=float))
    if mask is None:
        w = params
    else:
        w = params[:, np.asarray(mask, dtype=bool)]

    if p == 1:
        reg = np.sum(np.abs(w), axis=1)
    elif p == 2:
        reg = np.sum(np.square(w), axis=1)
    else:
        reg = np.power(np.sum(np.power(np.abs(w), p), axis=1), 1.0 / p)

    return lam * reg
