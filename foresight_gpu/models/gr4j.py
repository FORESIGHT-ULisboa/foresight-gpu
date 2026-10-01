"""GR4J — the default lumped conceptual rainfall-runoff model.

GR4J (Perrin et al., 2003, J. Hydrol.) is a parsimonious 4-parameter daily model:

* ``X1`` production store capacity [mm]
* ``X2`` groundwater exchange coefficient [mm]
* ``X3`` routing store capacity [mm]
* ``X4`` unit-hydrograph time base [days]

It contrasts with :class:`~foresight_gpu.models.mlp.MLPModel`: the parameter count is
**fixed at 4** regardless of ``n_features``, the parameters are physical (so bounds are
natural ranges and no search warp is applied), and inputs are read from named columns —
precipitation ``P`` and potential evapotranspiration ``PET`` (pair with
:class:`~foresight_gpu.utils.features.OudinPET` if you only have temperature).

The time loop is sequential but vectorised over the particle axis.
"""

import numpy as np

from .base import BaseForwardModel

#: Natural calibration ranges (low, high) for X1..X4.
_BOUNDS_LOW = np.array([1.0, -10.0, 1.0, 0.5])
_BOUNDS_HIGH = np.array([1500.0, 10.0, 500.0, 10.0])


class GR4JModel(BaseForwardModel):
    """GR4J daily rainfall-runoff model.

    Inputs
    ------
    X : array-like, shape (n_samples, n_features)
        Columns of precipitation ``P`` and potential evapotranspiration ``PET``.

    Parameters
    ----------
    p_index : int, optional
        Column of ``X`` holding precipitation ``P`` [mm/day].
    pet_index : int, optional
        Column of ``X`` holding potential evapotranspiration ``PET`` [mm/day].
    s_init, r_init : float, optional
        Initial fill fractions of the production and routing stores.
    """

    def __init__(self, p_index=0, pet_index=1, s_init=0.3, r_init=0.5):
        self.p_index = p_index
        self.pet_index = pet_index
        self.s_init = s_init
        self.r_init = r_init

    def n_parameters(self, n_features):
        return 4

    def parameter_bounds(self, n_features):
        return _BOUNDS_LOW.copy(), _BOUNDS_HIGH.copy()

    @staticmethod
    def _uh_ordinates(x4):
        """Unit-hydrograph ordinates UH1 ``[P, l1]`` and UH2 ``[P, l2]``."""
        x4 = np.asarray(x4, dtype=float)
        l1 = max(int(np.ceil(x4.max())), 1)
        l2 = max(int(np.ceil(2.0 * x4.max())), 2)

        t1 = np.arange(0, l1 + 1)[None, :]
        r1 = t1 / x4[:, None]
        sh1 = np.where(t1 < x4[:, None], np.power(np.clip(r1, 0, None), 2.5), 1.0)
        uh1 = sh1[:, 1:] - sh1[:, :-1]

        t2 = np.arange(0, l2 + 1)[None, :]
        r2 = t2 / x4[:, None]
        sh2 = np.where(
            t2 < x4[:, None],
            0.5 * np.power(np.clip(r2, 0, None), 2.5),
            np.where(
                t2 < 2.0 * x4[:, None],
                1.0 - 0.5 * np.power(np.clip(2.0 - r2, 0, None), 2.5),
                1.0,
            ),
        )
        uh2 = sh2[:, 1:] - sh2[:, :-1]
        return uh1, uh2

    def forward(self, X, params):
        X = np.asarray(X, dtype=float)
        params = np.atleast_2d(np.asarray(params, dtype=float))
        # Guard against invalid parameters reaching the physics.
        x1 = np.clip(params[:, 0], 1e-6, None)
        x2 = params[:, 1]
        x3 = np.clip(params[:, 2], 1e-6, None)
        x4 = np.clip(params[:, 3], 0.5, None)
        n_particles = params.shape[0]

        precip = X[:, self.p_index]
        pet = X[:, self.pet_index]
        n_steps = X.shape[0]

        uh1, uh2 = self._uh_ordinates(x4)
        st_uh1 = np.zeros_like(uh1)
        st_uh2 = np.zeros_like(uh2)

        store_s = self.s_init * x1
        store_r = self.r_init * x3
        out = np.empty((n_steps, n_particles), dtype=float)

        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(n_steps):
                p_t, e_t = precip[t], pet[t]
                if p_t >= e_t:
                    pn, en = p_t - e_t, 0.0
                else:
                    pn, en = 0.0, e_t - p_t

                sr = store_s / x1
                if pn > 0:
                    tanh_pn = np.tanh(pn / x1)
                    ps = x1 * (1.0 - sr**2) * tanh_pn / (1.0 + sr * tanh_pn)
                    store_s = store_s + ps
                else:
                    ps = np.zeros(n_particles)
                    if en > 0:
                        tanh_en = np.tanh(en / x1)
                        es = store_s * (2.0 - sr) * tanh_en / (1.0 + (1.0 - sr) * tanh_en)
                        store_s = store_s - es

                perc = store_s * (1.0 - (1.0 + (4.0 / 9.0 * store_s / x1) ** 4) ** -0.25)
                store_s = store_s - perc
                routing = perc + (pn - ps)

                prhu1 = 0.9 * routing
                prhu2 = 0.1 * routing
                st_uh1[:, :-1] = st_uh1[:, 1:] + uh1[:, :-1] * prhu1[:, None]
                st_uh1[:, -1] = uh1[:, -1] * prhu1
                q9 = st_uh1[:, 0]
                st_uh2[:, :-1] = st_uh2[:, 1:] + uh2[:, :-1] * prhu2[:, None]
                st_uh2[:, -1] = uh2[:, -1] * prhu2
                q1 = st_uh2[:, 0]

                exchange = x2 * (store_r / x3) ** 3.5
                store_r = np.maximum(0.0, store_r + q9 + exchange)
                qr = store_r * (1.0 - (1.0 + (store_r / x3) ** 4) ** -0.25)
                store_r = store_r - qr
                qd = np.maximum(0.0, q1 + exchange)
                out[t] = qr + qd

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
