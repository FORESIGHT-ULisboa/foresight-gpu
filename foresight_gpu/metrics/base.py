"""The :class:`Metric` wrapper used across ``foresight_gpu``.

A :class:`Metric` mirrors the convention of the companion ``forecast_performance``
package: it subclasses :class:`str`, so it **is its own name** and can be passed either
as a handle (``nse``) or a string (``"nse"``) interchangeably.

Two things are added on top of that convention for the GPU engine:

* ``greater_is_better`` — orientation of the raw metric value.
* :meth:`Metric.loss` — the **minimisation** form the optimiser consumes. Metrics where
  larger is better (NSE, KGE, KGE') are turned into ``1 - value``; error metrics
  (MAE, MSE, RMSE) are returned as-is. The engine only ever calls ``.loss``.
"""


class Metric(str):
    """A callable metric that equals its own name.

    Parameters
    ----------
    name : str
        Canonical metric name (what the object stringifies to).
    func : callable
        Underlying vectorised implementation ``func(sim, obs) -> ndarray``.
    kind : str, optional
        ``"deterministic"`` or ``"probabilistic"``.
    greater_is_better : bool, optional
        ``True`` when a larger raw value is better (efficiency-type metrics).
    aliases : iterable of str, optional
        Alternative names recognised when resolving a metric from a string.
    """

    def __new__(cls, name, func, kind="deterministic", greater_is_better=False, aliases=()):
        obj = super().__new__(cls, name)
        obj._func = func
        obj.__name__ = name
        obj.__doc__ = getattr(func, "__doc__", None)
        obj.kind = kind
        obj.greater_is_better = bool(greater_is_better)
        obj.aliases = tuple(aliases)
        return obj

    def __call__(self, *args, **kwargs):
        """Evaluate the raw metric (native orientation)."""
        return self._func(*args, **kwargs)

    def loss(self, sim, obs):
        """Minimisation form consumed by the optimiser (lower is better).

        Returns ``1 - value`` for ``greater_is_better`` metrics, else ``value``.
        """
        value = self._func(sim, obs)
        return 1.0 - value if self.greater_is_better else value

    def __repr__(self):
        return self.__name__

    # ``str`` is immutable; reconstruct from the name on (deep)copy/pickle by falling
    # back to a plain string, which compares equal to this Metric.
    def __reduce__(self):
        return (str, (str(self),))
