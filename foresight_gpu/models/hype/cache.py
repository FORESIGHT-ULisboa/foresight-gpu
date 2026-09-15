"""A bounded cache of simulated series, keyed by parameter vector.

HYPE simulates a continuous window whatever subset of dates is asked for, so the model runs
the whole window once per parameter vector and slices rows from the result. Caching those
series is what makes an early-stopping check free: the check re-evaluates the *whole*
population, and most of that population are survivors carried forward by the optimiser as
array slices, never re-evaluated between checks.

That access pattern is why a plain LRU sized at a small multiple of the population fails.
With ``population`` insertions per generation, a survivor is evicted after roughly a
generation and a half, so every check is a full miss and the cache buys nothing. Capacity must
therefore exceed ``(check_every + 1) * population``; the default is generous because a series
is small (20 years daily as float32 is about 29 KB).

Keys are ``(fingerprint, params.tobytes())``. The bytes give exact bit equality, so an error
can only ever cost a needless re-run, never return one particle's series for another - which
rounding the key would risk, since MOPSO perturbs particles by ~1e-3 of range. The fingerprint
covers everything that changes what a parameter vector *means* (slot layout, model options,
window, output selection, template contents), so a stale entry cannot survive a
reconfiguration.

Series are held as float32. The model quantises a fresh run to the same precision before
using it, so a cache hit and a cache miss return bit-identical values and ``cache_size=0``
changes only the run count, never the result. HYPE writes five significant figures anyway.
"""

from collections import OrderedDict

import numpy as np


class SeriesCache:
    """Bounded store of simulated series.

    Parameters
    ----------
    capacity : int
        Maximum number of series held. ``0`` disables caching entirely (every lookup
        misses), which is the honest way to run when output may not be a pure function of
        the parameters, or when memory is tight.
    fingerprint : str
        Configuration hash mixed into every key.

    Attributes
    ----------
    hits, misses : int
        Lookup counters, for the run accounting the model exposes.
    """

    def __init__(self, capacity, fingerprint=""):
        self.capacity = max(int(capacity), 0)
        self.fingerprint = fingerprint
        self.hits = 0
        self.misses = 0
        self._lru = OrderedDict()
        self._pinned = {}

    @property
    def enabled(self):
        return self.capacity > 0

    def __len__(self):
        return len(self._lru) + len(self._pinned)

    def key(self, params):
        """Exact key for one parameter row."""
        row = np.ascontiguousarray(params, dtype=np.float64) + 0.0  # normalise -0.0
        return self.fingerprint, row.tobytes()

    def get(self, key):
        """Return a cached series, or ``None``."""
        if not self.enabled:
            self.misses += 1
            return None
        if key in self._pinned:
            self.hits += 1
            return self._pinned[key]
        series = self._lru.get(key)
        if series is None:
            self.misses += 1
            return None
        self._lru.move_to_end(key)
        self.hits += 1
        return series

    def put(self, key, series):
        """Store a series, evicting the least recently used entry if full."""
        if not self.enabled:
            return
        self._lru[key] = np.asarray(series, dtype=np.float32)
        self._lru.move_to_end(key)
        while len(self._lru) + len(self._pinned) > self.capacity:
            if not self._lru:
                break
            self._lru.popitem(last=False)

    def pin(self, keys):
        """Protect a set of keys from eviction, replacing any previous pinned set.

        Called when a batch arrives that is mostly cache hits - the signature of an
        early-stopping or prediction pass over the surviving population, whose series are the
        ones worth keeping while candidate churn continues.
        """
        if not self.enabled:
            return
        keys = [k for k in keys if k in self._lru or k in self._pinned]
        if len(keys) >= self.capacity:
            return  # refuse to pin the whole cache; leave LRU in charge
        retained = {}
        for k in keys:
            series = self._pinned.get(k)
            if series is None:
                series = self._lru.pop(k, None)
            if series is not None:
                retained[k] = series
        for k, series in self._pinned.items():
            if k not in retained:
                self._lru[k] = series
        self._pinned = retained
        while len(self._lru) + len(self._pinned) > self.capacity and self._lru:
            self._lru.popitem(last=False)

    def clear(self):
        self._lru.clear()
        self._pinned.clear()
