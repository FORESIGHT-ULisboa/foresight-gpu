"""Date <-> ``X`` encoding for the HYPE forward model.

HYPE reads its forcing from ``Pobs.txt``/``Tobs.txt``, so ``X`` carries no features. Instead
``X[:, date_column]`` holds the **day ordinal** (days since 1970-01-01) of the sample, and
``forward`` returns the simulated value for exactly those dates, in that order.

Day resolution is deliberate: ``check_X_y(dtype=float)`` converts ``X`` to float64, which
represents integers exactly only up to 2**53. Day ordinals are ~2e4 (exact); nanosecond
ordinals are ~1.8e18 and would quantise to 256 ns steps. :func:`resolve_indices` rejects
anything that is not a whole day so the mistake cannot pass silently.
"""

import numpy as np

#: Reference epoch for day ordinals.
EPOCH = np.datetime64("1970-01-01", "D")

#: Ordinals beyond this are almost certainly a finer datetime64 unit mistaken for days.
_MAX_PLAUSIBLE_ORDINAL = 200_000  # year ~2517


def as_ordinals(dates):
    """Convert dates to float day-ordinals suitable for a column of ``X``.

    Parameters
    ----------
    dates : array-like
        Anything ``numpy`` can read as ``datetime64``: a ``DatetimeIndex``, an array of
        ``datetime64``, or ISO date strings.

    Returns
    -------
    ndarray
        Float64 day ordinals, shape ``[n_samples]``.
    """
    values = np.asarray(getattr(dates, "values", dates))
    if values.dtype.kind in "iuf":
        raise TypeError(
            "as_ordinals expects dates, not numbers. Pass a DatetimeIndex, datetime64 array "
            "or ISO strings; numeric input is already assumed to be ordinals."
        )
    return values.astype("datetime64[D]").astype(np.int64).astype(float)


def from_ordinals(ordinals):
    """Convert day ordinals back to ``datetime64[D]``."""
    values = np.asarray(ordinals, dtype=float)
    return np.rint(values).astype(np.int64).astype("datetime64[D]")


def as_X(dates):
    """Build a single-column ``X`` from dates, shape ``[n_samples, 1]``."""
    return as_ordinals(dates).reshape(-1, 1)


def window_ordinals(bdate, edate):
    """Return ``(t0, n_steps)`` for the inclusive daily window ``[bdate, edate]``."""
    t0 = int(as_ordinals([bdate])[0])
    t1 = int(as_ordinals([edate])[0])
    if t1 < t0:
        raise ValueError(f"edate ({edate}) precedes bdate ({bdate}).")
    return t0, t1 - t0 + 1


def check_ordinals(values, tol=1e-6):
    """Validate a column of day ordinals, returning it as a flat float array.

    Catches the two mistakes that would otherwise pass silently: a date column that has
    been transformed (a scaled value is not a whole number) and a finer datetime64 unit
    mistaken for days (nanosecond ordinals are ~1.8e18, past float64's exact-integer range).

    Parameters
    ----------
    values : array-like
        Day ordinals.
    tol : float
        Tolerance on integrality, to absorb float round-tripping.

    Returns
    -------
    ndarray
        The ordinals, flattened, dtype float64.

    Raises
    ------
    ValueError
        If any value is non-finite, not a whole day, or implausibly large.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return values

    if not np.all(np.isfinite(values)):
        raise ValueError("Date column contains non-finite values.")

    fractional = np.abs(values - np.rint(values))
    if fractional.max() > tol:
        bad = values[np.argmax(fractional)]
        raise ValueError(
            f"X[:, date_column] = {bad!r} is not a whole-day ordinal. HYPEModel expects the "
            "date as days since 1970-01-01 (see hype.as_X). Did a StandardScaler run in front "
            "of the estimator, or were nanosecond ordinals passed instead of days?"
        )

    if np.abs(values).max() > _MAX_PLAUSIBLE_ORDINAL:
        raise ValueError(
            f"Date ordinal {values[np.argmax(np.abs(values))]:.6g} is implausibly large for "
            "days since 1970-01-01. Convert with hype.as_X(dates), which uses "
            "datetime64[D]; finer units (e.g. nanoseconds) exceed float64's exact-integer "
            "range and cannot be recovered."
        )
    return values


def resolve_indices(values, t0, n_steps, tol=1e-6):
    """Map a column of day ordinals to row indices of a simulated window.

    Integer arithmetic rather than a search, so duplicates, gaps and arbitrary order all
    work and the cost is O(n).

    Parameters
    ----------
    values : ndarray
        Day ordinals, shape ``[n_samples]``.
    t0 : int
        Day ordinal of the first simulated step.
    n_steps : int
        Number of simulated steps.
    tol : float
        Tolerance on integrality, to absorb float round-tripping.

    Returns
    -------
    ndarray
        Row indices into the simulated series, shape ``[n_samples]``, dtype int64.

    Raises
    ------
    ValueError
        If any value is not a whole-day ordinal, or falls outside the window.
    """
    values = check_ordinals(values, tol)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)

    idx = np.rint(values).astype(np.int64) - int(t0)
    outside = (idx < 0) | (idx >= int(n_steps))
    if outside.any():
        offending = from_ordinals(values[outside])
        start, end = from_ordinals([t0, t0 + n_steps - 1])
        raise ValueError(
            f"{int(outside.sum())} date(s) fall outside the simulated window "
            f"{start}..{end}, e.g. {offending[:5]}. Widen bdate/edate on the HYPEModel."
        )
    return idx
