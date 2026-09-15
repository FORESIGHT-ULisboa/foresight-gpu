"""Supplying HYPE's forcing from Python instead of from whatever the folder happens to hold.

HYPE reads its meteorology from text files next to the executable, so by default a
:class:`~.model.HYPEModel` simulates whatever ``Pobs.txt``/``Tobs.txt`` are already in the
template folder. That is fine for a fixed catchment setup, but it makes the *data* invisible
to the caller: you cannot swap a reanalysis for a gauge series, run a different period, or
drive the model from a DataFrame you just built, without editing files by hand.

Passing ``forcing=`` fixes that. Each entry is written into every worker folder in HYPE's
exact format before the run, so the model is driven by data you control:

    HYPEModel(template_dir=..., forcing={"P": precip_df, "T": temp_df})

Values may be a ``DataFrame`` (index = dates, one column per forcing id), a ``Series`` (for a
single-id catchment), or a path to an existing HYPE-format file.

Two things follow automatically, because getting them wrong is silent:

* the matching ``read*obs`` switch is set in ``info.txt`` (supplying ``TMINobs`` is
  pointless while ``readtminobs n``);
* the simulation window can be **derived** from the data, so the period follows the forcing
  rather than a stale ``bdate`` in the template.
"""

import hashlib
import warnings

import numpy as np

from .files import _fmt, _read_lines, _write_lines

#: Canonical HYPE forcing/observation files, the ``info.txt`` switch that enables each, and
#: whether gaps are acceptable. Forcing must be complete - HYPE cannot integrate through a
#: hole - while an observation record normally has gaps and uses ``-9999``.
FORCING_FILES = {
    "Pobs.txt": dict(switch=None, gaps=False, what="precipitation (mm/day)"),
    "Tobs.txt": dict(switch=None, gaps=False, what="air temperature (degC)"),
    "TMINobs.txt": dict(switch="readtminobs", gaps=False, what="daily minimum temperature"),
    "TMAXobs.txt": dict(switch="readtmaxobs", gaps=False, what="daily maximum temperature"),
    "SFobs.txt": dict(switch="readsfobs", gaps=False, what="snowfall fraction"),
    "SWobs.txt": dict(switch="readswobs", gaps=False, what="shortwave radiation"),
    "RHobs.txt": dict(switch="readrhobs", gaps=False, what="relative humidity"),
    "UWobs.txt": dict(switch="readuobs", gaps=False, what="wind speed"),
    "Qobs.txt": dict(switch=None, gaps=True, what="observed discharge (m3/s)"),
    "Xobs.txt": dict(switch=None, gaps=True, what="auxiliary observations"),
}

#: Friendly names accepted in a ``forcing`` dict, mapped to the canonical file.
ALIASES = {
    "p": "Pobs.txt", "precip": "Pobs.txt", "precipitation": "Pobs.txt", "pobs": "Pobs.txt",
    "t": "Tobs.txt", "temp": "Tobs.txt", "temperature": "Tobs.txt", "tobs": "Tobs.txt",
    "tmin": "TMINobs.txt", "tminobs": "TMINobs.txt",
    "tmax": "TMAXobs.txt", "tmaxobs": "TMAXobs.txt",
    "sf": "SFobs.txt", "sfobs": "SFobs.txt", "snowfall": "SFobs.txt",
    "sw": "SWobs.txt", "swobs": "SWobs.txt", "radiation": "SWobs.txt",
    "rh": "RHobs.txt", "rhobs": "RHobs.txt", "humidity": "RHobs.txt",
    "uw": "UWobs.txt", "uwobs": "UWobs.txt", "wind": "UWobs.txt",
    "q": "Qobs.txt", "qobs": "Qobs.txt", "discharge": "Qobs.txt",
    "x": "Xobs.txt", "xobs": "Xobs.txt",
}

#: HYPE's missing-value code, used where a series is allowed to have gaps.
MISSING = -9999.0


def canonical_name(key):
    """Map a user-supplied forcing key to its HYPE file name."""
    text = str(key).strip()
    if text in FORCING_FILES:
        return text
    lowered = text.lower().removesuffix(".txt")
    if lowered in ALIASES:
        return ALIASES[lowered]
    if f"{text}.txt" in FORCING_FILES:
        return f"{text}.txt"
    raise ValueError(
        f"Unknown forcing {key!r}. Use a HYPE file name ({', '.join(sorted(FORCING_FILES))}) "
        f"or an alias ({', '.join(sorted(set(ALIASES)))})."
    )


def read_hype_table(path):
    """Read a HYPE-format forcing/observation file into a ``DataFrame``.

    Tab-separated, one header row naming the forcing ids, ISO dates in the first column,
    ``-9999`` treated as missing.
    """
    import pandas as pd

    lines = [line for line in _read_lines(path) if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty.")
    header = [f.strip() for f in lines[0].split("\t")]
    rows = [line.split("\t") for line in lines[1:]]
    index = pd.to_datetime([r[0].strip() for r in rows])
    data = np.array(
        [[float(f) if f.strip() else np.nan for f in r[1:len(header)]] for r in rows],
        dtype=float,
    )
    data[data == MISSING] = np.nan
    return pd.DataFrame(data, index=index, columns=header[1:])


def as_dated_frame(source, label, missing=None):
    """Coerce a source to a ``DataFrame`` with a unique, sorted, tz-naive ``DatetimeIndex``.

    Shared by the forcing and the observation readers, so both accept the same input shapes
    and reject the same mistakes with the same wording.

    Parameters
    ----------
    source : DataFrame, Series, or str or Path
        Data, or a path to an existing HYPE-format file.
    label : str
        How to name the argument in error messages, e.g. ``"forcing['Pobs.txt']"``.
    missing : float or None
        Value to map to NaN (pass :data:`MISSING` for HYPE's ``-9999``). ``None`` keeps it.

    Returns
    -------
    DataFrame
    """
    import pandas as pd

    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        return read_hype_table(source)
    if isinstance(source, pd.Series):
        frame = source.to_frame()
    elif isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        raise TypeError(
            f"{label} must be a DataFrame, a Series or a path to a HYPE-format "
            f"file, got {type(source).__name__}."
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.dtype.kind in "iuf":
            # pd.to_datetime would read a positional index as nanoseconds since the epoch
            # and hand back 1970-01-01 timestamps, so the window would look absurd rather
            # than wrong. Refuse instead.
            raise TypeError(
                f"{label} has a numeric index ({frame.index.dtype}); it needs "
                "actual dates. A positional index would be read as nanoseconds since 1970. "
                "Set a DatetimeIndex, e.g. df.set_index('DATE') or "
                "df.index = pd.to_datetime(df['date'])."
            )
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise TypeError(
                f"{label} needs a date index; could not read one ({exc})."
            ) from exc
    if frame.index.tz is not None:
        # Day ordinals floor in UTC, so a +09:00 index would move every date back a day
        # (2020-01-01 Asia/Tokyo becomes 2019-12-31) with nothing to show it happened.
        raise TypeError(
            f"{label} has a tz-aware index ({frame.index.tz}). Day ordinals are computed "
            "in UTC, which would shift every date. Use df.index.tz_localize(None) once you "
            "are sure the timestamps mean local days."
        )
    duplicated = frame.index.duplicated(keep="last")
    if duplicated.any():
        warnings.warn(
            f"{label} has {int(duplicated.sum())} duplicated date(s); keeping the last "
            "value for each. Check the source.",
            stacklevel=3,
        )
    frame = frame[~duplicated].sort_index()
    frame.columns = [str(c) for c in frame.columns]
    if missing is not None:
        values = frame.to_numpy(dtype=float)
        if (values == missing).any():
            frame = pd.DataFrame(
                np.where(values == missing, np.nan, values),
                index=frame.index, columns=frame.columns,
            )
    return frame


def as_frame(source, name):
    """Coerce a forcing entry to a ``DataFrame`` indexed by date.

    ``missing`` is deliberately left unmapped here: turning ``-9999`` into NaN for *forcing*
    would flip :func:`validate` from silently driving HYPE with -9999 mm of rain to raising,
    which is better but a separate behaviour change.
    """
    return as_dated_frame(source, f"forcing[{name!r}]")


def resolve(spec):
    """Normalise a ``forcing`` argument into ``{canonical file name: DataFrame}``."""
    if not spec:
        return {}
    if not hasattr(spec, "items"):
        raise TypeError(
            "forcing must be a mapping like {'P': precip_df, 'T': temp_df}; got "
            f"{type(spec).__name__}."
        )
    out = {}
    for key, source in spec.items():
        name = canonical_name(key)
        if name in out:
            raise ValueError(f"forcing has two entries resolving to {name!r}.")
        out[name] = as_frame(source, name)
    return out


def coverage(frames):
    """Common date range of the provided series, as ``datetime64[D]``.

    Returns
    -------
    (start, end) or (None, None)
        The **intersection**, since HYPE needs every enabled series over the whole window.
    """
    if not frames:
        return None, None
    starts = [f.index.min() for f in frames.values() if len(f)]
    ends = [f.index.max() for f in frames.values() if len(f)]
    if not starts:
        return None, None
    return (np.datetime64(max(starts), "D"), np.datetime64(min(ends), "D"))


def validate(frames, bdate, edate, expected_ids=None):
    """Check the provided forcing can actually drive the requested window.

    Raises
    ------
    ValueError
        If a series does not span ``[bdate, edate]``, if a gap-free series has holes there,
        or if a required forcing id is absent.
    """
    import pandas as pd

    if not frames:
        return
    window = pd.date_range(str(np.datetime64(bdate, "D")), str(np.datetime64(edate, "D")),
                           freq="D")
    for name, frame in frames.items():
        rules = FORCING_FILES[name]
        missing_dates = window.difference(frame.index)
        if len(missing_dates):
            raise ValueError(
                f"forcing {name!r} ({rules['what']}) does not cover the simulation window: "
                f"{len(missing_dates)} day(s) absent, first {missing_dates[0].date()}, "
                f"last {missing_dates[-1].date()}. It spans "
                f"{frame.index.min().date()}..{frame.index.max().date()} but the window is "
                f"{window[0].date()}..{window[-1].date()}."
            )
        inside = frame.loc[window]
        if not rules["gaps"] and not np.isfinite(inside.to_numpy(dtype=float)).all():
            holes = int((~np.isfinite(inside.to_numpy(dtype=float))).sum())
            raise ValueError(
                f"forcing {name!r} ({rules['what']}) has {holes} missing value(s) inside the "
                "simulation window. HYPE cannot integrate through a gap in the meteorology; "
                "fill or interpolate them before calibrating."
            )
        if expected_ids:
            absent = [i for i in expected_ids if i not in frame.columns]
            if absent:
                raise ValueError(
                    f"forcing {name!r} is missing column(s) {absent} for the subbasin(s) this "
                    f"setup needs; it has {list(frame.columns)}. Columns are forcing ids "
                    "(SUBIDs when the folder has no ForcKey.txt)."
                )
            extra = [c for c in frame.columns if c not in expected_ids]
            if extra:
                warnings.warn(
                    f"forcing {name!r} has extra column(s) {extra} that this setup does not "
                    "use; HYPE selects by header so they are harmless, but check the ids.",
                    stacklevel=3,
                )


def switches(frames):
    """``info.txt`` ``read*obs`` settings implied by the supplied series."""
    out = {}
    for name in frames:
        switch = FORCING_FILES[name]["switch"]
        if switch:
            out[switch] = "y"
    return out


def write(frame, path, allow_gaps):
    """Write one series in HYPE's format: tab-separated, CRLF, ISO dates."""
    values = frame.to_numpy(dtype=float)
    lines = ["DATE\t" + "\t".join(str(c) for c in frame.columns)]
    stamps = frame.index.to_numpy().astype("datetime64[D]").astype(str)
    for stamp, row in zip(stamps, values):
        cells = [
            _fmt(MISSING) if (allow_gaps and not np.isfinite(v)) else _fmt(v) for v in row
        ]
        lines.append(stamp + "\t" + "\t".join(cells))
    _write_lines(path, lines)
    return path


def stage(frames, directory, window=None):
    """Write every supplied series into ``directory``, ready to overlay a worker folder.

    Parameters
    ----------
    frames : dict of str -> DataFrame
    directory : Path
    window : (start, end) or None
        Trim to this range before writing. HYPE only needs the simulated period, and a
        trimmed file is faster for it to parse.

    Returns
    -------
    list of str
        File names written.
    """
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in frames.items():
        if window is not None:
            index = pd.date_range(str(np.datetime64(window[0], "D")),
                                  str(np.datetime64(window[1], "D")), freq="D")
            frame = frame.reindex(index)
        write(frame, directory / name, FORCING_FILES[name]["gaps"])
        written.append(name)
    return written


def digest(frames):
    """Stable hash of the supplied forcing, for the series cache key.

    Changing the meteorology changes the simulation for the *same* parameters, so the cache
    must not carry entries across a forcing change.
    """
    if not frames:
        return "no-forcing"
    hasher = hashlib.sha1()
    for name in sorted(frames):
        frame = frames[name]
        hasher.update(name.encode())
        hasher.update(",".join(map(str, frame.columns)).encode())
        hasher.update(frame.index.to_numpy().astype("datetime64[D]").astype(str).tobytes())
        hasher.update(np.ascontiguousarray(frame.to_numpy(dtype=float)).tobytes())
    return hasher.hexdigest()[:16]


def expected_ids(template_dir, subbasin=None):
    """Forcing ids this setup needs, inferred from the template.

    Prefers an existing ``Pobs.txt`` header (authoritative for what HYPE will look up),
    falls back to the ``SUBID`` column of ``GeoData.txt``, then to ``subbasin``.
    """
    from pathlib import Path

    template_dir = Path(template_dir)
    pobs = template_dir / "Pobs.txt"
    if pobs.exists():
        lines = [line for line in _read_lines(pobs) if line.strip()]
        if lines:
            return [f.strip() for f in lines[0].split("\t")[1:]]

    geodata = template_dir / "GeoData.txt"
    if geodata.exists():
        lines = [line for line in _read_lines(geodata) if line.strip()]
        if lines:
            header = [f.strip().upper() for f in lines[0].split("\t")]
            if "SUBID" in header:
                column = header.index("SUBID")
                return [
                    line.split("\t")[column].strip()
                    for line in lines[1:] if len(line.split("\t")) > column
                ]
    return [str(subbasin)] if subbasin is not None else []
