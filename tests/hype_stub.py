"""A stand-in for the HYPE executable, so the whole pipeline is testable without HYPE.

``HYPEModel`` takes its command as a parameter, so a test can point it at
``[sys.executable, hype_stub.__file__]``. The stub then behaves like HYPE: it reads
``info.txt`` and ``par.txt`` from its working directory and writes a result file into
``resultdir``.

The simulated series is a deterministic, cheap function of the parameters, so tests can
assert both that parameters *matter* and that identical parameters give identical output
(the premise the series cache rests on).

Failure modes are selected with ``FGPU_HYPE_STUB_MODE``:

``ok``              normal run (default)
``crash``           raise, leaving no output file
``nonzero``         exit non-zero after writing nothing
``truncated``       write only half the requested window
``missing_file``    exit 0 but write no output
``slow``            sleep, to exercise the timeout
``units_row``       basinoutput convention: header then a ``UNITS`` row (also the default)
``comment_header``  timeoutput convention: a leading ``!! model=...`` comment
``minus9999``       fill part of the series with the missing-value code
``statefile``       write a state file, to prove the runner clears it between runs
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

MODE_ENV = "FGPU_HYPE_STUB_MODE"
STATE_NAME = "stub_state.txt"


def _read_kv(path):
    """Minimal info.txt reader: whitespace-tokenised, ``!!`` comments skipped."""
    settings, options = {}, {}
    for raw in Path(path).read_text(encoding="latin-1").splitlines():
        if not raw.strip() or raw.lstrip().startswith("!!"):
            continue
        tokens = raw.replace("\t", " ").split()
        if tokens[0] == "modeloption" and len(tokens) >= 3:
            options[tokens[1]] = tokens[2]
        elif tokens[0] in ("basinoutput", "timeoutput", "mapoutput", "crit"):
            settings.setdefault(tokens[0] + " " + tokens[1], " ".join(tokens[2:]))
        elif len(tokens) >= 2:
            settings[tokens[0]] = " ".join(tokens[1:])
    return settings, options


def _read_par(path):
    values = {}
    for raw in Path(path).read_text(encoding="latin-1").splitlines():
        if not raw.strip() or raw.lstrip().startswith("!!"):
            continue
        fields = raw.split("\t")
        try:
            values[fields[0].strip()] = [float(f) for f in fields[1:] if f.strip()]
        except ValueError:
            continue
    return values


#: Parameters the stub responds to. A real HYPE run is sensitive to whichever parameters the
#: routines in force actually use; the stub just needs *enough* leverage that a swarm can
#: spread right across the non-exceedance axis, or no quantile band would ever be populated.
SENSITIVE = ("wcfc", "rrcs1", "rrcs2", "cmlt", "preccorr", "ttmp", "snalbmin", "cmrad")

#: Level the observations sit around, so a template-default parameter set is roughly unbiased.
BASE_LEVEL = 5.0

#: Nominal rainfall (mm/day) the routed signal is scaled against. Fixed on purpose - see
#: :func:`simulate` - so the output responds to the *magnitude* of the forcing.
NOMINAL_RAINFALL = 2.5

#: Untouched copy of the template ``par.txt``, replicated into every worker folder alongside
#: it. Each parameter's effect is measured as a *ratio* to its default, so a parameter the
#: test does not calibrate contributes exactly 1 instead of skewing the level.
REFERENCE_NAME = "par_reference.txt"


def read_forcing(folder, column=None):
    """Read ``Pobs.txt`` the way HYPE would: tab-separated, DATE plus one column per id.

    The stub responds to precipitation so that supplying ``forcing=`` is observable in the
    output - without this, a test could not tell a real forcing swap from a no-op.
    """
    path = Path(folder) / "Pobs.txt"
    if not path.exists():
        return None, None
    lines = [l for l in path.read_text(encoding="latin-1").splitlines() if l.strip()]
    if len(lines) < 2:
        return None, None
    header = [f.strip() for f in lines[0].split("\t")]
    index = 1
    if column is not None and str(column) in header:
        index = header.index(str(column))
    stamps, values = [], []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) <= index:
            continue
        stamps.append(fields[0].strip())
        try:
            values.append(float(fields[index]))
        except ValueError:
            values.append(np.nan)
    return (np.array(stamps, dtype="datetime64[D]"),
            np.asarray(values, dtype=float))


def simulate(dates, par, options, reference=None, precip=None):
    """A deterministic pseudo-hydrograph that depends on the parameters *and* the forcing.

    Not physics - a recession-smoothed rainfall signal whose *level* moves multiplicatively
    with the calibrated parameters, as a ratio to their template defaults. Two properties
    are load-bearing for the tests:

    * it responds to ``Pobs.txt``, so a ``forcing=`` swap is observable;
    * the level response is geometric and saturating, so the swarm can reach levels both
      well below and well above the observations. Without that the population never spreads
      across the exceedance axis, no quantile band is populated, and the GPU aggregation
      degenerates to NaN.
    """
    year_start = dates.astype("datetime64[Y]").astype("datetime64[D]")
    day_of_year = (dates - year_start).astype(int)
    season = 1.0 + 0.8 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)

    # Rainfall routed through a linear store, scaled by a *fixed* nominal rainfall rather
    # than by the series' own mean - dividing by its own mean would make the stub invariant
    # to the magnitude of the forcing, so tripling the rain would change nothing and a test
    # could not tell a real forcing swap from a no-op.
    shape = season
    if precip is not None and precip.size == dates.size:
        routed = np.empty(precip.size)
        store = 0.0
        for i, value in enumerate(np.nan_to_num(precip)):
            store = 0.88 * store + 0.12 * value
            routed[i] = store
        shape = routed / NOMINAL_RAINFALL

    reference = reference or {}
    decades = 0.0
    for name in SENSITIVE:
        vector = par.get(name)
        base = reference.get(name)
        if not vector or not base:
            continue
        current, default = float(np.mean(vector)), float(np.mean(base))
        if abs(default) < 1e-12:
            continue
        decades += np.log10(max(abs(current / default), 1e-9))

    # Saturating response, like a real catchment: the level moves over a bounded range
    # rather than as an unbounded product, and template defaults give gain 1. Without the
    # saturation, three independent multi-decade parameters push almost the whole search
    # box to a near-zero level and the swarm can never spread across the exceedance axis.
    gain = float(10.0 ** (1.2 * np.tanh(decades / 5.0)))

    # Model options shift the level, so a routine toggle is observable in the output.
    offset = sum(float(v) for v in options.values() if str(v).lstrip("-").isdigit())

    ripple = 0.4 * np.sin(np.arange(dates.size) * 0.7)
    return np.maximum(BASE_LEVEL * gain * shape + ripple + 0.05 * offset, 0.0)


def main():
    mode = os.environ.get(MODE_ENV, "ok")
    folder = Path.cwd()

    if mode == "crash":
        raise RuntimeError("stub: simulated HYPE crash")
    if mode == "nonzero":
        sys.stderr.write("stub: simulated HYPE error\n")
        return 3
    if mode == "slow":
        time.sleep(30)

    settings, options = _read_kv(folder / "info.txt")
    par = _read_par(folder / "par.txt")

    if mode == "statefile":
        state = folder / STATE_NAME
        previous = state.read_text() if state.exists() else ""
        state.write_text(previous + "run\n")
        # Contaminate the output with the run count, so a leak is detectable.
        options = dict(options, _leak=str(previous.count("run")))

    reference_path = folder / REFERENCE_NAME
    reference = _read_par(reference_path) if reference_path.exists() else None

    cdate = np.datetime64(settings["cdate"], "D")
    edate = np.datetime64(settings["edate"], "D")
    dates = np.arange(cdate, edate + np.timedelta64(1, "D"), dtype="datetime64[D]")

    # Precipitation over the output window, as HYPE would read it.
    stamps, values = read_forcing(folder, settings.get("basinoutput subbasins"))
    precip = None
    if stamps is not None:
        lookup = {str(s): v for s, v in zip(stamps, values)}
        if all(str(d) in lookup for d in dates):
            precip = np.array([lookup[str(d)] for d in dates], dtype=float)

    series = simulate(dates, par, options, reference, precip)

    if mode == "missing_file":
        return 0
    if mode == "truncated":
        dates, series = dates[: dates.size // 2], series[: series.size // 2]
    if mode == "minus9999":
        series = series.copy()
        series[: max(1, series.size // 10)] = -9999.0

    resultdir = folder / settings.get("resultdir", "./").strip()
    resultdir.mkdir(parents=True, exist_ok=True)

    subbasin = settings.get("basinoutput subbasins")
    variable = (settings.get("basinoutput variable")
                or settings.get("timeoutput variable") or "cout").split()[0]

    lines = []
    if subbasin is not None and mode != "comment_header":
        out = resultdir / f"{int(subbasin):07d}.txt"
        lines.append(f"DATE\t{variable}")
        lines.append("UNITS\tm3/s")
    else:
        out = resultdir / f"time{variable.upper()}.txt"
        lines.append(
            f"!! model=stub; variable={variable}; timestep=day; unit=m3/s;"
        )
        lines.append(f"DATE\t{subbasin or 1}")

    for date, value in zip(dates, series):
        lines.append(f"{date}\t{value:.5E}")

    with open(out, "w", encoding="latin-1", newline="") as handle:
        handle.write("\r\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
