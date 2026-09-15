"""Readers and writers for HYPE's text files.

Everything HYPE reads is tab-separated with **CRLF** line endings, and comment lines start
with ``!!``. Both are preserved on write: the template's comments document the user's setup,
and a commented-out parameter line (``!!<TAB>srrate<TAB>...``) is how HYPE users disable a
parameter, so it must not be resurrected by a round trip.

Two conventions the parsers absorb:

* ``info.txt`` writes directives either as ``modeloption <name><TAB><value>`` or as
  ``modeloption<TAB><name><TAB><value>`` - both appear in the same real file, so keys are
  tokenised on any whitespace.
* Output files come in two shapes: ``timeoutput`` files open with a ``!! model=...`` comment
  then the header, while ``basinoutput`` files put the header first and a ``UNITS`` row
  second. :func:`read_series` handles both without a flag.

Parameter arity is taken **per parameter** from ``par.txt``, never assumed from its class:
real files contain siblings of differing length (``macfrac`` has 6 values where the other
soil parameters have 5).
"""

import numpy as np

#: HYPE's missing-value code in result files.
MISSING = -9999.0

_COMMENT = "!!"
_NEWLINE = "\r\n"

#: ``info.txt`` keywords that take a sub-key before their value.
_SUBKEYED = ("modeloption", "basinoutput", "timeoutput", "mapoutput", "crit")


def _fmt(value):
    """Format a float for HYPE's free-format reader, losslessly.

    Shortest representation that round-trips exactly, in positional notation for ordinary
    magnitudes (Fortran reads ``1e-06``, but ``0.000001`` is unambiguous, so prefer it).

    ``precision`` is deliberately **not** passed: with ``unique=True`` numpy then emits the
    shortest string that reads back bit-identically, whereas a ``precision`` counts digits
    *after the decimal point* and so silently truncates the significant digits of a small
    value (``0.00082350123456789`` would lose its tail). Calibrated parameters routinely sit
    at 1e-4 and below, so that truncation would mean the value HYPE reads is not the value
    the optimiser scored.
    """
    value = float(value)
    if value == 0.0:
        return "0"
    if 1e-6 <= abs(value) < 1e15:
        return np.format_float_positional(value, unique=True, trim="-")
    return np.format_float_scientific(value, unique=True, trim="-")


def _read_lines(path):
    """Read a HYPE text file as a list of lines with terminators stripped."""
    with open(path, "r", encoding="latin-1", newline="") as handle:
        return [line.rstrip("\r") for line in handle.read().split("\n")]


def _write_lines(path, lines):
    """Write lines back with CRLF terminators."""
    with open(path, "w", encoding="latin-1", newline="") as handle:
        handle.write(_NEWLINE.join(line.rstrip("\r") for line in lines))


def _is_comment(line):
    return line.lstrip().startswith(_COMMENT)


def _tokens(line):
    """Split a directive line on any whitespace run (tabs and spaces both occur)."""
    return line.replace("\t", " ").split()


class ParFile:
    """A parsed ``par.txt``: named parameter vectors, with the original layout retained.

    Attributes
    ----------
    lines : list of str
        Every line of the template, verbatim.
    values : dict of str -> ndarray
        Active (non-commented) parameter vectors, in file order.
    """

    def __init__(self, lines, values, line_index):
        self.lines = lines
        self.values = values
        self._line_index = line_index  # parameter name -> index into ``lines``

    @classmethod
    def read(cls, path):
        """Parse ``par.txt``. Commented-out parameters are ignored, not revived."""
        lines = _read_lines(path)
        values, line_index = {}, {}
        for i, line in enumerate(lines):
            if not line.strip() or _is_comment(line):
                continue
            fields = line.split("\t")
            name = fields[0].strip()
            numbers = [f for f in fields[1:] if f.strip()]
            if not name or not numbers:
                continue
            try:
                values[name] = np.array([float(f) for f in numbers], dtype=float)
            except ValueError:  # a non-numeric row is not a parameter
                continue
            line_index[name] = i
        return cls(lines, values, line_index)

    def arity(self, name):
        """Number of values ``name`` carries in this file."""
        return int(self.values[name].size)

    def with_values(self, updates):
        """Return a copy whose named parameters carry new values.

        Parameters
        ----------
        updates : dict of str -> array-like
            New vectors. Each must match the arity found in the template.
        """
        lines = list(self.lines)
        values = dict(self.values)
        for name, new in updates.items():
            if name not in self._line_index:
                raise KeyError(f"{name!r} is not an active parameter in par.txt.")
            new = np.asarray(new, dtype=float).ravel()
            expected = self.arity(name)
            if new.size != expected:
                raise ValueError(
                    f"{name!r} expects {expected} value(s) in this par.txt, got {new.size}."
                )
            lines[self._line_index[name]] = name + "\t" + "\t".join(_fmt(v) for v in new)
            values[name] = new
        return ParFile(lines, values, dict(self._line_index))

    def write(self, path):
        _write_lines(path, self.lines)


class InfoFile:
    """A parsed ``info.txt``: settings, model options and output directives.

    Attributes
    ----------
    settings : dict of str -> str
        Single-keyword settings (``bdate``, ``resultdir``, ``calibration``, ...).
    options : dict of str -> str
        ``modeloption`` values, keyed by routine name.
    """

    def __init__(self, lines, settings, options):
        self.lines = lines
        self.settings = settings
        self.options = options

    @classmethod
    def read(cls, path):
        lines = _read_lines(path)
        settings, options = {}, {}
        for line in lines:
            if not line.strip() or _is_comment(line):
                continue
            tokens = _tokens(line)
            if not tokens:
                continue
            key = tokens[0]
            if key == "modeloption" and len(tokens) >= 3:
                options[tokens[1]] = tokens[2]
            elif key in _SUBKEYED:
                continue  # output/crit directives are rewritten wholesale, not merged
            elif len(tokens) >= 2:
                settings[key] = " ".join(tokens[1:])
        return cls(lines, settings, options)

    @staticmethod
    def _set_setting(lines, key, value):
        replacement = f"{key}\t{value}"
        for i, line in enumerate(lines):
            if _is_comment(line) or not line.strip():
                continue
            tokens = _tokens(line)
            if tokens and tokens[0] == key and tokens[0] not in _SUBKEYED:
                lines[i] = replacement
                return
        lines.append(replacement)

    @staticmethod
    def _set_option(lines, name, value):
        replacement = f"modeloption\t{name}\t{value}"
        for i, line in enumerate(lines):
            if _is_comment(line) or not line.strip():
                continue
            tokens = _tokens(line)
            if len(tokens) >= 2 and tokens[0] == "modeloption" and tokens[1] == name:
                lines[i] = replacement
                return
        lines.append(replacement)

    @staticmethod
    def _drop_directives(lines, keywords):
        """Comment out existing directives so ours are the only ones in force."""
        out = []
        for line in lines:
            tokens = [] if _is_comment(line) else _tokens(line)
            if tokens and tokens[0] in keywords:
                out.append(_COMMENT + "\t" + line)
            else:
                out.append(line)
        return out

    def configured(self, bdate, cdate, edate, options=None, subbasin=None,
                   output_variable="cout", resultdir="./results/", settings=None):
        """Return a copy set up for one deterministic GPU run.

        Pins the window, the model options and exactly one output directive, so the output
        file name and column layout are known rather than inferred.

        Parameters
        ----------
        settings : dict or None
            Extra plain settings to force, e.g. the ``read*obs`` switches implied by
            caller-supplied forcing. Applied after the fixed ones, so they win.
        """
        lines = self._drop_directives(
            list(self.lines), ("basinoutput", "timeoutput", "mapoutput")
        )
        for key, value in (
            ("bdate", bdate), ("cdate", cdate), ("edate", edate),
            ("resultdir", resultdir), ("calibration", "n"),
            ("readdaily", "y"), ("instate", "n"), ("parensemble", "n"),
        ):
            self._set_setting(lines, key, value)

        for key, value in (settings or {}).items():
            self._set_setting(lines, key, value)

        for name, value in (options or {}).items():
            self._set_option(lines, name, value)

        if subbasin is None:
            lines.append(f"timeoutput variable\t{output_variable}")
            lines.append("timeoutput meanperiod\t1")
        else:
            lines.append(f"basinoutput variable\t{output_variable}")
            lines.append("basinoutput meanperiod\t1")
            lines.append(f"basinoutput subbasins\t{subbasin}")

        merged = dict(self.options)
        merged.update({k: str(v) for k, v in (options or {}).items()})
        merged_settings = dict(self.settings)
        merged_settings.update({
            "bdate": str(bdate), "cdate": str(cdate), "edate": str(edate),
            "resultdir": resultdir, "calibration": "n",
        })
        merged_settings.update({k: str(v) for k, v in (settings or {}).items()})
        return InfoFile(lines, merged_settings, merged)

    def for_calibration(self, criteria, resultdir="./results/"):
        """Return a copy with HYPE's own DE-MC calibration switched on.

        Parameters
        ----------
        criteria : sequence of (str, str, str, float)
            ``(criterion, cvariable, rvariable, weight)`` per objective.
        """
        lines = self._drop_directives(list(self.lines), ("crit",))
        self._set_setting(lines, "calibration", "y")
        self._set_setting(lines, "resultdir", resultdir)
        lines.append("crit meanperiod\t1")
        lines.append("crit datalimit\t3")
        for n, (criterion, cvar, rvar, weight) in enumerate(criteria, start=1):
            lines.append(f"crit {n} criterion\t{criterion}")
            lines.append(f"crit {n} cvariable\t{cvar}")
            lines.append(f"crit {n} rvariable\t{rvar}")
            lines.append(f"crit {n} weight\t{_fmt(weight)}")
        return InfoFile(lines, dict(self.settings), dict(self.options))

    def write(self, path):
        _write_lines(path, self.lines)


def read_series(path, column=None, t0=None, n_steps=None):
    """Read one column of a HYPE result file.

    Handles both output conventions: a leading ``!! model=...`` comment then the header
    (``timeoutput``), or the header then a ``UNITS`` row (``basinoutput``). ``-9999``
    becomes NaN.

    Parameters
    ----------
    path : str or Path
    column : str or None
        Header name to select (a variable such as ``"cout"``, or a subbasin id). ``None``
        takes the first data column.
    t0, n_steps : int or None
        Expected window. When given, the file must match exactly - a truncated or shifted
        file raises rather than yielding partial NaNs, which would otherwise look like a
        plausible simulation to the objective function.

    Returns
    -------
    ndarray
        Float64 values, shape ``[n_rows]``.
    """
    from .dates import as_ordinals

    lines = [line for line in _read_lines(path) if line.strip()]
    cursor = 0
    while cursor < len(lines) and _is_comment(lines[cursor]):
        cursor += 1
    if cursor >= len(lines):
        raise ValueError(f"{path}: no header row found.")

    header = [f.strip() for f in lines[cursor].split("\t")]
    cursor += 1
    if cursor < len(lines) and lines[cursor].split("\t")[0].strip().upper() == "UNITS":
        cursor += 1

    if column is None:
        col = 1
    else:
        wanted = str(column).strip()
        if wanted not in header:
            raise ValueError(
                f"{path}: column {wanted!r} not in header {header}. For a basinoutput file "
                "pass the variable name (e.g. 'cout'); for a timeoutput file the subbasin id."
            )
        col = header.index(wanted)
    if col == 0:
        raise ValueError(f"{path}: column {column!r} is the date column.")

    stamps, values = [], []
    for line in lines[cursor:]:
        fields = line.split("\t")
        if len(fields) <= col:
            raise ValueError(f"{path}: row {fields[0]!r} has fewer than {col + 1} columns.")
        stamps.append(fields[0].strip())
        try:
            values.append(float(fields[col]))
        except ValueError:
            values.append(np.nan)

    series = np.asarray(values, dtype=float)
    series[series == MISSING] = np.nan

    if t0 is not None and n_steps is not None:
        if series.size != n_steps:
            raise ValueError(
                f"{path}: expected {n_steps} rows for the requested window, found "
                f"{series.size}. The run was truncated or the window is inconsistent."
            )
        got = as_ordinals(np.asarray(stamps, dtype="datetime64[D]"))
        if int(got[0]) != int(t0):
            raise ValueError(f"{path}: starts at ordinal {int(got[0])}, expected {int(t0)}.")
    return series


def write_optpar(path, entries, task=("DE", "WS"), ngen=100, npop=50,
                 gammascale=0.5, sigma=0.0, crossover=0.4, cal_log="Y",
                 block_line=22):
    """Write ``optpar.txt`` for HYPE's own DE-MC calibration.

    The parameter block is positional: HYPE expects it to start at a fixed line (22 in the
    reference files), and each parameter occupies **three** consecutive rows - lower bounds,
    upper bounds, step sizes - each with one value per class.

    Parameters
    ----------
    entries : sequence of (str, array-like, array-like, array-like)
        ``(name, low, high, step)``, each array one value per class.
    block_line : int
        1-based line at which the parameter block must start; the header is padded with
        blank lines to reach it.

    Returns
    -------
    int
        The line at which the block starts, so a caller can assert on a read-back.
    """
    header = ["Info optimization"]
    header += [f"task\t{value}" for value in task]
    header += [
        f"cal_log\t{cal_log}",
        f"DEMC_ngen\t{ngen}",
        f"DEMC_npop\t{npop}",
        f"DEMC_gammascale\t{_fmt(gammascale)}",
        f"DEMC_sigma\t{_fmt(sigma)}",
        f"DEMC_crossover\t{_fmt(crossover)}",
    ]
    if len(header) >= block_line:
        raise ValueError(
            f"optpar header needs {len(header)} lines but the parameter block must start at "
            f"line {block_line}; reduce the task rows or raise block_line."
        )
    lines = header + [""] * (block_line - 1 - len(header))

    for name, low, high, step in entries:
        rows = [np.asarray(a, dtype=float).ravel() for a in (low, high, step)]
        if len({r.size for r in rows}) != 1:
            raise ValueError(
                f"{name!r}: low/high/step have lengths "
                f"{'/'.join(str(r.size) for r in rows)}; all three rows must match the "
                "arity of the parameter in par.txt."
            )
        for row in rows:
            lines.append(name + "\t" + "\t".join(_fmt(v) for v in row))

    _write_lines(path, lines)
    return block_line


def read_respar(path):
    """Read ``respar.txt`` - the optimum found by HYPE's own calibration."""
    out = {}
    for line in _read_lines(path):
        if not line.strip() or _is_comment(line):
            continue
        fields = line.split()
        try:
            out[fields[0]] = np.array([float(f) for f in fields[1:]], dtype=float)
        except (ValueError, IndexError):
            continue
    return out


def read_bestsims(path):
    """Read ``bestsims.txt`` (comma-separated) into ``(header, rows)``."""
    lines = [line for line in _read_lines(path) if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty.")
    header = [f.strip() for f in lines[0].split(",")]
    rows = np.array(
        [[float(f) if f.strip() else np.nan for f in line.split(",")] for line in lines[1:]],
        dtype=float,
    )
    rows[rows == MISSING] = np.nan
    return header, rows


def arity_groups(par):
    """Group ``par.txt`` parameter names by how many values they carry.

    Used to sanity-check a declared dimension against the file. Arity is per parameter and
    authoritative - real files mix lengths within one class (``macfrac`` has 6 values where
    its soil siblings have 5).
    """
    groups = {}
    for name, values in par.values.items():
        groups.setdefault(int(values.size), []).append(name)
    return groups
