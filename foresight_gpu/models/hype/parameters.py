"""The HYPE parameter catalogue, routine gating and the search-space layout.

Three ideas do the work here.

**Mode.** A HYPE parameter is a vector with one value per landuse class, soil type or
parameter region. ``mode="substitute"`` gives the optimiser one search variable per value;
``mode="multiply"`` gives it a single multiplier that scales the whole template vector,
preserving the relative pattern the template encodes. Multiply therefore costs one dimension
regardless of how many classes exist, which is why the legacy setups used it for the
soil parameters.

**Routine gating.** Many parameters only exist if a ``modeloption`` is switched on -
``snalbmin`` is meaningless unless ``snowmeltmodel = 2``. Each spec carries a ``requires``
condition evaluated against the effective options, so the active parameter set follows the
model structure instead of being maintained by hand (the legacy code kept duplicate HYPE
folders per routine combination).

**Bounds live in physical space.** :meth:`Layout.bounds` returns real multipliers and real
values. The unit-box mapping the optimiser searches in is built from those by
:meth:`Layout.to_search` / :meth:`Layout.to_model`, which the model exposes through the
``search_transform`` hooks. This keeps MOPSO's absolute exploration noise (``c3``) meaningful
in every dimension - a raw physical box makes it 0.17 % of range in one dimension and 0.008 %
in another.
"""

import hashlib
import warnings
from dataclasses import dataclass, field, replace

import numpy as np

#: Smallest positive value used when guarding a log transform.
_TINY = np.finfo(float).tiny

#: Fallback bounds for a parameter absent from :data:`CATALOGUE`.
DEFAULT_SPEC = dict(low=0.001, high=200.0, log=True, mode="multiply")

_MODES = ("multiply", "substitute")


@dataclass(frozen=True)
class HypeParameter:
    """Search specification for one HYPE parameter.

    Parameters
    ----------
    name : str
        Parameter name as it appears in ``par.txt``.
    low, high : float
        Bounds in **physical** space. For ``mode="multiply"`` these bound the resulting
        parameter *values*, and the multiplier range is derived from the template vector so
        every class stays inside them.
    mode : {"multiply", "substitute"}
    log : bool
        Search this parameter logarithmically. Requires ``low > 0``.
    clip : tuple of (float, float) or None
        Hard physical clamp applied after mapping. Use it where a multiplier could push a
        bounded quantity (a fraction, a fill capacity) out of range and abort the run.
    requires : dict of str -> tuple or None
        Routine condition, e.g. ``{"snowmeltmodel": (2,)}``. The parameter is active only if
        every named option holds one of the listed values.
    dimension : str
        Documentation only (``"general"``, ``"landuse"``, ``"soil"``, ``"region"``). Arity is
        always read from ``par.txt``, which is authoritative - real files mix lengths within
        one class.
    """

    name: str
    low: float
    high: float
    mode: str = "multiply"
    log: bool = False
    clip: tuple = None
    requires: dict = None
    dimension: str = "unknown"

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(f"{self.name!r}: mode must be one of {_MODES}, got {self.mode!r}")
        if not self.high > self.low:
            raise ValueError(
                f"{self.name!r}: needs high > low, got low={self.low}, high={self.high}."
            )
        if self.log and self.low <= 0:
            raise ValueError(
                f"{self.name!r}: log search needs low > 0, got {self.low}. A non-positive "
                "lower bound makes log10(low) = -inf and every particle in that dimension "
                "becomes NaN."
            )


def _p(name, low, high, mode="multiply", log=False, **kw):
    return HypeParameter(name=name, low=low, high=high, mode=mode, log=log, **kw)


#: Curated search specifications, ported from the legacy calibration sets.
#:
#: ``requires`` is populated only where the routine dependency is documented or evidenced in
#: the legacy code; anything unverified is left ungated (always active) rather than guessed.
#: Extend or override per model via ``HYPEModel(parameter_specs=...)``.
CATALOGUE = {
    spec.name: spec
    for spec in (
        # -- soil / runoff response ------------------------------------------------------
        _p("wcfc", 0.0001, 1.0, "multiply", log=True, dimension="soil", clip=(1e-4, 0.95)),
        _p("wcwp", 0.0001, 1.0, "multiply", log=True, dimension="soil", clip=(1e-4, 0.95)),
        _p("wcep", 0.0001, 1.0, "multiply", log=True, dimension="soil", clip=(1e-4, 0.95)),
        _p("rrcs1", 0.0001, 5.0, "multiply", log=True, dimension="soil", clip=(1e-6, 1.0)),
        _p("rrcs2", 0.0001, 1.0, "multiply", log=True, dimension="soil", clip=(1e-6, 1.0)),
        _p("rrcs3", 0.0001, 1.0, "substitute", log=True, dimension="general"),
        _p("mperc1", 0.0001, 300.0, "multiply", log=True, dimension="soil"),
        _p("mperc2", 0.0001, 300.0, "multiply", log=True, dimension="soil"),
        _p("trrcs", 0.0001, 1.0, "multiply", log=True, dimension="soil", clip=(1e-6, 1.0)),
        _p("epotdist", 5.0, 20.0, "substitute", dimension="general"),
        _p("lp", 0.0, 25.0, "substitute", dimension="general"),
        # -- forcing corrections ---------------------------------------------------------
        _p("preccorr", -1.0, 15.0, "substitute", dimension="region"),
        _p("tempcorr", -25.0, 25.0, "substitute", dimension="region"),
        _p("tcalt", 0.0, 2.0, "substitute", dimension="general"),
        _p("pcelevadd", 0.0, 20.0, "substitute", dimension="general"),
        # -- evapotranspiration ----------------------------------------------------------
        _p("kc", 0.001, 25.0, "multiply", dimension="landuse"),
        # -- lakes and rivers ------------------------------------------------------------
        _p("gratp", 0.0001, 10.0, "substitute", log=True, dimension="general"),
        _p("gratk", 0.001, 10.0, "substitute", log=True, dimension="general"),
        _p("gldepi", 0.01, 100.0, "substitute", log=True, dimension="general"),
        _p("rivvel", 0.01, 100.0, "substitute", log=True, dimension="general"),
        _p("damp", 0.001, 1.0, "substitute", log=True, dimension="general"),
        # -- snow: temperature index (always available) ----------------------------------
        _p("ttmp", -3.0, 5.0, "substitute", dimension="landuse"),
        _p("cmlt", 0.001, 100.0, "multiply", log=True, dimension="landuse"),
        # -- snow: radiation index, only under snowmeltmodel 2 ---------------------------
        _p("snalbmin", 0.001, 1.0, "multiply", log=True, dimension="landuse",
           requires={"snowmeltmodel": (2,)}, clip=(0.01, 0.99)),
        _p("snalbmax", 0.001, 1.0, "multiply", log=True, dimension="landuse",
           requires={"snowmeltmodel": (2,)}, clip=(0.01, 0.99)),
        _p("snalbkexp", 0.001, 1.0, "multiply", log=True, dimension="landuse",
           requires={"snowmeltmodel": (2,)}),
        _p("cmrad", 0.001, 10.0, "multiply", log=True, dimension="landuse",
           requires={"snowmeltmodel": (2,)}),
        # -- snow cover fraction ---------------------------------------------------------
        _p("fscdist0", 0.001, 1.0, "multiply", log=True, dimension="landuse"),
        _p("fscdist1", 0.001, 10.0, "multiply", log=True, dimension="landuse"),
        _p("fscdistmax", 0.001, 1.0, "multiply", log=True, dimension="landuse"),
        _p("fsck1", 0.001, 2.0, "substitute", log=True, dimension="general"),
        _p("fsckexp", 1e-6, 1e-4, "substitute", log=True, dimension="general"),
        # -- surface runoff, only when the routine is on ---------------------------------
        _p("srrcs", 0.001, 1.0, "multiply", log=True, dimension="landuse",
           requires={"surfacerunoff": (1, 2, 3, 4)}, clip=(1e-6, 1.0)),
        _p("srrate", 0.0001, 1.0, "substitute", log=True, dimension="soil",
           requires={"surfacerunoff": (1, 2, 3, 4)}, clip=(1e-6, 1.0)),
        _p("srbeta", 0.0001, 20.0, "substitute", log=True, dimension="general",
           requires={"surfacerunoff": (1, 2, 3, 4)}),
        # -- macropore / infiltration routine --------------------------------------------
        _p("macrate", 0.001, 1.0, "multiply", log=True, dimension="soil",
           requires={"infiltration": (1, 2, 3)}, clip=(1e-6, 1.0)),
        _p("mactrinf", 0.001, 5.0, "multiply", log=True, dimension="soil",
           requires={"infiltration": (1, 2, 3)}),
        _p("mactrsm", 0.001, 1.0, "multiply", log=True, dimension="soil",
           requires={"infiltration": (1, 2, 3)}, clip=(1e-6, 1.0)),
        _p("macfrac", 0.001, 1.0, "multiply", log=True, dimension="soil",
           requires={"infiltration": (1, 2, 3)}, clip=(1e-6, 1.0)),
        # -- deep groundwater / aquifer routine ------------------------------------------
        _p("rcgrw", 0.0001, 1.0, "substitute", log=True, dimension="general",
           requires={"deepground": (1, 2)}, clip=(1e-6, 1.0)),
        _p("rcgrwst", 0.0001, 1.0, "multiply", log=True, dimension="soil",
           requires={"deepground": (1, 2)}, clip=(1e-6, 1.0)),
        # -- glacier ----------------------------------------------------------------------
        _p("glacttmp", -10.0, 5.0, "substitute", dimension="general"),
        _p("glaccmlt", 0.001, 100.0, "substitute", log=True, dimension="general"),
        _p("glaccmrad", 0.001, 101.0, "substitute", log=True, dimension="general"),
        _p("glaccmrefr", 0.001, 1.0, "substitute", log=True, dimension="general"),
        _p("glacalb", 0.001, 1.0, "substitute", log=True, dimension="general"),
    )
}


def effective_options(template_options, overrides=None):
    """Merge template ``modeloption`` values with user overrides, as ints where possible."""
    merged = dict(template_options or {})
    merged.update(overrides or {})
    out = {}
    for key, value in merged.items():
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def is_enabled(spec, options):
    """Whether ``spec`` is active given the effective ``options``."""
    for option, allowed in (spec.requires or {}).items():
        if options.get(option) not in tuple(allowed):
            return False
    return True


def multiplier_bounds(spec, base):
    """Multiplier range keeping every value of ``base`` inside the physical bounds.

    Mirrors the legacy derivation: the lower multiplier is set by the smallest template
    value and the upper by the largest, so no class can leave ``[low, high]``.
    """
    base = np.asarray(base, dtype=float)
    positive = base[base > 0]
    if positive.size == 0:
        return None  # inert or sign-ambiguous; caller warns and skips
    return spec.low / positive.min(), spec.high / positive.max()


@dataclass(frozen=True)
class Slot:
    """One search dimension: a whole parameter (multiply) or one of its values (substitute)."""

    label: str
    par_name: str
    mode: str
    index: int          # value index for substitute; -1 for multiply
    low: float
    high: float
    log: bool
    clip: tuple = None


@dataclass(frozen=True)
class Layout:
    """The resolved search space for one HYPE configuration.

    Attributes
    ----------
    slots : tuple of Slot
        Search dimensions, in a stable order.
    base : dict of str -> ndarray
        Template ``par.txt`` vectors for the active parameters.
    options : dict
        Effective model options this layout was built against.
    active : tuple of str
        Parameter names actually calibrated.
    dropped : dict of str -> str
        Requested parameters that were excluded, mapped to the reason.
    """

    slots: tuple
    base: dict
    options: dict
    active: tuple
    dropped: dict = field(default_factory=dict)

    # -- derived arrays, built once ----------------------------------------------------

    def __post_init__(self):
        lo = np.array([s.low for s in self.slots], dtype=float)
        hi = np.array([s.high for s in self.slots], dtype=float)
        log_mask = np.array([s.log for s in self.slots], dtype=bool)
        if np.any(hi <= lo):
            bad = [self.slots[i].label for i in np.where(hi <= lo)[0]]
            raise ValueError(f"Zero-width or inverted search bounds for: {bad}")
        if np.any(log_mask & (lo <= 0)):
            bad = [self.slots[i].label for i in np.where(log_mask & (lo <= 0))[0]]
            raise ValueError(f"Log-searched slots need low > 0: {bad}")
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(log_mask, np.log10(np.maximum(lo, _TINY)), lo)
            t_hi = np.where(log_mask, np.log10(np.maximum(hi, _TINY)), hi)
        object.__setattr__(self, "_lo", lo)
        object.__setattr__(self, "_hi", hi)
        object.__setattr__(self, "_log_mask", log_mask)
        object.__setattr__(self, "_t_lo", t_lo)
        object.__setattr__(self, "_span", t_hi - t_lo)

    @property
    def n_parameters(self):
        return len(self.slots)

    @property
    def names(self):
        """Slot labels, e.g. ``["wcfc", "preccorr[0]", ...]``."""
        return [s.label for s in self.slots]

    def bounds(self):
        """Per-slot ``(low, high)`` in **physical** space."""
        return self._lo.copy(), self._hi.copy()

    def to_model(self, params):
        """Map unit-box search values to physical values.

        Broadcasts over ``[k]`` and ``[n_particles, k]`` alike (``estimator`` calls the
        inverse on 1-D bounds and this on 2-D populations), and always allocates, so the
        caller's array is never aliased or mutated.
        """
        u = np.asarray(params, dtype=float)
        t = self._t_lo + u * self._span
        with np.errstate(over="ignore", invalid="ignore"):
            out = np.where(self._log_mask, np.power(10.0, t), t)
        return np.clip(out, self._lo, self._hi)

    def to_search(self, params):
        """Map physical values to the unit box (inverse of :meth:`to_model`)."""
        p = np.asarray(params, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(self._log_mask, np.log10(np.maximum(p, _TINY)), p)
        return (t - self._t_lo) / self._span

    def to_par_values(self, physical):
        """Expand one physical parameter row into full ``par.txt`` vectors.

        Parameters
        ----------
        physical : ndarray
            One row, shape ``[n_parameters]``, in physical space.

        Returns
        -------
        dict of str -> ndarray
        """
        physical = np.asarray(physical, dtype=float).ravel()
        if physical.size != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} values, got {physical.size}."
            )
        out = {name: self.base[name].astype(float).copy() for name in self.active}
        for value, slot in zip(physical, self.slots):
            if slot.mode == "multiply":
                out[slot.par_name] = self.base[slot.par_name] * value
            else:
                out[slot.par_name][slot.index] = value
        for name, vector in out.items():
            clip = self._clip_for(name)
            if clip is not None:
                np.clip(vector, clip[0], clip[1], out=vector)
        return out

    def _clip_for(self, par_name):
        for slot in self.slots:
            if slot.par_name == par_name:
                return slot.clip
        return None

    @property
    def fingerprint(self):
        """Stable hash of everything that changes what a parameter vector means."""
        parts = [
            "|".join(
                f"{s.label},{s.par_name},{s.mode},{s.index},{s.low!r},{s.high!r},{s.log}"
                for s in self.slots
            ),
            "|".join(f"{k}={self.options[k]}" for k in sorted(self.options)),
            "|".join(f"{k}:{self.base[k].tolist()}" for k in sorted(self.base)),
        ]
        return hashlib.sha1("||".join(parts).encode()).hexdigest()[:16]

    def describe(self):
        """Slot table as a ``DataFrame`` (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(
            [
                dict(label=s.label, parameter=s.par_name, mode=s.mode,
                     index=None if s.index < 0 else s.index,
                     low=s.low, high=s.high, log=s.log)
                for s in self.slots
            ]
        )


def build_layout(par, options, requested=None, specs=None, warn_unrequested=True):
    """Resolve the search space for a HYPE configuration.

    Parameters
    ----------
    par : ParFile
        Parsed ``par.txt`` - the authority on which parameters exist and their arity.
    options : dict
        Effective model options (see :func:`effective_options`).
    requested : sequence of str or None
        Parameter names to calibrate. ``None`` means every catalogued parameter present in
        ``par.txt`` and enabled by the options.
    specs : dict of str -> HypeParameter or None
        Overrides and additions to :data:`CATALOGUE`.
    warn_unrequested : bool
        Also warn about enabled, calibratable parameters that were *not* requested (their
        template values are used unchanged).

    Returns
    -------
    Layout
    """
    catalogue = dict(CATALOGUE)
    catalogue.update(specs or {})

    if requested is None:
        requested = [name for name in par.values if name in catalogue]
    requested = list(dict.fromkeys(requested))  # de-duplicate, keep order

    slots, base, active, dropped = [], {}, [], {}

    for name in requested:
        if name not in par.values:
            dropped[name] = (
                "absent from par.txt (or commented out there) - HYPE would ignore it"
            )
            continue
        spec = catalogue.get(name)
        if spec is None:
            spec = HypeParameter(name=name, **DEFAULT_SPEC)
            warnings.warn(
                f"{name!r} is not in the HYPE catalogue; using fallback bounds "
                f"[{spec.low}, {spec.high}] (log={spec.log}, mode={spec.mode}). Pass an "
                "explicit HypeParameter via parameter_specs to control it.",
                stacklevel=3,
            )
        if not is_enabled(spec, options):
            need = ", ".join(
                f"{opt}={'/'.join(str(v) for v in vals)}"
                for opt, vals in spec.requires.items()
            )
            got = ", ".join(f"{opt}={options.get(opt)!r}" for opt in spec.requires)
            dropped[name] = f"routine off (needs {need}; currently {got})"
            continue

        values = par.values[name]
        if spec.mode == "multiply":
            interval = multiplier_bounds(spec, values)
            if interval is None:
                dropped[name] = (
                    "multiply mode needs at least one positive template value; this row is "
                    "all zero or negative, so a multiplier could not change it"
                )
                continue
            lo, hi = interval
            slots.append(
                Slot(label=name, par_name=name, mode="multiply", index=-1,
                     low=lo, high=hi, log=spec.log, clip=spec.clip)
            )
        else:
            width = len(str(values.size - 1))
            for i in range(values.size):
                label = name if values.size == 1 else f"{name}[{i:0{width}d}]"
                slots.append(
                    Slot(label=label, par_name=name, mode="substitute", index=i,
                         low=spec.low, high=spec.high, log=spec.log, clip=spec.clip)
                )
        base[name] = values.astype(float).copy()
        active.append(name)

    for name, reason in dropped.items():
        warnings.warn(
            f"HYPE parameter {name!r} excluded: {reason}.", stacklevel=3
        )

    if warn_unrequested:
        idle = [
            name for name in par.values
            if name in catalogue and name not in active and name not in dropped
            and is_enabled(catalogue[name], options)
        ]
        if idle:
            warnings.warn(
                f"{len(idle)} calibratable parameter(s) are enabled but not requested, so "
                f"their par.txt values are used unchanged: {sorted(idle)}. Pass "
                "warn_unrequested=False to silence this.",
                stacklevel=3,
            )

    if not slots:
        raise ValueError(
            "No HYPE parameters left to calibrate. Requested "
            f"{requested!r}; exclusions: {dropped!r}."
        )

    layout = Layout(
        slots=tuple(slots), base=base, options=dict(options),
        active=tuple(active), dropped=dropped,
    )
    return layout


def optpar_entries(layout, step=None):
    """Build ``(name, low, high, step)`` rows for :func:`~.files.write_optpar`.

    HYPE's own calibration works on the parameter values directly, one bound per class, so a
    ``multiply`` slot is expanded back to the physical range it implies for each class.
    """
    entries = []
    for name in layout.active:
        base = layout.base[name]
        lo = np.empty(base.size)
        hi = np.empty(base.size)
        clip = None
        for slot in (s for s in layout.slots if s.par_name == name):
            clip = slot.clip  # every slot of one parameter shares its spec
            if slot.mode == "multiply":
                lo[:] = np.minimum(base * slot.low, base * slot.high)
                hi[:] = np.maximum(base * slot.low, base * slot.high)
            else:
                lo[slot.index] = slot.low
                hi[slot.index] = slot.high
        if clip is not None:
            lo = np.clip(lo, clip[0], clip[1])
            hi = np.clip(hi, clip[0], clip[1])
        steps = np.where(hi < 1.0, 0.001, 0.1) if step is None else np.full(base.size, step)
        entries.append((name, lo, hi, steps))
    return entries


__all__ = [
    "CATALOGUE", "DEFAULT_SPEC", "HypeParameter", "Layout", "Slot",
    "build_layout", "effective_options", "is_enabled", "multiplier_bounds",
    "optpar_entries", "replace",
]
