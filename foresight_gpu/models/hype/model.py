"""``HYPEModel`` - the SMHI HYPE hydrological model as a GPU forward model.

HYPE is a Fortran executable driven by a folder of text files, not an array function, so two
things differ from every other model in this package.

**The meteorology goes in through ``forcing``, not through ``X``.** HYPE reads precipitation
and temperature from its own files, so pass them as DataFrames (or paths) and they are
written into each worker folder before the run:

    HYPEModel(template_dir=..., forcing={"P": precip_df, "T": temp_df})

Omitting ``forcing`` falls back to whatever ``Pobs.txt``/``Tobs.txt`` the template already
contains - fine for a fixed setup, but then the data is invisible to the caller.

**``X`` carries dates, not features.** Because the forcing arrives by that other route,
there is nothing to put in a feature matrix; ``X[:, date_column]`` holds the day ordinal of
each sample (see :func:`~.dates.as_X`) and ``forward`` returns the simulated value for
exactly those dates, in that order. That is what lets early stopping,
``cross_val_score(cv=TimeSeriesSplit(...))`` and prediction over an arbitrary window work
with no change to the engine.

**One run covers the whole window.** HYPE integrates continuously from ``bdate`` regardless
of which dates are asked for, so a parameter vector is simulated once over ``[cdate, edate]``
and cached; ``forward`` then slices rows. Warmup is native: ``bdate`` starts the simulation,
``cdate`` starts the output, and the span between them never enters the loss.

Notes
-----
Two things to know before using it:

* **Do not put a scaler in front of it.** ``Pipeline([StandardScaler(), GPURegressor(...)])``
  transforms the date column outside the estimator, where nothing can undo it. ``forward``
  detects the damage and raises rather than returning nonsense.
* **Keep ``shuffle=False``** on the estimator (the default). Row order does not affect the
  simulation - HYPE owns that, and every metric is permutation-invariant given the date
  travels with the row - but a shuffled validation split interleaves held-out days with
  training days, and daily flows are autocorrelated enough that the score is then
  optimistically biased.

On Windows a script that sets ``n_workers > 1`` must guard its entry point with
``if __name__ == "__main__":``, because the process pool re-imports it.
"""

import atexit
import hashlib
import tempfile
import warnings
import weakref
from pathlib import Path

import numpy as np

from ..base import BaseForwardModel
from . import forcing as _forcing
from . import parameters as _params
from .cache import SeriesCache
from .dates import check_ordinals, resolve_indices, window_ordinals
from .files import InfoFile, ParFile
from .runner import HypeRunner, RunSpec

#: Files cleared before every run so no state or stale output survives into the next.
DEFAULT_VOLATILE = ("respar.txt", "bestsims.txt", "stub_state.txt")

#: How many individual failures are reported before switching to a summary.
_MAX_FAILURE_WARNINGS = 5


class HYPEModel(BaseForwardModel):
    """HYPE as a deterministic forward model over a particle swarm.

    Parameters
    ----------
    template_dir : str or Path
        The HYPE folder: the executable plus ``par.txt``, ``info.txt``, ``GeoData.txt`` and
        the static catchment description. It is never written to; each worker gets its own
        copy.
    forcing : dict or None
        The meteorology to drive the model with, as ``{name: data}`` where ``data`` is a
        ``DataFrame`` (index = dates, one column per forcing id), a ``Series``, or a path to
        an existing HYPE-format file. Names may be HYPE file names (``"Pobs.txt"``) or
        aliases (``"P"``, ``"precipitation"``, ``"T"``, ``"tmin"``, ``"Qobs"``, ...). Each
        entry is written into every worker folder, replacing whatever the template shipped
        with, and the matching ``read*obs`` switch is set in ``info.txt`` automatically.

        ``None`` falls back to the files already in ``template_dir`` - convenient for a
        fixed catchment setup, but the data is then invisible to the caller and cannot be
        varied from Python.
    bdate, cdate, edate : str or None
        Simulation start, output start and end. ``bdate < cdate`` gives a warmup span
        excluded from the output. Resolution order: an explicit argument wins; failing that
        the span of the supplied ``forcing`` (so the period follows the data); failing that
        the template ``info.txt``.
    parameters : sequence of str or None
        Parameter names to calibrate. ``None`` calibrates every catalogued parameter present
        in ``par.txt`` and enabled by the model options.
    model_options : dict or None
        ``modeloption`` overrides, e.g. ``{"snowmeltmodel": 2}``. Written into each worker's
        ``info.txt``, so switching a routine needs no second template folder.
    parameter_specs : dict or None
        Additions or overrides to :data:`~.parameters.CATALOGUE`, as
        ``{name: HypeParameter(...)}``.
    subbasin : int or None
        Subbasin id to read. Given, a ``basinoutput`` file is requested; otherwise a
        ``timeoutput`` file.
    output_variable : str
        HYPE variable to read (``"cout"`` is computed discharge).
    output_file : str or None
        Result file path relative to the working folder. ``None`` derives it from
        ``subbasin`` / ``output_variable``.
    date_column : int
        Column of ``X`` holding the day ordinal.
    executable : str or sequence of str
        Executable name inside the template folder, or a full command. A sequence lets tests
        substitute an interpreter plus script.
    n_workers : int
        Concurrent HYPE processes, each with its own working folder. ``1`` runs serially in
        process, with no pool.
    work_root : str or None
        Where working folders are created. ``None`` uses a fresh system temp directory -
        deliberately not the template or the project folder, since thousands of runs writing
        into a cloud-synced directory would keep the sync client busy indefinitely.
    copy_mode : {"link", "copy"}
        ``"link"`` hardlinks the large read-only inputs and copies only what HYPE rewrites.
    timeout : float or None
        Per-run timeout in seconds.
    cache_size : int
        Series cached (see :class:`~.cache.SeriesCache`). ``0`` disables caching. Keep it
        above ``(check_every + 1) * population`` or early-stopping checks stop being free.
    on_error : {"nan", "raise"}
        What a failed run yields. ``"nan"`` returns a NaN column and warns; ``"raise"``
        propagates, which is what you want while debugging a template.
    max_failure_fraction : float
        Raise if more than this fraction of a batch fails. Without it a wrong executable path
        produces a complete, plausible-looking fit over an entirely invalid front.
    warn_unrequested : bool
        Warn about enabled, calibratable parameters that were not requested.
    volatile : sequence of str or None
        Extra files to delete before each run.
    verbose : int
        Report each batch when non-zero.

    Attributes
    ----------
    parameter_names_ : list of str
        One label per search dimension, e.g. ``["wcfc", "preccorr[0]", ...]``.
    active_parameters_ : tuple of str
        Parameters actually calibrated, after routine gating.
    n_runs_, n_cache_hits_, n_failed_runs_ : int
        Run accounting.

    Notes
    -----
    ``GPURegressor.fit`` runs a ``clone`` of the model, so the instance you construct never
    executes anything: **read the counters and the workspace off ``estimator.model_``**, not
    off the model you passed in. That is also what keeps a ``cross_val_score`` or
    ``GridSearchCV`` fold from leaking state into the next.
    """

    def __init__(self, template_dir=None, forcing=None, bdate=None, cdate=None, edate=None,
                 parameters=None, model_options=None, parameter_specs=None,
                 subbasin=None, output_variable="cout", output_file=None,
                 date_column=0, executable="HYPEwithoutPopup4All.exe",
                 n_workers=1, work_root=None, copy_mode="link", timeout=None,
                 cache_size=8192, on_error="nan", max_failure_fraction=0.25,
                 warn_unrequested=True, volatile=None, verbose=0):
        # sklearn contract: store verbatim. ``clone`` asserts identity per parameter, so
        # normalising a list or resolving a None here would break GridSearchCV.
        self.template_dir = template_dir
        self.forcing = forcing
        self.bdate = bdate
        self.cdate = cdate
        self.edate = edate
        self.parameters = parameters
        self.model_options = model_options
        self.parameter_specs = parameter_specs
        self.subbasin = subbasin
        self.output_variable = output_variable
        self.output_file = output_file
        self.date_column = date_column
        self.executable = executable
        self.n_workers = n_workers
        self.work_root = work_root
        self.copy_mode = copy_mode
        self.timeout = timeout
        self.cache_size = cache_size
        self.on_error = on_error
        self.max_failure_fraction = max_failure_fraction
        self.warn_unrequested = warn_unrequested
        self.volatile = volatile
        self.verbose = verbose

    # -- lazy stage 1: layout (cheap, no side effects) -----------------------------------

    def _ensure_layout(self):
        """Parse the template and resolve the search space.

        Separate from the workspace because ``n_parameters`` / ``parameter_bounds`` are
        called before any run, and because ``fit`` works on a ``clone`` that owns no
        workspace at all.
        """
        if getattr(self, "_layout", None) is not None:
            return self._layout
        if self.template_dir is None:
            raise ValueError("HYPEModel needs template_dir (the HYPE folder).")
        template = Path(self.template_dir)
        if not template.is_dir():
            raise FileNotFoundError(f"HYPE template directory not found: {template}")

        par = ParFile.read(template / "par.txt")
        info = InfoFile.read(template / "info.txt")

        frames = _forcing.resolve(self.forcing)
        supplied_start, supplied_end = _forcing.coverage(frames)

        # Window precedence: an explicit argument, else the span of supplied forcing (so the
        # period follows the data rather than a stale bdate in the template), else the
        # template info.txt.
        bdate = self.bdate or (str(supplied_start) if supplied_start is not None else None) \
            or info.settings.get("bdate")
        edate = self.edate or (str(supplied_end) if supplied_end is not None else None) \
            or info.settings.get("edate")
        # cdate is the warmup boundary, not a period choice, so the template's value is kept
        # whenever it still lies inside the window. Supplying data should not silently throw
        # away the spin-up the template asked for; only a window that has moved past it does.
        cdate = self.cdate
        if cdate is None and bdate and edate:
            template_cdate = info.settings.get("cdate")
            if template_cdate and (
                np.datetime64(bdate, "D")
                <= np.datetime64(template_cdate, "D")
                <= np.datetime64(edate, "D")
            ):
                cdate = template_cdate
        cdate = cdate or bdate
        if not (bdate and cdate and edate):
            raise ValueError(
                "Simulation window incomplete: pass bdate/cdate/edate, supply forcing to "
                "derive them from, or provide them in the template info.txt (found "
                f"bdate={bdate!r}, cdate={cdate!r}, edate={edate!r})."
            )
        if np.datetime64(cdate, "D") < np.datetime64(bdate, "D"):
            raise ValueError(f"cdate ({cdate}) precedes bdate ({bdate}).")
        if np.datetime64(edate, "D") < np.datetime64(cdate, "D"):
            raise ValueError(f"edate ({edate}) precedes cdate ({cdate}).")

        if frames:
            _forcing.validate(
                frames, bdate, edate,
                expected_ids=_forcing.expected_ids(template, self.subbasin),
            )

        options = _params.effective_options(info.options, self.model_options)
        layout = _params.build_layout(
            par, options,
            requested=list(self.parameters) if self.parameters is not None else None,
            specs=self.parameter_specs,
            warn_unrequested=self.warn_unrequested,
        )

        self._par = par
        self._info = info
        self._dates = (str(bdate), str(cdate), str(edate))
        self._t0, self._n_steps = window_ordinals(cdate, edate)
        self._options = options
        self._forcing_frames = frames
        self._forcing_switches = _forcing.switches(frames)
        # The simulation depends on the meteorology as much as on the parameters, so the
        # forcing has to be part of the cache identity.
        self._fingerprint = hashlib.sha1(
            "|".join([
                layout.fingerprint, _forcing.digest(frames),
                str(bdate), str(cdate), str(edate),
                str(self.subbasin), str(self.output_variable),
                self._resolved_output,
            ]).encode()
        ).hexdigest()[:16]
        self._layout = layout
        return layout

    # -- forward-model contract ----------------------------------------------------------

    def n_parameters(self, n_features):
        """Number of search dimensions (independent of ``n_features``)."""
        return self._ensure_layout().n_parameters

    def parameter_bounds(self, n_features):
        """Per-dimension ``(low, high)`` in physical space."""
        return self._ensure_layout().bounds()

    def regularizable_mask(self, n_features):
        """No dimension is penalised: these are physical parameters, as with GR4J."""
        return np.zeros(self._ensure_layout().n_parameters, dtype=bool)

    def search_transform(self, params):
        """Unit box -> physical values."""
        return self._ensure_layout().to_model(params)

    def inverse_search_transform(self, params):
        """Physical values -> unit box."""
        return self._ensure_layout().to_search(params)

    # -- lazy stage 2: workspace ---------------------------------------------------------

    @property
    def _resolved_command(self):
        if isinstance(self.executable, (list, tuple)):
            return tuple(str(part) for part in self.executable)
        exe = Path(self.template_dir) / str(self.executable)
        if not exe.exists():
            available = sorted(
                p.name for p in Path(self.template_dir).iterdir()
                if p.suffix.lower() == ".exe"
            )
            raise FileNotFoundError(
                f"Executable {self.executable!r} not found in {self.template_dir}. "
                f"Executables present: {available or 'none'}."
            )
        return (str(exe),)

    @property
    def _resolved_output(self):
        if self.output_file is not None:
            return str(self.output_file).replace("\\", "/")
        if self.subbasin is not None:
            return f"results/{int(self.subbasin):07d}.txt"
        return f"results/time{str(self.output_variable).upper()}.txt"

    def _ensure_workspace(self):
        if getattr(self, "_runner", None) is not None:
            return self._runner
        layout = self._ensure_layout()
        bdate, cdate, edate = self._dates

        info = self._info.configured(
            bdate=bdate, cdate=cdate, edate=edate, options=self._options,
            subbasin=self.subbasin, output_variable=self.output_variable,
            resultdir="./results/", settings=self._forcing_switches,
        )
        root = self.work_root or tempfile.mkdtemp(prefix="foresight_hype_")
        Path(root).mkdir(parents=True, exist_ok=True)

        # Render the supplied forcing once here, not per worker: the files are identical
        # across workers, so each worker just links them over the template's own.
        forcing_dir = None
        if self._forcing_frames:
            forcing_dir = Path(root) / "_forcing"
            _forcing.stage(self._forcing_frames, forcing_dir, window=(bdate, edate))

        spec = RunSpec(
            template_dir=str(self.template_dir),
            work_root=str(root),
            command=self._resolved_command,
            par=self._par,
            info_lines=tuple(info.lines),
            output_relpath=self._resolved_output,
            # basinoutput columns are variables; a timeoutput header lists subbasin ids,
            # so fall back to the first data column there.
            output_column=self.output_variable if self.subbasin is not None else None,
            t0=self._t0,
            n_steps=self._n_steps,
            copy_mode=self.copy_mode,
            timeout=self.timeout,
            volatile=tuple(self.volatile or DEFAULT_VOLATILE),
            forcing_dir=None if forcing_dir is None else str(forcing_dir),
        )
        self._cache = SeriesCache(self.cache_size, self._fingerprint)
        self._runner = HypeRunner(spec, self.n_workers)
        self._finalizer = weakref.finalize(self, _release, self._runner)
        atexit.register(self._finalizer)
        return self._runner

    # -- evaluation ----------------------------------------------------------------------

    def forward(self, X, params):
        """Simulate the population and return the values at the dates in ``X``.

        Parameters
        ----------
        X : ndarray
            ``[n_samples, n_features]``; only ``date_column`` is read - HYPE takes its
            forcing from the template folder, so any other column is ignored.
        params : ndarray
            ``[n_particles, n_parameters]`` (or one row) in **physical** space.

        Returns
        -------
        ndarray
            ``[n_samples, n_particles]``, NaN for a failed run.
        """
        layout = self._ensure_layout()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D [n_samples, n_features].")
        column = X[:, int(self.date_column)]
        rows = resolve_indices(column, self._t0, self._n_steps)
        self._warn_if_unordered(column)

        params = np.atleast_2d(np.asarray(params, dtype=float))
        if params.shape[1] != layout.n_parameters:
            raise ValueError(
                f"params has {params.shape[1]} columns, expected {layout.n_parameters} "
                f"({', '.join(layout.names[:4])}...)."
            )
        low, high = layout.bounds()
        params = np.clip(params, low, high)

        series = self._simulate_population(params)
        return series[rows, :]

    def _simulate_population(self, params):
        """Return ``[n_steps, n_particles]`` for the whole window, using the cache."""
        runner = self._ensure_workspace()
        cache = self._cache

        unique, inverse = np.unique(params, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        keys = [cache.key(row) for row in unique]

        out = np.full((self._n_steps, unique.shape[0]), np.nan)
        pending = []
        for i, key in enumerate(keys):
            cached = cache.get(key)
            if cached is None:
                pending.append(i)
            else:
                out[:, i] = cached

        if pending:
            jobs = [(i, self._layout.to_par_values(unique[i])) for i in pending]
            failures = []
            for job_id, result, message in runner.run_many(jobs):
                if result is None:
                    failures.append((job_id, message))
                    continue
                # Quantise to the cache's precision *before* use, so a fresh run and a
                # cache hit are bit-identical and cache_size=0 changes only the run count.
                result = result.astype(np.float32)
                out[:, job_id] = result
                cache.put(keys[job_id], result)
            self._n_runs = getattr(self, "_n_runs", 0) + len(pending)
            self._handle_failures(failures, unique, len(pending))

        hit_fraction = 1.0 - len(pending) / max(unique.shape[0], 1)
        if hit_fraction >= 0.8 and unique.shape[0] > 1:
            # A batch that is mostly hits is a validation or prediction pass over the
            # surviving population: those series are the ones worth protecting from the
            # candidate churn that follows.
            cache.pin(keys)

        if self.verbose:
            print(
                f"[HYPEModel] {unique.shape[0]} unique parameter set(s): "
                f"{unique.shape[0] - len(pending)} cached, {len(pending)} run; "
                f"total runs {self.n_runs_}"
            )
        return out[:, inverse]

    def _handle_failures(self, failures, unique, dispatched):
        if not failures:
            return
        self._n_failed = getattr(self, "_n_failed", 0) + len(failures)
        fraction = len(failures) / max(dispatched, 1)

        detail = "; ".join(message for _, message in failures[:_MAX_FAILURE_WARNINGS])
        if self.on_error == "raise" or fraction > float(self.max_failure_fraction):
            raise RuntimeError(
                f"{len(failures)} of {dispatched} HYPE run(s) failed "
                f"({fraction:.0%} > max_failure_fraction="
                f"{self.max_failure_fraction:.0%}). First messages: {detail}"
            )
        if self.on_error != "nan":
            raise ValueError(f"on_error must be 'nan' or 'raise', got {self.on_error!r}")

        shown = min(len(failures), _MAX_FAILURE_WARNINGS)
        warnings.warn(
            f"{len(failures)} of {dispatched} HYPE run(s) failed and were returned as NaN. "
            f"A NaN column scores non-exceedance 0.0 exactly, which the double-Pareto sorter "
            f"can read as an unmatched tail model while the population is still narrow, so "
            f"keep an eye on n_failed_runs_. First {shown}: {detail}",
            stacklevel=3,
        )

    def _warn_if_unordered(self, column):
        if getattr(self, "_warned_order", False):
            return
        if column.size > 1 and np.any(np.diff(column) < 0):
            self._warned_order = True
            warnings.warn(
                "Dates reached HYPEModel out of chronological order. The simulation itself is "
                "unaffected (HYPE integrates the whole window), but if this came from "
                "shuffle=True then held-out days are interleaved with training days, and "
                "daily flows are autocorrelated enough to bias the score optimistically. "
                "Prefer shuffle=False for HYPE.",
                stacklevel=4,
            )

    # -- introspection -------------------------------------------------------------------

    @property
    def parameter_names_(self):
        return self._ensure_layout().names

    @property
    def active_parameters_(self):
        return self._ensure_layout().active

    @property
    def dropped_parameters_(self):
        """Requested parameters that were excluded, mapped to why."""
        return dict(self._ensure_layout().dropped)

    @property
    def model_options_(self):
        """Effective ``modeloption`` values (template merged with overrides)."""
        self._ensure_layout()
        return dict(self._options)

    @property
    def forcing_(self):
        """Supplied forcing as ``{HYPE file name: DataFrame}``.

        Empty when falling back to the files already in the template folder.
        """
        self._ensure_layout()
        return dict(self._forcing_frames)

    @property
    def forcing_switches_(self):
        """``read*obs`` settings that the supplied forcing turns on in ``info.txt``."""
        self._ensure_layout()
        return dict(self._forcing_switches)

    @property
    def fingerprint_(self):
        """Hash of everything a cached simulation depends on.

        Layout, forcing content, window and output selection - so a cache entry cannot
        survive a change to any of them.
        """
        self._ensure_layout()
        return self._fingerprint

    @property
    def simulation_window_(self):
        """``(bdate, cdate, edate)`` actually used."""
        self._ensure_layout()
        return self._dates

    @property
    def output_window_(self):
        """``(t0, n_steps)`` of the simulated output window, as day ordinals."""
        self._ensure_layout()
        return self._t0, self._n_steps

    @property
    def dates_(self):
        """Every date the model can be asked for, as ``datetime64[D]``.

        The output window ``[cdate, edate]``; the warmup span before ``cdate`` is simulated
        but never returned. Use it to build ``X`` (see :func:`~.dates.as_X`) or to restrict
        observations to what the model can answer.
        """
        from .dates import from_ordinals

        self._ensure_layout()
        return from_ordinals(np.arange(self._t0, self._t0 + self._n_steps))

    def _x_from_ordinals(self, ordinals):
        """Build an ``X`` with the day ordinal in ``date_column`` and zeros elsewhere."""
        ordinals = np.asarray(ordinals, dtype=float).ravel()
        X = np.zeros((ordinals.size, int(self.date_column) + 1))
        X[:, int(self.date_column)] = ordinals
        return X

    def observations(self, source, column=None, drop_missing=True, **kwargs):
        """Read observations and restrict them to what the model can answer for.

        The one-call route from a data source to ``(X, y)`` ready for ``fit``: it puts the
        date in this model's ``date_column`` and drops days outside the simulated output
        window (and, by default, days with no observation).

        Parameters
        ----------
        source : DataFrame, Series, or str or Path
            Observations. A frame or series must carry a ``DatetimeIndex``; a path is read
            as in :func:`load_observations`.
        column : str or None
            Which column holds the observations. ``None`` is only allowed when there is
            exactly one.
        drop_missing : bool
            Drop days whose observation is missing (required before ``fit``).
        **kwargs
            Forwarded to :func:`load_observations` for the path route (``sep``,
            ``dayfirst``, ``date_name``).

        Returns
        -------
        X, y : ndarray
        """
        self._ensure_layout()
        ordinals, values = _read_observations(source, column=column, **kwargs)
        return self.align(self._x_from_ordinals(ordinals), values,
                          drop_missing=drop_missing)

    def align(self, X, y=None, drop_missing=True, column=None):
        """Prepare observations for ``fit``: keep only usable, in-window rows.

        Two filters, both necessary in practice:

        * dates outside the simulated output window, which ``forward`` would reject;
        * rows whose observation is missing, when ``drop_missing`` (the default). Gaps are
          the normal case in a discharge record, and ``GPURegressor.fit`` cannot take them:
          scikit-learn validates ``y`` strictly, so a NaN there raises before the estimator
          sees it. Dropping rows is safe here precisely because the date travels with the
          row, so what remains stays correctly aligned.

        How ``y`` is matched to ``X`` depends on whether it carries dates:

        * a pandas ``Series`` or one-column ``DataFrame`` with a ``DatetimeIndex`` is
          **joined on dates**, so the two may come from different sources and cover
          different periods;
        * anything else is matched **positionally** and must have one value per row of
          ``X``. A length mismatch raises rather than guessing which rows correspond.

        The join preserves ``X``'s row order *and* multiplicity. Sorting or de-duplicating
        would change the effective weighting of both the loss and the exceedance objective,
        and would desynchronise any parallel array the caller is holding.

        Parameters
        ----------
        X : array-like
            ``[n_samples, n_features]`` with day ordinals in ``date_column``.
        y : array-like, Series or DataFrame, optional
            Observations.
        drop_missing : bool
            Also drop rows where the observation is not finite. Ignored when ``y`` is
            ``None``.
        column : str or None
            Which column of a multi-column ``y`` holds the observations.

        Returns
        -------
        X, y : ndarray
            Filtered arrays (``y`` omitted if it was not given).
        """
        self._ensure_layout()
        t0, n_steps = self._t0, self._n_steps
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D [n_samples, n_features], got shape {X.shape}. To turn a "
                "dated observation series into (X, y), use HYPEModel.observations()."
            )
        ordinals = check_ordinals(X[:, int(self.date_column)])
        in_window = (ordinals >= t0) & (ordinals < t0 + n_steps)
        if y is None:
            return X[in_window]

        dated = _dated_values(y, column)
        if dated is None:
            y = np.asarray(y, dtype=float).ravel()
            if y.size != X.shape[0]:
                raise ValueError(
                    f"X has {X.shape[0]} row(s) but y has {y.size} value(s), and y carries "
                    "no dates, so the two cannot be matched. Either pass y as a pandas "
                    "Series/DataFrame with a DatetimeIndex (align then joins on dates), or "
                    "build both from one source with HYPEModel.observations(source)."
                )
            keep = in_window & np.isfinite(y) if drop_missing else in_window
            return X[keep], y[keep]

        # Join on dates. The output window is contiguous, so the exact join is one integer
        # offset - the same arithmetic resolve_indices uses, O(n) and safe against
        # duplicates and arbitrary order on either side.
        obs_ordinals, obs_values = dated
        lookup = np.full(n_steps, np.nan)
        obs_in = (obs_ordinals >= t0) & (obs_ordinals < t0 + n_steps)
        lookup[(obs_ordinals[obs_in] - t0).astype(np.int64)] = obs_values[obs_in]

        joined = np.full(X.shape[0], np.nan)
        joined[in_window] = lookup[(ordinals[in_window] - t0).astype(np.int64)]

        keep = in_window & np.isfinite(joined) if drop_missing else in_window
        if not (in_window & np.isfinite(joined)).any():
            raise ValueError(
                "align found no date shared by X and y. X spans "
                f"{_span(ordinals)}, the observations span {_span(obs_ordinals)}, and the "
                f"model output window is {_span([t0, t0 + n_steps - 1])}. Check "
                "bdate/cdate/edate and that both sides carry real dates."
            )
        self._warn_alignment(ordinals, in_window, obs_ordinals, obs_in, joined)
        return X[keep], joined[keep]

    def _warn_alignment(self, ordinals, in_window, obs_ordinals, obs_in, joined):
        """Warn when X asked for in-window days the observations do not cover.

        Only that direction is reported. Observations outside X are *not* a mismatch: X is
        routinely a deliberate subset (a fold, a plotting window), so warning about the days
        that were not asked for would fire on ordinary use. And a plain window trim stays
        silent too - dropping the warmup span is this method's documented purpose, and the
        ``hype_observations`` fixture depends on it.
        """
        unmatched = in_window & ~np.isfinite(joined)
        if not unmatched.any():
            return
        kept = in_window & np.isfinite(joined)
        warnings.warn(
            f"align joined X and y on dates, and {int(unmatched.sum())} of the "
            f"{int(in_window.sum())} in-window day(s) in X have no observation "
            f"(e.g. {_span(ordinals[unmatched])}). Continuing with the "
            f"{int(kept.sum())} day(s) the two share, {_span(ordinals[kept])}. The "
            f"observations span {_span(obs_ordinals)}, of which "
            f"{int((~obs_in).sum())} day(s) fall outside the model window.",
            stacklevel=3,
        )

    @property
    def n_runs_(self):
        return getattr(self, "_n_runs", 0)

    @property
    def n_failed_runs_(self):
        return getattr(self, "_n_failed", 0)

    @property
    def n_cache_hits_(self):
        cache = getattr(self, "_cache", None)
        return 0 if cache is None else cache.hits

    def describe(self, params=None):
        """Slot table, optionally with physical values for one or many parameter rows.

        Returns
        -------
        DataFrame
            One row per search dimension. With ``params`` of shape ``[n_models, k]`` the
            per-dimension quantiles across models are added, which is the parameter-spread
            view the legacy code drew by hand.
        """
        import pandas as pd

        layout = self._ensure_layout()
        table = layout.describe()
        if params is None:
            return table
        values = np.atleast_2d(np.asarray(params, dtype=float))
        if values.shape[1] != layout.n_parameters:
            raise ValueError(
                f"params has {values.shape[1]} columns, expected {layout.n_parameters}."
            )
        if values.shape[0] == 1:
            table["value"] = values[0]
        else:
            table["p10"] = np.nanpercentile(values, 10, axis=0)
            table["median"] = np.nanmedian(values, axis=0)
            table["p90"] = np.nanpercentile(values, 90, axis=0)
        return table

    def write_par(self, path, params):
        """Write one physical parameter row into a ``par.txt`` at ``path``.

        Useful for handing a calibrated member back to HYPE outside this package.
        """
        layout = self._ensure_layout()
        values = np.asarray(params, dtype=float).ravel()
        self._par.with_values(layout.to_par_values(values)).write(path)
        return path

    # -- lifecycle -----------------------------------------------------------------------

    def close(self):
        """Shut down workers and remove the working folders."""
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer()
            try:
                atexit.unregister(finalizer)
            except Exception:  # pragma: no cover - unregister is best effort
                pass
        self._runner = None
        self._finalizer = None
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    #: State that must not be pickled: a process pool is unpicklable and a cache is large.
    _TRANSIENT = ("_runner", "_cache", "_finalizer")

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in self._TRANSIENT:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for key in self._TRANSIENT:
            setattr(self, key, None)


def _release(runner):
    """Module-level so ``weakref.finalize`` does not keep the model alive."""
    try:
        runner.close()
    except Exception:  # pragma: no cover - teardown must not raise
        pass


def _span(ordinals):
    """``"start..end"`` for a set of day ordinals, for error and warning messages."""
    from .dates import from_ordinals

    ordinals = np.asarray(ordinals, dtype=float).ravel()
    if ordinals.size == 0:
        return "(empty)"
    start, end = from_ordinals([ordinals.min(), ordinals.max()])
    return f"{start}..{end}"


def _pick_column(frame, column, label):
    """Select one column, refusing to guess when the choice matters.

    A single-column frame resolves implicitly; with two or more, guessing would silently
    calibrate against the wrong gauge.
    """
    if column is not None:
        if column not in frame.columns:
            raise KeyError(
                f"{label}: column {column!r} not found. Available: {list(frame.columns)}."
            )
        return frame[column]
    if frame.shape[1] == 1:
        return frame.iloc[:, 0]
    raise ValueError(
        f"{label} has {frame.shape[1]} columns {list(frame.columns)}; pass column= to "
        "choose one. Guessing would risk calibrating against the wrong series."
    )


def _dated_values(y, column=None):
    """``(ordinals, values)`` when ``y`` carries dates, ``None`` when it is positional."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency in practice
        return None
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        return None

    from .dates import as_ordinals

    series = _pick_column(y, column, "y") if isinstance(y, pd.DataFrame) else y
    index = series.index
    if not isinstance(index, pd.DatetimeIndex):
        if index.dtype.kind in "iuf":
            return None  # a RangeIndex-ed Series is just an array
        raise TypeError(
            f"y has a {type(index).__name__} of dtype {index.dtype}; align cannot tell "
            "whether it holds dates or row labels. Either set "
            "y.index = pd.to_datetime(y.index) to join on dates, or pass y.to_numpy() to "
            "match X positionally."
        )
    if index.tz is not None:
        raise TypeError(
            f"y has a tz-aware index ({index.tz}). Day ordinals are computed in UTC, which "
            "would shift every date. Use y.tz_localize(None)."
        )
    duplicated = index.duplicated(keep="last")
    if duplicated.any():
        warnings.warn(
            f"y has {int(duplicated.sum())} duplicated date(s); keeping the last value "
            "for each. Check the source.",
            stacklevel=4,
        )
        series = series[~duplicated]
    return as_ordinals(series.index), series.to_numpy(dtype=float)


def _read_observations(source, column=None, date_name="DATE", sep="\t", **kwargs):
    """``(ordinals, values)`` from a DataFrame, Series or path."""
    import pandas as pd

    from . import forcing as _fc
    from .dates import as_ordinals

    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        # read_csv rather than read_hype_table, so sep=';' and dayfirst=True keep working
        # for the observation files that are not in HYPE's own format.
        raw = pd.read_csv(source, sep=sep, **kwargs)
        name = date_name if date_name in raw.columns else raw.columns[0]
        stamps = pd.to_datetime(
            raw[name], **({"dayfirst": True} if kwargs.get("dayfirst") else {})
        )
        source = raw.drop(columns=[name]).set_index(pd.DatetimeIndex(stamps))
    elif isinstance(source, pd.DataFrame) and date_name in getattr(source, "columns", ()):
        # A frame that still carries the date as a column rather than an index.
        source = source.set_index(pd.DatetimeIndex(pd.to_datetime(source[date_name]))).drop(
            columns=[date_name]
        )

    frame = _fc.as_dated_frame(source, "observations", missing=_fc.MISSING)
    series = _pick_column(frame, column, "observations")
    return as_ordinals(series.index), series.to_numpy(dtype=float)


def load_observations(source, column=None, date_name="DATE", sep="\t", **kwargs):
    """Read observations into ``(X, y)`` ready for ``fit``.

    Parameters
    ----------
    source : DataFrame, Series, or str or Path
        Observations. A frame or series may carry the dates on its index or, for a frame,
        in a ``date_name`` column. A path is read with :func:`pandas.read_csv`, so
        ``sep``/``dayfirst`` cover the observation files that are not in HYPE's own format.
    column : str or None
        Observation column. ``None`` is only allowed when there is exactly one, so a
        multi-gauge file never resolves to the wrong series by accident.
    date_name : str
        Name of the date column, when the dates are in a column rather than the index.
    sep : str
        Field separator for the path route.
    **kwargs
        Passed to :func:`pandas.read_csv` (e.g. ``dayfirst=True``).

    Returns
    -------
    X : ndarray
        ``[n_samples, 1]`` of day ordinals.
    y : ndarray
        ``[n_samples]`` observations, with HYPE's ``-9999`` mapped to NaN.

    Notes
    -----
    ``X`` is always one column wide. For a model with ``date_column`` other than 0, use
    :meth:`HYPEModel.observations`, which knows where the date belongs.
    """
    ordinals, values = _read_observations(
        source, column=column, date_name=date_name, sep=sep, **kwargs
    )
    return ordinals.reshape(-1, 1), values
