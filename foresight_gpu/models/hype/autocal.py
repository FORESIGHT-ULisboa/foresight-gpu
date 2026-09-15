"""HYPE's own DE-MC automatic calibration, driven from the same parameter catalogue.

This is an *alternative* to the GPU method, not part of it: HYPE optimises a single weighted
criterion and returns one best parameter set, where GPU evolves a population spread across
the non-exceedance axis. It is here because the comparison is the obvious benchmark, and
because deriving ``optpar.txt`` from the catalogue removes the step the legacy workflow did
by hand.

It is deliberately a function rather than arguments on :class:`~.model.HYPEModel`: every
constructor argument becomes a ``get_params`` key that ``clone`` must round-trip, and none of
these knobs affect ``forward``.
"""

import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .files import read_bestsims, read_respar, read_series, write_optpar
from .parameters import optpar_entries

#: HYPE criterion codes, weighted and summed into its objective.
DEFAULT_CRITERIA = (("MR2", "cout", "rout", 1.0),)


@dataclass
class AutoCalResult:
    """Outcome of one HYPE automatic calibration.

    Attributes
    ----------
    best_params : dict of str -> ndarray
        Optimum read from ``respar.txt``, per parameter.
    accepted : ndarray or None
        Rows of ``bestsims.txt`` (accepted DE-MC candidates), or ``None`` if absent.
    accepted_columns : list of str
        Column names for ``accepted``.
    simulation : ndarray or None
        Simulated series of the optimum, if HYPE wrote one.
    work_dir : Path
        Folder the calibration ran in; kept when ``keep=True`` so its logs can be read.
    returncode : int
        Exit status of the executable.
    """

    best_params: dict = field(default_factory=dict)
    accepted: object = None
    accepted_columns: list = field(default_factory=list)
    simulation: object = None
    work_dir: object = None
    returncode: int = 0

    def to_frame(self):
        """Optimum as a tidy ``DataFrame`` (requires pandas)."""
        import pandas as pd

        rows = [
            {"parameter": name, "index": i, "value": float(value)}
            for name, values in self.best_params.items()
            for i, value in enumerate(np.atleast_1d(values))
        ]
        return pd.DataFrame(rows)


def autocalibrate(model, criteria=DEFAULT_CRITERIA, ngen=100, npop=50, step=None,
                  gammascale=0.5, sigma=0.0, crossover=0.4, task=("DE", "WS"),
                  block_line=22, work_dir=None, keep=False, timeout=None, verbose=0):
    """Run HYPE's built-in DE-MC calibration over a model's active parameters.

    Bounds come from the model's :class:`~.parameters.Layout`, so the two calibration routes
    explore the same space - a ``multiply`` slot is expanded back to the physical range it
    implies for each class, because HYPE optimises values directly.

    Parameters
    ----------
    model : HYPEModel
        Supplies the template, the window, the model options and the parameter layout.
    criteria : sequence of (str, str, str, float)
        ``(criterion, cvariable, rvariable, weight)`` per objective, e.g.
        ``[("MR2", "cout", "rout", 1.0)]``.
    ngen, npop : int
        DE-MC generations and population. Keep both small for a smoke test - HYPE runs
        ``ngen * npop`` simulations in one process.
    step : float or None
        Step size per parameter; ``None`` picks 0.001 for sub-unit ranges, else 0.1.
    work_dir : str or Path or None
        Folder to run in. ``None`` creates a temp copy of the template.
    keep : bool
        Keep the working folder (and HYPE's logs) after returning.
    timeout : float or None
        Seconds before the run is killed.

    Returns
    -------
    AutoCalResult

    Raises
    ------
    FileNotFoundError
        If ``Qobs.txt`` is missing - HYPE cannot compute a criterion without observations.
    RuntimeError
        If the executable fails, or writes no ``respar.txt``.
    """
    layout = model._ensure_layout()
    template = Path(model.template_dir)
    bdate, cdate, edate = model.simulation_window_

    if not (template / "Qobs.txt").exists():
        raise FileNotFoundError(
            f"{template} has no Qobs.txt. HYPE's own calibration reads observations from the "
            "model folder (the GPU route takes them from y instead), so add Qobs.txt with a "
            "column named for the subbasin id."
        )

    created = work_dir is None
    root = Path(work_dir) if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="foresight_hype_autocal_")
    )
    root.mkdir(parents=True, exist_ok=True)
    # Skip the template's own results: a folder that has calibrated before can hold tens of
    # thousands of numbered output files, and copying them would swamp the setup.
    shutil.copytree(template, root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("results"))
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    info = model._info.configured(
        bdate=bdate, cdate=cdate, edate=edate, options=model.model_options_,
        subbasin=model.subbasin, output_variable=model.output_variable,
        resultdir="./results/",
    ).for_calibration(criteria, resultdir="./results/")
    info.write(root / "info.txt")

    entries = optpar_entries(layout, step=step)
    write_optpar(
        root / "optpar.txt", entries, task=task, ngen=ngen, npop=npop,
        gammascale=gammascale, sigma=sigma, crossover=crossover,
        block_line=block_line,
    )
    if verbose:
        print(
            f"[autocalibrate] {len(entries)} parameter(s), "
            f"{sum(int(np.size(e[1])) for e in entries)} value(s); "
            f"DEMC_ngen={ngen} DEMC_npop={npop} -> up to {ngen * npop} HYPE simulations"
        )

    command = list(model._resolved_command)
    completed = subprocess.run(
        command, cwd=str(root), capture_output=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        tail = (completed.stderr or b"").decode("latin-1", "replace")[-800:]
        raise RuntimeError(
            f"HYPE calibration exited {completed.returncode} in {root}. stderr: {tail.strip()}"
        )

    respar = _first_existing(root, ("respar.txt", "results/respar.txt"))
    if respar is None:
        raise RuntimeError(
            f"HYPE calibration wrote no respar.txt in {root}. Check the hyss_*.log there; "
            "the usual cause is a criterion that never had enough observations "
            "(crit datalimit) or an optpar.txt whose parameter block is misaligned."
        )
    best = read_respar(respar)

    accepted, columns = None, []
    bestsims = _first_existing(root, ("bestsims.txt", "results/bestsims.txt"))
    if bestsims is not None:
        try:
            columns, accepted = read_bestsims(bestsims)
        except ValueError as exc:
            warnings.warn(f"Could not read {bestsims}: {exc}", stacklevel=2)

    output = _best_simulation_path(root / model._resolved_output, columns, accepted)
    simulation = None
    if output is not None:
        try:
            simulation = read_series(
                output,
                model.output_variable if model.subbasin is not None else None,
            )
        except ValueError as exc:
            warnings.warn(f"Could not read {output}: {exc}", stacklevel=2)

    result = AutoCalResult(
        best_params=best, accepted=accepted, accepted_columns=columns,
        simulation=simulation, work_dir=root, returncode=completed.returncode,
    )
    if created and not keep:
        shutil.rmtree(root, ignore_errors=True)
        result.work_dir = None
    return result


def _first_existing(root, candidates):
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def _best_simulation_path(plain, columns, accepted):
    """Locate the simulation of the optimum.

    During calibration HYPE does not overwrite one result file: it writes a numbered file
    per accepted candidate (``0050675_000001.txt``, ...). ``bestsims.txt`` names each
    candidate in its ``NO`` column and scores it in ``CRIT`` (lower is better), so the
    optimum's file can be identified rather than guessed. Falls back to the plain file, then
    to the highest-numbered one.
    """
    if plain.exists():
        return plain

    numbered = sorted(plain.parent.glob(f"{plain.stem}_*{plain.suffix}"))
    if not numbered:
        return None

    if accepted is not None and accepted.size and {"NO", "CRIT"} <= set(columns):
        no = accepted[:, columns.index("NO")]
        crit = accepted[:, columns.index("CRIT")]
        finite = np.isfinite(crit)
        if finite.any():
            best = int(no[finite][int(np.nanargmin(crit[finite]))])
            candidate = plain.with_name(f"{plain.stem}_{best:06d}{plain.suffix}")
            if candidate.exists():
                return candidate
    return numbered[-1]
