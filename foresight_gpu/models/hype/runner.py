"""Running the HYPE executable, one working folder per worker process.

A folder cannot be shared by two *simultaneous* runs - each needs its own ``par.txt`` and its
own result file - so concurrency needs one folder per worker. Nothing needs a folder per
*task*, though: a worker claims its folder on first use, keyed by its own pid, and keeps it
for life. That is the same folder count as the legacy design without the per-task rent from a
shared queue, which leaked a folder on every error path and eventually deadlocked the pool.

Two details that protect the series cache, which assumes output is a pure function of
``par.txt``:

* the result file is deleted before every run, so a stale file can never be read back as a
  fresh result;
* ``volatile`` files (HYPE state dumps, logs) are cleared too, so run *i+1* cannot inherit
  state from run *i*.

Worker entry points are module-level and the payload is a small picklable :class:`RunSpec`
shipped once via ``initargs`` - the legacy version pickled the whole model object for every
parameter set.
"""

import multiprocessing
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .files import ParFile, read_series

#: Files HYPE rewrites, so each worker needs a private copy rather than a hardlink.
MUTABLE = ("par.txt", "info.txt", "optpar.txt")

#: How much of a failing run's stderr to keep.
_STDERR_TAIL = 600


@dataclass(frozen=True)
class RunSpec:
    """Everything a worker needs, picklable and shipped once per worker."""

    template_dir: str
    work_root: str
    command: tuple
    par: ParFile
    info_lines: tuple
    output_relpath: str
    output_column: object
    t0: int
    n_steps: int
    copy_mode: str = "link"
    timeout: float = None
    volatile: tuple = ()
    #: Directory of caller-supplied forcing files, written once by the parent and overlaid
    #: on top of the template so they win over whatever the folder shipped with.
    forcing_dir: str = None


_SPEC = None      # set by _worker_init
_WORKDIR = None   # created on first task


def _link_or_copy(src, dst, copy_mode):
    """Hardlink read-only inputs where possible; copy what HYPE rewrites."""
    if dst.exists():
        return
    if copy_mode == "link" and src.name not in MUTABLE:
        try:
            os.link(src, dst)
            return
        except OSError:  # different volume, or a filesystem without hardlinks
            pass
    shutil.copy2(src, dst)


def _build_workspace(spec, workdir):
    """Replicate the template into ``workdir``, then write the configured ``info.txt``."""
    template = Path(spec.template_dir)
    if not template.is_dir():
        raise FileNotFoundError(
            f"HYPE template directory not found: {template}. A fitted HYPEModel keeps only "
            "the path, so the folder must still exist to predict; use hype.freeze() for a "
            "portable artifact."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    result_rel = Path(spec.output_relpath).parent
    result_dir = workdir / result_rel

    for src in template.rglob("*"):
        relative = src.relative_to(template)
        # Never replicate previous results. A folder that has run HYPE's own calibration
        # can hold tens of thousands of numbered output files, and copying them into every
        # worker would dominate setup while contributing nothing.
        if result_rel != Path(".") and result_rel in relative.parents:
            continue
        dst = workdir / relative
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(src, dst, spec.copy_mode)

    result_dir.mkdir(parents=True, exist_ok=True)

    # Caller-supplied forcing overlays the template's own, replacing e.g. Pobs.txt entirely.
    if spec.forcing_dir:
        staged = Path(spec.forcing_dir)
        for src in sorted(staged.glob("*.txt")):
            dst = workdir / src.name
            if dst.exists():
                dst.unlink()
            _link_or_copy(src, dst, spec.copy_mode)

    with open(workdir / "info.txt", "w", encoding="latin-1", newline="") as handle:
        handle.write("\r\n".join(spec.info_lines))
    return workdir


def _worker_dir(spec):
    """This process's folder, built on first use."""
    global _WORKDIR
    if _WORKDIR is None:
        _WORKDIR = _build_workspace(spec, Path(spec.work_root) / f"w{os.getpid()}")
    return _WORKDIR


def _worker_init(spec):
    """Pool initialiser: stash the spec; the folder is built lazily on the first task."""
    global _SPEC, _WORKDIR
    _SPEC, _WORKDIR = spec, None


def _worker_run(job):
    """Run HYPE once.

    Parameters
    ----------
    job : tuple
        ``(job_id, {parameter_name: values})``.

    Returns
    -------
    tuple
        ``(job_id, series or None, message)``. ``series`` is ``None`` on failure and
        ``message`` then says why.
    """
    job_id, updates = job
    spec = _SPEC
    try:
        workdir = _worker_dir(spec)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return job_id, None, f"workspace setup failed: {exc}"

    output = workdir / spec.output_relpath
    try:
        spec.par.with_values(updates).write(workdir / "par.txt")
        # Clear anything that could survive into the next run and break determinism.
        for name in (spec.output_relpath,) + tuple(spec.volatile):
            stale = workdir / name
            if stale.exists():
                stale.unlink()
    except Exception as exc:  # noqa: BLE001
        return job_id, None, f"could not stage the run: {exc}"

    try:
        completed = subprocess.run(
            list(spec.command), cwd=str(workdir), capture_output=True,
            timeout=spec.timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return job_id, None, f"timed out after {spec.timeout}s"
    except OSError as exc:
        return job_id, None, f"could not launch {spec.command!r}: {exc}"

    if completed.returncode != 0:
        tail = (completed.stderr or b"").decode("latin-1", "replace")[-_STDERR_TAIL:]
        return job_id, None, f"exit code {completed.returncode}; stderr: {tail.strip()}"

    if not output.exists():
        tail = (completed.stdout or b"").decode("latin-1", "replace")[-_STDERR_TAIL:]
        return job_id, None, (
            f"exit code 0 but {spec.output_relpath} was not written; stdout: {tail.strip()}"
        )

    try:
        series = read_series(output, spec.output_column, spec.t0, spec.n_steps)
    except Exception as exc:  # noqa: BLE001
        return job_id, None, f"could not read {spec.output_relpath}: {exc}"
    return job_id, series.astype(np.float64), ""


class HypeRunner:
    """Dispatches HYPE runs, serially or across a process pool.

    ``n_workers=1`` takes the same code path with no pool at all, so a serial run and a
    parallel run are directly comparable (a test asserts they agree exactly).
    """

    def __init__(self, spec, n_workers=1):
        self.spec = spec
        self.n_workers = max(int(n_workers), 1)
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None and self.n_workers > 1:
            try:
                self._pool = multiprocessing.Pool(
                    processes=self.n_workers,
                    initializer=_worker_init,
                    initargs=(self.spec,),
                )
            except AssertionError:
                # Daemonic worker processes cannot have children (e.g. inside a joblib
                # multiprocessing backend); degrade rather than fail the fit.
                warnings.warn(
                    "Cannot start a HYPE process pool here (already inside a daemonic "
                    "worker); falling back to serial runs.",
                    stacklevel=2,
                )
                self.n_workers = 1
        return self._pool

    def run_many(self, jobs):
        """Yield ``(job_id, series or None, message)`` for each job, in completion order."""
        jobs = list(jobs)
        if not jobs:
            return
        pool = self._ensure_pool()
        if pool is None:
            _worker_init(self.spec)
            for job in jobs:
                yield _worker_run(job)
            return
        try:
            for result in pool.imap_unordered(_worker_run, jobs, chunksize=1):
                yield result
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            self._pool = None
            raise

    def close(self):
        """Shut the pool down and delete the worker folders."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        global _WORKDIR
        _WORKDIR = None
        root = Path(self.spec.work_root)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
