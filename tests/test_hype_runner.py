"""HYPE runner: worker folders, determinism, parallel equivalence, failure handling.

Every test here drives the real subprocess machinery through the stub, so the code path is
the one a real HYPE run takes - file staging, launch, output parsing - with no HYPE present.
"""

import numpy as np
import pytest
from hype_stub import MODE_ENV, STATE_NAME  # tests/ is on sys.path under pytest

from foresight_gpu.models.hype import HYPEModel
from foresight_gpu.models.hype.runner import MUTABLE


def _work_root(model):
    """The folder the model's runner is using."""
    from pathlib import Path

    return Path(model._runner.spec.work_root)


def _worker_dirs(model):
    root = _work_root(model)
    return [p for p in root.iterdir() if p.is_dir()] if root.exists() else []


@pytest.fixture
def model_factory(hype_template, stub_command, tmp_path):
    """Builds independent models over the shared template; each is closed on teardown."""
    made = []

    def build(**kw):
        options = dict(
            template_dir=hype_template, subbasin=1234, executable=stub_command,
            parameters=["wcfc", "rrcs1", "cmlt"],
            work_root=tmp_path / f"work{len(made)}", warn_unrequested=False,
        )
        options.update(kw)
        model = HYPEModel(**options)
        made.append(model)
        return model

    yield build
    for model in made:
        model.close()


@pytest.fixture
def stub_mode(monkeypatch):
    """Select a stub failure mode for the child processes."""
    def use(mode):
        monkeypatch.setenv(MODE_ENV, mode)
    yield use
    monkeypatch.delenv(MODE_ENV, raising=False)


def _params(model, n=3, seed=0):
    rng = np.random.default_rng(seed)
    return model.search_transform(rng.uniform(0, 1, (n, model.n_parameters(1))))


def _X(model, count=50):
    from foresight_gpu.models.hype import as_X

    return as_X(model.dates_[:count])


class TestWorkspace:
    def test_creates_one_folder_per_worker(self, model_factory):
        model = model_factory(n_workers=1)
        model.forward(_X(model), _params(model, 2))
        folders = _worker_dirs(model)
        assert len(folders) == 1
        assert folders[0].name.startswith("w")

    def test_template_is_never_written_to(self, model_factory, hype_template):
        before = {p.name: p.read_bytes() for p in hype_template.iterdir() if p.is_file()}
        model = model_factory()
        model.forward(_X(model), _params(model, 2))
        after = {p.name: p.read_bytes() for p in hype_template.iterdir() if p.is_file()}
        assert before == after

    def test_mutable_files_are_copies_not_links(self, model_factory):
        model = model_factory(copy_mode="link")
        model.forward(_X(model), _params(model, 1))
        workdir = _worker_dirs(model)[0]
        for name in MUTABLE:
            candidate = workdir / name
            if candidate.exists():
                assert candidate.stat().st_nlink == 1, f"{name} must not be a hardlink"

    def test_copy_mode_copy_also_works(self, model_factory):
        model = model_factory(copy_mode="copy")
        out = model.forward(_X(model), _params(model, 2))
        assert np.isfinite(out).all()

    def test_previous_results_are_not_replicated(self, model_factory, hype_template):
        """A folder that has run HYPE's own calibration holds thousands of outputs.

        Copying them into every worker dominates setup and contributes nothing - on a real
        folder it was ~12000 files per worker and 5x the wall clock.
        """
        stale = hype_template / "results"
        stale.mkdir(exist_ok=True)
        for i in range(20):
            (stale / f"0001234_{i:06d}.txt").write_text("stale")

        model = model_factory()
        model.forward(_X(model), _params(model, 1))
        results = _worker_dirs(model)[0] / "results"
        assert results.is_dir(), "the output directory must still exist"
        assert not list(results.glob("*_0000*.txt")), "stale results were copied"

    def test_close_removes_the_working_folders(self, model_factory):
        model = model_factory()
        model.forward(_X(model), _params(model, 1))
        root = _work_root(model)
        assert root.exists()
        model.close()
        assert not root.exists()

    def test_missing_template_raises_actionably(self, stub_command, tmp_path):
        model = HYPEModel(template_dir=tmp_path / "nope", executable=stub_command)
        with pytest.raises(FileNotFoundError, match="template directory not found"):
            model.n_parameters(1)

    def test_missing_executable_lists_what_is_there(self, hype_template, tmp_path):
        model = HYPEModel(template_dir=hype_template, subbasin=1234,
                          executable="NoSuchHype.exe", parameters=["cmlt"],
                          work_root=tmp_path / "w", warn_unrequested=False)
        with pytest.raises(FileNotFoundError, match="Executables present"):
            model.forward(_X(model), _params(model, 1))
        model.close()


class TestDeterminism:
    def test_same_parameters_give_the_same_series(self, model_factory):
        """The premise the cache rests on: output is a function of par.txt alone."""
        model = model_factory(cache_size=0)
        X, params = _X(model), _params(model, 1)
        first = model.forward(X, params)
        second = model.forward(X, params)
        assert model.n_runs_ == 2  # genuinely re-run, not served from cache
        np.testing.assert_array_equal(first, second)

    def test_no_state_leaks_between_consecutive_runs(self, model_factory, stub_mode):
        """A vector run *after a different vector* must give the same answer as alone.

        The stub's ``statefile`` mode writes a state file and folds the run count into its
        output, so a leak is detectable. The runner clears volatile files before each run.
        """
        stub_mode("statefile")
        model = model_factory(cache_size=0, volatile=(STATE_NAME,))
        X = _X(model)
        a, b = _params(model, 1, seed=1), _params(model, 1, seed=2)
        alone = model.forward(X, a)
        model.forward(X, b)
        after_other = model.forward(X, a)
        np.testing.assert_array_equal(alone, after_other)

    def test_stale_output_is_never_reused(self, model_factory, stub_mode):
        """A run that writes nothing must fail, not return the previous run's file."""
        model = model_factory(cache_size=0, max_failure_fraction=1.0)
        X, params = _X(model), _params(model, 1)
        assert np.isfinite(model.forward(X, params)).all()
        stub_mode("missing_file")
        with pytest.warns(UserWarning, match="failed"):
            out = model.forward(X, _params(model, 1, seed=9))
        assert np.isnan(out).all()


class TestParallel:
    def test_pool_matches_serial_exactly(self, model_factory):
        serial, parallel = model_factory(n_workers=1), model_factory(n_workers=3)
        X = _X(serial)
        params = _params(serial, 6)
        np.testing.assert_array_equal(
            serial.forward(X, params), parallel.forward(X, params)
        )
        assert parallel.n_runs_ == 6

    def test_pool_creates_one_folder_per_worker_not_per_task(self, model_factory):
        model = model_factory(n_workers=2)
        model.forward(_X(model), _params(model, 8))
        # 8 tasks over 2 workers: the legacy per-task rent would have needed 8 folders.
        assert 1 <= len(_worker_dirs(model)) <= 2

    def test_results_reassemble_in_request_order(self, model_factory):
        """imap_unordered returns out of order; the job id must restore the column order."""
        serial, parallel = model_factory(n_workers=1), model_factory(n_workers=4)
        X = _X(serial)
        params = _params(serial, 8, seed=3)
        np.testing.assert_array_equal(
            serial.forward(X, params), parallel.forward(X, params)
        )


class TestFailures:
    @pytest.mark.parametrize("mode", ["crash", "nonzero", "missing_file", "truncated"])
    def test_failure_yields_nan_and_counts(self, model_factory, stub_mode, mode):
        stub_mode(mode)
        model = model_factory(max_failure_fraction=1.0)
        with pytest.warns(UserWarning, match="failed"):
            out = model.forward(_X(model), _params(model, 2))
        assert np.isnan(out).all()
        assert model.n_failed_runs_ == 2

    def test_warning_explains_the_front_hazard(self, model_factory, stub_mode):
        stub_mode("crash")
        model = model_factory(max_failure_fraction=1.0)
        with pytest.warns(UserWarning, match="non-exceedance 0.0"):
            model.forward(_X(model), _params(model, 1))

    def test_on_error_raise_propagates(self, model_factory, stub_mode):
        stub_mode("crash")
        model = model_factory(on_error="raise")
        with pytest.raises(RuntimeError, match="HYPE run"):
            model.forward(_X(model), _params(model, 2))

    def test_max_failure_fraction_aborts_a_broken_setup(self, model_factory, stub_mode):
        """A wrong executable must not yield a plausible-looking fit over a garbage front."""
        stub_mode("nonzero")
        model = model_factory(max_failure_fraction=0.25)
        with pytest.raises(RuntimeError, match="max_failure_fraction"):
            model.forward(_X(model), _params(model, 4))

    def test_failure_message_carries_the_cause(self, model_factory, stub_mode):
        stub_mode("nonzero")
        model = model_factory(on_error="raise")
        with pytest.raises(RuntimeError, match="exit code 3"):
            model.forward(_X(model), _params(model, 1))

    def test_truncated_output_is_rejected_not_padded(self, model_factory, stub_mode):
        """Partial NaNs would give a plausible exceedance with a bad loss - worse than NaN."""
        stub_mode("truncated")
        model = model_factory(on_error="raise")
        with pytest.raises(RuntimeError, match="expected .* rows"):
            model.forward(_X(model), _params(model, 1))

    def test_timeout_kills_a_hung_run(self, model_factory, stub_mode):
        stub_mode("slow")
        model = model_factory(timeout=2.0, on_error="raise")
        with pytest.raises(RuntimeError, match="timed out"):
            model.forward(_X(model), _params(model, 1))

    def test_missing_values_become_nan(self, model_factory, stub_mode):
        stub_mode("minus9999")
        model = model_factory()
        out = model.forward(_X(model, count=model.output_window_[1]), _params(model, 1))
        assert np.isnan(out).any() and np.isfinite(out).any()


class TestOutputConventions:
    def test_timeoutput_file_is_read(self, model_factory, stub_mode):
        """Without a subbasin the model asks for a timeoutput file instead."""
        stub_mode("comment_header")
        model = model_factory(subbasin=None, output_variable="cout")
        out = model.forward(_X(model), _params(model, 2))
        assert np.isfinite(out).all()
