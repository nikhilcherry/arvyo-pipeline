"""End-to-end smoke test for the foldr -> fitr -> batchr glue layer.

This module IS the pipeline's documentation: every command shown in the
README's glue-layer quickstart is exercised here first.

Requires the `[pipeline]` extra (`pip install -e ".[pipeline]"`); skipped
with a clear message if foldr/fitr/batchr aren't importable, since they're
install-time deps kept out of the base repo to stay lean.

Runtime note: these tiny fixtures (200 points / 5-day baseline) are the
same ones used for contract/schema tests elsewhere in the suite. They're
too sparse for a blind BLS/TLS period search to recover their injected
periods (verified empirically: even feeding fitr each fixture's own
catalog `period_days` directly, bypassing search, still yields
"no_significant_signal" — SNR/SDE are simply too low). That's expected for
data this small, not a pipeline bug, so the "sane verdicts" checks below
are tolerant of `no_period`/`no_significant_signal` for every class. A
separate pair of tests below injects a *realistic* transit signal (same
approach as tests/test_smoke.py) to prove the "clear signal" happy path
actually recovers the right class end-to-end.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arvyo._toolchain import ToolNotFoundError, tool_command
from arvyo.contract import load_sample
from arvyo.result_schema import validate_result
from arvyo.worker import process_target

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent


def _tools_available() -> bool:
    try:
        for name in ("foldr", "fitr", "batchr"):
            tool_command(name)
        return True
    except ToolNotFoundError:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _tools_available(),
        reason="foldr/fitr/batchr not installed; pip install -e '.[pipeline]'",
    ),
]

FIXTURE_NAMES = ["planet", "eb", "starspot", "null", "unknown"]

# "Not obviously wrong": no crash, and (when a class is expected to have a
# signal) a period-search/model-fit failure is an acceptable outcome given
# these fixtures' low SNR — see module docstring.
SANE_VERDICTS = {
    "planet": {"clear", "ambiguous", "no_significant_signal", "no_period"},
    "eb": {"clear", "ambiguous", "no_significant_signal", "no_period"},
    "starspot": {"clear", "ambiguous", "no_significant_signal", "no_period"},
    "null": {"no_period", "no_significant_signal"},
    "unknown": {"clear", "ambiguous", "no_significant_signal", "no_period"},
}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_produces_schema_valid_result(name):
    result = process_target(FIXTURES / f"fixture_{name}.npz")
    validate_result(result)  # raises SchemaError on any violation
    assert result["error"] is None or result["verdict"] in {"error", "no_period"}


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_verdict_is_sane(name):
    result = process_target(FIXTURES / f"fixture_{name}.npz")
    assert result["verdict"] in SANE_VERDICTS[name]

    true_label = load_sample(FIXTURES / f"fixture_{name}.npz")["label"]
    if result["verdict"] == "clear":
        # planet/blend are a documented degenerate pair in fitr (blend
        # generalizes planet with an extra dilution parameter) — a "blend"
        # winner for a planet-labeled fixture is a known-correct outcome,
        # not a misclassification.
        acceptable = {true_label, "blend"} if true_label == "planet" else {true_label}
        assert result["winner"] in acceptable
    elif result["verdict"] == "ambiguous":
        tied = set(result["model_fit"]["tied_models"])
        acceptable = tied | ({"blend"} if true_label == "planet" else set())
        assert true_label in acceptable or "blend" in tied


def test_corrupt_npz_yields_error_verdict_no_exception(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(
        path, time=np.linspace(0, 1, 10), flux=np.ones(10),
        tic_id=1, label="planet", sector=1,
    )  # missing flux_err

    result = process_target(path)  # must not raise

    validate_result(result)
    assert result["verdict"] == "error"
    assert result["error"]["stage"] in {"period_search", "contract_validation"}
    assert result["period_search"] is None
    assert result["model_fit"] is None


def _write_injected_transit_npz(path, *, kind: str, seed: int = 0):
    """A higher-fidelity synthetic light curve (see tests/test_smoke.py),
    unlike the tiny 200-point fixtures, actually has a recoverable signal.
    """
    from arvyo.synthesis.forward_models import eclipsing_binary, planet

    period, t0 = 1.8, 0.3
    time = np.arange(0, 5, 2.0 / 60 / 24)
    rng = np.random.default_rng(seed)

    if kind == "planet":
        params = {"period": period, "rp": 0.12, "t0": t0, "a": 10.0, "inc": 88.0}
        flux = planet(params, time) + rng.normal(0, 0.0002, time.size)
        flux_err = np.full(time.size, 0.0002)
    elif kind == "eb":
        params = {
            "period": period, "rp": 0.15, "t0": t0, "a": 8.0, "inc": 86.0,
            "secondary_scale": 0.3,
        }
        flux = eclipsing_binary(params, time) + rng.normal(0, 0.0003, time.size)
        flux_err = np.full(time.size, 0.0003)
    else:
        raise ValueError(kind)

    np.savez(
        path, time=time, flux=flux, flux_err=flux_err,
        tic_id=1, label=kind, sector=1, period_days=period, epoch_btjd=t0,
    )


def test_happy_path_planet_signal_recovered(tmp_path):
    path = tmp_path / "synth_planet.npz"
    _write_injected_transit_npz(path, kind="planet")

    result = process_target(path)
    validate_result(result)

    assert result["period_search"]["passed_gate"] is True
    assert result["verdict"] == "clear"
    assert result["winner"] in {"planet", "blend"}  # see degeneracy note above


def test_happy_path_eb_signal_recovered(tmp_path):
    path = tmp_path / "synth_eb.npz"
    _write_injected_transit_npz(path, kind="eb")

    result = process_target(path)
    validate_result(result)

    assert result["period_search"]["passed_gate"] is True
    assert result["verdict"] == "clear"
    assert result["winner"] == "eb"


def _run_pipeline_cli(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "arvyo.run", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=180,
    )


def test_bulk_run_via_manifest_and_resume(tmp_path):
    manifest_dir = tmp_path / "manifest_run"
    manifest_dir.mkdir()
    names = ["planet", "eb", "null"]
    for name in names:
        (manifest_dir / f"fixture_{name}.npz").write_bytes(
            (FIXTURES / f"fixture_{name}.npz").read_bytes()
        )

    manifest_path = manifest_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        for name in names:
            writer.writerow([f"fixture_{name}.npz"])  # relative to manifest dir

    results_dir = manifest_dir / "results"

    proc1 = _run_pipeline_cli(
        ["all", str(manifest_path), "--results-dir", str(results_dir)], cwd=manifest_dir
    )
    assert proc1.returncode in (0, 1), proc1.stderr  # 1 only if a target itself failed
    assert "3 item(s)" in proc1.stdout

    result_files = sorted(results_dir.glob("*.json"))
    assert len(result_files) == 3
    for rf in result_files:
        validate_result(json.loads(rf.read_text()))

    summarize = _run_pipeline_cli(["summarize", str(results_dir)], cwd=manifest_dir)
    assert summarize.returncode == 0
    assert "Verdicts over 3 result(s)" in summarize.stdout

    # Re-run: batchr's content-hash cache must make this a no-op.
    proc2 = _run_pipeline_cli(
        ["all", str(manifest_path), "--results-dir", str(results_dir)], cwd=manifest_dir
    )
    assert "0 ok, 3 cached, 0 failed" in proc2.stdout


def test_cwd_independence_of_run_one(tmp_path):
    foreign_dir = tmp_path / "somewhere_else"
    foreign_dir.mkdir()
    fixtures_copy = tmp_path / "fixtures"
    fixtures_copy.mkdir()
    (fixtures_copy / "fixture_null.npz").write_bytes(
        (FIXTURES / "fixture_null.npz").read_bytes()
    )

    proc = _run_pipeline_cli(
        ["one", "../fixtures/fixture_null.npz"], cwd=foreign_dir
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    validate_result(result)
    assert result["input"]["label"] == "null"
