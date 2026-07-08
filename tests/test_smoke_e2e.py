"""End-to-end smoke test for scripts/smoke_run.py's direct-module pipeline.

Distinct from tests/test_end_to_end.py (which exercises the foldr/fitr/batchr
glue layer): this drives arvyo's modules directly (contract, views, TLS,
synthesis, inference, vetting, viz) the same way scripts/smoke_run.py does,
against real .npz samples from the sibling arvyo-data repo. Skipped cleanly
if that data isn't present locally (e.g. in CI without arvyo-data checked out).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from smoke_run import ALL_LABELS, STAGE_NAMES, main, pick_samples  # noqa: E402

DATA_ROOT = REPO_ROOT.parent / "arvyo-data" / "data" / "processed"


def _has_any_sample() -> bool:
    return bool(pick_samples(str(DATA_ROOT), ALL_LABELS))


@pytest.mark.skipif(not _has_any_sample(), reason=(
    f"no contract-valid .npz samples found under {DATA_ROOT}; "
    "checkout arvyo-data alongside arvyo-pipeline to run this test"
))
def test_smoke_run_end_to_end(tmp_path):
    out_dir = tmp_path / "smoke_out"
    exit_code = main([
        "--data-root", str(DATA_ROOT),
        "--out-dir", str(out_dir),
        "--seed", "42",
        "--fast",
    ])
    assert exit_code == 0

    report_files = sorted(out_dir.glob("*.json"))
    per_sample_reports = [f for f in report_files if f.name != "report.json"]
    assert per_sample_reports, "expected at least one per-sample report"

    for path in per_sample_reports:
        report = json.loads(path.read_text())
        assert set(report["stages"].keys()) == set(STAGE_NAMES)
        for stage_name in STAGE_NAMES:
            assert report["stages"][stage_name]["status"] in ("pass", "fail", "skip")
