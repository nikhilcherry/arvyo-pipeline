"""Direct TLS-recovery regression for the regenerated fixtures.

Distinct from tests/test_end_to_end.py's tolerant, glue-layer-driven
`test_fixture_verdict_is_sane`: this calls arvyo.search.tls_search.run_tls
directly on tests/fixtures/fixture_planet.npz and fixture_null.npz and pins
down the two claims scripts/regenerate_fixtures.py's docstring makes —
that the planet period is actually recoverable, and that null actually
stays quiet — rather than assuming them.
"""

from __future__ import annotations

from pathlib import Path

from arvyo.contract import load_sample
from arvyo.pipeline_config import PipelineConfig
from arvyo.search.tls_search import run_tls

FIXTURES = Path(__file__).parent / "fixtures"

TRUE_PLANET_PERIOD_DAYS = 3.14


def test_planet_period_recovered_within_one_percent():
    sample = load_sample(FIXTURES / "fixture_planet.npz")
    result = run_tls(sample["time"], sample["flux"], show_progress_bar=False)

    rel_diff = abs(result["period"] - TRUE_PLANET_PERIOD_DAYS) / TRUE_PLANET_PERIOD_DAYS
    assert rel_diff < 0.01, (
        f"recovered period {result['period']:.4f}d is {rel_diff:.2%} off "
        f"the injected {TRUE_PLANET_PERIOD_DAYS}d (must be <1%)"
    )
    assert result["SDE"] >= PipelineConfig().sde_min


def test_null_does_not_produce_a_confident_detection():
    sample = load_sample(FIXTURES / "fixture_null.npz")
    result = run_tls(sample["time"], sample["flux"], show_progress_bar=False)

    sde_min = PipelineConfig().sde_min
    assert result["SDE"] < sde_min, (
        f"null fixture produced SDE={result['SDE']:.2f} >= gate {sde_min} "
        "— white+red noise alone should never pass the detection gate"
    )
