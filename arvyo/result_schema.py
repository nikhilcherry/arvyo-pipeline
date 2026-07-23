"""The frozen arvyo-pipeline per-target result schema.

This is the output contract of `arvyo.worker.process_target`: every .npz
target produces exactly one dict matching this shape. Any change requires
bumping SCHEMA_VERSION.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.1"

VERDICTS = {"clear", "ambiguous", "no_significant_signal", "no_period", "error"}

TOP_LEVEL_KEYS = {
    "schema_version",
    "input",
    "period_search",
    "model_fit",
    "verdict",
    "winner",
    "centroid_vetting",
    "error",
    "runtime_s",
    "versions",
}

REQUIRED_INPUT_KEYS = {"path", "tic_id", "label", "sector"}
REQUIRED_PERIOD_SEARCH_KEYS = {
    "engine", "period", "t0", "duration_hours", "depth_ppm", "snr", "sde", "passed_gate",
}


class SchemaError(ValueError):
    """Raised when a result dict violates the frozen result schema."""


def validate_result(result: dict) -> None:
    """Raise SchemaError if `result` does not match the frozen schema."""
    missing = TOP_LEVEL_KEYS - set(result)
    if missing:
        raise SchemaError(f"result missing top-level key(s): {sorted(missing)}")

    if result["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"schema_version {result['schema_version']!r} != {SCHEMA_VERSION!r}"
        )

    missing = REQUIRED_INPUT_KEYS - set(result["input"])
    if missing:
        raise SchemaError(f"result['input'] missing key(s): {sorted(missing)}")

    if result["period_search"] is not None:
        missing = REQUIRED_PERIOD_SEARCH_KEYS - set(result["period_search"])
        if missing:
            raise SchemaError(f"result['period_search'] missing key(s): {sorted(missing)}")

    if result["verdict"] not in VERDICTS:
        raise SchemaError(f"result['verdict'] {result['verdict']!r} not in {VERDICTS}")

    if result["error"] is not None:
        if not {"stage", "message"} <= set(result["error"]):
            raise SchemaError("result['error'] must have 'stage' and 'message' keys")

    for key in ("foldr", "fitr", "total"):
        if key not in result["runtime_s"]:
            raise SchemaError(f"result['runtime_s'] missing key {key!r}")

    for key in ("foldr", "fitr", "arvyo_pipeline"):
        if key not in result["versions"]:
            raise SchemaError(f"result['versions'] missing key {key!r}")


def new_result(
    *,
    input: dict,
    period_search: dict | None,
    model_fit: dict | None,
    verdict: str,
    winner: str | None,
    error: dict | None,
    runtime_s: dict,
    versions: dict,
    centroid_vetting: dict | None = None,
) -> dict:
    """Build a schema-v1.1 result dict with a stable key order.

    `centroid_vetting` is new in 1.1 (localizr's on-target/off-target-blend
    check) and always defaults to None so every pre-1.1 call site keeps
    working unchanged: it's only ever non-None when centroid vetting was
    actually attempted for this target (see worker.py's
    _run_centroid_vetting), which itself is opt-in
    (PipelineConfig.centroid_vetting_enabled) since it needs live network
    access to MAST/Gaia.
    """
    result = {
        "schema_version": SCHEMA_VERSION,
        "input": input,
        "period_search": period_search,
        "model_fit": model_fit,
        "verdict": verdict,
        "winner": winner,
        "centroid_vetting": centroid_vetting,
        "error": error,
        "runtime_s": runtime_s,
        "versions": versions,
    }
    validate_result(result)
    return result
