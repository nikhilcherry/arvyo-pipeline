"""arvyo/worker.py — glue layer: foldr (period search) -> fitr (4-model fit).

Both tools are invoked ONLY via their CLI + JSON stdout + exit codes, never
imported as libraries: their published CLI/JSON/exit-code interface is
frozen the same way arvyo/contract.py's .npz schema is. See each tool's
README for the interface this module relies on:
  https://github.com/nikhilcherry/foldr
  https://github.com/nikhilcherry/fitr
"""

from __future__ import annotations

import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from . import __version__ as ARVYO_PIPELINE_VERSION
from ._toolchain import ToolNotFoundError, tool_command
from .contract import ContractError, load_sample
from .pipeline_config import PipelineConfig
from .result_schema import new_result


@lru_cache(maxsize=None)
def _tool_version(name: str) -> str:
    """`foldr --version` / `fitr version`, cached once per process."""
    try:
        cmd = tool_command(name)
    except ToolNotFoundError:
        return "unknown"
    argv = [cmd, "version"] if name == "fitr" else [cmd, "--version"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return proc.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _invoke_tool(name: str, args: list[str], timeout_s: float):
    """Resolve and run a tool's CLI, parsing stdout as JSON.

    Returns (returncode, payload, stderr, duration_s, tool_error).
    `tool_error` is a human-readable message set when the tool couldn't be
    resolved or timed out; in both cases returncode/payload are None.
    `payload` is None whenever stdout wasn't valid JSON (e.g. a usage
    error, which most of these tools report on stderr with no stdout JSON).
    """
    start = time.monotonic()
    try:
        cmd = tool_command(name)
    except ToolNotFoundError as exc:
        return None, None, "", time.monotonic() - start, str(exc)

    try:
        proc = subprocess.run(
            [cmd, *args], capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        return None, None, stderr, time.monotonic() - start, f"timed out after {timeout_s}s"

    duration = time.monotonic() - start
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = None
    return proc.returncode, payload, proc.stderr, duration, None


def _passed_gate(sde, snr, config: PipelineConfig) -> bool:
    sde_ok = sde is not None and sde >= config.sde_min
    snr_ok = snr is not None and snr >= config.snr_min
    return sde_ok or snr_ok


def _failure_message(returncode, stderr, tool_error, tool_name) -> str:
    if tool_error is not None:
        return tool_error
    stderr = (stderr or "").strip()
    if stderr:
        return stderr
    return f"{tool_name} exited {returncode} with no parseable JSON on stdout"


def process_target(
    npz_path: str | Path,
    config: PipelineConfig | None = None,
    use_catalog_period: bool = False,
) -> dict:
    """Run foldr then fitr on one .npz target; always returns a result dict.

    Never raises for tool-level failures (bad file, foldr/fitr crash or
    timeout, a non-clear fitr verdict) — every input produces exactly one
    schema-valid result dict. Only programmer errors (a bad `config` type)
    raise.

    `use_catalog_period=True` skips foldr's period search entirely and
    folds at the `period_days`/`epoch_btjd` already present in the .npz
    metadata instead — useful for demoing fitr on targets with a known
    ephemeris, and for sidestepping foldr's inherent search-resolution
    phase error (a real, small period/epoch mismatch that a rigid
    `t0_shift` can't fully absorb and that model-comparison can pick up
    on; catalog values don't have this error).
    """
    if config is not None and not isinstance(config, PipelineConfig):
        raise TypeError(f"config must be a PipelineConfig or None, got {type(config)!r}")
    config = config or PipelineConfig.load()

    npz_path = Path(npz_path)
    total_start = time.monotonic()
    runtime_s = {"foldr": 0.0, "fitr": 0.0, "total": 0.0}
    versions = {
        "foldr": _tool_version("foldr"),
        "fitr": _tool_version("fitr"),
        "arvyo_pipeline": ARVYO_PIPELINE_VERSION,
    }

    def _finish(**kwargs) -> dict:
        runtime_s["total"] = time.monotonic() - total_start
        return new_result(runtime_s=runtime_s, versions=versions, **kwargs)

    # --- step 1: validate against the frozen .npz contract ---------------
    try:
        sample = load_sample(npz_path)
    except ContractError as exc:
        return _finish(
            input={"path": str(npz_path), "tic_id": None, "label": None, "sector": None},
            period_search=None,
            model_fit=None,
            verdict="error",
            winner=None,
            error={"stage": "contract_validation", "message": str(exc)},
        )

    input_meta = {
        "path": str(npz_path),
        "tic_id": sample.get("tic_id"),
        "label": sample.get("label"),
        "sector": sample.get("sector"),
    }

    # --- step 2: period source: catalog ephemeris, or foldr's search -----
    if use_catalog_period:
        period_days = sample.get("period_days")
        epoch_btjd = sample.get("epoch_btjd")
        if period_days is None or epoch_btjd is None:
            return _finish(
                input=input_meta,
                period_search=None,
                model_fit=None,
                verdict="error",
                winner=None,
                error={
                    "stage": "period_search",
                    "message": (
                        "use_catalog_period=True but period_days/epoch_btjd "
                        "are missing from the .npz metadata"
                    ),
                },
            )
        period_search = {
            "engine": "catalog",
            "period": period_days,
            "t0": epoch_btjd,
            "duration_hours": None,
            "depth_ppm": None,
            "snr": None,
            "sde": None,
            "passed_gate": True,
        }
    else:
        returncode, payload, stderr, duration, tool_error = _invoke_tool(
            "foldr", [str(npz_path), "--json", "--no-plot"], config.foldr_timeout_s
        )
        runtime_s["foldr"] = duration

        if payload is None:
            return _finish(
                input=input_meta,
                period_search=None,
                model_fit=None,
                verdict="no_period",
                winner=None,
                error={
                    "stage": "period_search",
                    "message": _failure_message(returncode, stderr, tool_error, "foldr"),
                },
            )

        sde = payload.get("sde")
        snr = payload.get("snr")
        passed_gate = _passed_gate(sde, snr, config)
        period_search = {
            "engine": payload.get("engine"),
            "period": payload.get("period_days"),
            "t0": payload.get("t0"),
            "duration_hours": payload.get("duration_hours"),
            "depth_ppm": payload.get("depth_ppm"),
            "snr": snr,
            "sde": sde,
            "passed_gate": passed_gate,
        }

        # --- step 3: gate the period; short-circuit if not trustworthy ---
        if not passed_gate:
            return _finish(
                input=input_meta,
                period_search=period_search,
                model_fit=None,
                verdict="no_period",
                winner=None,
                error=None,
            )

    # --- step 4: fitr's 4-model fit at foldr's candidate period -----------
    returncode, payload, stderr, duration, tool_error = _invoke_tool(
        "fitr",
        [
            "fit", str(npz_path),
            "--period", str(period_search["period"]),
            "--epoch", str(period_search["t0"]),
            "--json",
        ],
        config.fitr_timeout_s,
    )
    runtime_s["fitr"] = duration

    exit_verdicts = {0: "clear", 3: "ambiguous", 4: "no_significant_signal"}
    if returncode not in exit_verdicts or payload is None:
        return _finish(
            input=input_meta,
            period_search=period_search,
            model_fit=None,
            verdict="error",
            winner=None,
            error={
                "stage": "model_fit",
                "message": _failure_message(returncode, stderr, tool_error, "fitr"),
            },
        )

    verdict = exit_verdicts[returncode]
    winner = payload.get("winner") if verdict == "clear" else None

    return _finish(
        input=input_meta,
        period_search=period_search,
        model_fit=payload,
        verdict=verdict,
        winner=winner,
        error=None,
    )
