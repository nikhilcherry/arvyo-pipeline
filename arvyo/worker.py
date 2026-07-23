"""arvyo/worker.py — glue layer: foldr (period search) -> fitr (4-model fit)
-> localizr (opt-in centroid-offset vetting for the planet/blend degeneracy).

Every tool is invoked ONLY via its CLI + JSON stdout + exit codes, never
imported as a library: their published CLI/JSON/exit-code interface is
frozen the same way arvyo/contract.py's .npz schema is. See each tool's
README for the interface this module relies on:
  https://github.com/nikhilcherry/foldr
  https://github.com/nikhilcherry/fitr
  https://github.com/nikhilcherry/localizr
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


def _run_centroid_vetting(
    sample: dict,
    period_search: dict,
    verdict: str,
    winner: str | None,
    model_fit: dict | None,
    config: PipelineConfig,
    centroid_vet: bool,
) -> dict | None:
    """localizr's on-target/off-target-blend check -- opt-in, since it needs
    live network access to MAST + Gaia for a real target pixel file.

    Only ever attempted for the exact case localizr exists to resolve, per
    fitr's own README: "blend vs planet is often genuinely degenerate from
    photometry alone -- the real discriminator is a centroid offset, which
    fitr never sees." That's either a "clear" planet/blend winner (localizr
    can still catch a diluted blend fitr called "planet" with high
    confidence) or an "ambiguous" verdict where planet and blend are tied.
    Any other verdict returns None (not attempted, not applicable) rather
    than a dict, so untouched results stay byte-identical to before this
    field existed.
    """
    if not centroid_vet:
        return None

    candidates: set[str] = set()
    if verdict == "clear" and winner in ("planet", "blend"):
        candidates = {winner}
    elif verdict == "ambiguous" and model_fit is not None:
        tied = set(model_fit.get("tied_models") or [])
        if "planet" in tied and "blend" in tied:
            candidates = {"planet", "blend"}
    if not candidates:
        return {"ran": False, "skipped_reason": "verdict/winner is not a planet/blend case"}

    tic_id = sample.get("tic_id")
    if tic_id is None:
        return {"ran": False, "skipped_reason": "no tic_id/kic_id in .npz metadata"}

    period = period_search.get("period")
    epoch = period_search.get("t0")
    duration_hours = period_search.get("duration_hours")
    if period is None or epoch is None or duration_hours is None:
        # The --use-catalog-period path leaves duration_hours unset (foldr
        # never ran to measure it), and localizr requires it explicitly.
        return {
            "ran": False,
            "skipped_reason": "period/epoch/duration_hours not all available "
            "(centroid vetting needs foldr's search output, not --use-catalog-period)",
        }

    mission = str(sample.get("mission") or "tess").strip().lower()
    id_flag = "--kic-id" if mission == "kepler" else "--tic-id"

    returncode, payload, stderr, duration, tool_error = _invoke_tool(
        "localizr",
        [
            "localize", id_flag, str(tic_id),
            "--period", str(period),
            "--epoch", str(epoch),
            "--duration-hours", str(duration_hours),
            "--json",
        ],
        config.localizr_timeout_s,
    )
    if payload is None:
        return {
            "ran": False,
            "skipped_reason": _failure_message(returncode, stderr, tool_error, "localizr"),
        }
    return {"ran": True, "runtime_s": duration, **payload}


def process_target(
    npz_path: str | Path,
    config: PipelineConfig | None = None,
    use_catalog_period: bool = False,
    centroid_vet: bool | None = None,
) -> dict:
    """Run foldr then fitr (then, opt-in, localizr) on one .npz target;
    always returns a result dict.

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

    `centroid_vet` runs localizr's on-target/off-target-blend check when
    the verdict is a planet/blend case (see _run_centroid_vetting); it
    needs live network access to MAST/Gaia, so it defaults to
    `config.centroid_vetting_enabled` (itself off by default) rather than
    always-on. Pass True/False explicitly to override the config.
    """
    if config is not None and not isinstance(config, PipelineConfig):
        raise TypeError(f"config must be a PipelineConfig or None, got {type(config)!r}")
    config = config or PipelineConfig.load()
    if centroid_vet is None:
        centroid_vet = config.centroid_vetting_enabled

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

    # --- step 5: localizr's centroid-offset vetting (opt-in) --------------
    centroid_vetting = _run_centroid_vetting(
        sample, period_search, verdict, winner, payload, config, centroid_vet
    )

    return _finish(
        input=input_meta,
        period_search=period_search,
        model_fit=payload,
        verdict=verdict,
        winner=winner,
        centroid_vetting=centroid_vetting,
        error=None,
    )
