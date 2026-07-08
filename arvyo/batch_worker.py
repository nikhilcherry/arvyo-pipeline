"""arvyo/batch_worker.py — the `module:function` batchr invokes per item.

batchr's `run_batch`/CLI calls `fn(item: str) -> Any` in a worker process
(see https://github.com/nikhilcherry/batchr — "Non-picklable functions"):
`fn` must be a module-level function taking a single item string, since
workers are separate `ProcessPoolExecutor` processes. Notably, batchr does
NOT pass its `--config` through to `fn` — that config only feeds the cache
key. So the pipeline config each worker process should use is threaded
through the `ARVYO_PIPELINE_CONFIG` env var, set by `arvyo.run all` before
invoking `batchr run`; forked worker processes inherit it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .pipeline_config import PipelineConfig
from .worker import process_target


def _load_config() -> PipelineConfig:
    config_path = os.environ.get("ARVYO_PIPELINE_CONFIG")
    return PipelineConfig.load(config_path) if config_path else PipelineConfig.load()


def _result_filename(result: dict) -> str:
    tic_id = result["input"].get("tic_id")
    sector = result["input"].get("sector")
    if tic_id is not None and sector is not None:
        return f"{tic_id}_{sector}.json"
    return f"{Path(result['input']['path']).stem}.json"


def run_one(item: str) -> dict:
    """batchr's per-item worker: run the pipeline, write one result JSON.

    Returns a small summary that batchr caches in its own ledger — the
    full result dict lives at ``{results_dir}/{tic_id}_{sector}.json``.
    """
    config = _load_config()
    result = process_target(item, config)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / _result_filename(result)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, out_path)

    return {
        "item": str(item),
        "tic_id": result["input"].get("tic_id"),
        "sector": result["input"].get("sector"),
        "verdict": result["verdict"],
        "winner": result["winner"],
        "result_path": str(out_path),
    }
