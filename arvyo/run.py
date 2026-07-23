"""python -m arvyo.run — CLI driver for the foldr -> fitr per-target pipeline.

    python -m arvyo.run one PATH.npz            # single target, JSON to stdout
    python -m arvyo.run all MANIFEST.csv        # bulk via batchr, resumable
    python -m arvyo.run summarize RESULTS_DIR   # verdict counts + confusion vs labels

`all` shells out to batchr's CLI (never imports batchr's Python API) with
PYTHONPATH set to the repo root. batchr's `--fn module:function` importer
only adds the *current* cwd to sys.path (batchr/cli.py `_import_fn`), so if
`arvyo` isn't already importable from wherever `batchr run` happens to
execute, it raises ModuleNotFoundError on `arvyo.batch_worker`. See the
README's "batchr PYTHONPATH workaround" note.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from ._toolchain import tool_command
from .pipeline_config import PipelineConfig
from .worker import process_target

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cmd_one(args: argparse.Namespace) -> int:
    config = PipelineConfig.load(args.config) if args.config else PipelineConfig.load()
    centroid_vet = True if args.centroid_vet else (False if args.no_centroid_vet else None)
    result = process_target(
        args.path, config,
        use_catalog_period=args.use_catalog_period,
        centroid_vet=centroid_vet,
    )
    print(json.dumps(result, indent=2))
    return 0


def _read_manifest(manifest_path: Path) -> list[str]:
    """Read a manifest CSV with a `path` column (or single unheaded column).

    Relative paths are resolved against the manifest file's own directory
    and made absolute, so the resulting items list is unambiguous no
    matter what cwd `batchr run` ends up executing in.
    """
    items: list[str] = []
    with open(manifest_path, newline="") as f:
        rows = [row for row in csv.reader(f) if row and row[0].strip()]
    if not rows:
        return items

    start = 1 if rows[0][0].strip().lower() == "path" else 0
    for row in rows[start:]:
        p = Path(row[0].strip())
        if not p.is_absolute():
            p = (manifest_path.parent / p).resolve()
        items.append(str(p))
    return items


def _load_result_jsons(results_dir: Path):
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def _verdict_counts(results_dir: Path) -> Counter:
    counts: Counter = Counter()
    for result in _load_result_jsons(results_dir):
        counts[result.get("verdict", "unknown")] += 1
    return counts


def _confusion_table(results_dir: Path) -> dict[str, Counter]:
    """label -> Counter(winner_or_verdict) for targets whose npz carried a label."""
    table: dict[str, Counter] = defaultdict(Counter)
    for result in _load_result_jsons(results_dir):
        label = result.get("input", {}).get("label")
        if not label:
            continue
        outcome = result.get("winner") or result.get("verdict")
        table[label][outcome] += 1
    return table


def _cmd_all(args: argparse.Namespace) -> int:
    config = PipelineConfig.load(args.config) if args.config else PipelineConfig.load()

    results_dir = Path(args.results_dir) if args.results_dir else Path(config.results_dir)
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (results_dir / ".batchr")

    effective_config = config.as_dict()
    effective_config["results_dir"] = str(results_dir)

    manifest_path = Path(args.manifest).resolve()
    items = _read_manifest(manifest_path)
    if not items:
        print(f"No items found in manifest {manifest_path}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        items_path = Path(tmp) / "items.txt"
        items_path.write_text("\n".join(items) + "\n")

        config_path = Path(tmp) / "pipeline_config.json"
        config_path.write_text(json.dumps({"pipeline": effective_config}))

        env = os.environ.copy()
        pythonpath = str(REPO_ROOT)
        if env.get("PYTHONPATH"):
            pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = pythonpath
        env["ARVYO_PIPELINE_CONFIG"] = str(config_path)

        batchr_cmd = [
            tool_command("batchr"), "run",
            "--fn", "arvyo.batch_worker:run_one",
            "--items", str(items_path),
            "--cache-dir", str(cache_dir),
            "--serializer", "json",
            "--config", str(config_path),
        ]
        if args.workers:
            batchr_cmd += ["--workers", str(args.workers)]

        proc = subprocess.run(
            batchr_cmd, env=env, cwd=str(REPO_ROOT), text=True, capture_output=True
        )
        print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)

    counts = _verdict_counts(results_dir)
    print(
        f"results written to {results_dir}: "
        + ", ".join(f"{v}={n}" for v, n in sorted(counts.items()))
    )

    _maybe_log_trackr(manifest_path, effective_config, counts)

    return proc.returncode


def _maybe_log_trackr(manifest_path: Path, config: dict, counts: Counter) -> None:
    try:
        import trackr
    except ImportError:
        return
    try:
        trackr_config = dict(config)
        trackr_config["manifest_path"] = str(manifest_path)
        run = trackr.init(project="arvyo-pipeline", name="batch-run", config=trackr_config)
        run.log({f"verdict_{k}": float(v) for k, v in counts.items()})
        run.finish(status="completed")
    except Exception:
        # trackr is a nice-to-have; never let logging failures break the run.
        pass


def _cmd_summarize(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"{results_dir} does not exist", file=sys.stderr)
        return 1

    counts = _verdict_counts(results_dir)
    total = sum(counts.values())
    print(f"Verdicts over {total} result(s) in {results_dir}:")
    for verdict, n in sorted(counts.items()):
        print(f"  {verdict}: {n}")

    confusion = _confusion_table(results_dir)
    if confusion:
        print()
        print("Label vs. winner/verdict (labeled targets only):")
        outcomes = sorted({o for row in confusion.values() for o in row})
        print("  " + "label".ljust(10) + "".join(o.rjust(14) for o in outcomes))
        for label in sorted(confusion):
            row = confusion[label]
            print(
                "  " + label.ljust(10)
                + "".join(str(row.get(o, 0)).rjust(14) for o in outcomes)
            )

        agreeing = sum(row.get(label, 0) for label, row in confusion.items())
        labeled_total = sum(sum(row.values()) for row in confusion.values())
        if labeled_total:
            pct = 100.0 * agreeing / labeled_total
            print(f"\n  agreement with catalog label: {agreeing}/{labeled_total} ({pct:.1f}%)")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arvyo.run")
    sub = parser.add_subparsers(dest="command", required=True)

    p_one = sub.add_parser("one", help="Run the pipeline on a single .npz target")
    p_one.add_argument("path")
    p_one.add_argument("--config", default=None, help="path to a pipeline config yaml")
    p_one.add_argument(
        "--use-catalog-period", action="store_true",
        help=(
            "Skip foldr's period search and fold at the .npz's own "
            "period_days/epoch_btjd metadata instead. Useful for demoing "
            "fitr on targets with a known ephemeris."
        ),
    )
    p_one.add_argument(
        "--centroid-vet", action="store_true",
        help=(
            "Run localizr's centroid-offset check when the verdict is a "
            "planet/blend case, overriding config's centroid_vetting_enabled "
            "to on. Needs live network access to MAST/Gaia."
        ),
    )
    p_one.add_argument(
        "--no-centroid-vet", action="store_true",
        help="Force centroid vetting off, overriding config's centroid_vetting_enabled.",
    )
    p_one.set_defaults(func=_cmd_one)

    p_all = sub.add_parser("all", help="Run the pipeline over a manifest CSV via batchr")
    p_all.add_argument("manifest", help="CSV with a 'path' column (or single unheaded column)")
    p_all.add_argument("--config", default=None)
    p_all.add_argument("--results-dir", default=None)
    p_all.add_argument(
        "--cache-dir", default=None, help="batchr cache dir (default: {results_dir}/.batchr)"
    )
    p_all.add_argument("--workers", type=int, default=0)
    p_all.set_defaults(func=_cmd_all)

    p_sum = sub.add_parser("summarize", help="Verdict counts + label-vs-winner confusion table")
    p_sum.add_argument("results_dir")
    p_sum.set_defaults(func=_cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
