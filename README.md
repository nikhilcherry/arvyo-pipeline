# arvyo-pipeline

Team Arvyo's BAH2026 exoplanet transit-detection model/analysis repo:
analysis-by-synthesis over TESS/Kepler light curves — search for transit
candidates, generate competing forward-model hypotheses (planet / EB /
blend / starspot), and fit + vet each one.

This repo is SEPARATE from `arvyo-data` (the dataset builder). It never
writes into `arvyo-data/`; it only reads processed `.npz` files via the
configurable `data.processed_root` path in `configs/default.yaml`
(default `../arvyo-data/data/processed`).

## Data Contract

The `.npz` schema produced by `arvyo-data` is the ONLY interface between
the two repos, frozen in `arvyo/contract.py`:

```python
SCHEMA_VERSION = "1.0"

REQUIRED_ARRAYS = ["time", "flux", "flux_err"]
OPTIONAL_ARRAYS = ["flux_raw"]          # present for starspot/null classes
REQUIRED_META = ["tic_id", "label", "sector"]
OPTIONAL_META = ["period_days", "epoch_btjd", "crowdsap", "mission",
                 "augmented", "injection_params"]
LABELS = ["planet", "eb", "blend", "starspot", "null", "unknown"]
```

**Any change to this schema requires bumping `SCHEMA_VERSION` and updating
BOTH repos' READMEs.**

Validate a processed data directory with:

```bash
python -m arvyo.contract /path/to/processed
```

## Repo layout

```
arvyo-pipeline/
├── configs/default.yaml     # data paths, schema version, hyperparams
├── arvyo/
│   ├── contract.py          # the frozen .npz schema: loader + validator
│   ├── data/dataset.py      # PyTorch Dataset over arvyo-data .npz files
│   ├── views/views.py       # global/local/secondary view generation
│   ├── search/tls_search.py # TLS wrapper + MES/SDE extraction
│   ├── models/              # dual-branch 1D CNN encoder + classifier/novelty heads
│   ├── synthesis/           # batman x4 hypothesis forward models
│   ├── inference/           # emcee (primary) + sbi (fallback) posteriors
│   ├── vetting/vet.py       # odd-even + secondary-eclipse checks
│   └── viz/plots.py         # phase-fold + model-fit + residual figure
├── app/dashboard.py         # Streamlit skeleton
├── benchmarks/              # how to run ExoMiner + triceratops (comparison oracles)
├── third_party/             # vendored code, license-gated (see below)
└── tests/                   # pytest: contract, views, end-to-end smoke
```

The glue layer (see below) lives alongside the above in `arvyo/`:
`worker.py` (per-target foldr->fitr), `batch_worker.py` (batchr's
per-item entry point), `run.py` (`python -m arvyo.run` CLI),
`pipeline_config.py`, `result_schema.py`, `_toolchain.py`.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # or: pip install -r requirements.txt --break-system-packages

python -m arvyo.contract ../arvyo-data/data/processed   # validate data
pytest                                                    # run tests
streamlit run app/dashboard.py                            # explore a sample
```

## Glue layer: foldr -> fitr -> batchr

`arvyo/worker.py`, `arvyo/batch_worker.py`, and `arvyo/run.py` are pure
orchestration — they never import `foldr`/`fitr` internals, only invoke
their published CLIs and parse stdout JSON + exit codes, since those
interfaces (like the `.npz` contract above) are frozen. Per target:

1. `arvyo.contract.load_sample` validates the `.npz` against the schema above.
2. **foldr** runs a period search (`foldr FILE.npz --json --no-plot`). Its
   `sde`/`snr` are gated against `pipeline.sde_min`/`pipeline.snr_min` in
   `configs/default.yaml` (default 7.0, pass if *either* is met). If the
   gate fails, the target short-circuits to verdict `no_period` — **fitr
   is not run**.
3. **fitr** fits all 4 forward models at foldr's candidate period/epoch
   (`fitr fit FILE.npz --period P --epoch T0 --json`) and its exit code
   becomes the verdict: `0`→`clear`, `3`→`ambiguous`, `4`→`no_significant_signal`,
   anything else (or unparseable stdout) → `error`. fitr's JSON is embedded
   verbatim under `model_fit` — never reshaped.
4. **batchr** (`arvyo.run all`) drives this over a manifest with caching/resume;
   **trackr**, if installed, gets a best-effort run summary (optional, never
   fails the run).

Install the pipeline tools (kept out of the base deps to stay lean):

```bash
pip install -e ".[pipeline]"
```

### Quickstart

```bash
# single target -> one JSON result dict on stdout
python -m arvyo.run one tests/fixtures/fixture_planet.npz | python -m json.tool | head -30

# full suite, including the e2e glue-layer tests (skipped automatically
# if foldr/fitr/batchr aren't installed); use `-m "not e2e"` to skip them
# explicitly for a fast run
pytest tests/ -q

# bulk run over a manifest CSV (a `path` column, or one path per line),
# resumable via batchr's content-hash cache
python -m arvyo.run all manifest.csv --results-dir results/
python -m arvyo.run summarize results/
```

### Result JSON schema (v1.0)

Frozen in `arvyo/result_schema.py`; every target produces exactly one of
these, even on tool failure/timeout/bad input — `process_target` never
raises for tool-level errors, only for programmer errors (bad `config`).

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | `"1.0"` |
| `input` | dict | `path`, `tic_id`, `label`, `sector` from the `.npz` |
| `period_search` | dict \| null | foldr's output: `engine`, `period`, `t0`, `duration_hours`, `depth_ppm`, `snr`, `sde`, `passed_gate`; `null` if foldr never ran |
| `model_fit` | dict \| null | fitr's JSON, embedded verbatim; `null` unless verdict is `clear`/`ambiguous`/`no_significant_signal` |
| `verdict` | str | see vocabulary below |
| `winner` | str \| null | fitr's winning model, only set when `verdict == "clear"` |
| `error` | dict \| null | `{"stage": ..., "message": ...}`; `stage` is one of `contract_validation`, `period_search`, `model_fit` |
| `runtime_s` | dict | `foldr`, `fitr`, `total` wall-clock seconds |
| `versions` | dict | `foldr`, `fitr`, `arvyo_pipeline` version strings |

Verdict vocabulary:

| Verdict | Meaning |
|---|---|
| `clear` | fitr found one model that fits significantly better than the rest (exit 0) |
| `ambiguous` | fitr couldn't separate 2+ models (exit 3) — see `model_fit.tied_models` |
| `no_significant_signal` | fitr fit the models but none is a significant improvement over flat (exit 4) |
| `no_period` | foldr's `sde`/`snr` didn't pass the configured gate — fitr never ran |
| `error` | contract validation, or a tool crash/timeout/unparseable-output failure; see `error.stage`/`error.message` |

Note: the tiny 200-point/5-day-baseline fixtures in `tests/fixtures/` are
shared with the contract/schema tests and are too sparse for a blind
period search to recover their injected periods — expect `no_period` /
`no_significant_signal` from them even for planet/eb/starspot. This is a
fixture-size limitation, not a pipeline bug; `tests/test_end_to_end.py`
also injects realistic higher-fidelity transits to prove the `clear`
happy path recovers the right class end-to-end.

### batchr PYTHONPATH workaround

batchr's `--fn module:function` importer (`batchr/cli.py::_import_fn`)
only adds the *current working directory* to `sys.path` before importing
the worker module — it does not use the caller's own `sys.path`. So if
`arvyo` isn't already importable from wherever `batchr run` happens to
execute (e.g. it isn't pip-installed), resolving `arvyo.batch_worker` raises
`ModuleNotFoundError`. `arvyo.run all` works around this by setting
`PYTHONPATH` to the repo root (and running with `cwd` set to the repo
root) before shelling out to `batchr run`, so `arvyo.batch_worker:run_one`
is always resolvable regardless of install state or invocation cwd. Also
note: batchr's `--config` only feeds its cache-key hash, it's never passed
to the worker function itself — the pipeline config each worker process
should use is threaded through separately via the `ARVYO_PIPELINE_CONFIG`
env var, which forked worker processes inherit.

## Vendored components

| Component | Source | License | What / why |
|---|---|---|---|
| `third_party/nigraha_views/preprocess.py` | github.com/ExoplanetML/Nigraha @ `c4365b41` | MIT | Reference for global/local view generation (201/81-bin median folding) and ±1.5×duration in-transit masking during detrend. Adapted (not copied) into `arvyo/views/views.py`. |

Skipped: Astronet-Triage's TCE label CSVs (GPLv3 — not vendored; see
`third_party/SKIPPED.md`), and `exoplanet-ml`/AstroNet view utils (not
needed — Nigraha's sufficed). Both are documented in
`third_party/SKIPPED.md`.

## Reference repos

- **ExoMiner** (Valizadegan et al.) — comparison oracle; see `benchmarks/README.md` for running its container against a TIC list.
- **Nigraha** (Rao 2021) — source of the vendored view-generation logic (`third_party/nigraha_views/`).
- **Astronet-Triage / Astronet-Vetting** (Yu et al.) — TCE triage/vetting reference; its label CSVs were GPL-blocked from vendoring, see `third_party/SKIPPED.md`.
- **TESS-ExoClass** — reference for TESS-specific TCE classification/vetting heuristics informing `arvyo/vetting/vet.py`.
- **DAVE** (Data Validation Explorer) — reference for interactive vetting-plot conventions informing `arvyo/viz/plots.py`.
- **Astromer / StarCLR** — candidate self-supervised light-curve backbones; see Open Questions below.

## Open questions

- **SSL backbone core-vs-stretch** pending hackathon ruling on pre-trained
  weights — decides whether `arvyo/models/` gains a SimCLR trainer or an
  Astromer fine-tune wrapper.
