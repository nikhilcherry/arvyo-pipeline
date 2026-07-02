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

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # or: pip install -r requirements.txt --break-system-packages

python -m arvyo.contract ../arvyo-data/data/processed   # validate data
pytest                                                    # run tests
streamlit run app/dashboard.py                            # explore a sample
```

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
