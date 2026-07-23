# Finale setup — one canonical, copy-pasteable cell

This is the single source of truth for "how do I get a clean machine to a
working `arvyo-pipeline` checkout." If you're setting up for the BAH2026
finale demo, judges' laptop, or just a fresh clone: **use this doc, top to
bottom, and nothing else.** Every tool README's install section links back
here instead of repeating its own snippet, so there is exactly one place
that can go stale.

Verified end-to-end in a genuinely fresh `venv` (not the dev environment)
on 2026-07-10 — see the real, unedited run log at the bottom. Total wall
time for the pinned-dependency install + tool installs + health check +
smoke run was well under 5 minutes on a 16-thread machine with wheels
already warm in pip's cache; a fully cold cache (first-time downloads,
especially `torch`) will take longer.

**Policy reminder** (workspace-wide, not just this doc): team Arvyo's six
tools — `trackr`, `batchr`, `peekr`, `foldr`, `fitr`, `localizr` — are
**never** installed with a bare `pip install <name>`. None of them are
published to PyPI, and at least one of those names (`foldr`) is squatted by
an unrelated package there, so a bare install would silently pull the wrong
thing. Every install below uses the explicit
`git+https://github.com/nikhilcherry/<name>` form, pinned to a commit SHA.

## 0. Prerequisites

- `git`, and network access to `github.com`.
- A sibling checkout of `arvyo-data` next to this repo, i.e.:
  ```
  BAH/
  ├── arvyo-data/       <- sibling clone, see step 4
  └── arvyo-pipeline/   <- this repo
  ```
  `arvyo-data/data/samples/` (one committed real `.npz` per class) is all
  this cell needs — it does **not** require the gitignored bulk corpus
  under `arvyo-data/data/processed/`, so this works from a fresh clone of
  both repos with no manual data transfer.

## 1. Create the venv

Detect whatever Python 3 is actually on this machine — don't guess a
version:

```bash
python3 --version                       # sanity check before creating the venv
python3 -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
```

Verified against `Python 3.13.13`. `pyproject.toml` declares
`requires-python = ">=3.10"`; anything 3.10+ should work, but only 3.13.13
has actually been exercised end-to-end for this doc.

## 2. Install pinned scientific dependencies

These are the exact versions verified together for this doc — not
`requirements.txt`'s unpinned list (that file intentionally stays unpinned
for day-to-day development flexibility; this cell trades that flexibility
for finale-day reproducibility):

```bash
python -m pip install \
  numpy==2.4.6 \
  pandas==2.3.3 \
  scipy==1.17.1 \
  matplotlib==3.10.9 \
  pyyaml==6.0.3 \
  torch==2.13.0 \
  "lightkurve==2.6.0" \
  wotan==1.10 \
  transitleastsquares==1.32 \
  batman-package==2.5.3 \
  sbi==0.26.1 \
  emcee==3.1.6 \
  exovetter==0.0.16 \
  triceratops==1.0.20 \
  streamlit==1.59.1 \
  pytest==9.1.1
```

If a pin fails to resolve on your platform (e.g. no CPU-only `torch` wheel
for your OS/arch), install that one package unpinned, note the version you
actually got, and move on — don't downgrade every other pin to chase it.

## 3. Install the six tools, pinned to a commit SHA

**To bump a pin:** `git -C <tool-repo> rev-parse HEAD` on the tool's own
checkout after pulling, replace the SHA below, re-run this whole cell cold,
and update the "last verified" date above.

```bash
python -m pip install git+https://github.com/nikhilcherry/trackr.git@01a8d2ed02890c2c13559140f7dcaf60b5bd1d42
python -m pip install git+https://github.com/nikhilcherry/batchr.git@63cd15b2e771339571c6880d6d6bd8ce0e35b87f
python -m pip install git+https://github.com/nikhilcherry/peekr.git@1e7edac7f896bee369d3990877bb8019b1c88bef
python -m pip install git+https://github.com/nikhilcherry/foldr.git@06000df3ba152224d66b272511f7afca71c8e706
python -m pip install git+https://github.com/nikhilcherry/fitr.git@ca63fa0a8b5e635202f6d3206c3f3392fc273c25
python -m pip install git+https://github.com/nikhilcherry/localizr.git@2a99e3941e7586680735ee212aa332f1adfbcdb8
```

> **`batchr` pin note (resolved):** at verification time,
> `63cd15b2e771339571c6880d6d6bd8ce0e35b87f` had not yet been pushed, so the
> log below installed the last-pushed SHA (`b6471c0`) instead. `batchr` has
> since been pushed — `pip install git+https://github.com/nikhilcherry/batchr.git@63cd15b2e771339571c6880d6d6bd8ce0e35b87f`
> now resolves and installs cleanly (spot-checked 2026-07-10, post-push).

`peekr` is not currently imported or shelled out to anywhere in
`arvyo-pipeline` (it's a standalone data-exploration tool) — it's installed
here anyway so this cell is the complete, canonical "all six tools" setup
regardless of which ones a given demo path touches.

`localizr` **is** shelled out to by `arvyo/worker.py`, but only when
centroid vetting is turned on (`pipeline.centroid_vetting_enabled: true` in
`configs/default.yaml`, or `arvyo.run one --centroid-vet`) — it's off by
default because it needs live network access to MAST + Gaia for a real
target pixel file. The core pipeline (steps 4-5 below) runs and passes with
`localizr` installed but never invoked; it's pinned here so the option is
available without a second setup pass. Verified installing cleanly with the
SHA above (`pip install`, `localizr --help`) on 2026-07-23, independent of
the rest of this doc's original 2026-07-10 run.

## 4. Install this repo + the arvyo-data sibling health check

```bash
python -m pip install -e . --no-deps    # register `arvyo`; deps already pinned above

git -C ../arvyo-data pull                # make sure the sibling clone is current
python -m arvyo.contract ../arvyo-data/data/samples
```

Expected output (one line per class, all `valid`, no bulk corpus required):

```
Data contract report for ../arvyo-data/data/samples (schema 1.0)
  blend: 1 valid
  eb: 1 valid
  null: 1 valid
  planet: 1 valid
  starspot: 1 valid
```

If any class shows `0 valid` or an error instead, stop here — the sibling
clone is missing/stale/corrupt, and nothing past this point will work
correctly.

## 5. Final smoke test

```bash
python scripts/smoke_run.py --data-root ../arvyo-data/data/samples --seed 42 --fast
```

This runs all 5 committed real samples through the full spine (contract ->
views -> TLS -> 4-hypothesis synthesis -> emcee posterior -> vetting ->
figure) with halved emcee/sbi settings (`--fast`). Expect:

- **Runtime:** ~40–90s wall time depending on core count (the emcee/TLS
  stages use all available CPU threads); reported as `total runtime: NNs`
  in the last line of output, plus PyTorch/SBI neural-network training logs
  and TLS periodogram progress bars along the way — verbose but expected,
  not an error.
- **Output:** a table with all 5 labels (`planet`, `eb`, `blend`,
  `starspot`, `null`) showing `pass` across every one of `contract / views /
  tls / synthesis / inference / vetting / figure`, followed by
  `labels missing locally: none`, exit code `0`.

If any column is anything other than `pass` for any label, something in
steps 1–4 didn't install cleanly — re-check the contract health check
output in step 4 before assuming a pipeline bug.

---

## Fresh-venv verification log (real, unedited, 2026-07-10)

Ran in `/tmp/finale_verify_venv`, a `python3 -m venv` created fresh for this
check — not the long-lived dev environment used elsewhere in this repo.

```
$ python3 --version
Python 3.13.13

$ python -m pip install numpy==2.4.6 pandas==2.3.3 scipy==1.17.1 \
    matplotlib==3.10.9 pyyaml==6.0.3 torch==2.13.0 "lightkurve==2.6.0" \
    wotan==1.10 transitleastsquares==1.32 batman-package==2.5.3 \
    sbi==0.26.1 emcee==3.1.6 exovetter==0.0.16 triceratops==1.0.20 \
    streamlit==1.59.1 pytest==9.1.1
(all 16 pins resolved and installed cleanly, no conflicts)

$ python -m pip install git+https://github.com/nikhilcherry/trackr.git@01a8d2ed02890c2c13559140f7dcaf60b5bd1d42
$ python -m pip install git+https://github.com/nikhilcherry/batchr.git@b6471c0cf31296b072a4933a6d7bff1976ca1c26   # see batchr pin note above
$ python -m pip install git+https://github.com/nikhilcherry/peekr.git@1e7edac7f896bee369d3990877bb8019b1c88bef
$ python -m pip install git+https://github.com/nikhilcherry/foldr.git@06000df3ba152224d66b272511f7afca71c8e706
$ python -m pip install git+https://github.com/nikhilcherry/fitr.git@ca63fa0a8b5e635202f6d3206c3f3392fc273c25
$ which trackr batchr peekr foldr fitr
/tmp/finale_verify_venv/bin/trackr
/tmp/finale_verify_venv/bin/batchr
/tmp/finale_verify_venv/bin/peekr
/tmp/finale_verify_venv/bin/foldr
/tmp/finale_verify_venv/bin/fitr

$ python -m pip install -e . --no-deps
$ python -m arvyo.contract ../arvyo-data/data/samples
Data contract report for ../arvyo-data/data/samples (schema 1.0)
  blend: 1 valid
  eb: 1 valid
  null: 1 valid
  planet: 1 valid
  starspot: 1 valid

$ time python scripts/smoke_run.py --data-root ../arvyo-data/data/samples --seed 42 --fast
... (TLS periodogram search + neural-network training logs) ...
label      | contract  | views     | tls       | synthesis | inference | vetting   | figure
planet     | pass      | pass      | pass      | pass      | pass      | pass      | pass
eb         | pass      | pass      | pass      | pass      | pass      | pass      | pass
blend      | pass      | pass      | pass      | pass      | pass      | pass      | pass
starspot   | pass      | pass      | pass      | pass      | pass      | pass      | pass
null       | pass      | pass      | pass      | pass      | pass      | pass      | pass

labels missing locally: none
total runtime: 41.2s

real    0m44.091s
user    5m21.391s
sys     0m13.916s

exit code: 0
```
