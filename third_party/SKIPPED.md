# Skipped vendoring targets

## 2. Astronet-Triage (github.com/yuliang419/Astronet-Triage) — TCE CSV label files

- **Commit checked:** 5675a57dd41dd0321df480453451096dc5a4a6b0
- **License found:** GNU GPLv3 (repo-wide `LICENSE`, no separate license on `astronet/tces.csv`).
- **Decision:** DO NOT VENDOR. GPL/AGPL/LGPL is excluded by ground rule #4 even
  for what is nominally a data file, since it ships under the repo's GPLv3
  terms and there's no independent license grant on the CSV itself.
- **Alternative:** the TCE label data isn't itself an algorithm to
  reimplement — `arvyo`'s own `contract.py` schema (`LABELS` in
  `arvyo/contract.py`) defines label taxonomy independently, and TCE
  candidates from arvyo-data / your own TLS searches (`arvyo/search/tls_search.py`)
  serve the same role this CSV would have. No functional gap.

## 3. exoplanet-ml / AstroNet (github.com/google-research/exoplanet-ml) — median-binning / view utils

- **Decision:** NOT CLONED. Per the handout, this was only a fallback "IF
  Nigraha's [view utils] are insufficient." Nigraha's `data/preprocess.py`
  (vendored at `third_party/nigraha_views/`, MIT license) already provides
  working global-view (201-bin median fold) and local-view (81-bin median
  fold over ±2 transit durations) generation matching the paper description,
  so this fallback was not needed. No license issue — just unnecessary.
