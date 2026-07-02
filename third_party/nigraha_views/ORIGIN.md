# Origin

- **Repo:** https://github.com/ExoplanetML/Nigraha
- **Commit:** c4365b41dd02b187c3210189ffe8e3ead584f4f5
- **Date vendored:** 2026-07-02
- **License:** MIT (Copyright (c) 2021 Sriram Rao) — see `LICENSE` in this directory, copied verbatim.
- **Files taken:**
  - `data/preprocess.py` -> `preprocess.py`
    (contains `get_folded_lightcurve`, `process_lightcurve`, `build_halfphase_views`:
    in-transit masking at ±1.5x transit duration during detrend, global view
    (201-bin median fold), local view (81-bin median fold over ±2 transit
    durations), odd/even half-phase views.)
- **Local modifications:** none. This file is kept as reference material only;
  it is NOT imported by `arvyo/`. The runnable adaptation lives in
  `arvyo/views/views.py`, credited via docstring back to this commit.
