#!/usr/bin/env python
"""(Re)generate tests/fixtures/fixture_*.npz with recoverable signals.

The original fixtures (200 points / ~5-day baseline) were too sparse for a
blind BLS/TLS period search to recover their injected periods, and their
period_days/epoch_btjd catalog fields weren't even in the same time frame as
the `time` array (epoch_btjd=1000.0 against a time array starting at 0) — so
tests/test_end_to_end.py had to carry a whole "these fixtures can't actually
be searched" caveat. This script fixes both: a ~27-day baseline (one TESS
sector) at 10-minute cadence gives a 3.14-day planet period 8+ transits
(comfortably >7 SDE/SNR through foldr's real search), and epoch_btjd is now
set in the *same frame* as `time` so `--use-catalog-period` (arvyo/run.py)
folds correctly too.

Load-bearing detail: fitr's PlanetModel/BlendModel fit with FIXED
limb-darkening coefficients u=[0.4, 0.25] (fitr/models/planet.py) — they are
not free parameters. If you inject a transit with *different* LD coefficients
(e.g. forward_models.planet's own default [0.3, 0.2]), fitr's planet fit
carries a small but systematic shape residual that its `blend` model can
partially absorb via its extra free `dilution` parameter, and that
model-mismatch signal easily survives blend's ΔBIC penalty at these point
counts — verdict stays "clear" but winner is spuriously "blend" instead of
"planet" (this is exactly what an earlier version of this script did;
verified empirically). Passing u=[0.4, 0.25] to eliminate that systematic
mismatch is what makes `winner == "planet"` a reliable outcome instead of a
coin flip against the model's own degenerate blend alternative.

Run: python scripts/generate_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arvyo.synthesis.forward_models import eclipsing_binary, planet, starspot as starspot_model

FIXTURES_DIR = ROOT / "tests" / "fixtures"

CADENCE_DAYS = 10.0 / 60 / 24   # 10-minute TESS FFI-like cadence
BASELINE_DAYS = 27.0            # ~one TESS sector

# Matches fitr/models/planet.py's fixed LD_COEFFS exactly (see module
# docstring) so injected planet/eb transits don't create a spurious
# planet/blend systematic mismatch.
FITR_LD_COEFFS = [0.4, 0.25]


def _time_grid(seed_jitter: float = 0.0) -> np.ndarray:
    return np.arange(0.0, BASELINE_DAYS, CADENCE_DAYS) + seed_jitter


def make_planet_fixture() -> dict:
    period, t0, rp = 3.14, 1.5, 0.10
    time = _time_grid()
    rng = np.random.default_rng(300494)

    params = {
        "period": period, "rp": rp, "t0": t0, "a": 20.0, "inc": 89.0,
        "u": FITR_LD_COEFFS,
    }
    flux_err_val = 0.0003
    flux = planet(params, time) + rng.normal(0, flux_err_val, time.size)
    flux_err = np.full(time.size, flux_err_val)

    return dict(
        time=time, flux=flux, flux_err=flux_err,
        tic_id=300494, label="planet", sector=5,
        period_days=period, epoch_btjd=t0, crowdsap=0.9,
    )


def make_eb_fixture() -> dict:
    period, t0, rp = 1.5, 0.4, 0.15
    time = _time_grid()
    rng = np.random.default_rng(999954)

    params = {
        "period": period, "rp": rp, "t0": t0, "a": 8.0, "inc": 86.0,
        "secondary_scale": 0.3, "u": FITR_LD_COEFFS,
    }
    flux_err_val = 0.0004
    flux = eclipsing_binary(params, time) + rng.normal(0, flux_err_val, time.size)
    flux_err = np.full(time.size, flux_err_val)

    return dict(
        time=time, flux=flux, flux_err=flux_err,
        tic_id=999954, label="eb", sector=5,
        period_days=period, epoch_btjd=t0, crowdsap=0.85,
    )


def make_starspot_fixture() -> dict:
    prot = 6.2
    time = _time_grid()
    rng = np.random.default_rng(116797)

    params = {"prot": prot, "amp1": 0.012, "amp2": 0.004, "phase1": 0.7, "phase2": 1.9}
    signal = starspot_model(params, time)

    flux_err_val = 0.0004
    flux = signal + rng.normal(0, flux_err_val, time.size)
    # flux_raw: the pre-detrend curve with an added slow instrumental drift,
    # never read by foldr/fitr (contract.py's OPTIONAL_ARRAYS) but exercised
    # by arvyo's own view/detrend code paths downstream.
    drift = 0.002 * (time / BASELINE_DAYS)
    flux_raw = signal + drift + rng.normal(0, flux_err_val, time.size)
    flux_err = np.full(time.size, flux_err_val)

    return dict(
        time=time, flux=flux, flux_err=flux_err, flux_raw=flux_raw,
        tic_id=116797, label="starspot", sector=5,
    )


def make_null_fixture() -> dict:
    time = _time_grid()
    rng = np.random.default_rng(352471)

    flux_err_val = 0.0004
    flux = 1.0 + rng.normal(0, flux_err_val, time.size)
    flux_err = np.full(time.size, flux_err_val)

    return dict(
        time=time, flux=flux, flux_err=flux_err,
        tic_id=352471, label="null", sector=5,
    )


def make_unknown_fixture() -> dict:
    time = _time_grid()
    rng = np.random.default_rng(933939)

    # Deliberately ambiguous: broadband noise a bit louder than `null`, no
    # designed periodic signal — this class exists to exercise the
    # low-confidence/novelty path, not a specific recoverable period.
    flux_err_val = 0.0006
    flux = 1.0 + rng.normal(0, flux_err_val, time.size)
    flux_err = np.full(time.size, flux_err_val)

    return dict(
        time=time, flux=flux, flux_err=flux_err,
        tic_id=933939, label="unknown", sector=5, mission="TESS",
    )


BUILDERS = {
    "planet": make_planet_fixture,
    "eb": make_eb_fixture,
    "starspot": make_starspot_fixture,
    "null": make_null_fixture,
    "unknown": make_unknown_fixture,
}


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        data = builder()
        out_path = FIXTURES_DIR / f"fixture_{name}.npz"
        np.savez(out_path, **data)
        print(f"wrote {out_path.relative_to(ROOT)} ({data['time'].size} points, "
              f"{BASELINE_DAYS:.0f}-day baseline)")


if __name__ == "__main__":
    main()
