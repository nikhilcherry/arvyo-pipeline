#!/usr/bin/env python
"""(Re)generate tests/fixtures/fixture_*.npz with recoverable signals.

The original fixtures (200 points / ~5-day baseline) were too sparse for a
blind BLS/TLS period search to recover their injected periods, and their
period_days/epoch_btjd catalog fields weren't even in the same time frame as
the `time` array (epoch_btjd=1000.0 against a time array starting at 0) — so
tests/test_end_to_end.py had to carry a whole "these fixtures can't actually
be searched" caveat. This script fixes both: a 27.4-day baseline (one TESS
sector) at 30-minute cadence (~1315 points) gives the 3.14-day planet period
8 full transits — comfortably recoverable (>7 SDE) through foldr's real
search — and epoch_btjd is now set in the *same frame* as `time` so
`--use-catalog-period` (arvyo/run.py) folds correctly too.

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

`fixture_unknown.npz` is deliberately ambiguous by construction, not by
omission: it's broadband white noise louder than `null` with no injected
periodic signal at all, so it has no "correct" period to recover. It exists
to exercise the pipeline's low-confidence/novelty path (a real target that
looks like nothing in particular), not to be a harder version of `planet`
or `eb`.

Run: python scripts/regenerate_fixtures.py [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arvyo.synthesis.forward_models import eclipsing_binary, planet, starspot as starspot_model

FIXTURES_DIR = ROOT / "tests" / "fixtures"

CADENCE_DAYS = 30.0 / 60 / 24   # 30-minute TESS FFI cadence
BASELINE_DAYS = 27.4            # one TESS sector

# Matches fitr/models/planet.py's fixed LD_COEFFS exactly (see module
# docstring) so injected planet/eb transits don't create a spurious
# planet/blend systematic mismatch.
FITR_LD_COEFFS = [0.4, 0.25]

FIXTURE_NAMES = ["planet", "eb", "starspot", "null", "unknown"]


def _time_grid() -> np.ndarray:
    return np.arange(0.0, BASELINE_DAYS, CADENCE_DAYS)


def _rngs(seed: int) -> dict[str, np.random.Generator]:
    """Independent, reproducible RNG streams for each fixture, all derived
    from one master --seed (default 42) so the whole regeneration run is
    reproducible from a single number without the streams overlapping."""
    children = np.random.SeedSequence(seed).spawn(len(FIXTURE_NAMES))
    return {name: np.random.default_rng(child) for name, child in zip(FIXTURE_NAMES, children)}


def _red_noise(rng: np.random.Generator, n: int, *, sigma: float, phi: float = 0.98) -> np.ndarray:
    """Mild AR(1) red noise: correlated, slowly-wandering, but stationary
    (no drift/trend) — distinct from a designed periodic signal, so it must
    not trip a period search."""
    innovations = rng.normal(0.0, sigma, n)
    red = np.empty(n)
    red[0] = innovations[0]
    for i in range(1, n):
        red[i] = phi * red[i - 1] + innovations[i]
    return red


def make_planet_fixture(rng: np.random.Generator) -> dict:
    period, t0, rp = 3.14, 1.5, 0.10
    time = _time_grid()

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


def make_eb_fixture(rng: np.random.Generator) -> dict:
    period, t0, rp = 1.5, 0.4, 0.15
    time = _time_grid()

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


def make_starspot_fixture(rng: np.random.Generator) -> dict:
    prot = 6.2
    time = _time_grid()

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


def make_null_fixture(rng: np.random.Generator) -> dict:
    time = _time_grid()

    # White noise + mild red noise only — no injected signal, periodic or
    # otherwise. The red component (correlated, ~3x quieter than the white
    # floor) is what real quiet TESS targets actually look like; a search
    # must not mistake it for a period.
    flux_err_val = 0.0004
    white = rng.normal(0, flux_err_val, time.size)
    red = _red_noise(rng, time.size, sigma=flux_err_val * 0.3)
    flux = 1.0 + white + red
    flux_err = np.full(time.size, flux_err_val)

    return dict(
        time=time, flux=flux, flux_err=flux_err,
        tic_id=352471, label="null", sector=5,
    )


def make_unknown_fixture(rng: np.random.Generator) -> dict:
    time = _time_grid()

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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rngs = _rngs(args.seed)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        data = builder(rngs[name])
        out_path = FIXTURES_DIR / f"fixture_{name}.npz"
        np.savez(out_path, **data)
        size_kb = out_path.stat().st_size / 1024
        print(f"wrote {out_path.relative_to(ROOT)} ({data['time'].size} points, "
              f"{BASELINE_DAYS:.1f}-day baseline, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
