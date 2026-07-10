#!/usr/bin/env python
"""Convert frozen-contract .npz fixtures into arvyo-wiz's sample JSON format.

Sources (see README notes in both repos for why):
  planet/eb/starspot/null <- tests/fixtures/fixture_*.npz (this repo)
  blend                   <- ../arvyo-data/data/samples/blend/4281068.npz
                             (KIC 4281068 / KOI K07689.01, a real Robovetter
                             koi_fpflag_co==1 centroid-offset false positive,
                             deterministically selected by arvyo-data's
                             scripts/08_select_blend_sample.py catalog filter
                             for being the top-SNR candidate that also clears
                             arvyo-pipeline's TLS SDE>=7.0 detection gate
                             (SDE ~18.9) — arvyo-data's TESS `blend` class is
                             synthetic-only, so this is the one real example
                             available. Pulled from arvyo-data's committed
                             data/samples/ (not the gitignored bulk
                             data/kepler/processed/ pool) so this script
                             reproduces from a fresh clone of both repos.

Each output JSON feeds arvyo-wiz's VerdictPanel/GraphDetail/LightCurvePanel
directly. The light curve is phase-folded (true orbital phase when
period_days/epoch_btjd are known, else a centered/normalized time axis) and
decimated to <=200 points. Classifier confidence, per-model deltaLogZ, and
SBI posterior params are hand-authored illustrative values, not run through
the real classifier/SBI stack — same spirit as arvyo-wiz's physics.js
`verdictFor`, which is explicitly labeled "SIMULATED VERDICT".
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arvyo.contract import load_sample
FIXTURES_DIR = ROOT / "tests" / "fixtures"
BLEND_SOURCE = ROOT.parent / "arvyo-data" / "data" / "samples" / "blend" / "4281068.npz"
OUT_DIR = ROOT.parent / "arvyo-wiz" / "src" / "data" / "samples"

MAX_POINTS = 200

MODEL_DEFS = [
    ("TRANSIT", "planet"),
    ("ECL.BINARY", "eb"),
    ("BLEND", "blend"),
    ("STARSPOT", "starspot"),
]

# Hand-authored illustrative verdict + SBI-posterior blocks, one per class.
# Not computed from real inference — see module docstring.
VERDICT_BY_CLASS = {
    "planet": {"confidence": 0.96, "scores": {"planet": 0.0, "eb": -145.2, "blend": -58.3, "starspot": -310.5}},
    "eb": {"confidence": 0.93, "scores": {"eb": 0.0, "planet": -180.4, "blend": -42.1, "starspot": -260.0}},
    "blend": {"confidence": 0.71, "scores": {"blend": 0.0, "eb": -12.6, "planet": -35.0, "starspot": -190.0}},
    "starspot": {"confidence": 0.89, "scores": {"starspot": 0.0, "planet": -95.0, "eb": -110.0, "blend": -88.0}},
    "null": {"confidence": 0.12, "scores": {"planet": -18.4, "eb": -21.1, "blend": -19.7, "starspot": -24.5}},
}

SBI_EXTRA_BY_CLASS = {
    # starspot has no orbiting companion, so its posterior is over
    # rotation/spot-morphology params instead of transit geometry.
    "starspot": {
        "rotationDays": {"p16": 11.2, "p50": 11.8, "p84": 12.5},
        "spotCoverage": {"p16": 0.24, "p50": 0.28, "p84": 0.33},
        "spotContrast": {"p16": 0.66, "p50": 0.72, "p84": 0.78},
    },
    "null": {},
}


def build_verdict(cls):
    spec = VERDICT_BY_CLASS[cls]
    models = [
        {"name": name, "modelClass": model_class, "deltaLogZ": spec["scores"][model_class],
         "win": spec["scores"][model_class] == 0.0}
        for name, model_class in MODEL_DEFS
    ]
    return {"confidence": spec["confidence"], "models": models}


def decimate(length, max_points=MAX_POINTS):
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.round(np.linspace(0, length - 1, max_points)).astype(int))


def compute_phase(time, period_days, epoch_btjd):
    """True orbital phase in [-0.5, 0.5) when period info is known, else a
    centered/baseline-normalized time axis (still roughly [-0.5, 0.5]) so
    GraphDetail's default view bounds work unchanged for every sample."""
    if period_days and epoch_btjd is not None:
        phase = ((time - epoch_btjd) / period_days + 0.5) % 1.0 - 0.5
        folded = True
    else:
        span = time.max() - time.min()
        center = (time.max() + time.min()) / 2.0
        phase = (time - center) / span if span > 0 else np.zeros_like(time)
        folded = False
    order = np.argsort(phase)
    return phase[order], order, folded


def estimate_transit_params(phase, flux):
    """Rough Rp/Rs estimate from observed depth, just to keep the
    illustrative SBI posterior in the same ballpark as the real curve."""
    depth = max(0.0, 1.0 - float(np.min(flux)))
    rp_rs = max(0.02, min(0.5, np.sqrt(depth)))
    return rp_rs


def build_sbi_params(cls, sample, phase, flux):
    if cls == "starspot" or cls == "null":
        return SBI_EXTRA_BY_CLASS.get(cls, {})

    period_days = sample.get("period_days")
    rp_rs = estimate_transit_params(phase, flux)
    params = {
        "radiusRatio": {"p16": round(rp_rs * 0.93, 4), "p50": round(rp_rs, 4), "p84": round(rp_rs * 1.07, 4)},
        "aOverRs": {"p16": 6.7, "p50": 7.0, "p84": 7.3} if cls != "eb" else {"p16": 5.2, "p50": 5.5, "p84": 5.8},
        "inclinationDeg": {"p16": 88.6, "p50": 89.1, "p84": 89.6} if cls != "eb" else {"p16": 83.6, "p50": 84.1, "p84": 84.6},
    }
    if period_days:
        jitter = period_days * 1e-4
        params["periodDays"] = {
            "p16": round(period_days - jitter, 6),
            "p50": round(period_days, 6),
            "p84": round(period_days + jitter, 6),
        }
    return params


def convert(path, cls, sample_id, source_repo, source_path, mission_default):
    sample = load_sample(path)
    time = sample["time"]
    flux = sample["flux"]
    flux_err = sample["flux_err"]

    keep = decimate(len(time))
    time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    period_days = sample.get("period_days")
    epoch_btjd = sample.get("epoch_btjd")
    phase, order, folded = compute_phase(time, period_days, epoch_btjd)
    flux, flux_err = flux[order], flux_err[order]

    verdict = build_verdict(cls)
    sbi_params = build_sbi_params(cls, sample, phase, flux)

    out = {
        "sampleId": sample_id,
        "class": cls,
        "source": {
            "repo": source_repo,
            "path": source_path,
            "ticId": int(sample["tic_id"]),
            "sector": int(sample["sector"]),
            "mission": sample.get("mission", mission_default),
        },
        "meta": {
            "periodDays": period_days,
            "epochBtjd": epoch_btjd,
            "crowdsap": sample.get("crowdsap"),
            "folded": folded,
        },
        "lightCurve": {
            "phase": [round(float(p), 6) for p in phase],
            "flux": [round(float(f), 6) for f in flux],
            "fluxErr": [round(float(e), 6) for e in flux_err],
        },
        "verdict": verdict,
        "sbiParams": sbi_params,
    }
    return out


SPECS = [
    dict(path=FIXTURES_DIR / "fixture_planet.npz", cls="planet",
         sample_id="pipeline-planet-300494", source_repo="arvyo-pipeline",
         source_path="tests/fixtures/fixture_planet.npz", mission_default="TESS"),
    dict(path=FIXTURES_DIR / "fixture_eb.npz", cls="eb",
         sample_id="pipeline-eb-999954", source_repo="arvyo-pipeline",
         source_path="tests/fixtures/fixture_eb.npz", mission_default="TESS"),
    dict(path=FIXTURES_DIR / "fixture_starspot.npz", cls="starspot",
         sample_id="pipeline-starspot-116797", source_repo="arvyo-pipeline",
         source_path="tests/fixtures/fixture_starspot.npz", mission_default="TESS"),
    dict(path=FIXTURES_DIR / "fixture_null.npz", cls="null",
         sample_id="pipeline-null-352471", source_repo="arvyo-pipeline",
         source_path="tests/fixtures/fixture_null.npz", mission_default="TESS"),
    dict(path=BLEND_SOURCE, cls="blend",
         sample_id="pipeline-blend-4281068", source_repo="arvyo-data",
         source_path="data/samples/blend/4281068.npz", mission_default="Kepler"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        out = convert(**spec)
        out_path = OUT_DIR / f"{spec['cls']}.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {out_path.relative_to(OUT_DIR.parent.parent)} "
              f"({len(out['lightCurve']['phase'])} pts, tic_id={out['source']['ticId']})")


if __name__ == "__main__":
    main()
