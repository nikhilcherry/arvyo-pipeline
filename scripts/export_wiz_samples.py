#!/usr/bin/env python
"""Convert frozen-contract .npz samples into arvyo-wiz's sample JSON format.

Sources: ALL five classes now come from arvyo-data's committed real targets
under ../arvyo-data/data/samples/<label>/*.npz (see that repo's
data/samples/PROVENANCE.md for how each one was selected/exported). This
script no longer reads tests/fixtures/ at all — those are arvyo-pipeline's
own synthetic unit-test fixtures (see tests/make_fixtures.py /
scripts/regenerate_fixtures.py), a different concern from "what does
arvyo-wiz show a user," and `blend` in particular was already real-data-only
(arvyo-data's synthetic `blend` generation isn't a thing; the one real
example is KIC 4281068 / KOI K07689.01, a Robovetter koi_fpflag_co==1
centroid-offset false positive that clears arvyo-pipeline's TLS SDE>=7.0
gate at SDE~18.9 — see arvyo-data's scripts/08_select_blend_sample.py).
Routing planet/eb/starspot/null onto arvyo-data's real samples the same way
just makes all five classes consistent, and reproducible from a fresh clone
of both repos without the gitignored bulk corpus.

Each output JSON feeds arvyo-wiz's VerdictPanel/GraphDetail/LightCurvePanel
directly. The light curve is phase-folded (true orbital phase when
period_days/epoch_btjd are known and finite, else a centered/normalized time
axis) and decimated to <=200 points. Classifier confidence, per-model
deltaLogZ, and SBI posterior params are hand-authored illustrative values,
not run through the real classifier/SBI stack — same spirit as arvyo-wiz's
physics.js `verdictFor`, which is explicitly labeled "SIMULATED VERDICT".
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arvyo.contract import load_sample

DATA_ROOT = ROOT.parent / "arvyo-data"
DATA_SAMPLES_DIR = DATA_ROOT / "data" / "samples"
OUT_DIR = ROOT.parent / "arvyo-wiz" / "src" / "data" / "samples"

CLASSES = ["planet", "eb", "blend", "starspot", "null"]

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


def _finite(x):
    """True only for a real, finite number. Real arvyo-data samples store
    period_days/epoch_btjd as NaN (not a missing key) when unknown — e.g.
    starspot's epoch_btjd, null's period_days/epoch_btjd — so a plain
    truthiness check (`if period_days:`) is wrong here: NaN is truthy in
    Python and would silently fold the light curve into a NaN phase axis."""
    return x is not None and np.isfinite(x)


def compute_phase(time, period_days, epoch_btjd):
    """True orbital phase in [-0.5, 0.5) when period info is known, else a
    centered/baseline-normalized time axis (still roughly [-0.5, 0.5]) so
    GraphDetail's default view bounds work unchanged for every sample."""
    if _finite(period_days) and _finite(epoch_btjd):
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
    if _finite(period_days):
        jitter = period_days * 1e-4
        params["periodDays"] = {
            "p16": round(period_days - jitter, 6),
            "p50": round(period_days, 6),
            "p84": round(period_days + jitter, 6),
        }
    return params


def convert(path, cls, source_path):
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

    tic_id = int(sample["tic_id"])
    out = {
        "sampleId": f"real-{cls}-{tic_id}",
        "class": cls,
        "source": {
            "repo": "arvyo-data",
            "path": source_path,
            "ticId": tic_id,
            "sector": int(sample["sector"]),
            # only `blend` (Kepler) carries an explicit mission key; the
            # TESS classes omit it entirely, hence the get()-with-default.
            "mission": sample.get("mission", "TESS"),
        },
        "meta": {
            # NaN is not valid JSON (breaks JSON.parse in the browser) and
            # also isn't meaningful here — starspot has no epoch, null has
            # neither — so both come through as JSON null, not NaN.
            "periodDays": float(period_days) if _finite(period_days) else None,
            "epochBtjd": float(epoch_btjd) if _finite(epoch_btjd) else None,
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


def discover_specs():
    """One committed real sample per class, from arvyo-data/data/samples/.

    Raises loudly (missing dir / no file / more than one file) rather than
    silently skipping a class or picking an arbitrary one — a class-name
    typo or an extra file left behind should fail the export, not produce a
    quietly wrong sample set.
    """
    specs = []
    for cls in CLASSES:
        class_dir = DATA_SAMPLES_DIR / cls
        matches = sorted(class_dir.glob("*.npz"))
        if not matches:
            raise FileNotFoundError(
                f"no committed .npz sample for class {cls!r} under {class_dir} "
                "(expected arvyo-data as a sibling checkout with data/samples/ populated)"
            )
        if len(matches) > 1:
            raise ValueError(
                f"expected exactly one committed sample for class {cls!r} under "
                f"{class_dir}, found {len(matches)}: {[m.name for m in matches]}"
            )
        path = matches[0]
        source_path = str(path.relative_to(DATA_ROOT))
        specs.append(dict(path=path, cls=cls, source_path=source_path))
    return specs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in discover_specs():
        out = convert(**spec)
        out_path = OUT_DIR / f"{spec['cls']}.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {out_path.relative_to(OUT_DIR.parent.parent)} "
              f"({len(out['lightCurve']['phase'])} pts, tic_id={out['source']['ticId']})")


if __name__ == "__main__":
    main()
