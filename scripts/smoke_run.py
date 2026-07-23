"""End-to-end smoke runner over the arvyo-pipeline spine.

Runs one small .npz sample per class through: contract validation -> view
generation -> TLS period search -> 4-hypothesis synthesis -> best_explanation
ranking -> emcee posterior -> vetting -> diagnostic figure. Never raises on a
stage failure; every stage records pass/fail/skip and the run continues to
whatever downstream stages can still run.

    python scripts/smoke_run.py --data-root ../arvyo-data/data/processed
    python scripts/smoke_run.py --make-docs-figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arvyo.contract import ContractError, load_sample
from arvyo.inference.posteriors import fit_emcee, fit_sbi
from arvyo.search.tls_search import run_tls
from arvyo.synthesis.forward_models import SIMULATORS, best_explanation
from arvyo.vetting.vet import vet
from arvyo.views.views import make_views, phase_fold
from arvyo.viz.plots import plot_fit

DEFAULT_DATA_ROOT = "../arvyo-data/data/processed"
DEFAULT_DOCS_SAMPLES_ROOT = "../arvyo-data/data/samples"
ALL_LABELS = ["planet", "eb", "blend", "starspot", "null"]
STAGE_NAMES = ["contract", "views", "tls", "synthesis", "inference", "vetting", "figure"]
MODEL_NAMES = ["planet", "eb", "blend", "starspot"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN/Infinity) with None.

    `json.dumps` writes bare NaN/Infinity tokens for these by default, which
    is not valid JSON — reports must parse with any standard JSON reader.
    """
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def pick_samples(data_root: str, labels: list[str]) -> dict[str, Path]:
    """Return {label: path} - the first contract-valid .npz per label.

    Skips zero-byte/unloadable files. Missing labels are simply absent.
    Prefers the smallest valid file per label (fastest smoke).
    """
    root = Path(data_root)
    picks: dict[str, Path] = {}
    for label in labels:
        label_dir = root / label
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.npz"), key=lambda p: p.stat().st_size):
            try:
                load_sample(path)
            except ContractError:
                continue
            picks[label] = path
            break
    return picks


def _build_candidates(period, t0, duration_days, depth):
    rp0 = float(np.clip(np.sqrt(max(depth, 1e-6)), 0.01, 0.3))
    prot0 = max(period, 0.1)
    return [
        {"name": "planet",
         "params": {"period": period, "rp": rp0, "t0": t0, "a": 15.0, "inc": 89.0},
         "free": ["t0", "rp"]},
        {"name": "eb",
         "params": {"period": period, "rp": rp0, "t0": t0, "a": 15.0, "inc": 89.0,
                    "secondary_scale": 0.3},
         "free": ["t0", "rp", "secondary_scale"]},
        {"name": "blend",
         "params": {"period": period, "rp": rp0, "t0": t0, "a": 15.0, "inc": 89.0,
                    "source": "planet", "dilution": 0.5},
         "free": ["t0", "rp", "dilution"]},
        {"name": "starspot",
         "params": {"prot": prot0, "amp1": 0.01, "amp2": 0.0, "phase1": 0.0, "phase2": 0.0},
         "free": ["amp1", "phase1"]},
    ]


def _process_sample(npz_path: Path, *, seed=42, fast=False, need_chain=False):
    """Run every stage once. Returns (stages, sample, artifacts).

    `stages` is the JSON-safe per-stage status/detail dict. `sample` is the
    raw loaded contract dict (arrays + metadata), or None on contract
    failure. `artifacts` holds non-JSON-safe intermediates (raw TLS result,
    fitted candidate params, views, emcee chain) for docs-figure reuse.
    """
    stages = {name: {"status": "skip", "detail": {}} for name in STAGE_NAMES}
    artifacts: dict = {}

    try:
        sample = load_sample(npz_path)
    except Exception as exc:
        stages["contract"] = {"status": "fail", "detail": {}, "error": str(exc)}
        return stages, None, artifacts

    time_arr, flux, flux_err = sample["time"], sample["flux"], sample["flux_err"]
    stages["contract"] = {"status": "pass", "detail": {"n_points": int(time_arr.size)}}

    tls_result = None
    tls_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tls_result = run_tls(time_arr, flux, show_progress_bar=False)
            # Excludes DeprecationWarning: TLS's internal multiprocessing pool
            # (used for its T0 search) triggers stdlib's fork-safety warning,
            # which embeds a nondeterministic PID and isn't a TLS/science
            # signal, unlike the UserWarnings TLS itself emits (e.g. period
            # aliasing) that we want to keep and report.
            tls_warnings = sorted({
                str(w.message) for w in caught
                if not issubclass(w.category, DeprecationWarning)
            })
        period = tls_result["period"]
        t0 = tls_result["T0"]
        duration_days = tls_result["duration"]
        depth = tls_result["depth"]
        tls_detail = {
            "period": period, "t0": t0, "sde": tls_result["SDE"],
            "duration_hours": duration_days * 24.0, "depth_ppm": depth * 1e6,
        }
        meta_period = sample.get("period_days")
        if meta_period is not None and np.isfinite(meta_period):
            meta_period = float(meta_period)
            tls_detail["metadata_period_days"] = meta_period
            tls_detail["metadata_epoch_btjd"] = float(sample.get("epoch_btjd", float("nan")))
            tls_detail["period_rel_diff"] = (
                abs(period - meta_period) / meta_period if meta_period else float("nan")
            )
        if tls_warnings:
            tls_detail["warnings"] = tls_warnings
        stages["tls"] = {"status": "pass", "detail": tls_detail}
        artifacts["tls_result"] = tls_result
    except Exception as exc:
        stages["tls"] = {"status": "fail", "detail": {}, "error": str(exc)}

    if tls_result is not None:
        try:
            global_view, local_view = make_views(time_arr, flux, period, t0, duration_days)
            combined = np.concatenate([global_view, local_view])
            stages["views"] = {
                "status": "pass",
                "detail": {
                    "global_bins": int(global_view.size),
                    "local_bins": int(local_view.size),
                    "finite_fraction": float(np.mean(np.isfinite(combined))),
                },
            }
            artifacts["views"] = (global_view, local_view)
        except Exception as exc:
            stages["views"] = {"status": "fail", "detail": {}, "error": str(exc)}
    else:
        stages["views"] = {"status": "skip", "detail": {},
                            "error": "no period estimate (tls did not produce one)"}

    ranked = None
    if tls_result is not None:
        try:
            candidates = _build_candidates(period, t0, duration_days, depth)
            ranked = best_explanation(time_arr, flux, flux_err, candidates)
            chi2_by_name = {name: None for name in MODEL_NAMES}
            bic_by_name = {name: None for name in MODEL_NAMES}
            for r in ranked:
                chi2_by_name[r["name"]] = r["chi2"]
                bic_by_name[r["name"]] = r["bic"]
            stages["synthesis"] = {
                "status": "pass",
                "detail": {"chi2": chi2_by_name, "bic": bic_by_name, "best": ranked[0]["name"]},
            }
            artifacts["ranked"] = ranked
        except Exception as exc:
            stages["synthesis"] = {"status": "fail", "detail": {}, "error": str(exc)}
    else:
        stages["synthesis"] = {"status": "skip", "detail": {},
                                "error": "no TLS candidate to seed synthesis"}

    if tls_result is not None:
        try:
            nwalkers = 16 if fast else 32
            nsteps = 100 if fast else 200
            posteriors = fit_emcee(
                time_arr, flux, flux_err, t0=t0, period_init=period,
                duration_init=max(duration_days, 1e-3), depth_init=max(depth, 1e-4),
                nwalkers=nwalkers, nsteps=nsteps, seed=seed, return_chain=need_chain,
            )
            chain = posteriors.pop("chain", None)
            if chain is not None:
                artifacts["chain"] = chain
            detail = {"engine": "emcee", **posteriors}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_sbi(time_arr, flux, num_simulations=15 if fast else 30)
                detail["sbi"] = "available and ran"
            except RuntimeError as sbi_exc:
                detail["sbi"] = f"unavailable: {sbi_exc}"
            except Exception as sbi_exc:
                detail["sbi"] = f"attempted, failed: {sbi_exc}"
            stages["inference"] = {"status": "pass", "detail": detail}
        except Exception as exc:
            stages["inference"] = {"status": "fail", "detail": {}, "error": str(exc)}
    else:
        stages["inference"] = {"status": "skip", "detail": {},
                                "error": "no TLS candidate to seed inference"}

    if tls_result is not None:
        try:
            vet_result = vet(time_arr, flux, period, t0, duration_days)
            stages["vetting"] = {"status": "pass", "detail": vet_result}
            artifacts["vet"] = vet_result
        except Exception as exc:
            stages["vetting"] = {"status": "fail", "detail": {}, "error": str(exc)}
    else:
        stages["vetting"] = {"status": "skip", "detail": {},
                              "error": "no TLS candidate to seed vetting"}

    if tls_result is not None and ranked is not None:
        try:
            best_cand = ranked[0]
            artifacts["model_flux"] = SIMULATORS[best_cand["name"]](best_cand["params"], time_arr)
            artifacts["best_cand"] = best_cand
        except Exception as exc:
            artifacts["model_flux_error"] = str(exc)

    return stages, sample, artifacts


def run_sample(npz_path: Path, out_dir: Path, *, seed: int = 42, fast: bool = False) -> dict:
    """Run one sample through every stage.

    Returns the per-sample report dict (also written to out_dir as JSON).
    NEVER raises on a stage failure - records
    {"stage": name, "status": "fail", "error": str} and continues to
    whatever downstream stages can still run.
    """
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages, sample, artifacts = _process_sample(npz_path, seed=seed, fast=fast)

    label = sample["label"] if sample is not None else None
    tic_id = sample["tic_id"] if sample is not None else None
    tic_id_str = str(int(tic_id)) if isinstance(tic_id, (int, np.integer)) else str(tic_id)

    if "model_flux" in artifacts:
        try:
            fig_path = out_dir / f"{label}_{tic_id_str}.png"
            plot_fit(sample["time"], sample["flux"], artifacts["model_flux"],
                      artifacts["tls_result"]["period"], artifacts["tls_result"]["T0"],
                      save_path=fig_path)
            stages["figure"] = {"status": "pass", "detail": {"path": str(fig_path)}}
        except Exception as exc:
            stages["figure"] = {"status": "fail", "detail": {}, "error": str(exc)}
    elif "model_flux_error" in artifacts:
        stages["figure"] = {"status": "fail", "detail": {}, "error": artifacts["model_flux_error"]}
    else:
        stages["figure"] = {"status": "skip", "detail": {},
                             "error": "no fitted model available to plot"}

    consistency = {
        "best_matches_label": (artifacts["ranked"][0]["name"] == label) if "ranked" in artifacts else None,
        "note": "informational only — a mismatch is NOT a stage failure",
    }

    report = {
        "sample": {
            "path": str(npz_path),
            "label": label,
            "tic_id": tic_id_str,
            "n_points": int(sample["time"].size) if sample is not None else None,
        },
        "stages": stages,
        "consistency": consistency,
    }

    report_path = out_dir / f"{label or 'unknown'}_{tic_id_str}.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2))
    return report


def _stage_status_table(all_reports: dict[str, dict]) -> str:
    lines = [f"{'label':10s} | " + " | ".join(f"{s:9s}" for s in STAGE_NAMES)]
    for label, report in all_reports.items():
        row = [report["stages"][s]["status"] for s in STAGE_NAMES]
        lines.append(f"{label:10s} | " + " | ".join(f"{s:9s}" for s in row))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default="smoke_out")
    parser.add_argument("--labels", nargs="+", default=ALL_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="halve emcee/sbi settings")
    parser.add_argument("--make-docs-figures", action="store_true",
                         help="regenerate docs/figures/ from arvyo-data's committed "
                              "data/samples/ (not the gitignored bulk corpus) and exit")
    parser.add_argument("--docs-samples-root", default=DEFAULT_DOCS_SAMPLES_ROOT,
                         help="root for --make-docs-figures (must be the committed "
                              "one-sample-per-class dir, so figures are reproducible "
                              "from a fresh clone without the bulk corpus)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    if args.make_docs_figures:
        docs_picks = pick_samples(args.docs_samples_root, ALL_LABELS)
        generate_docs_figures(docs_picks, seed=args.seed, fast=args.fast)
        print(f"Regenerated docs figures into {DOCS_FIGURES_DIR}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = pick_samples(args.data_root, args.labels)
    missing_labels = [label for label in args.labels if label not in picks]

    if not picks:
        print(f"No samples found under {args.data_root} for labels {args.labels}", file=sys.stderr)
        return 2

    start = _time.time()
    all_reports = {}
    any_fail = False
    for label, path in picks.items():
        report = run_sample(path, out_dir, seed=args.seed, fast=args.fast)
        all_reports[label] = report
        for stage in STAGE_NAMES:
            if report["stages"][stage]["status"] == "fail":
                any_fail = True
    runtime_s = _time.time() - start

    aggregate = {
        "seed": args.seed,
        "data_root": str(args.data_root),
        "labels_requested": args.labels,
        "labels_missing": missing_labels,
        "labels_run": list(picks.keys()),
        "runtime_s": runtime_s,
        "any_stage_failed": any_fail,
    }
    (out_dir / "report.json").write_text(json.dumps(_json_safe(aggregate), indent=2))

    print(_stage_status_table(all_reports))
    print(f"\nlabels missing locally: {missing_labels or 'none'}")
    print(f"total runtime: {runtime_s:.1f}s")

    return 1 if any_fail else 0


def generate_docs_figures(picks: dict[str, Path], *, seed: int = 42, fast: bool = False):
    """Regenerate the 8 README walkthrough figures from the real smoke samples."""
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if "planet" not in picks:
        raise RuntimeError("no local 'planet' sample available to build docs figures from")

    planet_path = picks["planet"]
    sample = load_sample(planet_path)
    time_arr, flux = sample["time"], sample["flux"]
    stages, _, artifacts = _process_sample(planet_path, seed=seed, fast=fast, need_chain=True)

    # 01: raw light curve
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.scatter(time_arr, flux, s=3, color="0.3", alpha=0.6)
    ax.set_xlabel("Time (BTJD, days)")
    ax.set_ylabel("Relative flux")
    ax.set_title(f"Raw light curve — planet TIC {sample['tic_id']}")
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES_DIR / "01_raw_lightcurve.png", dpi=150)
    plt.close(fig)

    tls_result = artifacts.get("tls_result")
    period = tls_result["period"] if tls_result else None
    t0 = tls_result["T0"] if tls_result else None
    duration_days = tls_result["duration"] if tls_result else None

    # 02: global/local views
    if "views" in artifacts:
        global_view, local_view = artifacts["views"]
        fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(10, 4))
        ax_g.plot(np.linspace(-0.5, 0.5, global_view.size), global_view, color="0.2")
        ax_g.set_title(f"Global view ({global_view.size} bins)")
        ax_g.set_xlabel("Phase")
        ax_l.plot(np.linspace(-0.5, 0.5, local_view.size), local_view, color="0.2")
        ax_l.set_title(f"Local view ({local_view.size} bins)")
        ax_l.set_xlabel("Phase (zoomed)")
        fig.tight_layout()
        fig.savefig(DOCS_FIGURES_DIR / "02_views.png", dpi=150)
        plt.close(fig)

    # 03: TLS periodogram
    if tls_result is not None:
        raw = tls_result["raw"]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(raw.periods, raw.power, color="0.3", linewidth=0.8)
        ax.axvline(period, color="orange", linestyle="--",
                    label=f"best period={period:.3f}d, SDE={tls_result['SDE']:.1f}")
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("SDE power")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(DOCS_FIGURES_DIR / "03_tls_periodogram.png", dpi=150)
        plt.close(fig)

    # 04: phase-folded + binned curve
    if tls_result is not None:
        phase = phase_fold(time_arr, t0, period)
        order = np.argsort(phase)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.scatter(phase[order], flux[order], s=3, color="0.6", alpha=0.4, label="data")
        if "views" in artifacts:
            global_view, _ = artifacts["views"]
            ax.plot(np.linspace(-0.5, 0.5, global_view.size),
                     global_view * np.nanstd(flux) + np.nanmedian(flux),
                     color="orange", linewidth=1.5, label="binned (rescaled)")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Relative flux")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(DOCS_FIGURES_DIR / "04_phase_fold.png", dpi=150)
        plt.close(fig)

    # 05: four hypotheses 2x2 grid
    if "ranked" in artifacts and tls_result is not None:
        ranked = artifacts["ranked"]
        best_name = ranked[0]["name"]
        phase = phase_fold(time_arr, t0, period)
        order = np.argsort(phase)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
        by_name = {r["name"]: r for r in ranked}
        for ax, name in zip(axes.flat, MODEL_NAMES):
            cand = by_name[name]
            model_flux = SIMULATORS[name](cand["params"], time_arr)
            ax.scatter(phase[order], flux[order], s=3, color="0.6", alpha=0.4)
            ax.plot(phase[order], model_flux[order], color="orange", linewidth=1.2)
            title = f"{name} (chi2={cand['chi2']:.1f})"
            if name == best_name:
                title += "  ← best"
                for spine in ax.spines.values():
                    spine.set_edgecolor("orange")
                    spine.set_linewidth(2)
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(DOCS_FIGURES_DIR / "05_four_hypotheses.png", dpi=150)
        plt.close(fig)

    # 06: emcee posteriors
    if "chain" in artifacts:
        chain = artifacts["chain"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, name in zip(axes, ("period", "duration", "depth")):
            samples = chain[name]
            p16, p50, p84 = np.percentile(samples, [16, 50, 84])
            ax.hist(samples, bins=30, color="0.5")
            for val, style in ((p16, ":"), (p50, "-"), (p84, ":")):
                ax.axvline(val, color="orange", linestyle=style)
            ax.set_title(f"{name}\np16={p16:.4g} p50={p50:.4g} p84={p84:.4g}", fontsize=9)
        fig.tight_layout()
        fig.savefig(DOCS_FIGURES_DIR / "06_posteriors.png", dpi=150)
        plt.close(fig)

    # 07: vetting
    if "vet" in artifacts:
        vet_result = artifacts["vet"]
        fig, (ax_oe, ax_sec) = plt.subplots(1, 2, figsize=(10, 4))
        oe = vet_result["odd_even"]
        ax_oe.bar(["odd", "even"], [oe["odd_depth"], oe["even_depth"]], color=["0.4", "0.7"])
        ax_oe.set_title(f"Odd/even depth diff={oe['depth_diff']:.2e}")
        sec = vet_result["secondary"]
        ax_sec.bar(["primary", "secondary"],
                    [sec["primary_depth"], sec["secondary_depth"]], color=["0.4", "0.7"])
        ax_sec.set_title(f"Secondary/primary ratio={sec['secondary_to_primary_ratio']:.2f}")
        fig.tight_layout()
        fig.savefig(DOCS_FIGURES_DIR / "07_vetting.png", dpi=150)
        plt.close(fig)

    # 08: class gallery
    gallery_labels = [l for l in ["planet", "eb", "starspot", "null", "blend"] if l in picks]
    fig, axes = plt.subplots(1, len(gallery_labels), figsize=(4 * len(gallery_labels), 3.5))
    if len(gallery_labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, gallery_labels):
        s = load_sample(picks[label])
        _, _, arts = _process_sample(picks[label], seed=seed, fast=fast)
        if "tls_result" in arts:
            p, t0_ = arts["tls_result"]["period"], arts["tls_result"]["T0"]
        else:
            p, t0_ = s.get("period_days") or 1.0, s.get("epoch_btjd") or 0.0
        ph = phase_fold(s["time"], t0_, p)
        order = np.argsort(ph)
        ax.scatter(ph[order], s["flux"][order], s=3, color="0.4", alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("Phase")
    missing = [l for l in ["planet", "eb", "starspot", "null", "blend"] if l not in picks]
    if missing:
        fig.suptitle(f"(no local sample available for: {', '.join(missing)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES_DIR / "08_class_gallery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
