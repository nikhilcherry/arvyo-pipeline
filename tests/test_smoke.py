import numpy as np

from arvyo.inference.posteriors import fit_emcee
from arvyo.models.encoder import DualBranchEncoder
from arvyo.models.heads import ClassifierHead
from arvyo.search.tls_search import run_tls
from arvyo.synthesis.forward_models import best_explanation, planet
from arvyo.views.views import make_views
from arvyo.viz.plots import plot_fit


def test_end_to_end_smoke(tmp_path):
    period = 1.8
    t0 = 0.3
    rp = 0.12
    params = {"period": period, "rp": rp, "t0": t0, "a": 10.0, "inc": 88.0}

    # 5-day baseline, 2-min cadence: keeps TLS's JIT-compile + search under ~2 min.
    time = np.arange(0, 5, 2.0 / 60 / 24)
    flux_err = np.full(time.size, 0.0005)

    rng = np.random.default_rng(0)
    flux = planet(params, time) + rng.normal(0, 0.0002, time.size)

    tls_result = run_tls(time, flux)
    assert abs(tls_result["period"] - period) / period < 0.05

    global_view, local_view = make_views(
        time, flux, tls_result["period"], tls_result["T0"], tls_result["duration"])
    assert global_view.shape == (201,)
    assert local_view.shape == (81,)

    import torch

    encoder = DualBranchEncoder()
    head = ClassifierHead(encoder.feature_dim, num_classes=4)
    g = torch.tensor(global_view, dtype=torch.float32).unsqueeze(0)
    l = torch.tensor(local_view, dtype=torch.float32).unsqueeze(0)
    aux = torch.zeros((1, 2), dtype=torch.float32)
    logits = head(encoder(g, l, aux))
    assert logits.shape == (1, 4)

    candidates = [
        {"name": "planet", "params": dict(params), "free": []},
        {"name": "starspot", "params": {"prot": 5.0, "amp1": 0.001}, "free": []},
    ]
    ranked = best_explanation(time, flux, flux_err, candidates)
    assert ranked[0]["name"] == "planet"

    posteriors = fit_emcee(
        time, flux, flux_err, t0=tls_result["T0"],
        period_init=tls_result["period"], duration_init=tls_result["duration"],
        depth_init=max(tls_result["depth"], 1e-4),
        nwalkers=32, nsteps=200,
    )
    for key in ("period", "duration", "depth"):
        for pct in ("p16", "p50", "p84"):
            assert np.isfinite(posteriors[key][pct])

    model_flux = planet(params, time)
    out_path = tmp_path / "fit.png"
    plot_fit(time, flux, model_flux, period, t0, save_path=out_path)
    assert out_path.exists()
