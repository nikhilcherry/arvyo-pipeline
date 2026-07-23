import numpy as np

from arvyo.synthesis.forward_models import best_explanation, starspot


def test_best_explanation_penalizes_extra_free_params_via_bic():
    """A model family with spurious extra free parameters can only match or
    beat a simpler fixed-parameter fit on raw chi2 (it's a strict
    generalization), even when those extra parameters aren't doing anything
    real -- that's exactly the bias documented in
    scripts/regenerate_fixtures.py (blend's free `dilution` "partially
    absorbing" a mismatch). Ranking must be by BIC, not raw chi2, so the
    unnecessarily flexible model doesn't automatically win.
    """
    prot = 5.0
    true_amp1 = 0.02
    time = np.linspace(0, 20, 2000)
    rng = np.random.default_rng(0)
    flux_err = np.full(time.size, 2e-4)
    flux = starspot({"prot": prot, "amp1": true_amp1}, time) + rng.normal(0, 2e-4, time.size)

    candidates = [
        {
            "name": "starspot",
            "params": {"prot": prot, "amp1": true_amp1},
            "free": [],
        },
        {
            "name": "starspot",
            "params": {"prot": prot, "amp1": true_amp1, "amp2": 0.0, "phase2": 0.0},
            "free": ["amp2", "phase2"],
        },
    ]
    ranked = best_explanation(time, flux, flux_err, candidates)

    simple = next(r for r in ranked if r["n_params"] == 0)
    flexible = next(r for r in ranked if r["n_params"] == 2)

    # The extra-df fit can only match or beat the fixed fit on raw chi2.
    assert flexible["chi2"] <= simple["chi2"] + 1e-6
    # But BIC must prefer the simpler model -- it's the actual ranking used.
    assert ranked[0] is simple
    assert simple["bic"] < flexible["bic"]


def test_best_explanation_reports_bic_and_aic_fields():
    time = np.linspace(0, 10, 200)
    flux_err = np.full(time.size, 1e-3)
    flux = starspot({"prot": 3.0, "amp1": 0.01}, time)

    candidates = [{"name": "starspot", "params": {"prot": 3.0, "amp1": 0.01}, "free": []}]
    ranked = best_explanation(time, flux, flux_err, candidates)

    r = ranked[0]
    assert r["n_params"] == 0
    assert r["bic"] == r["chi2"]  # n_params=0 -> BIC/AIC reduce to chi2 exactly
    assert r["aic"] == r["chi2"]
