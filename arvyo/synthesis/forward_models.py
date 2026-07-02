"""Forward models for analysis-by-synthesis: four competing hypotheses for
what produced an observed dip (planet, eclipsing binary, blend, starspot),
plus chi2 scoring and a best_explanation ranker.

batman installs as `batman-package` but imports as `batman`; imported
inside functions so this module is importable without it.
"""

import numpy as np
from scipy.optimize import minimize


def planet(params, time):
    """Transiting planet: batman, quadratic limb darkening.

    Required params: period, rp (Rp/R*). Optional: t0, a (a/R*), inc (deg),
    ecc, w (deg), u (2-element quadratic limb-darkening coeffs).
    """
    import batman

    bp = batman.TransitParams()
    bp.t0 = params.get("t0", 0.0)
    bp.per = params["period"]
    bp.rp = params["rp"]
    bp.a = params.get("a", 15.0)
    bp.inc = params.get("inc", 89.0)
    bp.ecc = params.get("ecc", 0.0)
    bp.w = params.get("w", 90.0)
    bp.u = params.get("u", [0.3, 0.2])
    bp.limb_dark = "quadratic"

    model = batman.TransitModel(bp, time)
    return model.light_curve(bp)


def eclipsing_binary(params, time):
    """batman primary eclipse + scaled secondary at phase 0.5.

    Extra params beyond `planet`: secondary_scale (secondary depth as a
    fraction of the primary depth; radius ratio scales as sqrt of it).
    """
    primary_flux = planet(params, time)

    secondary_scale = params.get("secondary_scale", 0.1)
    secondary_params = dict(params)
    secondary_params["t0"] = params.get("t0", 0.0) + params["period"] / 2.0
    secondary_params["rp"] = params["rp"] * np.sqrt(secondary_scale)
    secondary_flux = planet(secondary_params, time)

    return primary_flux + (secondary_flux - 1.0)


def blend(params, time):
    """Planet or EB signal diluted by a blend factor.

    Extra params: source ("planet" or "eb", default "planet"), dilution
    (fraction of the true depth retained, 0 < f <= 1).
    """
    source = params.get("source", "planet")
    dilution = params.get("dilution", 0.5)
    raw_flux = eclipsing_binary(params, time) if source == "eb" else planet(params, time)
    return 1.0 + dilution * (raw_flux - 1.0)


def starspot(params, time):
    """Sum of 1-2 sinusoids at Prot and Prot/2 (rotational modulation, no eclipse)."""
    prot = params["prot"]
    amp1 = params.get("amp1", 0.01)
    amp2 = params.get("amp2", 0.0)
    phase1 = params.get("phase1", 0.0)
    phase2 = params.get("phase2", 0.0)

    flux = 1.0 + amp1 * np.sin(2 * np.pi * time / prot + phase1)
    if amp2:
        flux = flux + amp2 * np.sin(2 * np.pi * time / (prot / 2.0) + phase2)
    return flux


SIMULATORS = {
    "planet": planet,
    "eb": eclipsing_binary,
    "blend": blend,
    "starspot": starspot,
}


def chi2(model_flux, flux, flux_err):
    return float(np.sum(((flux - model_flux) / flux_err) ** 2))


def _fit_free_params(sim_fn, base_params, free_keys, time, flux, flux_err):
    x0 = np.array([base_params[k] for k in free_keys], dtype=float)

    def objective(x):
        params = dict(base_params)
        params.update(zip(free_keys, x))
        try:
            model_flux = sim_fn(params, time)
        except Exception:
            return 1e12
        return chi2(model_flux, flux, flux_err)

    result = minimize(objective, x0, method="Nelder-Mead")

    fitted_params = dict(base_params)
    fitted_params.update(zip(free_keys, result.x))
    return fitted_params, float(result.fun)


def best_explanation(time, flux, flux_err, candidates):
    """Fit and rank competing hypotheses.

    `candidates`: list of `{"name", "params", "free"}` dicts, where `name`
    is one of SIMULATORS' keys, `params` is the full parameter dict (fixed
    values), and `free` (optional) lists param keys to fit via
    scipy.optimize.minimize. Returns a list of `{"name", "params", "chi2"}`
    sorted ascending by chi2 (best fit first).
    """
    ranked = []
    for cand in candidates:
        name = cand["name"]
        sim_fn = SIMULATORS[name]
        free_keys = cand.get("free", [])
        base_params = cand["params"]

        if free_keys:
            fitted_params, chi2_val = _fit_free_params(
                sim_fn, base_params, free_keys, time, flux, flux_err)
        else:
            fitted_params = base_params
            chi2_val = chi2(sim_fn(base_params, time), flux, flux_err)

        ranked.append({"name": name, "params": fitted_params, "chi2": chi2_val})

    ranked.sort(key=lambda r: r["chi2"])
    return ranked
