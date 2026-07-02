"""Posterior inference: emcee MCMC (primary) with an sbi SNPE fallback.

sbi pins recent torch; install torch first, then sbi, to let pip resolve.
Both entry points import their MCMC/SBI dependency inside the function so
this module stays importable even if those packages failed to install.
"""

import numpy as np


def _box_transit(time, t0, period, duration, depth):
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phase) < (duration / period) / 2.0
    flux = np.ones_like(time)
    flux[in_transit] -= depth
    return flux


def fit_emcee(time, flux, flux_err, t0, period_init, duration_init, depth_init,
              nwalkers=32, nsteps=200, burn=50, seed=0):
    """Tiny emcee fit of period+duration+depth on a box-transit planet model.

    Returns {"period": {p16,p50,p84}, "duration": {...}, "depth": {...}}.
    """
    import emcee

    def log_prior(theta):
        period, duration, depth = theta
        if period <= 0 or duration <= 0 or duration >= period or not (0.0 < depth < 1.0):
            return -np.inf
        return 0.0

    def log_likelihood(theta):
        period, duration, depth = theta
        model_flux = _box_transit(time, t0, period, duration, depth)
        resid = (flux - model_flux) / flux_err
        return -0.5 * np.sum(resid ** 2)

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    ndim = 3
    theta0 = np.array([period_init, duration_init, depth_init], dtype=float)
    rng = np.random.default_rng(seed)
    scale = np.abs(theta0) * 0.01 + 1e-6
    pos = theta0 + scale * rng.normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(pos, nsteps, progress=False)

    chain = sampler.get_chain(discard=min(burn, nsteps - 1), flat=True)
    pcts = np.percentile(chain, [16, 50, 84], axis=0)

    names = ["period", "duration", "depth"]
    return {
        name: {"p16": float(pcts[0, i]), "p50": float(pcts[1, i]), "p84": float(pcts[2, i])}
        for i, name in enumerate(names)
    }


def fit_sbi(time, observed_flux, period_bounds=(0.5, 10.0), rp_bounds=(0.01, 0.3),
            num_simulations=200):
    """Minimal SNPE round over the batman `planet` simulator (period, rp).

    Raises RuntimeError with a clear message if sbi is not installed.
    """
    try:
        import torch
        from sbi.inference import SNPE
        from sbi.utils import BoxUniform
    except ImportError as exc:
        raise RuntimeError(
            "sbi is not installed; fit_sbi is unavailable in this "
            "environment. Install `sbi` (torch first, then sbi — see "
            "requirements.txt) or use fit_emcee instead."
        ) from exc

    from arvyo.synthesis.forward_models import planet

    low = torch.tensor([period_bounds[0], rp_bounds[0]], dtype=torch.float32)
    high = torch.tensor([period_bounds[1], rp_bounds[1]], dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)

    def simulate(theta):
        period, rp = theta.tolist()
        flux = planet({"period": period, "rp": rp, "t0": 0.0}, time)
        return torch.as_tensor(flux, dtype=torch.float32)

    theta = prior.sample((num_simulations,))
    x = torch.stack([simulate(t) for t in theta])

    inference = SNPE(prior=prior)
    inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior()

    x_obs = torch.as_tensor(observed_flux, dtype=torch.float32)
    samples = posterior.sample((1000,), x=x_obs).numpy()

    pcts = np.percentile(samples, [16, 50, 84], axis=0)
    return {
        "period": {"p16": float(pcts[0, 0]), "p50": float(pcts[1, 0]), "p84": float(pcts[2, 0])},
        "rp": {"p16": float(pcts[0, 1]), "p50": float(pcts[1, 1]), "p84": float(pcts[2, 1])},
    }
