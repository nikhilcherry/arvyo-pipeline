"""Transit Least Squares wrapper + MES/SDE extraction.

TLS's first call JIT-compiles via numba (~30s) — keep that outside timed
test assertions.
"""


def run_tls(time, flux, **tls_kwargs):
    """Run Transit Least Squares on one light curve.

    Returns a dict with period, T0, duration, depth, SDE (and the raw TLS
    result under `raw` for callers that need more).
    """
    from transitleastsquares import transitleastsquares

    model = transitleastsquares(time, flux)
    result = model.power(**tls_kwargs)

    return {
        "period": float(result.period),
        "T0": float(result.T0),
        "duration": float(result.duration),
        "depth": float(1.0 - result.depth),  # result.depth is the relative flux level at transit bottom
        "SDE": float(result.SDE),
        "raw": result,
    }
