"""Phase-fold + model-fit + residual figure."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fit(time, flux, model_flux, period, epoch, save_path=None):
    """Phase-folded data + model-fit line + residual strip.

    Data points in grey, orange model line, residual panel below (matches
    the dashboard mockup style). If `save_path` is given, saves a PNG
    there. Returns the matplotlib Figure either way.
    """
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]
    model_sorted = model_flux[order]
    residual = flux_sorted - model_sorted

    fig, (ax_fit, ax_resid) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_fit.scatter(phase_sorted, flux_sorted, s=4, color="0.4", alpha=0.6, label="data")
    ax_fit.plot(phase_sorted, model_sorted, color="orange", linewidth=1.5, label="model")
    ax_fit.set_ylabel("Relative flux")
    ax_fit.legend(loc="lower right")

    ax_resid.axhline(0.0, color="orange", linewidth=1.0)
    ax_resid.scatter(phase_sorted, residual, s=4, color="0.4", alpha=0.6)
    ax_resid.set_xlabel("Phase")
    ax_resid.set_ylabel("Residual")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100)

    return fig
