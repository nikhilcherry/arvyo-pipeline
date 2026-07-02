"""Global/local/secondary phase-folded view generation.

Adapted from Nigraha (Rao, S. 2021), github.com/ExoplanetML/Nigraha,
commit c4365b41dd02b187c3210189ffe8e3ead584f4f5 (MIT license), reference
copy at third_party/nigraha_views/preprocess.py. Rewritten here in pure
numpy (no lightkurve/TensorFlow dependency) so it runs standalone.

Global view: 201 bins over phase [-0.5, 0.5). Local view: 61-81 bins over
±2 transit durations. Median binning; normalized so the out-of-transit
baseline is ~0 and the deepest in-transit bin is ~-1.
"""

import numpy as np


def phase_fold(time, epoch, period):
    """Fold `time` onto [-0.5, 0.5) phase relative to `epoch`/`period`."""
    return ((time - epoch + 0.5 * period) % period) / period - 0.5


def median_bin(phase, flux, num_bins, width, center=0.0):
    """Median-bin `flux` over `phase` into `num_bins` bins spanning `width`
    centered on `center`. Empty bins are filled by linear interpolation."""
    edges = np.linspace(center - width / 2, center + width / 2, num_bins + 1)
    binned = np.full(num_bins, np.nan)
    bin_idx = np.clip(np.digitize(phase, edges) - 1, 0, num_bins - 1)
    for i in range(num_bins):
        mask = bin_idx == i
        if mask.any():
            binned[i] = np.median(flux[mask])

    nan_mask = np.isnan(binned)
    if nan_mask.any() and not nan_mask.all():
        idx = np.arange(num_bins)
        binned[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], binned[~nan_mask])
    elif nan_mask.all():
        binned[:] = 0.0

    return binned


def _normalize(view):
    """Baseline (median) -> 0, deepest in-transit bin -> -1."""
    view = view - np.nanmedian(view)
    min_val = np.nanmin(view)
    if min_val < 0:
        view = view / (-min_val)
    return view


def make_views(time, flux, period, epoch, duration, global_bins=201, local_bins=81):
    """Build the global and local phase-folded views for one candidate.

    `time`, `epoch`, `period`, `duration` share units (days). Returns
    `(global_view, local_view)`, each a 1D float array normalized so the
    out-of-transit baseline is ~0 and the transit minimum is ~-1.
    """
    phase = phase_fold(time, epoch, period)
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]

    global_view = median_bin(phase_sorted, flux_sorted, global_bins, width=1.0)

    duration_phase = duration / period
    local_width = min(4.0 * duration_phase, 1.0)
    local_mask = np.abs(phase_sorted) <= local_width
    if local_mask.sum() < local_bins:
        local_mask = np.ones_like(phase_sorted, dtype=bool)
        local_width = 1.0
    local_view = median_bin(phase_sorted[local_mask], flux_sorted[local_mask],
                             local_bins, width=local_width)

    return _normalize(global_view), _normalize(local_view)


def make_secondary_view(time, flux, period, epoch, duration, bins=81):
    """Local view centered at phase 0.5 (secondary eclipse search window)."""
    phase = phase_fold(time, epoch, period)
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]

    duration_phase = duration / period
    width = min(4.0 * duration_phase, 1.0)

    shifted = np.where(phase_sorted < 0, phase_sorted + 1.0, phase_sorted)
    mask = np.abs(shifted - 0.5) <= width
    if mask.sum() < bins:
        mask = np.ones_like(shifted, dtype=bool)
        width = 0.5

    view = median_bin(shifted[mask], flux_sorted[mask], bins, width=width, center=0.5)
    return _normalize(view)
