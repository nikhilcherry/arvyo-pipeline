"""Vetting checks: odd-even depth comparison, secondary eclipse search.

exovetter's API is organized around TCE objects with astropy units, which
fights straightforward integration for a skeleton. Per the handout's
explicit fallback, this module implements odd-even and secondary checks
directly (~30 lines each) rather than wrapping exovetter.
"""

import numpy as np


def odd_even_test(time, flux, period, epoch, duration):
    """Compare in-transit depths of odd- vs even-numbered transits.

    A large odd/even depth mismatch suggests an eclipsing binary at twice
    the true period rather than a genuine planet.
    """
    cycle = np.floor((time - epoch + 0.5 * period) / period).astype(int)
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    half_dur_phase = (duration / period) / 2.0
    in_transit = np.abs(phase) < half_dur_phase

    odd_mask = in_transit & (cycle % 2 != 0)
    even_mask = in_transit & (cycle % 2 == 0)
    baseline = np.median(flux[~in_transit]) if (~in_transit).any() else 1.0

    odd_depth = float(baseline - np.median(flux[odd_mask])) if odd_mask.any() else float("nan")
    even_depth = float(baseline - np.median(flux[even_mask])) if even_mask.any() else float("nan")
    depth_diff = (abs(odd_depth - even_depth)
                  if np.isfinite(odd_depth) and np.isfinite(even_depth) else float("nan"))

    return {"odd_depth": odd_depth, "even_depth": even_depth, "depth_diff": depth_diff}


def secondary_search(time, flux, period, epoch, duration):
    """Search for a secondary eclipse near phase 0.5, relative to the
    primary in-transit depth."""
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    half_dur_phase = (duration / period) / 2.0

    primary_mask = np.abs(phase) < half_dur_phase
    shifted = np.where(phase < 0, phase + 1.0, phase)
    secondary_mask = np.abs(shifted - 0.5) < half_dur_phase
    out_of_transit = ~primary_mask & ~secondary_mask

    baseline = np.median(flux[out_of_transit]) if out_of_transit.any() else 1.0
    primary_depth = float(baseline - np.median(flux[primary_mask])) if primary_mask.any() else float("nan")
    secondary_depth = float(baseline - np.median(flux[secondary_mask])) if secondary_mask.any() else float("nan")

    if np.isfinite(primary_depth) and primary_depth != 0:
        ratio = secondary_depth / primary_depth
    else:
        ratio = float("nan")

    return {
        "secondary_depth": secondary_depth,
        "primary_depth": primary_depth,
        "secondary_to_primary_ratio": ratio,
    }


def vet(time, flux, period, epoch, duration):
    """Run all vetting checks and return a combined report."""
    return {
        "odd_even": odd_even_test(time, flux, period, epoch, duration),
        "secondary": secondary_search(time, flux, period, epoch, duration),
    }
