import numpy as np

from arvyo.views.views import make_views


def test_local_view_min_near_center_and_deep():
    rng = np.random.default_rng(0)
    period = 3.0
    epoch = 1.5
    duration = 0.1
    time = np.linspace(0, 15, 5000)
    flux = 1.0 + rng.normal(0, 0.0005, time.size)
    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    in_transit = np.abs(phase) < (duration / period) / 2
    flux[in_transit] -= 0.01

    global_view, local_view = make_views(time, flux, period, epoch, duration)

    assert global_view.shape == (201,)
    assert local_view.shape == (81,)

    center = len(local_view) // 2
    min_idx = int(np.argmin(local_view))
    assert abs(min_idx - center) <= 5
    assert local_view.min() < -0.5
