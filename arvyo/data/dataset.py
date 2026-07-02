"""PyTorch Dataset over arvyo-data processed .npz files."""

from pathlib import Path

import numpy as np

from arvyo import contract
from arvyo.views.views import make_views


def _log10_period(period_days):
    if period_days is None or not np.isfinite(period_days) or period_days <= 0:
        period_days = 1.0
    return float(np.log10(period_days))


def _crowdsap(value):
    if value is None or not np.isfinite(value):
        return 1.0
    return float(value)


class ArvyoDataset:
    """PyTorch-style Dataset over `{root}/{label}/*.npz` files.

    Returns `(global_view, local_view, aux, label_idx)` torch tensors per
    `arvyo.contract`. `aux = [crowdsap, log10(period_days or 1)]`, with
    NaN/missing values mapped to sensible defaults.
    """

    def __init__(self, root, labels, global_bins=201, local_bins=81):
        import torch  # local import: keep torch out of module import time

        self._torch = torch
        self.root = Path(root)
        self.labels = list(labels)
        self.global_bins = global_bins
        self.local_bins = local_bins
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

        self.samples = []
        for label in self.labels:
            label_dir = self.root / label
            if not label_dir.exists():
                continue
            for npz_path in sorted(label_dir.glob("*.npz")):
                if npz_path.stat().st_size == 0:
                    continue
                self.samples.append(npz_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        torch = self._torch
        path = self.samples[idx]
        sample = contract.load_sample(path)

        period = sample.get("period_days")
        epoch = sample.get("epoch_btjd")
        if period and epoch and np.isfinite(period) and np.isfinite(epoch):
            duration = 0.05 * period  # fallback: 5% of period if no TLS duration on hand
            global_view, local_view = make_views(
                sample["time"], sample["flux"], period, epoch, duration,
                global_bins=self.global_bins, local_bins=self.local_bins,
            )
        else:
            global_view = np.zeros(self.global_bins, dtype=np.float64)
            local_view = np.zeros(self.local_bins, dtype=np.float64)

        aux = np.array([
            _crowdsap(sample.get("crowdsap")),
            _log10_period(period),
        ], dtype=np.float32)

        label_idx = self.label_to_idx[sample["label"]]

        return (
            torch.tensor(global_view, dtype=torch.float32),
            torch.tensor(local_view, dtype=torch.float32),
            torch.tensor(aux, dtype=torch.float32),
            torch.tensor(label_idx, dtype=torch.long),
        )
