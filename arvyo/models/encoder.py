"""1D CNN encoder stub with dual global/local input branches.

Conceptually mirrors the dual-branch design used by AstroNet/Nigraha-style
transit classifiers; this is an independent PyTorch reimplementation, not
a port of their (TF1) code.
"""

import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    def __init__(self, in_channels=1, channels=(16, 32), kernel_size=5):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    def forward(self, x):
        x = self.net(x)
        return x.mean(dim=-1)  # global average pool over remaining length -> (batch, out_channels)


class DualBranchEncoder(nn.Module):
    """Dual global/local 1D CNN encoder. A few conv blocks per branch,
    global-average-pooled and concatenated with the aux vector."""

    def __init__(self, global_channels=(16, 32), local_channels=(16, 32), aux_dim=2):
        super().__init__()
        self.global_branch = ConvBranch(1, global_channels)
        self.local_branch = ConvBranch(1, local_channels)
        self.feature_dim = self.global_branch.out_channels + self.local_branch.out_channels + aux_dim

    def forward(self, global_view, local_view, aux):
        """global_view: (B, L_g), local_view: (B, L_l), aux: (B, aux_dim)."""
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        return torch.cat([g, l, aux], dim=-1)
