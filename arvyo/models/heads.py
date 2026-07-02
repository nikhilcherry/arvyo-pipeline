"""Classifier + novelty-score heads on top of DualBranchEncoder features."""

import torch.nn as nn


class ClassifierHead(nn.Module):
    """2 dense layers -> num_classes logits (softmax externally, e.g. via CrossEntropyLoss)."""

    def __init__(self, in_dim, num_classes=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)


class NoveltyHead(nn.Module):
    """Placeholder anomaly/novelty score head — one scalar per sample."""

    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)
