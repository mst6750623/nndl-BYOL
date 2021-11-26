import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLPHead, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.layer(x)