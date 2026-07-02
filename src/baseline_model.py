"""LFCC-CNN baseline for spoof detection."""
import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import BASELINE
from src.features import LFCCFrontend


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class LFCCCNN(nn.Module):
    """Treats [3*n_lfcc, T] LFCC map as a 1-channel 2D image.

    ~470K params — trains in minutes/epoch on a P100.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.frontend = LFCCFrontend()
        self.blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.frontend(wav)          # [B, 180, T] — fixed DSP, not learned
        x = feats.unsqueeze(1)                  # [B, 1, 180, T]
        x = self.blocks(x)
        x = self.pool(x)
        return self.head(x)