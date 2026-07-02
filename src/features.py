"""Feature extraction: LFCC frontend for the classical baseline."""
import torch
import torchaudio

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import AUDIO, BASELINE


class LFCCFrontend(torch.nn.Module):
    """Linear-Frequency Cepstral Coefficients + delta + delta-delta.

    LFCC (linear filterbank) is preferred over MFCC (mel filterbank) in
    anti-spoofing: vocoder/TTS artifacts concentrate in high frequencies,
    which mel spacing compresses but linear spacing preserves.
    Output: [B, 3*n_lfcc, T] — static, delta, delta-delta stacked.
    """

    def __init__(self):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=AUDIO.sample_rate,
            n_lfcc=BASELINE.n_lfcc,
            speckwargs={
                "n_fft": BASELINE.n_fft,
                "hop_length": BASELINE.hop_length,
                "win_length": BASELINE.win_length,
            },
        )
        self.deltas = torchaudio.transforms.ComputeDeltas()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, samples] -> lfcc: [B, n_lfcc, T]
        f = self.lfcc(wav)
        d1 = self.deltas(f)
        d2 = self.deltas(d1)
        return torch.cat([f, d1, d2], dim=1)  # [B, 3*n_lfcc, T]