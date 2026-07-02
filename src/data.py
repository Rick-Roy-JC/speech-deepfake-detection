"""ASVspoof 2019 LA data pipeline."""
import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import AUDIO

# Kaggle mount point for awsaf49/asvpoof-2019-dataset
# Kaggle mount point for awsaf49/asvpoof-2019-dataset
KAGGLE_ROOT = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA"

PROTOCOL_FILES = {
    "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev":   "ASVspoof2019_LA_dev/flac",
    "eval":  "ASVspoof2019_LA_eval/flac",
}


def load_protocol(split: str, root: str = KAGGLE_ROOT) -> pd.DataFrame:
    """Parse a CM protocol file into a DataFrame with labels.

    Protocol format: SPEAKER_ID UTT_ID - SYSTEM_ID KEY
    KEY is 'bonafide' or 'spoof'. Label: bonafide=1, spoof=0.
    """
    path = os.path.join(root, PROTOCOL_FILES[split])
    df = pd.read_csv(
        path, sep=" ", header=None,
        names=["speaker_id", "utt_id", "unused", "system_id", "key"],
    )
    df["label"] = (df["key"] == "bonafide").astype(int)
    df["path"] = df["utt_id"].apply(
        lambda u: os.path.join(root, AUDIO_DIRS[split], f"{u}.flac")
    )
    return df[["utt_id", "speaker_id", "system_id", "key", "label", "path"]]


def pad_or_truncate(x: np.ndarray, max_samples: int = AUDIO.max_samples) -> np.ndarray:
    """Fix waveform length: repeat-pad short clips, center-crop long ones."""
    n = len(x)
    if n >= max_samples:
        start = (n - max_samples) // 2
        return x[start:start + max_samples]
    # repeat-pad (standard practice in anti-spoofing; preserves artifacts
    # better than zero-padding)
    n_repeats = int(np.ceil(max_samples / n))
    return np.tile(x, n_repeats)[:max_samples]


class ASVspoofDataset(Dataset):
    """Returns (waveform_tensor [max_samples], label) pairs."""

    def __init__(self, split: str, root: str = KAGGLE_ROOT, subset_frac: float | None = None, seed: int = 42):
        self.df = load_protocol(split, root)
        if subset_frac is not None:
            # stratified subsample for quick smoke tests
            self.df = (
                self.df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(frac=subset_frac, random_state=seed))
                .reset_index(drop=True)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, sr = sf.read(row["path"], dtype="float32")
        assert sr == AUDIO.sample_rate, f"Unexpected sample rate {sr}"
        x = pad_or_truncate(x)
        return torch.from_numpy(x.copy()), torch.tensor(row["label"], dtype=torch.long)


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss([spoof, bonafide])."""
    counts = df["label"].value_counts().sort_index()  # index 0=spoof, 1=bonafide
    w = len(df) / (2.0 * counts)
    return torch.tensor(w.values, dtype=torch.float32)