"""Central config for speech deepfake detection project."""
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 16000       # ASVspoof LA is 16kHz; wav2vec2 requires 16kHz
    max_duration_s: float = 4.0    # truncate/pad all clips to 4s
    max_samples: int = int(16000 * 4.0)

@dataclass
class BaselineConfig:
    # LFCC frontend for classical baseline
    n_lfcc: int = 60
    n_fft: int = 512
    hop_length: int = 160          # 10ms hop at 16kHz
    win_length: int = 320          # 20ms window

@dataclass
class ModelConfig:
    w2v2_checkpoint: str = "facebook/wav2vec2-base"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")
    classifier_hidden: int = 256
    num_classes: int = 2           # bonafide vs spoof

@dataclass
class TrainConfig:
    batch_size: int = 16
    eval_batch_size: int = 16      # explicit — no repeat of the missing-eval-batch-size OOM
    lr: float = 1e-4
    epochs: int = 5
    fp16: bool = True              # T4/P100 — fp16, NOT bf16 (Project 2 lesson)
    seed: int = 42
    num_workers: int = 2

AUDIO = AudioConfig()
BASELINE = BaselineConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()