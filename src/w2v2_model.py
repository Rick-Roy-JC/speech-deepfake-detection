"""wav2vec2-base classifier with frozen / LoRA variants."""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from peft import LoraConfig, get_peft_model

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import MODEL


class W2V2Classifier(nn.Module):
    """wav2vec2-base encoder + mean-pool + MLP head.

    mode='frozen': encoder fully frozen, only the head trains (~200K params).
    mode='lora':   encoder frozen except LoRA adapters on attention q/v
                   projections (~300K adapter params) + head.
    """

    def __init__(self, mode: str = "lora"):
        super().__init__()
        assert mode in ("frozen", "lora")
        self.mode = mode
        self.encoder = Wav2Vec2Model.from_pretrained(MODEL.w2v2_checkpoint)

        # The CNN feature extractor is always frozen (standard practice).
        self.encoder.feature_extractor._freeze_parameters()

        if mode == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            # Freeze everything, then inject trainable LoRA adapters.
            for p in self.encoder.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=MODEL.lora_r,
                lora_alpha=MODEL.lora_alpha,
                lora_dropout=MODEL.lora_dropout,
                target_modules=list(MODEL.lora_target_modules),
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        hidden = self.encoder.config.hidden_size if mode == "frozen" \
            else self.encoder.base_model.model.config.hidden_size  # 768

        self.head = nn.Sequential(
            nn.Linear(hidden, MODEL.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(MODEL.classifier_hidden, MODEL.num_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, samples], already 16kHz float32 in [-1, 1]
        out = self.encoder(wav).last_hidden_state    # [B, T', 768]
        pooled = out.mean(dim=1)                     # mean pool over time
        return self.head(pooled)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)