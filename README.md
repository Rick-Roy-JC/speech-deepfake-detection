# Speech Deepfake Detection: Classical DSP vs. Self-Supervised Representations 
 **[🎙️ Live demo on Hugging Face Spaces](https://huggingface.co/spaces/this-is-rickroy/speech-deepfake-detection)**

Detecting spoofed/synthetic speech on ASVspoof 2019 (Logical Access), comparing a
classical signal-processing baseline (LFCC + CNN) against wav2vec2-based models,
including a LoRA fine-tuned variant. All experiments run on free-tier Kaggle GPUs
(single T4, ~7.5 GPU-hours total).

## TL;DR

A handcrafted-feature baseline achieves near-perfect results on known attack
types (0.27% EER) but collapses on unseen attacks (21.27% EER) — it memorizes
attack signatures rather than learning what synthetic speech *is*. Adapting
just 0.5% of a self-supervised speech model's parameters with LoRA achieves
the best of both: 0.13% EER on known attacks and 1.22% on unseen ones —
a 17x generalization improvement over the classical baseline.

| Model | Trainable params | Dev EER | Eval EER (unseen attacks) |
|-------|------------------|---------|---------------------------|
| LFCC-CNN baseline | ~470K | 0.27% | 21.27% |
| wav2vec2-base frozen + head | 197K | 4.18% | 8.90% |
| wav2vec2-base + LoRA (r=8) | 492K (0.5% of model) | **0.13%** | **1.22%** |

## Results

Evaluated with Equal Error Rate (EER). Dev shares attack types with training
(A01–A06); Eval contains only unseen attacks (A07–A19), making it the true
test of generalization.

| Model | Frontend | Dev EER | Eval EER (unseen attacks) |
|-------|----------|---------|---------------------------|
| CNN baseline (5 epochs, T4) | LFCC (60 + Δ + ΔΔ) | 0.27% | **21.27%** |
| wav2vec2-base frozen + head | raw waveform | 4.18% | 8.90% | 
| wav2vec2-base + LoRA | raw waveform | TBD | TBD |

![Eval results: DET curves and LoRA score distributions](results/figures/eval_results.png)
 
## Why the dev/eval gap is the whole story

ASVspoof 2019 LA is deliberately constructed so that the training and
development sets share attack algorithms (A01–A06: known TTS/VC systems),
while the evaluation set contains only *unseen* attacks (A07–A19). Dev
performance therefore measures fit; eval performance measures generalization
to novel spoofing technology — the actual deployment scenario, since attackers
don't announce their vocoders in advance.

**LFCC-CNN (0.27% → 21.27%, ~79x degradation).** Linear-frequency cepstral
features are designed to expose vocoder artifacts in high-frequency bands.
The CNN learns these artifact signatures extremely well — for the six attacks
it has seen. Novel synthesis pipelines leave different fingerprints, and the
model has no fallback representation of natural speech to lean on.

**wav2vec2 frozen + head (4.18% → 8.90%, ~2.1x).** Features pretrained by
self-supervision on 960h of natural speech encode general speech structure.
A 197K-parameter head on top is the *weakest* model on known attacks — but
degrades least on unseen ones, because the representation was never fit to
any attack in the first place.

**wav2vec2 + LoRA (0.13% → 1.22%, ~9x).** Injecting low-rank adapters
(r=8, q/v attention projections, 492K trainable params = 0.5% of the model)
lets the encoder specialize for anti-spoofing while staying anchored to its
pretrained representation. It beats the baseline even on the baseline's home
turf, and generalizes 17x better where it counts. Note the degradation factor
(9x) sits between frozen (2.1x) and LFCC (79x): adaptation does reintroduce
some attack-specific fit — it just starts from features robust enough that
the eval floor stays low.

## Method
- **Data:** ASVspoof 2019 LA. Train: 25,380 utts (2,580 bonafide / 22,800 spoof);
  Eval: 71,237 utts. Class imbalance handled with inverse-frequency loss weights.
- **Audio pipeline:** 16 kHz, 4-second crops; short clips repeat-padded (tiling
  preserves vocoder artifact density; zero-padding dilutes it).
- **Baseline:** 60 LFCCs + Δ + ΔΔ (180-dim), 3-block CNN (~470K params).
- **wav2vec2 models:** `facebook/wav2vec2-base`, mean-pooled encoder output,
  2-layer MLP head. Frozen variant trains the head only; LoRA variant adds
  adapters (peft) on attention q/v projections.
- **Training:** AdamW, lr 1e-4, batch 16, 5 epochs, fp16 (AMP), seed 42.
- **Metric:** Equal Error Rate from ROC; scoring uses P(bonafide).

## Reproducing on Kaggle

1. Create a Kaggle notebook, accelerator **GPU T4 x2** (not P100 — see notes).
2. Add Input → search "ASVspoof 2019" → attach the `awsaf49` mirror (~24GB).
3. In the first cell:
!git clone https://github.com/Rick-Roy-JC/speech-deepfake-detection.git /kaggle/working/speech-deepfake-detection
%cd /kaggle/working/speech-deepfake-detection
!pip install -q -U peft torchao
4. Train: `from src.train_baseline import main; main()` for the baseline, or
   `from src.train_w2v2 import main; main(mode="frozen")` / `main(mode="lora")`.
5. Smoke-test first on CPU with `main(subset_frac=0.01, epochs=1)` before
   spending GPU quota.
6. `src/data.py` auto-detects the dataset mount path across Kaggle layouts.

Trained LoRA adapter weights are included at `results/lora_adapter/` and can be
loaded onto `facebook/wav2vec2-base` via peft to reproduce the best model
without retraining. 

## Practical Notes

Hard-won environment lessons from running this on free-tier compute:

- **Kaggle P100 is dead for modern PyTorch:** current preinstalled builds
  dropped sm_60 support. Use the T4 x2 accelerator.
- **peft/torchao clash:** latest peft's LoRA dispatcher requires torchao ≥ 0.16;
  Kaggle ships 0.10. `pip install -U peft torchao` fixes it (torchao's compiled
  kernels stay disabled on torch 2.10 — harmless, only the version check matters).
- **Dataset mount nesting:** the awsaf49 mirror mounts double-nested
  (`.../LA/LA/`), and the mount root differs across Kaggle environment versions;
  data loading auto-detects candidates and validates the protocols folder exists.
- **Kaggle version outputs don't accumulate:** each Quick Save snapshots only
  the current session's `/kaggle/working`. Carry forward previous artifacts
  before saving, or they're only retrievable from older version outputs.
- **Repeat-padding over zero-padding** for anti-spoofing audio (artifact
  density preservation). 

## Limitations & future work

- Single seed per configuration; EER variance across seeds not measured.
- No min-tDCF (the challenge's official co-metric) — EER only.
- LoRA rank fixed at r=8; a rank ablation (r=4/8/16) would map the
  capacity/generalization trade-off (cf. my Text-to-SQL QLoRA project).
- Cross-dataset evaluation (e.g., ASVspoof 2021 LA with codec noise, or
  In-the-Wild) would test robustness beyond the clean-audio setting.

## Project Structure
- `src/` — data pipeline, features, models, training, evaluation
- `configs/` — central configuration
- `notebooks/` — Colab/Kaggle training notebooks
- `results/` — metrics, figures, score files

## Dataset
ASVspoof 2019 LA. Audio files are not committed; see setup instructions.