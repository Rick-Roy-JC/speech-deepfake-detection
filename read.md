# Speech Deepfake Detection: Classical DSP vs. Self-Supervised Representations

Detecting spoofed/synthetic speech on ASVspoof 2019 (Logical Access), comparing a
classical signal-processing baseline (LFCC + CNN) against a LoRA-fine-tuned
wav2vec2 model. Evaluated with Equal Error Rate (EER).

## Results
_(to be filled)_

| Model | Frontend | Trainable Params | EER (%) |
|-------|----------|------------------|---------|
| CNN baseline | LFCC | TBD | TBD |
| wav2vec2 (frozen) + head | raw waveform | TBD | TBD |
| wav2vec2 + LoRA | raw waveform | TBD | TBD |

## Project Structure
- `src/` — data pipeline, features, models, training, evaluation
- `configs/` — central configuration
- `notebooks/` — Colab/Kaggle training notebooks
- `results/` — metrics, figures, score files

## Dataset
ASVspoof 2019 LA. Audio files are not committed; see setup instructions.