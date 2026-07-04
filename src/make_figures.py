"""Generate DET curves and score distribution figures from saved eval scores."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

MODELS = {
    "LFCC-CNN baseline": ("baseline_eval_scores.npy", "baseline_eval_labels.npy", "tab:red"),
    "wav2vec2 frozen": ("w2v2_frozen_eval_scores.npy", "w2v2_frozen_eval_labels.npy", "tab:orange"),
    "wav2vec2 + LoRA": ("w2v2_lora_eval_scores.npy", "w2v2_lora_eval_labels.npy", "tab:blue"),
}


def det_points(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # clip zeros for probit transform
    fpr = np.clip(fpr, 1e-6, 1 - 1e-6)
    fnr = np.clip(fnr, 1e-6, 1 - 1e-6)
    return norm.ppf(fpr), norm.ppf(fnr), fpr, fnr


def eer_of(labels, scores):
    _, _, fpr, fnr = det_points(labels, scores)
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Panel 1: DET curves ----
    ax = axes[0]
    ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5])  # as fractions
    for name, (sf, lf, color) in MODELS.items():
        spath, lpath = os.path.join(RESULTS, sf), os.path.join(RESULTS, lf)
        if not (os.path.exists(spath) and os.path.exists(lpath)):
            print(f"skipping {name} (files not found)")
            continue
        scores, labels = np.load(spath), np.load(lpath)
        x, y, fpr, fnr = det_points(labels, scores)
        eer = eer_of(labels, scores)
        ax.plot(x, y, color=color, label=f"{name} (EER {eer*100:.2f}%)")
        # mark EER point
        e = norm.ppf(np.clip(eer, 1e-6, 1))
        ax.plot(e, e, "o", color=color, ms=6)
    lim = norm.ppf([0.0008, 0.6])
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)  # EER diagonal
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t*100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t*100:g}" for t in ticks])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("False Alarm Rate (%)"); ax.set_ylabel("Miss Rate (%)")
    ax.set_title("DET curves — ASVspoof 2019 LA eval (unseen attacks)")
    ax.legend(); ax.grid(alpha=0.3)

    # ---- Panel 2: LoRA score distributions ----
    ax = axes[1]
    scores = np.load(os.path.join(RESULTS, "w2v2_lora_eval_scores.npy"))
    labels = np.load(os.path.join(RESULTS, "w2v2_lora_eval_labels.npy"))
    ax.hist(scores[labels == 0], bins=60, alpha=0.6, label="spoof (n=63,882)",
            color="tab:red", density=True)
    ax.hist(scores[labels == 1], bins=60, alpha=0.6, label="bonafide (n=7,355)",
            color="tab:green", density=True)
    ax.set_xlabel("P(bonafide)"); ax.set_ylabel("density")
    ax.set_title("wav2vec2 + LoRA — eval score distributions")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGS, "eval_results.png")
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()