"""Train the LFCC-CNN baseline."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import TRAIN
from src.data import ASVspoofDataset, class_weights, KAGGLE_ROOT
from src.baseline_model import LFCCCNN
from src.metrics import compute_eer

OUT_DIR = "/kaggle/working/checkpoints_baseline"


def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for wav, y in loader:
            wav = wav.to(device)
            logits = model(wav)
            probs = torch.softmax(logits.float(), dim=1)[:, 1]  # P(bonafide)
            all_scores.append(probs.cpu().numpy())
            all_labels.append(y.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    eer, thr = compute_eer(labels, scores)
    return eer, scores, labels


def main(subset_frac=None, epochs=None):
    torch.manual_seed(TRAIN.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    epochs = epochs or TRAIN.epochs

    train_ds = ASVspoofDataset("train", root=KAGGLE_ROOT, subset_frac=subset_frac)
    dev_ds = ASVspoofDataset("dev", root=KAGGLE_ROOT, subset_frac=subset_frac)
    train_dl = DataLoader(train_ds, batch_size=TRAIN.batch_size, shuffle=True,
                          num_workers=TRAIN.num_workers, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=TRAIN.eval_batch_size, shuffle=False,
                        num_workers=TRAIN.num_workers, pin_memory=True)

    model = LFCCCNN().to(device)
    w = class_weights(train_ds.df).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN.fp16)

    best_eer = 1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (wav, y) in enumerate(train_dl):
            wav, y = wav.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=TRAIN.fp16):
                loss = criterion(model(wav), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {running / (step + 1):.4f}", flush=True)

        eer, _, _ = evaluate(model, dev_dl, device)
        print(f"== epoch {epoch} done in {time.time()-t0:.0f}s | dev EER {eer*100:.2f}% ==")
        torch.save(model.state_dict(), f"{OUT_DIR}/last.pt")
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), f"{OUT_DIR}/best.pt")
            print(f"   new best EER {eer*100:.2f}% -> saved best.pt")

    print(f"Training complete. Best dev EER: {best_eer*100:.2f}%")


if __name__ == "__main__":
    main()