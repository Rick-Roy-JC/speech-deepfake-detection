"""Train wav2vec2 classifier (frozen or LoRA variant)."""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.config import TRAIN
from src.data import ASVspoofDataset, class_weights, KAGGLE_ROOT
from src.w2v2_model import W2V2Classifier
from src.train_baseline import evaluate  # reuse the same eval/EER loop


def main(mode="lora", subset_frac=None, epochs=None, batch_size=None):
    torch.manual_seed(TRAIN.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = epochs or TRAIN.epochs
    batch_size = batch_size or TRAIN.batch_size
    out_dir = f"/kaggle/working/checkpoints_w2v2_{mode}"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = ASVspoofDataset("train", root=KAGGLE_ROOT, subset_frac=subset_frac)
    dev_ds = ASVspoofDataset("dev", root=KAGGLE_ROOT, subset_frac=subset_frac)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    dev_dl = DataLoader(dev_ds, batch_size=TRAIN.eval_batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = W2V2Classifier(mode=mode).to(device)
    print(f"mode={mode} | trainable params: {model.trainable_params():,}")

    w = class_weights(train_ds.df).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=TRAIN.lr)
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
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), f"{out_dir}/best.pt")
            if mode == "lora":
                # tiny adapter-only save — committable to git (Project 2 pattern)
                model.encoder.save_pretrained(f"{out_dir}/lora_adapter")
            print(f"   new best EER {eer*100:.2f}% -> saved")

    print(f"[{mode}] complete. Best dev EER: {best_eer*100:.2f}%")
    return best_eer


if __name__ == "__main__":
    main()