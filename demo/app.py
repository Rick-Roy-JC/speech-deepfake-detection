import numpy as np
import torch
import torch.nn as nn
import librosa
import gradio as gr
from transformers import Wav2Vec2Model
from peft import PeftModel

SR, MAX_S = 16000, 16000 * 4
ADAPTER = "lora_adapter"

def build_model():
    enc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    enc = PeftModel.from_pretrained(enc, ADAPTER)
    head = nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2)
    )
    head_sd = torch.load(f"{ADAPTER}/lora_head.pt", map_location="cpu")
    head.load_state_dict({k.replace("head.", ""): v for k, v in head_sd.items()})
    enc.eval(); head.eval()
    return enc, head

ENC, HEAD = build_model()

def pad_or_truncate(x):
    n = len(x)
    if n >= MAX_S:
        s = (n - MAX_S) // 2
        return x[s:s + MAX_S]
    return np.tile(x, int(np.ceil(MAX_S / n)))[:MAX_S]

def predict(audio_path):
    if audio_path is None:
        return {"error": 0.0}
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    wav = pad_or_truncate(wav.astype(np.float32))
    with torch.no_grad():
        h = ENC(torch.from_numpy(wav)[None, :]).last_hidden_state.mean(dim=1)
        probs = torch.softmax(HEAD(h), dim=1)[0]
    return {"bonafide (real)": float(probs[1]), "spoof (fake)": float(probs[0])}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath",
                    label="Speech clip (a few seconds is enough)"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Speech Deepfake Detection — wav2vec2 + LoRA",
    description=(
        "Detects synthetic/spoofed speech. wav2vec2-base with LoRA adapters "
        "(0.5% params trained), 1.22% EER on ASVspoof 2019 LA eval (unseen attacks). "
        "⚠️ Trained on clean 16kHz studio-condition audio — scores on noisy "
        "microphone recordings or heavily compressed audio are less reliable. "
        "[Code & analysis](https://github.com/Rick-Roy-JC/speech-deepfake-detection)"
    ),
)
demo.launch()