import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. GLOBAL SETUP & GPU CHECK
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" usando dispositivo: {device}")

# 2. TEXT MODEL (Transformers)
TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL).to(device)
text_model.eval()

# 3. AUDIO MODEL (SERNet)
class SERNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)
        )
    def forward(self, x):
        return self.fc(x)

audio_model = SERNet().to(device)
audio_model.eval()
AUDIO_LABELS = ["angry", "happy", "sad", "fear", "neutral", "surprise"]

# 4. DETECTION FUNCTIONS
def get_text_emotion(text):
    inputs = text_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    idx = torch.argmax(probs).item()
    return text_model.config.id2label[idx], probs[0][idx].item()

def get_audio_emotion(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        x = torch.tensor(mfcc, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = audio_model(x)
        probs = F.softmax(logits, dim=0)
        idx = torch.argmax(probs).item()
        return AUDIO_LABELS[idx], probs[idx].item()
    except Exception:
        return "neutral", 0.0

# 5. FUSION LOGIC
def fuse(t_res, a_res):
    # Weighting: 60% Text (usually more reliable), 40% Audio
    t_emo, t_conf = t_res
    a_emo, a_conf = a_res
    
    combined_conf = (t_conf * 0.6) + (a_conf * 0.4)
    # If they agree, boost confidence
    final_emo = t_emo if t_conf >= a_conf else a_emo
    
    return {
        "final_emotion": final_emotion,
        "confidence": round(combined_conf, 4),
        "details": {"text": t_res, "audio": a_res}
    }

if __name__ == "__main__":
    # Test Run
    sample_text = "I am so impressed with this 3D crown fade!"
    print(f"Result: {get_text_emotion(sample_text)}")
