import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2
import os
import tensorflow as tf

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model

# 1. DEVICE SETUP
# PyTorch stays on GPU (it's working fine)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Force TensorFlow to CPU to avoid 4090 CUDNN Autotuner errors
tf.config.set_visible_devices([], 'GPU')

# 2. TEXT EMOTION
TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME).to(device)

def detect_text_emotion(text):
    inputs = text_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    emotion_id = torch.argmax(probs).item()
    return text_model.config.id2label[emotion_id], probs[0][emotion_id].item()

# 3. AUDIO EMOTION
class SERNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 6))
    def forward(self, x): return self.fc(x)

audio_model = SERNet().to(device)
AUDIO_EMOTIONS = ["angry", "happy", "sad", "fear", "neutral", "surprise"]

def detect_audio_emotion(path):
    if not os.path.exists(path): return "neutral", 0.0
    y, sr = librosa.load(path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    x = torch.tensor(mfcc, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = audio_model(x)
    probs = F.softmax(logits, dim=0)
    emotion_id = torch.argmax(probs).item()
    return AUDIO_EMOTIONS[emotion_id], float(probs[emotion_id].detach().cpu())

# 4. FACIAL EMOTION (CPU FORCED)
FACE_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def detect_face_emotion(image_path):
    if not os.path.exists("emotion_model.h5") or not os.path.exists(image_path):
        return "neutral", 0.0
    
    with tf.device('/CPU:0'):
        face_model = load_model("emotion_model.h5", compile=False)
        img = cv2.imread(image_path)
        if img is None: return "no_face", 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0: return "no_face", 0.0

        (x, y, w, h) = faces[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64)).astype("float") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        preds = face_model.predict(roi, verbose=0)[0]
        return FACE_EMOTIONS[np.argmax(preds)], float(np.max(preds))

# 5. FUSION
def analyze_emotion(text, audio_path, image_path):
    t_res = detect_text_emotion(text)
    a_res = detect_audio_emotion(audio_path)
    f_res = detect_face_emotion(image_path)

    scores = {}
    for res, weight in zip([t_res, a_res, f_res], [0.4, 0.3, 0.3]):
        emo, conf = res
        if emo != "no_face":
            scores[emo] = scores.get(emo, 0) + (conf * weight)
    
    if not scores: return {"error": "No emotions detected"}
    final_emo = max(scores, key=scores.get)
    return {"final": final_emo, "confidence": round(scores[final_emo], 4), "breakdown": {"text": t_res, "audio": a_res, "face": f_res}}

if __name__ == "__main__":
    print("\n🚀 ENGINE STARTING (STABLE HYBRID MODE)...")
    result = analyze_emotion("I am excited about this!", "sample.wav", "finaltest.png")
    print("\n--- FINAL MULTIMODAL RESULT ---")
    print(result)
