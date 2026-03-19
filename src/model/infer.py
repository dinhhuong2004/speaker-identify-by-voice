import numpy as np
import faiss
import torch
import librosa
import os

from transformers import AutoFeatureExtractor, WavLMModel

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/wavlm-base"

FAISS_PATH = "data/faiss.index"
LABEL_PATH = "data/faiss_labels.npy"

TARGET_SR = 16000
MAX_DURATION = 3  # seconds


# =========================
# LOAD MODEL
# =========================
print("🔧 Loading model...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# =========================
# LOAD FAISS
# =========================
print("📦 Loading FAISS index...")

index = faiss.read_index(FAISS_PATH)
labels = np.load(LABEL_PATH)


# =========================
# AUDIO PREPROCESS (match training)
# =========================
def preprocess_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ File not found: {audio_path}")

    audio, sr = librosa.load(audio_path, sr=TARGET_SR)

    # trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    if len(audio) == 0:
        raise ValueError("❌ Audio is empty after trimming")

    # normalize
    audio = librosa.util.normalize(audio)

    # fix length
    max_len = TARGET_SR * MAX_DURATION
    if len(audio) > max_len:
        audio = audio[:max_len]

    return audio


# =========================
# EXTRACT EMBEDDING
# =========================
def extract_embedding(audio):
    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state.mean(dim=1)
    emb = emb.squeeze().cpu().numpy()

    # normalize vector
    emb = emb / np.linalg.norm(emb)

    return emb.astype("float32")


# =========================
# PREDICT
# =========================
def predict(audio_path, k=1):
    audio = preprocess_audio(audio_path)

    emb = extract_embedding(audio).reshape(1, -1)

    scores, indices = index.search(emb, k)

    pred_label = labels[indices[0][0]]
    score = scores[0][0]

    return pred_label, score


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    test_path = "data/test.wav"  # 👉 đặt file ở đây

    pred, score = predict(test_path)

    print(f"\n🎤 Speaker ID: {pred}")
    print(f"📊 Similarity score: {score:.4f}")