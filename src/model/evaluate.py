import numpy as np
import faiss
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoFeatureExtractor, WavLMModel

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/wavlm-base"

FAISS_PATH = "data/faiss.index"
LABEL_PATH = "data/faiss_labels.npy"

TARGET_SR = 16000


# =========================
# LOAD MODEL
# =========================
print("🔧 Loading model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# =========================
# LOAD DATA + INDEX
# =========================
dataset = load_from_disk("data/processed")

index = faiss.read_index(FAISS_PATH)
labels = np.load(LABEL_PATH)


# =========================
# EMBEDDING
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

    emb = emb / np.linalg.norm(emb)

    return emb.astype("float32")


# =========================
# EVALUATE
# =========================
correct = 0
total = 0

print("🚀 Evaluating on test set...")

for sample in tqdm(dataset["test"], desc="Evaluating"):
    audio = sample["audio"]["array"]
    true_label = sample["speaker_id"]

    emb = extract_embedding(audio).reshape(1, -1)

    scores, indices = index.search(emb, k=1)

    pred_label = labels[indices[0][0]]

    if pred_label == true_label:
        correct += 1

    total += 1


accuracy = correct / total

print("\n====================")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✔ Correct: {correct}/{total}")
print("====================")