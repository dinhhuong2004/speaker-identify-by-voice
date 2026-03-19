import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel
from tqdm import tqdm
from datasets import load_from_disk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "microsoft/wavlm-base"

# =========================
# LOAD MODEL
# =========================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# =========================
# EXTRACT EMBEDDING
# =========================
def extract_embedding(audio):
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    embedding = embedding.squeeze().cpu().numpy()

    # 🔥 normalize vector (CRITICAL)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# =========================
# BUILD EMBEDDING DATASET
# =========================
if __name__ == "__main__":
    dataset = load_from_disk("data/processed")

    embeddings = []
    labels = []

    print("🚀 Extracting embeddings...")

    for sample in tqdm(dataset["train"], desc="Embedding", unit="sample"):
        audio = sample["audio"]["array"]
        speaker_id = sample["speaker_id"]

        emb = extract_embedding(audio)

        embeddings.append(emb)
        labels.append(speaker_id)

    embeddings = np.array(embeddings).astype("float32")
    labels = np.array(labels)

    np.save("data/embeddings.npy", embeddings)
    np.save("data/labels.npy", labels)

    print("✅ Embedding done!")
    print("Shape:", embeddings.shape)