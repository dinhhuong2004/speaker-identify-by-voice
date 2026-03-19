import faiss
import numpy as np

# =========================
# LOAD DATA
# =========================
embeddings = np.load("data/embeddings.npy").astype("float32")
labels = np.load("data/labels.npy")

dim = embeddings.shape[1]

print("Embedding shape:", embeddings.shape)

# =========================
# BUILD INDEX (COSINE)
# =========================
index = faiss.IndexFlatIP(dim)  # Inner Product = cosine

index.add(embeddings)

print("Total vectors in index:", index.ntotal)

# =========================
# SAVE
# =========================
faiss.write_index(index, "data/faiss.index")
np.save("data/faiss_labels.npy", labels)

print("✅ FAISS index built!")