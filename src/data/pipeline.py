from datasets import load_dataset, concatenate_datasets, Audio
import librosa
from tqdm import tqdm
import os
import time

# =========================
# CONFIG
# =========================
DATASET_NAME = "thucdangvan020999/speaker_identification_100_speakers_"
NUM_SPEAKERS = 100
TARGET_SR = 16000
MAX_DURATION = 3  # seconds
SAVE_PATH = "data/processed"


# =========================
# LOAD + MERGE
# =========================
def load_and_merge():
    start = time.time()

    speakers = [f"speaker_{i:03d}" for i in range(1, NUM_SPEAKERS + 1)]
    datasets_list = []

    print("🚀 Loading speakers...")

    for i, spk in enumerate(tqdm(speakers, desc="Loading speakers", unit="spk")):
        try:
            ds = load_dataset(DATASET_NAME, spk, split="train")

            ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
            ds = ds.map(lambda x: {"speaker_id": i})

            datasets_list.append(ds)

        except Exception as e:
            print(f"skip {spk}: {e}")

    print("📦 Merging datasets...")
    dataset = concatenate_datasets(datasets_list)

    print(f"⏱ Load time: {time.time() - start:.2f}s")
    return dataset


# =========================
# PREPROCESS (manual tqdm)
# =========================
def preprocess_dataset(dataset):
    start = time.time()

    print("🧹 Preprocessing audio (with ETA)...")

    processed = []

    for sample in tqdm(dataset, desc="Preprocessing", unit="sample"):
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]

        # trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        if len(audio) == 0:
            continue

        # normalize
        audio = librosa.util.normalize(audio)

        # fix length
        max_len = TARGET_SR * MAX_DURATION
        if len(audio) > max_len:
            audio = audio[:max_len]

        sample["audio"]["array"] = audio
        processed.append(sample)

    print(f"⏱ Preprocess time: {time.time() - start:.2f}s")
    return processed


# =========================
# MAIN PIPELINE
# =========================
def build_pipeline():
    total_start = time.time()

    dataset = load_and_merge()

    dataset = preprocess_dataset(dataset)

    from datasets import Dataset
    dataset = Dataset.from_list(dataset)

    print("🔀 Splitting train/test...")
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"⏱ Total pipeline time: {time.time() - total_start:.2f}s")

    return dataset


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    dataset = build_pipeline()

    print(dataset)
    print("Sample:", dataset["train"][0])

    total_samples = len(dataset["train"]) + len(dataset["test"])
    num_speakers = len(set(dataset["train"]["speaker_id"]))

    print(f"✅ Total samples: {total_samples}")
    print(f"👤 Number of speakers: {num_speakers}")

    os.makedirs("data", exist_ok=True)
    dataset.save_to_disk(SAVE_PATH)

    print(f"💾 Saved to {SAVE_PATH}")