
"""
Audio Data Processing Pipeline for Machine Learning
DESCRIPTION:
This module implements a complete audio preprocessing pipeline for machine learning tasks.
It handles loading raw audio files, preprocessing them, and creating stratified train/test splits.
MAIN FUNCTIONS:
1. load_local_data() - Loads audio files from organized label directories
2. preprocess_audio() - Applies audio processing (trimming, normalization, padding)
3. preprocess_dataset() - Batch processes all audio samples
4. train_test_split() - Creates stratified train/test splits
5. save_dataset() - Exports processed data as NumPy arrays
INPUT:
- Raw audio files (.wav, .mp3, .flac) organized in subdirectories by label/class
- Located at: "D:\\Master DS\\Intro_to_DS\\data_real"
- Directory structure: data_real/[label_1]/[audio_files], data_real/[label_2]/[audio_files]
PROCESSING STEPS:
1. Load audio files at target sample rate (16kHz)
2. Trim silence from audio
3. Normalize amplitude
4. Pad/truncate to fixed length (4 seconds)
5. Filter out invalid/corrupted files
6. Stratified split: 75% train, 25% test
OUTPUT:
- train_audios.npy: Training audio samples (float32 array)
- train_labels.npy: Training labels (integer class indices)
- test_audios.npy: Test audio samples (float32 array)
- test_labels.npy: Test labels (integer class indices)
- Output location: "D:\\Master DS\\Intro_to_DS\\data_processed"
CONFIGURATION:
- TARGET_SR: 16000 Hz (sample rate)
- MAX_DURATION: 4 seconds
- TEST_SIZE: 25% (0.25 split ratio)
"""
import librosa
from tqdm import tqdm
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_split

# =========================
# CONFIG
# =========================
RAW_DATA_PATH = "D:\\Master DS\\Intro_to_DS\\data_real"  
TARGET_SR = 16000
MAX_DURATION = 4  
SAVE_PATH = "D:\\Master DS\\Intro_to_DS\\data_processed"
TEST_SIZE = 0.2  # 80/20

# =========================
# LOAD LOCAL DATA
# =========================
def load_local_data(data_path=RAW_DATA_PATH):
    """Load audio từ local folders"""
    start = time.time()
    print("🚀 Loading audio from local folders...")
    
    samples = []
    labels = []
    label_dirs = sorted([d for d in os.listdir(data_path) 
                        if os.path.isdir(os.path.join(data_path, d))])
    
    for label_id, label_dir in enumerate(tqdm(label_dirs, desc="Loading labels")):
        label_path = os.path.join(data_path, label_dir)
        audio_files = [f for f in os.listdir(label_path) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for audio_file in tqdm(audio_files, desc=f"  {label_dir}", leave=False):
            try:
                audio_path = os.path.join(label_path, audio_file)
                audio, sr = librosa.load(audio_path, sr=TARGET_SR)
                samples.append(audio)
                labels.append(label_id)
            except Exception as e:
                print(f"  ⚠ Skip {audio_file}: {e}")
    
    print(f"✅ Loaded {len(samples)} samples | ⏱ {time.time() - start:.2f}s")
    return samples, labels

# =========================
# PREPROCESS AUDIO
# =========================
def preprocess_audio(audio):
    """Xử lý audio"""
    # Trim silence from audio (remove leading/trailing silence)
    # cắt bỏ phần im lặng ở đầu và cuối file audio
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    if len(audio) == 0:
        return None
    
    # Normalize
    # util.normalize() sẽ chuẩn hóa âm thanh sao cho biên độ lớn nhất là 1.0, giúp giảm thiểu sự khác biệt về âm lượng giữa các file audio
    audio = librosa.util.normalize(audio)
    
    # Fix length
    # Pad or truncate to fixed length (4 seconds)
    max_len = TARGET_SR * MAX_DURATION
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        # Nếu audio ngắn hơn 4 giây, chúng ta sẽ thêm phần padding (giá trị 0) vào cuối audio để đảm bảo tất cả các mẫu có cùng độ dài. Điều này giúp mô hình học được từ dữ liệu có kích thước đồng nhất.
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    
    return audio

# =========================
# PREPROCESS DATASET
# =========================
def preprocess_dataset(samples):
    """Preprocess tất cả"""
    print("🧹 Preprocessing audio...")
    processed = []
    valid_indices = []
    
    for idx, audio in enumerate(tqdm(samples, desc="Preprocessing")):
        audio = preprocess_audio(audio)
        if audio is not None:
            processed.append(audio)
            valid_indices.append(idx)
    
    return processed, valid_indices

# =========================
# TRAIN/TEST SPLIT
# =========================
def train_test_split(samples, labels, test_size=TEST_SIZE):
    """Chia train/test"""
    print(f"🔀 Splitting ({100*(1-test_size):.0f}/{100*test_size:.0f})...")
    
    indices = np.arange(len(samples))
    train_idx, test_idx = sklearn_split(
        indices, test_size=test_size, stratify=labels, random_state=42
    )
    
    train_samples = np.array([samples[i] for i in train_idx], dtype='float32')
    train_labels = np.array([labels[i] for i in train_idx])
    
    test_samples = np.array([samples[i] for i in test_idx], dtype='float32')
    test_labels = np.array([labels[i] for i in test_idx])

    # In ra số lượng mẫu và phân bố nhãn
    print(f"✅ Train: {len(train_samples)} samples | Test: {len(test_samples)} samples")
    print(f"   Train label distribution: {np.bincount(train_labels)}")
    print(f"   Test label distribution: {np.bincount(test_labels)}")
    return {
        'train': {'audios': train_samples, 'labels': train_labels},
        'test': {'audios': test_samples, 'labels': test_labels}
    }

# =========================
# SAVE
# =========================
def save_dataset(dataset, save_path=SAVE_PATH):
    """Lưu dữ liệu"""
    os.makedirs(save_path, exist_ok=True)
    
    np.save(os.path.join(save_path, 'train_audios.npy'), dataset['train']['audios'])
    np.save(os.path.join(save_path, 'train_labels.npy'), dataset['train']['labels'])
    np.save(os.path.join(save_path, 'test_audios.npy'), dataset['test']['audios'])
    np.save(os.path.join(save_path, 'test_labels.npy'), dataset['test']['labels'])
    
    print(f"💾 Saved to {save_path}")
    print(f"  train: {dataset['train']['audios'].shape}, {dataset['train']['labels'].shape}")
    print(f"  test: {dataset['test']['audios'].shape}, {dataset['test']['labels'].shape}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Load
    samples, labels = load_local_data(RAW_DATA_PATH)
    print(f"✅ {len(samples)} samples")
    
    # Preprocess
    samples, valid_indices = preprocess_dataset(samples)
    labels = [labels[i] for i in valid_indices]
    print(f"✅ {len(samples)} samples after preprocess")
    
    # # Split
    dataset = train_test_split(samples, labels, test_size=TEST_SIZE)
    
    # # Save
    save_dataset(dataset, SAVE_PATH)
    
    print("\n" + "="*50)
    print("✅ DONE!")