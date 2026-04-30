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
TARGET_SR = 16000   # sampling rate: audio sẽ được resample về 16000Hz
MAX_DURATION = 3    # max duration: audio sẽ được cắt hoặc padding về 3 giây (16000 * 3 = 48000 samples)
SAVE_PATH = "D:\\Master DS\\Intro_to_DS\\data_processed"  # nơi lưu dữ liệu đã xử lý
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
    # Trim silence: loại bỏ phần im lặng ở đầu và cuối
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    if len(audio) == 0:
        return None
    
    # Normalize: Chuẩn hóa volume, Scale audio về khoảng [-1, 1]
    audio = librosa.util.normalize(audio)
    
    # Fix length: cắt hoặc padding về độ dài cố định (3 giây)
    max_len = TARGET_SR * MAX_DURATION
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    
    return audio

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
    
    # # Preprocess
    samples, valid_indices = preprocess_dataset(samples)
    labels = [labels[i] for i in valid_indices]
    print(f"✅ {len(samples)} samples after preprocess")
    
    # # Split
    dataset = train_test_split(samples, labels, test_size=TEST_SIZE)
    
    # # Save
    save_dataset(dataset, SAVE_PATH)
    
    print("\n" + "="*50)
    print("✅ DONE!")