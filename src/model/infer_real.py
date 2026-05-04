import numpy as np
import torch
import faiss
import os
import argparse
import librosa
from transformers import AutoFeatureExtractor, AutoModel

# =========================
# CONFIG - MODELS
# =========================
MODELS = {
    'wavlm': {
        'name': 'microsoft/wavlm-base',
        'dim': 768,
        'description': 'WavLM-Base (Microsoft) - Tốt cho speaker identification'
    },
    'wav2vec2': {
        'name': 'facebook/wav2vec2-base',
        'dim': 768,
        'description': 'Wav2Vec2-Base (Meta) - Phổ biến, mục đích chung'
    },
    'hubert': {
        'name': 'facebook/hubert-base-ls960',
        'dim': 768,
        'description': 'HuBERT-Base - Tập trung vào speech representation'
    },
    'unispeech': {
        'name': 'microsoft/unispeech-sat-base',
        'dim': 768,
        'description': 'UniSpeech-SAT - Chuyên biệt cho speaker verification'
    },
}

# =========================
# CONFIG - PATH
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "D:\\Master DS\\Intro_to_DS\\data_processed"
TARGET_SR = 16000
MAX_DURATION = 3

# =========================
# LOAD COMPONENTS
# =========================
def load_model(model_key):
    """Load model"""
    if model_key not in MODELS:
        print(f"❌ Model '{model_key}' không hỗ trợ!")
        exit(1)
    
    model_info = MODELS[model_key]
    print(f"🔧 Loading model: {model_key}")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_info['name'])
    model = AutoModel.from_pretrained(model_info['name']).to(DEVICE)
    model.eval()
    
    return feature_extractor, model

def load_faiss_index(model_key):
    """Load FAISS index"""
    index_file = os.path.join(DATA_PATH, f'faiss_{model_key}.index')
    labels_file = os.path.join(DATA_PATH, f'faiss_labels_{model_key}.npy')
    
    if not os.path.exists(index_file):
        print(f"❌ FAISS index for {model_key} not found!")
        print(f"   Please run: python evaluate_real.py --model {model_key}")
        exit(1)
    
    print(f"📦 Loading FAISS index...")
    index = faiss.read_index(index_file)
    labels = np.load(labels_file)
    
    print(f"   Index: {index.ntotal} vectors")
    
    return index, labels

# =========================
# PREPROCESS AUDIO
# =========================
def preprocess_audio(audio_path):
    """Preprocess audio (giống training)"""
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return None
    
    print(f"🎵 Loading audio: {audio_path}")
    
    # Load
    audio, sr = librosa.load(audio_path, sr=TARGET_SR)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    if len(audio) == 0:
        print("❌ Audio is empty after trimming")
        return None
    
    # Normalize
    audio = librosa.util.normalize(audio)
    
    # Fix length
    max_len = TARGET_SR * MAX_DURATION
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    
    print(f"✅ Audio preprocessed: {len(audio)} samples ({len(audio)/TARGET_SR:.2f}s)")
    
    return audio

# =========================
# EXTRACT EMBEDDING
# =========================
def extract_embedding(audio, feature_extractor, model):
    """Extract embedding từ audio"""
    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if hasattr(outputs, 'last_hidden_state'):
        emb = outputs.last_hidden_state.mean(dim=1)
    else:
        emb = outputs.pooler_output
    
    emb = emb.squeeze().cpu().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    
    return emb.astype('float32')

# =========================
# PREDICT
# =========================
def predict(audio_path, model_key='wavlm', k=1):
    """
    Predict speaker từ audio file
    
    Args:
        audio_path: path to audio file
        model_key: which model to use
        k: number of neighbors to return
    
    Returns:
        predictions: list of (label, confidence) tuples
    """
    print("\n" + "="*60)
    print(f"🎯 INFERENCE - Model: {model_key.upper()}")
    print("="*60)
    
    # Load model
    feature_extractor, model = load_model(model_key)
    
    # Load FAISS index
    index, faiss_labels = load_faiss_index(model_key)
    
    # Preprocess audio
    audio = preprocess_audio(audio_path)
    if audio is None:
        return None
    
    # Extract embedding
    print(f"🚀 Extracting embedding...")
    emb = extract_embedding(audio, feature_extractor, model).reshape(1, -1)
    print(f"   Embedding: {emb.shape}")
    
    # Search FAISS
    print(f"🔍 Searching FAISS (k={k})...")
    scores, indices = index.search(emb, k)
    
    # Results
    results = []
    for i in range(k):
        label = faiss_labels[indices[0][i]]
        score = scores[0][i]
        results.append((int(label), float(score)))
    
    return results

# =========================
# PRINT RESULTS
# =========================
def print_predict_results(results, audio_path, model_key):
    """In kết quả dự đoán"""
    print("\n" + "="*60)
    print(f"📊 RESULTS")
    print("="*60)
    print(f"Audio: {os.path.basename(audio_path)}")
    print(f"Model: {model_key}")
    print()
    
    print("Top predictions:")
    for rank, (label, confidence) in enumerate(results, 1):
        bar = "█" * int(confidence * 30)
        print(f"  {rank}. Label {label:2d} | {confidence:.4f} {bar}")
    
    print("\n" + "="*60)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description='Predict speaker from audio')
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to audio file to predict'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='wavlm',
        choices=list(MODELS.keys()),
        help='Model to use (default: wavlm)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of top predictions to show (default: 3)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Predict using all available models'
    )
    
    args = parser.parse_args()
    
    # Check audio file
    if not os.path.exists(args.audio_path):
        print(f"❌ Audio file not found: {args.audio_path}")
        exit(1)
    
    models_to_use = list(MODELS.keys()) if args.all else [args.model]
    
    for model_key in models_to_use:
        results = predict(args.audio_path, model_key=model_key, k=args.k)
        
        if results is not None:
            print_predict_results(results, args.audio_path, model_key)

if __name__ == "__main__":
    main()