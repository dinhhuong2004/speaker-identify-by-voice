import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

# =========================
# CONFIG - CÁC MODEL CÓ SẴN
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
SAVE_PATH = "D:\\Master DS\\Intro_to_DS\\data_processed"
TARGET_SR = 16000

# =========================
# LOAD DATA
# =========================
def load_data():
    """Load training data"""
    print("📦 Loading training data...")
    train_audios = np.load(os.path.join(DATA_PATH, 'train_audios.npy'))
    train_labels = np.load(os.path.join(DATA_PATH, 'train_labels.npy'))
    
    print(f"✅ Loaded {train_audios.shape[0]} training samples")
    print(f"   Shape: {train_audios.shape}")
    
    return train_audios, train_labels

# =========================
# LOAD MODEL
# =========================
def load_model(model_key):
    """Load model từ key"""
    if model_key not in MODELS:
        print(f"❌ Model '{model_key}' không hỗ trợ!")
        print(f"   Các model có sẵn: {list(MODELS.keys())}")
        exit(1)
    
    model_info = MODELS[model_key]
    model_name = model_info['name']
    
    print(f"\n🔧 Loading model: {model_key}")
    print(f"   HuggingFace: {model_name}")
    print(f"   Dimension: {model_info['dim']}")
    print(f"   {model_info['description']}")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    print(f"   ✅ Loaded on {DEVICE}")
    
    return feature_extractor, model, model_info['dim']

# =========================
# EXTRACT EMBEDDING
# =========================
def extract_embedding(audio, feature_extractor, model):
    """
    Extract single embedding from audio
    
    audio: numpy array (48000,)
    output: embedding normalized
    """
    try:
        # Feature extraction
        inputs = feature_extractor(
            audio, 
            sampling_rate=TARGET_SR, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get last hidden state and mean pool
        if hasattr(outputs, 'last_hidden_state'):
            emb = outputs.last_hidden_state.mean(dim=1)  # (1, D)
        else:
            emb = outputs.pooler_output  # Fallback
        
        emb = emb.squeeze().cpu().numpy()  # (D,)
        
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        return emb.astype('float32')
    
    except Exception as e:
        print(f"⚠️ Error extracting embedding: {e}")
        return None

# =========================
# BATCH EXTRACT
# =========================
def extract_all_embeddings(train_audios, feature_extractor, model):
    """Extract embeddings từ tất cả audios"""
    print("\n🚀 Extracting embeddings...")
    embeddings = []
    
    for audio in tqdm(train_audios, desc="Extracting", unit="sample"):
        emb = extract_embedding(audio, feature_extractor, model)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings, dtype='float32')
    
    print(f"✅ Extracted {embeddings.shape[0]} embeddings")
    print(f"   Shape: {embeddings.shape}")
    
    return embeddings

# =========================
# SAVE EMBEDDINGS
# =========================
def save_embeddings(embeddings, train_labels, model_key):
    """Save embeddings với tên model"""
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Tên file include model name
    emb_file = os.path.join(SAVE_PATH, f'embeddings_{model_key}.npy')
    label_file = os.path.join(SAVE_PATH, f'embeddings_labels_{model_key}.npy')
    
    print(f"\n💾 Saving embeddings...")
    np.save(emb_file, embeddings)
    np.save(label_file, train_labels)
    
    print(f"✅ Saved to {SAVE_PATH}")
    print(f"   {os.path.basename(emb_file)}: {embeddings.shape}")
    print(f"   {os.path.basename(label_file)}: {train_labels.shape}")
    
    return emb_file, label_file

# =========================
# PRINT HELP
# =========================
def print_available_models():
    """In danh sách model có sẵn"""
    print("\n" + "="*60)
    print("📋 AVAILABLE MODELS")
    print("="*60)
    for key, info in MODELS.items():
        print(f"\n  {key.upper()}")
        print(f"    HuggingFace: {info['name']}")
        print(f"    Dimension: {info['dim']}")
        print(f"    {info['description']}")
    print("\n" + "="*60)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description='Extract embeddings using different models')
    parser.add_argument(
        '--model', 
        type=str, 
        default='wavlm',
        choices=list(MODELS.keys()),
        help=f'Model to use (default: wavlm)'
    )
    parser.add_argument(
        '--list-models', 
        action='store_true',
        help='List all available models'
    )
    
    args = parser.parse_args()
    
    # Nếu user muốn xem danh sách model
    if args.list_models:
        print_available_models()
        return
    
    print("="*60)
    print(f"🎯 EXTRACT EMBEDDINGS - Model: {args.model.upper()}")
    print("="*60)
    
    # Load data
    train_audios, train_labels = load_data()
    
    # Load model
    feature_extractor, model, _ = load_model(args.model)
    
    # Extract embeddings
    embeddings = extract_all_embeddings(train_audios, feature_extractor, model)
    
    # Save
    emb_file, label_file = save_embeddings(embeddings, train_labels, args.model)
    
    print("\n" + "="*60)
    print("✅ EMBEDDING EXTRACTION DONE!")
    print("="*60)

if __name__ == "__main__":
    main()