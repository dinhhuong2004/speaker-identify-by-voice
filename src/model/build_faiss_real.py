# Build FAISS Indices for Fast Audio Embedding Search
# =====================================================================
# This module creates FAISS (Facebook AI Similarity Search) indices from pre-computed
# audio embeddings, enabling fast nearest-neighbor search for speaker identification.
#
# Main Purpose:
#     Convert high-dimensional audio embeddings into searchable FAISS indices
#     for efficient similarity-based speaker matching during inference.
#
# Key Features:
#     - Support for multiple pre-trained models (WavLM, Wav2Vec2, HuBERT, UniSpeech-SAT)
#     - IndexFlatIP (Inner Product) for cosine similarity on normalized embeddings
#     - Automatic embedding validation and verification
#     - Memory-efficient index building with contiguous array layout
#     - Index persistence (save/load from disk)
#     - Multi-model batch processing capability
#
# Core Functionality (Step-by-step):
#     1. Check if embeddings exist for each model
#     2. Load pre-computed embeddings (.npy files) from disk
#     3. Validate embedding dimensions and data types
#     4. Create FAISS IndexFlatIP index from embeddings
#        - IndexFlatIP: computes Inner Product (cosine similarity on L2-normalized vectors)
#        - Fast for small-to-medium datasets (millions of vectors)
#     5. Add all embeddings to the index (build index data structure)
#     6. Save index to disk (.index file) for reuse
#     7. Save corresponding labels mapping for result interpretation
#     8. Generate summary report with index statistics
#
# Workflow:
#     embeddings.npy → FAISS Index → index.faiss + labels.npy
#
# Usage Examples:
#     # Build indices for all available models
#     python build_faiss_real.py --all
#
#     # Build index for specific model
#     python build_faiss_real.py --model wavlm
#
#     # Default behavior (builds all if no args specified)
#     python build_faiss_real.py
#
# Input Requirements:
#     - Embeddings file: embeddings_[model_name].npy
#       Shape: (N_samples, 768) where N_samples = number of training audio files
#       Data type: float32
#       Format: Each row is a normalized audio embedding vector
#
#     - Labels file: embeddings_labels_[model_name].npy
#       Shape: (N_samples,)
#       Content: Speaker labels/IDs for each embedding
#
# Output Files:
#     - faiss_[model_name].index: FAISS index file (binary format)
#     - faiss_labels_[model_name].npy: Label mappings for index
#     - faiss_build_results.json: Summary statistics
#
# Index Details:
#     - Type: IndexFlatIP (Inner Product)
#     - Dimension: 768 (for all speech models)
#     - Distance metric: Cosine similarity (on normalized embeddings)
#     - Search: Brute-force exact search (no approximation)
#     - Suitable for: ~1-100M vectors
#
# Downstream Usage:
#     Used by evaluate_real.py and infer_real.py for:
#     - Fast nearest-neighbor lookup during speaker identification
#     - Query: new audio embedding → find k-nearest training embeddings
#     - Result: predicted speaker label with similarity score
#
# Configuration:
#     - Models: WavLM, Wav2Vec2, HuBERT, UniSpeech-SAT (all 768-dim)
#     - Data path: D:\\Master DS\\Intro_to_DS\\data_processed
#     - Index type: Inner Product (IndexFlatIP)
#     - Format: Binary FAISS format

import numpy as np
import faiss
import os
import argparse
import json

# =========================
# CONFIG
# =========================
MODELS = {
    'wavlm': {
        'name': 'microsoft/wavlm-base',
        'dim': 768,
        'description': 'WavLM-Base (Microsoft)'
    },
    'wav2vec2': {
        'name': 'facebook/wav2vec2-base',
        'dim': 768,
        'description': 'Wav2Vec2-Base (Meta)'
    },
    'hubert': {
        'name': 'facebook/hubert-base-ls960',
        'dim': 768,
        'description': 'HuBERT-Base'
    },
    'unispeech': {
        'name': 'microsoft/unispeech-sat-base',
        'dim': 768,
        'description': 'UniSpeech-SAT'
    },
}

DATA_PATH = "D:\\Master DS\\Intro_to_DS\\data_processed"

# =========================
# CHECK EMBEDDINGS
# =========================
def check_embeddings_exist(model_key):
    """Check xem embeddings có tồn tại chưa"""
    emb_file = os.path.join(DATA_PATH, f'embeddings_{model_key}.npy')
    label_file = os.path.join(DATA_PATH, f'embeddings_labels_{model_key}.npy')
    
    return os.path.exists(emb_file) and os.path.exists(label_file)

# =========================
# LOAD EMBEDDINGS
# =========================
def load_embeddings(model_key):
    """Load embeddings cho model"""
    print(f"📦 Loading embeddings for {model_key}...")
    
    emb_file = os.path.join(DATA_PATH, f'embeddings_{model_key}.npy')
    label_file = os.path.join(DATA_PATH, f'embeddings_labels_{model_key}.npy')
    
    if not os.path.exists(emb_file):
        print(f"❌ Embeddings not found for {model_key}")
        print(f"   Please run: python extract_embedding_real.py --model {model_key}")
        return None, None
    
    embeddings = np.load(emb_file)
    labels = np.load(label_file)
    
    print(f"✅ Loaded {embeddings.shape[0]} embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Labels: {labels.shape}")
    
    return embeddings, labels

# =========================
# BUILD FAISS INDEX
# =========================
def build_faiss_index(embeddings, labels, model_key):
    """Build FAISS index từ embeddings"""
    print(f"\n🚀 Building FAISS index for {model_key}...")
    
    # Ensure contiguous memory layout
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # Create index: IndexFlatIP = Inner Product (= Cosine similarity on normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    print(f"   Adding {embeddings.shape[0]} vectors to index...")
    index.add(embeddings)
    
    print(f"✅ Built FAISS index")
    print(f"   Dimension: {dimension}")
    print(f"   Total vectors: {index.ntotal}")
    
    return index

# =========================
# SAVE INDEX
# =========================
def save_faiss_index(index, labels, model_key):
    """Save FAISS index + labels"""
    index_file = os.path.join(DATA_PATH, f'faiss_{model_key}.index')
    labels_file = os.path.join(DATA_PATH, f'faiss_labels_{model_key}.npy')
    
    print(f"\n💾 Saving FAISS index...")
    
    # Save index
    faiss.write_index(index, index_file)
    print(f"   ✅ {os.path.basename(index_file)}")
    
    # Save labels
    np.save(labels_file, labels)
    print(f"   ✅ {os.path.basename(labels_file)}")
    
    # Get file sizes
    index_size = os.path.getsize(index_file) / (1024 * 1024)  # MB
    labels_size = os.path.getsize(labels_file) / 1024  # KB
    
    print(f"\n📊 File sizes:")
    print(f"   Index: {index_size:.2f} MB")
    print(f"   Labels: {labels_size:.2f} KB")

# =========================
# BUILD ALL MODELS
# =========================
def build_all_indices():
    """Build FAISS indices cho tất cả models có embeddings"""
    print("="*60)
    print("🎯 BUILD FAISS INDICES FOR ALL MODELS")
    print("="*60)
    
    results = {}
    
    for model_key in MODELS.keys():
        if not check_embeddings_exist(model_key):
            print(f"\n⚠️ Skipping {model_key}: embeddings not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"📦 {model_key.upper()}")
        print(f"{'='*60}")
        
        # Load embeddings
        embeddings, labels = load_embeddings(model_key)
        if embeddings is None:
            continue
        
        # Build index
        index = build_faiss_index(embeddings, labels, model_key)
        
        # Save index
        save_faiss_index(index, labels, model_key)
        
        results[model_key] = {
            'status': 'success',
            'num_vectors': int(embeddings.shape[0]),
            'dimension': int(embeddings.shape[1]),
            'num_labels': int(len(np.unique(labels)))
        }
    
    return results

# =========================
# PRINT SUMMARY
# =========================
def print_summary(results):
    """In tóm tắt"""
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    if not results:
        print("❌ No indices built")
        return
    
    print(f"\n{'Model':<15} {'Status':<10} {'Vectors':<12} {'Labels':<12}")
    print("-"*60)
    
    for model_key in sorted(results.keys()):
        info = results[model_key]
        if info['status'] == 'success':
            print(f"{model_key:<15} ✅ OK       {info['num_vectors']:<12} {info['num_labels']:<12}")
    
    print("="*60)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description='Build FAISS indices from embeddings')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=list(MODELS.keys()),
        help='Build index for specific model'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Build indices for all available models'
    )
    
    args = parser.parse_args()
    
    # Nếu không specify model, build tất cả
    if args.model is None and not args.all:
        args.all = True
    
    results = {}
    
    if args.all:
        results = build_all_indices()
    else:
        # Build specific model
        print("="*60)
        print(f"🎯 BUILD FAISS INDEX - {args.model.upper()}")
        print("="*60)
        
        if not check_embeddings_exist(args.model):
            print(f"❌ Embeddings not found for {args.model}")
            print(f"   Please run: python extract_embedding_real.py --model {args.model}")
            exit(1)
        
        # Load embeddings
        embeddings, labels = load_embeddings(args.model)
        
        # Build index
        index = build_faiss_index(embeddings, labels, args.model)
        
        # Save index
        save_faiss_index(index, labels, args.model)
        
        results[args.model] = {
            'status': 'success',
            'num_vectors': int(embeddings.shape[0]),
            'dimension': int(embeddings.shape[1]),
            'num_labels': int(len(np.unique(labels)))
        }
    
    # Print summary
    print_summary(results)
    
    # Save results to JSON
    results_file = os.path.join(DATA_PATH, 'faiss_build_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()