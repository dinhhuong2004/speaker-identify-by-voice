import numpy as np
import torch
import faiss
import os
import argparse
import gc # Garbage collection để giải phóng bộ nhớ sau mỗi model
from tqdm import tqdm
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import json
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))
from metric_losses import (
    focal_loss,
    confidence_loss,
    compute_top_k_accuracy
)

# =========================
# UTILITY FUNCTIONS
# =========================
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

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

# =========================
# LOAD DATA
# =========================
def load_data():
    """Load test data"""
    print("📦 Loading test data...")
    test_audios = np.load(os.path.join(DATA_PATH, 'test_audios.npy'))
    test_labels = np.load(os.path.join(DATA_PATH, 'test_labels.npy'), allow_pickle=True)
    
    print(f"✅ Loaded {test_audios.shape[0]} test samples")
    return test_audios, test_labels

def load_model(model_key):
    """Load pre-trained model"""
    if model_key not in MODELS:
        print(f"❌ Model '{model_key}' không hỗ trợ!")
        exit(1)
    
    model_info = MODELS[model_key]
    print(f"\n🔧 Loading model: {model_key}")
    print(f"   {model_info['description']}")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_info['name'])
    model = AutoModel.from_pretrained(model_info['name']).to(DEVICE)
    model.eval()
    
    print(f"   ✅ on {DEVICE}")
    return feature_extractor, model

def load_embeddings(model_key):
    """Load pre-computed embeddings"""
    print(f"\n📦 Loading embeddings for {model_key}...")
    
    emb_file = os.path.join(DATA_PATH, f'embeddings_{model_key}.npy')
    label_file = os.path.join(DATA_PATH, f'embeddings_labels_{model_key}.npy')
    
    # Kiểm tra embeddings có tồn tại
    if not os.path.exists(emb_file):
        print(f"❌ Embeddings not found for {model_key}")
        print(f"   Please run first: python extract_embedding_real.py --model {model_key}")
        exit(1)
    
    embeddings = np.load(emb_file)
    labels = np.load(label_file, allow_pickle=True)
    
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    print(f"✅ Loaded labels: {labels.shape}")
    
    return embeddings, labels

def load_faiss_index(embeddings, labels, model_key):
    """Load FAISS index hoặc build nếu chưa có"""
    print(f"\n🔍 Loading FAISS index for {model_key}...")
    
    index_file = os.path.join(DATA_PATH, f'faiss_{model_key}.index')
    labels_file = os.path.join(DATA_PATH, f'faiss_labels_{model_key}.npy')
    
    # Nếu index tồn tại, load
    if os.path.exists(index_file):
        print(f"✅ Loading existing index...")
        index = faiss.read_index(index_file)
        faiss_labels = np.load(labels_file, allow_pickle=True)
        print(f"   Index: {index.ntotal} vectors")
        return index, faiss_labels
    
    # Nếu không, build mới
    print(f"⚠️ Index not found. Building now...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, index_file)
    np.save(labels_file, labels)
    
    print(f"✅ Built and saved index: {index.ntotal} vectors")
    
    return index, labels

# =========================
# EXTRACT EMBEDDING
# =========================
def extract_embedding(audio, feature_extractor, model):
    """Extract embedding từ audio"""
    try:
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
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

# =========================
# EVALUATE
# =========================
def evaluate(test_audios, test_labels, feature_extractor, model, index, faiss_labels):
    """Evaluate model trên test set"""
    print("\n🎯 Evaluating on test set...")
    predictions = []
    confidences = []
    
    for audio in tqdm(test_audios, desc="Evaluating", unit="sample"):
        emb = extract_embedding(audio, feature_extractor, model)
        
        # Handle None embeddings
        if emb is None:
            predictions.append(None)
            confidences.append(0.0)
            continue
        
        emb = emb.reshape(1, -1)
        
        # Search k=1 nearest neighbor
        scores, indices = index.search(emb, 1)
        pred_label = faiss_labels[indices[0][0]]
        confidence = scores[0][0]
        
        predictions.append(pred_label)
        confidences.append(confidence)
    
    predictions = np.array(predictions, dtype=object)
    confidences = np.array(confidences)
    
    return predictions, confidences

# =========================
# COMPUTE METRICS
# =========================
def compute_metrics(test_labels, predictions, confidences):
    """Tính toán metrics - Accuracy, Precision, Recall, F1-Score, Focal Loss, etc."""
    # Debug: Check predictions
    print(f"\nDebug: Total predictions: {len(predictions)}")
    print(f"Debug: Prediction dtype: {predictions.dtype}")
    print(f"Debug: Label dtype: {test_labels.dtype}")
    
    # Filter out None predictions
    valid_mask = np.array([p is not None for p in predictions])
    valid_predictions = np.array([p for p in predictions], dtype=test_labels.dtype)[valid_mask]
    valid_labels = test_labels[valid_mask]
    valid_confidences = confidences[valid_mask]
    
    print(f"Debug: Valid predictions: {len(valid_predictions)}")
    print(f"Debug: Valid predictions dtype: {valid_predictions.dtype}")
    
    if len(valid_predictions) == 0:
        print("❌ No valid predictions!")
        return None
    
    try:
        # Accuracy
        accuracy = accuracy_score(valid_labels, valid_predictions)
        
        # Precision, Recall, F1-Score (Weighted)
        precision_weighted = precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
        
        # Precision, Recall, F1-Score (Macro)
        precision_macro = precision_score(valid_labels, valid_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(valid_labels, valid_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(valid_labels, valid_predictions, average='macro', zero_division=0)
        
        # Top-1 Accuracy (same as accuracy for FAISS k=1)
        top_1_accuracy = accuracy
        
        # Focal Loss (giúp detect hard examples)
        # Convert to one-hot for focal_loss calculation
        n_classes = len(np.unique(valid_labels))
        y_pred_proba = np.zeros((len(valid_predictions), n_classes))
        for i, pred in enumerate(valid_predictions):
            y_pred_proba[i, pred] = valid_confidences[i]  # Use confidence as proxy for prob
        
        try:
            focal_loss_val = focal_loss(valid_labels, y_pred_proba, alpha=0.25, gamma=2.0)
        except:
            focal_loss_val = 0.0
        
        # Confidence Loss Analysis
        confidence_metrics = confidence_loss(valid_confidences, valid_labels, valid_predictions)
        # Convert confidence metrics to JSON serializable format
        confidence_metrics = convert_to_serializable(confidence_metrics)
        
        correct = np.sum(valid_predictions == valid_labels)
        total = len(valid_predictions)
        
        results = {
            # Classification metrics (cần thiết)
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            
            # Confidence analysis (cần thiết)
            'confidence_metrics': confidence_metrics,
            
            # Stats (cần thiết)
            'correct': int(correct),
            'total': int(total),
            'skipped': int(np.sum(~valid_mask)),
            'mean_confidence': float(valid_confidences.mean()),
        }

        
        # Confusion Matrix (as list for JSON serialization)
        cm = confusion_matrix(valid_labels, valid_predictions)
        results['confusion_matrix'] = cm.tolist()
        
        return results
    except Exception as e:
        print(f"❌ Error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# =========================
# PRINT RESULTS (TABLE FORMAT)
# =========================
def print_results_table(results, model_key):
    """In kết quả dưới dạng 1 bảng duy nhất với các metric cần thiết"""
    print("\n" + "="*100)
    print(f"📊 EVALUATION RESULTS - {model_key.upper()}")
    print("="*100)
    
    conf = results['confidence_metrics']
    
    # Bảng chứa các metric cần thiết
    print(f"{'Metric':<35} {'Value':<30}")
    print("-"*100)
    
    # Row 1: Accuracy
    print(f"{'Accuracy':<35} {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Row 2: Precision
    print(f"{'Precision':<35} {results['precision_weighted']:.4f}")
    
    # Row 3: Recall
    print(f"{'Recall':<35} {results['recall_weighted']:.4f}")
    
    # Row 4: F1-Score
    print(f"{'F1-Score':<35} {results['f1_weighted']:.4f}")
    
    # Row 5: Mean Confidence
    print(f"{'Mean Confidence':<35} {results['mean_confidence']:.4f}")
    
    # Row 6: Avg Confidence (Correct)
    print(f"{'Avg Confidence (Correct)':<35} {conf['avg_confidence_correct']:.4f}")
    
    # Row 7: Avg Confidence (Incorrect)
    print(f"{'Avg Confidence (Incorrect)':<35} {conf['avg_confidence_incorrect']:.4f}")
    
    # Row 8: Correct Predictions
    print(f"{'Correct Predictions':<35} {results['correct']}/{results['total']}")
    
    # Row 9: Incorrect Predictions
    print(f"{'Incorrect Predictions':<35} {results['total'] - results['correct']}")
    
    # Row 10: Success Rate
    print(f"{'Success Rate':<35} {(results['correct']/results['total']*100):.2f}%")
    
    # Row 11: Skipped Samples
    print(f"{'Skipped Samples':<35} {results['skipped']}")
    
    print("="*100)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description='Evaluate embeddings using different models')
    parser.add_argument(
        '--model',
        type=str,
        default='wavlm',
        choices=list(MODELS.keys()),
        help='Model to evaluate (default: wavlm)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluate all available models'
    )
    
    args = parser.parse_args()
    
    # Load test data
    test_audios, test_labels = load_data()
    
    models_to_eval = list(MODELS.keys()) if args.all else [args.model]
    all_results = {}
    
    for model_key in models_to_eval:
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATING: {model_key.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load model
            feature_extractor, model = load_model(model_key)
            
            # Load embeddings (đã extract trước)
            embeddings, train_labels = load_embeddings(model_key)
            
            # Load FAISS index (build nếu chưa có)
            index, faiss_labels = load_faiss_index(embeddings, train_labels, model_key)
            
            # Evaluate
            predictions, confidences = evaluate(
                test_audios, test_labels, feature_extractor, model, index, faiss_labels
            )
            
            # Compute metrics
            results = compute_metrics(test_labels, predictions, confidences)
            
            if results is None:
                print("⚠️ Skipping this model due to metric computation error")
                continue
            
            all_results[model_key] = results
            
            # Print results (dạng bảng)
            print_results_table(results, model_key)
            
            # Clean up memory
            del model, feature_extractor, embeddings, index
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error evaluating {model_key}: {e}")
            continue
    
    # Compare if evaluating multiple models
    if len(all_results) > 1:
        print("\n" + "="*110)
        print("📊 MODEL COMPARISON")
        print("="*110)
        print(f"{'Model':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Mean Conf':<15}")
        print("-"*110)
        for model_key in sorted(all_results.keys()):
            acc = all_results[model_key]['accuracy']
            prec = all_results[model_key]['precision_weighted']
            rec = all_results[model_key]['recall_weighted']
            f1 = all_results[model_key]['f1_weighted']
            conf = all_results[model_key]['mean_confidence']
            print(f"{model_key:<15} {acc*100:>6.2f}%       {prec:>6.4f}       {rec:>6.4f}       {f1:>6.4f}       {conf:>6.4f}")
        print("="*110)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(DATA_PATH, f'evaluation_results_{timestamp}.json')
    # Convert all results to JSON serializable format
    all_results_serializable = convert_to_serializable(all_results)
    with open(results_file, 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()