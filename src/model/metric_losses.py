"""
Hàm Loss và Metric để đánh giá mô hình speaker identification
Sử dụng cho evaluation và model comparison
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)


# =========================
# CLASSIFICATION METRICS
# =========================
def compute_accuracy(y_true, y_pred):
    """Tính Accuracy"""
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true, y_pred, average='weighted'):
    """Tính Precision
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'weighted', 'macro', 'micro', None
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true, y_pred, average='weighted'):
    """Tính Recall
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'weighted', 'macro', 'micro', None
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true, y_pred, average='weighted'):
    """Tính F1-Score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'weighted', 'macro', 'micro', None
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(y_true, y_pred):
    """Tính Confusion Matrix"""
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(y_true, y_pred, target_names=None):
    """Tính Classification Report"""
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


# =========================
# LOSS FUNCTIONS
# =========================
def cross_entropy_loss(y_true, y_pred_proba):
    """
    Cross Entropy Loss (Log Loss)
    Đo lường sự khác biệt giữa true distribution và predicted distribution
    
    Args:
        y_true: True class labels (0 to num_classes-1)
        y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
    
    Returns:
        float: Cross entropy loss
    """
    # Ensure proper shapes
    if len(y_pred_proba.shape) == 1:
        raise ValueError("y_pred_proba should be 2D: (n_samples, n_classes)")
    
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # Calculate log loss
    return log_loss(y_true, y_pred_proba)


def focal_loss(y_true, y_pred_proba, alpha=0.25, gamma=2.0):
    """
    Focal Loss - tập trung vào hard examples
    Hữu ích khi có class imbalance
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    
    Returns:
        float: Focal loss
    """
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # Convert to one-hot
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    
    # Calculate focal loss
    p_t = np.sum(y_true_one_hot * y_pred_proba, axis=1)
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss_val = -np.mean(focal_weight * np.log(p_t))
    
    return focal_loss_val


def confidence_loss(confidences, y_true, y_pred):
    """
    Confidence Loss - đo lường sự tự tin của mô hình
    Mục tiêu: Mô hình nên tự tin khi đúng, không tự tin khi sai
    
    Args:
        confidences: Mảng confidence scores (0-1)
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Statistics về confidence loss
    """
    correct_mask = (y_true == y_pred)
    incorrect_mask = ~correct_mask
    
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[incorrect_mask]
    
    results = {
        'avg_confidence_correct': np.mean(correct_confidences) if len(correct_confidences) > 0 else 0,
        'avg_confidence_incorrect': np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0,
        'std_confidence_correct': np.std(correct_confidences) if len(correct_confidences) > 0 else 0,
        'std_confidence_incorrect': np.std(incorrect_confidences) if len(incorrect_confidences) > 0 else 0,
    }
    
    # Loss: penalize high confidence on wrong predictions
    if len(incorrect_confidences) > 0:
        results['wrong_high_confidence_loss'] = np.mean(incorrect_confidences)
    else:
        results['wrong_high_confidence_loss'] = 0.0
    
    return results


# =========================
# RANKING METRICS
# =========================
def compute_top_k_accuracy(y_true, y_pred_proba, k=1):
    """
    Tính Top-K Accuracy
    Kiểm tra xem true label có trong top-k predictions không
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (shape: n_samples x n_classes)
        k: Number of top predictions to consider
    
    Returns:
        float: Top-k accuracy
    """
    n_samples = len(y_true)
    correct = 0
    
    for i in range(n_samples):
        top_k_preds = np.argsort(y_pred_proba[i])[-k:]
        if y_true[i] in top_k_preds:
            correct += 1
    
    return correct / n_samples


def compute_mean_average_precision(y_true, y_pred_scores, k=None):
    """
    Tính Mean Average Precision (MAP)
    Hữu ích cho ranking/retrieval tasks
    
    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred_scores: Predicted scores/confidence
        k: Top-k to evaluate (None = all)
    
    Returns:
        float: MAP score
    """
    # Sort by predicted scores (descending)
    sorted_idx = np.argsort(-y_pred_scores)
    y_true_sorted = y_true[sorted_idx]
    
    if k is not None:
        y_true_sorted = y_true_sorted[:k]
    
    # Calculate precision at each position
    hits = 0
    ap_sum = 0
    
    for i, label in enumerate(y_true_sorted):
        if label == 1:
            hits += 1
            ap_sum += hits / (i + 1)
    
    if hits == 0:
        return 0.0
    
    return ap_sum / hits


# =========================
# COMPREHENSIVE EVALUATION
# =========================
def evaluate_model(y_true, y_pred, y_pred_proba=None, confidences=None, target_names=None):
    """
    Hàm tổng hợp để đánh giá mô hình
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        confidences: Confidence scores (optional)
        target_names: Names of labels (optional)
    
    Returns:
        dict: Toàn bộ metrics
    """
    results = {}
    
    # Classification Metrics
    results['accuracy'] = compute_accuracy(y_true, y_pred)
    results['precision_weighted'] = compute_precision(y_true, y_pred, average='weighted')
    results['recall_weighted'] = compute_recall(y_true, y_pred, average='weighted')
    results['f1_weighted'] = compute_f1(y_true, y_pred, average='weighted')
    results['precision_macro'] = compute_precision(y_true, y_pred, average='macro')
    results['recall_macro'] = compute_recall(y_true, y_pred, average='macro')
    results['f1_macro'] = compute_f1(y_true, y_pred, average='macro')
    
    # Confusion Matrix
    results['confusion_matrix'] = compute_confusion_matrix(y_true, y_pred).tolist()
    
    # Classification Report
    if target_names:
        results['classification_report'] = compute_classification_report(
            y_true, y_pred, target_names=target_names
        )
    
    # Loss Functions
    if y_pred_proba is not None:
        results['cross_entropy_loss'] = cross_entropy_loss(y_true, y_pred_proba)
        results['focal_loss'] = focal_loss(y_true, y_pred_proba)
        results['top_1_accuracy'] = compute_top_k_accuracy(y_true, y_pred_proba, k=1)
        results['top_3_accuracy'] = compute_top_k_accuracy(y_true, y_pred_proba, k=3)
        results['top_5_accuracy'] = compute_top_k_accuracy(y_true, y_pred_proba, k=5)
    
    # Confidence Metrics
    if confidences is not None:
        results['confidence_metrics'] = confidence_loss(confidences, y_true, y_pred)
        results['mean_confidence'] = np.mean(confidences)
        results['std_confidence'] = np.std(confidences)
    
    return results


def print_evaluation_report(results, model_name="Model"):
    """In báo cáo evaluation"""
    print("\n" + "="*80)
    print(f"📊 EVALUATION REPORT - {model_name}")
    print("="*80)
    
    # Classification Metrics
    print("\n📈 CLASSIFICATION METRICS:")
    print(f"  Accuracy:              {results.get('accuracy', 0):.4f}")
    print(f"  Precision (Weighted):  {results.get('precision_weighted', 0):.4f}")
    print(f"  Recall (Weighted):     {results.get('recall_weighted', 0):.4f}")
    print(f"  F1-Score (Weighted):   {results.get('f1_weighted', 0):.4f}")
    print(f"  Precision (Macro):     {results.get('precision_macro', 0):.4f}")
    print(f"  Recall (Macro):        {results.get('recall_macro', 0):.4f}")
    print(f"  F1-Score (Macro):      {results.get('f1_macro', 0):.4f}")
    
    # Loss Functions
    if 'cross_entropy_loss' in results:
        print("\n🔴 LOSS FUNCTIONS:")
        print(f"  Cross Entropy Loss:    {results['cross_entropy_loss']:.4f}")
        print(f"  Focal Loss:            {results['focal_loss']:.4f}")
    
    # Top-k Accuracy
    if 'top_1_accuracy' in results:
        print("\n🎯 TOP-K ACCURACY:")
        print(f"  Top-1:                 {results['top_1_accuracy']:.4f}")
        print(f"  Top-3:                 {results['top_3_accuracy']:.4f}")
        print(f"  Top-5:                 {results['top_5_accuracy']:.4f}")
    
    # Confidence Metrics
    if 'confidence_metrics' in results:
        print("\n💪 CONFIDENCE ANALYSIS:")
        conf = results['confidence_metrics']
        print(f"  Avg Confidence (Correct):   {conf['avg_confidence_correct']:.4f}")
        print(f"  Avg Confidence (Incorrect): {conf['avg_confidence_incorrect']:.4f}")
        print(f"  Wrong Pred High Conf Loss:  {conf['wrong_high_confidence_loss']:.4f}")
    
    print("\n" + "="*80)


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # Example
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    confidences = np.random.rand(n_samples)
    
    # Evaluate
    results = evaluate_model(y_true, y_pred, y_pred_proba, confidences)
    print_evaluation_report(results, "Example Model")
