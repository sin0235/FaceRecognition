"""
Evaluation Module
Threshold tuning, metrics computation, ROC curves, confusion matrix
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
) -> Dict:
    """
    Tinh cac metrics co ban
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Danh sach ten labels
    
    Returns:
        Dict chua cac metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'total_samples': len(y_true),
        'correct': int(np.sum(y_true == y_pred)),
        'wrong': int(np.sum(y_true != y_pred))
    }


def threshold_sweep(
    similarities: np.ndarray,
    y_true: np.ndarray,
    y_pred_identities: np.ndarray,
    thresholds: List[float] = None
) -> Dict:
    """
    Sweep qua cac threshold de tim optimal
    
    Args:
        similarities: Mang confidence scores
        y_true: Ground truth labels
        y_pred_identities: Predicted identities (truoc khi ap dung threshold)
        thresholds: Danh sach thresholds can test
    
    Returns:
        Dict chua ket qua sweep
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95, 0.05)
    
    results = []
    
    for thresh in thresholds:
        y_pred = np.where(similarities >= thresh, y_pred_identities, -1)
        
        known_mask = y_pred != -1
        
        if np.sum(known_mask) == 0:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            full_pred = np.where(y_pred == -1, -1, y_pred)
            
            correct = np.sum((y_pred == y_true) & known_mask)
            total_predicted = np.sum(known_mask)
            total_true_positives = correct
            
            accuracy = correct / len(y_true) if len(y_true) > 0 else 0
            precision = correct / total_predicted if total_predicted > 0 else 0
            recall = correct / len(y_true) if len(y_true) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        known_ratio = np.sum(known_mask) / len(y_true) if len(y_true) > 0 else 0
        
        results.append({
            'threshold': float(thresh),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'known_ratio': float(known_ratio),
            'num_known': int(np.sum(known_mask)),
            'num_unknown': int(np.sum(~known_mask))
        })
    
    best_f1_idx = np.argmax([r['f1'] for r in results])
    best_acc_idx = np.argmax([r['accuracy'] for r in results])
    
    return {
        'results': results,
        'best_f1_threshold': results[best_f1_idx]['threshold'],
        'best_f1_score': results[best_f1_idx]['f1'],
        'best_accuracy_threshold': results[best_acc_idx]['threshold'],
        'best_accuracy_score': results[best_acc_idx]['accuracy']
    }


def plot_threshold_sweep(
    sweep_results: Dict,
    output_path: str = None,
    show: bool = True
) -> None:
    """
    Ve bieu do threshold sweep
    """
    results = sweep_results['results']
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1'] for r in results]
    known_ratios = [r['known_ratio'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(thresholds, accuracies, 'b-o', label='Accuracy', linewidth=2)
    ax1.plot(thresholds, f1_scores, 'r-s', label='F1 Score', linewidth=2)
    ax1.axvline(x=sweep_results['best_f1_threshold'], color='r', linestyle='--', 
                label=f"Best F1: {sweep_results['best_f1_threshold']:.2f}")
    ax1.axvline(x=sweep_results['best_accuracy_threshold'], color='b', linestyle='--',
                label=f"Best Acc: {sweep_results['best_accuracy_threshold']:.2f}")
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Threshold vs Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(thresholds, known_ratios, 'g-^', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Known Ratio')
    ax2.set_title('Threshold vs Known Ratio')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved threshold sweep plot: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_path: str = None,
    show: bool = True
) -> Dict:
    """
    Ve ROC curve cho binary classification (known vs unknown)
    
    Args:
        y_true: Binary labels (1 = correct prediction, 0 = wrong)
        y_scores: Confidence scores
    
    Returns:
        Dict chua fpr, tpr, thresholds, auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.scatter([eer], [1-eer], color='red', s=100, zorder=5, 
               label=f'EER = {eer:.3f} (thresh={eer_threshold:.2f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    output_path: str = None,
    show: bool = True,
    max_classes: int = 20
) -> np.ndarray:
    """
    Ve confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape[0] > max_classes:
        print(f"Confusion matrix qua lon ({cm.shape[0]} classes), chi hien thi {max_classes} classes dau")
        cm = cm[:max_classes, :max_classes]
        if labels:
            labels = labels[:max_classes]
    
    figsize = max(8, cm.shape[0] * 0.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm


def evaluate_recognition_engine(
    engine,
    test_images: List[str],
    test_labels: List[str],
    output_dir: str = "results/evaluation"
) -> Dict:
    """
    Danh gia RecognitionEngine tren test set
    
    Args:
        engine: RecognitionEngine instance
        test_images: List duong dan anh test
        test_labels: List ground truth labels
        output_dir: Thu muc luu ket qua
    
    Returns:
        Dict chua ket qua danh gia
    """
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = []
    confidences = []
    
    print(f"Evaluating {len(test_images)} images...")
    
    for img_path in test_images:
        result = engine.recognize(img_path)
        predictions.append(result['identity'])
        confidences.append(result['confidence'])
    
    y_true = np.array(test_labels)
    y_pred = np.array(predictions)
    y_scores = np.array(confidences)
    
    unique_labels = list(set(test_labels))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    
    y_true_ids = np.array([label_to_id.get(l, -1) for l in y_true])
    y_pred_ids = np.array([label_to_id.get(p, -1) for p in y_pred])
    
    metrics = compute_metrics(y_true, y_pred, unique_labels)
    
    y_correct = (y_true == y_pred).astype(int)
    roc_results = plot_roc_curve(
        y_correct, y_scores,
        output_path=os.path.join(output_dir, "roc_curve.png"),
        show=False
    )
    
    cm = plot_confusion_matrix(
        y_true, y_pred,
        labels=unique_labels,
        output_path=os.path.join(output_dir, "confusion_matrix.png"),
        show=False
    )
    
    sweep_results = threshold_sweep(
        y_scores, y_true_ids, y_pred_ids
    )
    plot_threshold_sweep(
        sweep_results,
        output_path=os.path.join(output_dir, "threshold_sweep.png"),
        show=False
    )
    
    report = generate_report(metrics, roc_results, sweep_results, output_dir)
    
    return {
        'metrics': metrics,
        'roc': roc_results,
        'threshold_sweep': sweep_results,
        'confusion_matrix': cm,
        'predictions': predictions,
        'confidences': confidences
    }


def generate_report(
    metrics: Dict,
    roc_results: Dict,
    sweep_results: Dict,
    output_dir: str
) -> str:
    """
    Tao report markdown
    """
    report = f"""# Evaluation Report

## Summary Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision (weighted) | {metrics['precision_weighted']:.4f} |
| Recall (weighted) | {metrics['recall_weighted']:.4f} |
| F1 Score (weighted) | {metrics['f1_weighted']:.4f} |
| Total Samples | {metrics['total_samples']} |
| Correct | {metrics['correct']} |
| Wrong | {metrics['wrong']} |

## ROC Analysis

| Metric | Value |
|--------|-------|
| AUC | {roc_results['auc']:.4f} |
| EER | {roc_results['eer']:.4f} |
| EER Threshold | {roc_results['eer_threshold']:.4f} |

## Threshold Tuning

| Recommendation | Threshold | Score |
|----------------|-----------|-------|
| Best F1 | {sweep_results['best_f1_threshold']:.2f} | {sweep_results['best_f1_score']:.4f} |
| Best Accuracy | {sweep_results['best_accuracy_threshold']:.2f} | {sweep_results['best_accuracy_score']:.4f} |

## Visualizations

- ROC Curve: `roc_curve.png`
- Confusion Matrix: `confusion_matrix.png`
- Threshold Sweep: `threshold_sweep.png`
"""
    
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved report: {report_path}")
    
    return report


if __name__ == "__main__":
    print("="*60)
    print("EVALUATION MODULE TEST")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 100
    n_classes = 10
    
    y_true = np.random.randint(0, n_classes, n_samples)
    
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    y_scores = np.random.uniform(0.3, 1.0, n_samples)
    y_scores[y_pred == y_true] += 0.2
    y_scores = np.clip(y_scores, 0, 1)
    
    metrics = compute_metrics(y_true, y_pred)
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    sweep = threshold_sweep(y_scores, y_true, y_pred)
    print(f"\nThreshold Sweep:")
    print(f"  Best F1 threshold: {sweep['best_f1_threshold']:.2f}")
    print(f"  Best accuracy threshold: {sweep['best_accuracy_threshold']:.2f}")
    
    print("\nModule test thanh cong!")
