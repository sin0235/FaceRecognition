"""Module tìm threshold tối ưu cho LBPH model."""
import numpy as np
from .evaluate_lbph import evaluate_lbph


def find_optimal_threshold(model, val_faces, val_labels, 
                          min_coverage=0.3, 
                          threshold_range=None):
    """
    Tìm threshold tối ưu cho LBPH model dựa trên validation set.
    
    Strategy: 
        - Maximize (accuracy * coverage) 
        - Subject to: coverage >= min_coverage
    
    Args:
        model: LBPH model đã train
        val_faces: List ảnh validation (grayscale)
        val_labels: numpy array labels tương ứng
        min_coverage: Coverage tối thiểu cần đảm bảo (default=0.3)
        threshold_range: Range các threshold để thử (default=range(40,121,5))
        
    Returns:
        best_threshold (int): Threshold tối ưu
        metrics_history (list): List of (threshold, accuracy, coverage, score) tuples
        
    Example:
        >>> best_thr, history = find_optimal_threshold(model, val_faces, val_labels)
        >>> print(f"Best threshold: {best_thr}")
    """
    if threshold_range is None:
        threshold_range = range(40, 121, 5)
    
    best_threshold = None
    best_score = -1
    metrics_history = []

    for threshold in threshold_range:
        # Evaluate với threshold này
        accuracy, coverage, used, _ = evaluate_lbph(
            model, val_faces, val_labels, threshold
        )
        
        # Tính score (chỉ xét nếu coverage đủ lớn)
        if coverage >= min_coverage:
            # Trade-off score: cân bằng giữa accuracy và coverage
            score = accuracy * coverage
            
            metrics_history.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'coverage': coverage,
                'used': used,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold

    if best_threshold is None:
        # Fallback: nếu không threshold nào thỏa min_coverage, chọn threshold cao nhất
        print(f"Warning: No threshold achieves min_coverage={min_coverage}")
        best_threshold = max(threshold_range)
    
    return best_threshold, metrics_history
