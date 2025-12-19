"""Module tìm threshold tối ưu cho LBPH model."""
import numpy as np
import logging
from tqdm import tqdm


def find_optimal_threshold(model, faces, labels, 
                          min_coverage=0.3, 
                          threshold_range=None):
    """
    Tìm threshold tối ưu cho LBPH model dựa trên validation set.
    
    Strategy: 
        - Maximize (accuracy * coverage) 
        - Subject to: coverage >= min_coverage
    
    Optimization:
        - Chỉ gọi model.predict() 1 lần cho mỗi sample
        - Cache kết quả predictions và tái sử dụng cho tất cả thresholds
    
    Args:
        model: LBPH model đã train
        faces: List ảnh validation (grayscale)
        labels: numpy array labels tương ứng
        min_coverage: Coverage tối thiểu cần đảm bảo (default=0.3)
        threshold_range: Range các threshold để thử (default=range(40,121,5))
        
    Returns:
        best_threshold (int): Threshold tối ưu
        best_score (float): Score tốt nhất (accuracy * coverage)
        threshold_results (list): List of (threshold, accuracy, coverage, score) tuples
        
    Example:
        >>> best_thr, best_sc, results = find_optimal_threshold(model, val_faces, val_labels)
        >>> print(f"Best threshold: {best_thr}, Score: {best_sc}")
    """
    if threshold_range is None:
        threshold_range = range(40, 121, 5)
    
    # BƯỚC 1: Predict cho tất cả samples 1 lần duy nhất
    import time
    logging.info(f"[THRESHOLD] Predicting {len(faces)} validation samples...")
    predictions = []
    confidences = []
    start_time = time.time()
    
    for i, img in enumerate(tqdm(faces, desc="[THRESHOLD] Predicting faces", unit="face")):
        pred, conf = model.predict(img)
        predictions.append(pred)
        confidences.append(conf)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    total_time = time.time() - start_time
    print(f"[THRESHOLD] Prediction hoàn thành trong {total_time/60:.2f} phút!")
    
    # BƯỚC 2: Đánh giá với từng threshold (không cần predict lại)
    print(f"[THRESHOLD] Đang đánh giá {len(threshold_range)} thresholds...")
    best_threshold = None
    best_score = -1
    threshold_results = []
    
    for threshold in threshold_range:
        # Tìm samples được accept (confidence < threshold)
        accepted_mask = confidences < threshold
        used = np.sum(accepted_mask)
        
        if used > 0:
            # Tính accuracy trên samples được accept
            accepted_preds = predictions[accepted_mask]
            accepted_labels = labels[accepted_mask]
            correct = np.sum(accepted_preds == accepted_labels)
            accuracy = correct / used
        else:
            accuracy = 0.0
        
        coverage = used / len(labels) if len(labels) > 0 else 0.0
        
        # Tính score (chỉ xét nếu coverage đủ lớn)
        if coverage >= min_coverage:
            score = accuracy * coverage
            threshold_results.append((threshold, accuracy, coverage, score))
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    print(f"[THRESHOLD] Đánh giá hoàn thành!")
    
    if best_threshold is None:
        # Fallback: nếu không threshold nào thỏa min_coverage, chọn threshold cao nhất
        print(f"[WARNING] Không có threshold nào đạt min_coverage={min_coverage}")
        best_threshold = max(threshold_range)
        best_score = 0.0
    
    return best_threshold, best_score, threshold_results
