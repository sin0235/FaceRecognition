import numpy as np
from tqdm import tqdm

def evaluate_lbph(model, faces, labels, threshold):
    """
    Đánh giá model LBPH với threshold cho trước.
    
    Args:
        model: LBPH model đã train (cv2.face.LBPHFaceRecognizer)
        faces: List các ảnh mặt grayscale
        labels: numpy array các label tương ứng
        threshold: Ngưỡng confidence để accept prediction (lower is better)
        
    Returns:
        accuracy (float): Độ chính xác (0-1) trên samples được accept
        coverage (float): Tỷ lệ samples được accept (0-1)  
        used (int): Số lượng samples được accept
        confidences (np.array): Array các confidence values của tất cả predictions
        
    Note:
        - LBPH confidence: giá trị càng THẤP càng tốt (khác với cosine similarity)
        - Chỉ tính accuracy trên samples có conf < threshold
        - Coverage = used / total_samples
    """
    correct = 0
    used = 0
    confidences = []
    
    print(f"Evaluating {len(faces)} images with threshold={threshold}...")

    for img, true_label in tqdm(zip(faces, labels), total=len(faces), desc="Evaluating"):
        pred, conf = model.predict(img)
        confidences.append(conf)

        # Chỉ accept predictions có confidence < threshold
        if conf < threshold:
            used += 1
            if pred == true_label:
                correct += 1

    # Tính metrics dạng decimal (0-1), không phải percentage
    accuracy = (correct / used) if used > 0 else 0.0
    coverage = (used / len(labels)) if len(labels) > 0 else 0.0

    return accuracy, coverage, used, np.array(confidences)

