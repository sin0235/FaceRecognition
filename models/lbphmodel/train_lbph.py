import cv2
import numpy as np

def train_lbph_model(faces, labels, radius=1, neighbors=8, grid_x=8, grid_y=8):
    """
    Train LBPH (Local Binary Patterns Histograms) face recognizer.
    
    Args:
        faces: List các ảnh mặt grayscale để train
        labels: numpy array các label tương ứng (phải là int)
        radius: Bán kính của LBP operator (default=1)
        neighbors: Số neighbors của LBP operator (default=8)  
        grid_x: Số cells theo chiều ngang khi chia histogram grid (default=8)
        grid_y: Số cells theo chiều dọc khi chia histogram grid (default=8)
        
    Returns:
        model: LBPH face recognizer đã train (cv2.face.LBPHFaceRecognizer)
        
    Note:
        - LBPH parameters ảnh hưởng đến độ chính xác và tốc độ
        - radius=1, neighbors=8 là config phổ biến nhất
        - grid càng nhỏ (ít cells) thì model càng robust nhưng kém detail
    """
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y
    )
    
    # Convert labels sang numpy array nếu chưa phải
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=np.int32)
    
    model.train(faces, labels)
    return model
