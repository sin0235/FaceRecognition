import os
import cv2
import numpy as np

def load_data_no_haar(data_dir, max_images_per_identity=None):
    """
    Load dữ liệu ảnh mặt từ thư mục.
    
    Args:
        data_dir: Đường dẫn thư mục chứa các folder identity
        max_images_per_identity: Số ảnh tối đa mỗi identity (None = lấy tất cả)
    
    Returns:
        faces: List các ảnh grayscale
        labels: numpy array các label tương ứng
    """
    faces = []
    labels = []

    for label in sorted(os.listdir(data_dir), key=lambda x: int(x) if x.isdigit() else x):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        img_count = 0
        for img_name in os.listdir(label_path):
            if max_images_per_identity is not None and img_count >= max_images_per_identity:
                break
                
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(int(label))
            img_count += 1

    return faces, np.array(labels)
