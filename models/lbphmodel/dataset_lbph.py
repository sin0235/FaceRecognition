import os
import cv2
import numpy as np

def load_data_no_haar(data_dir, max_images_per_identity=None, max_identities=None):
    """
    Load dữ liệu ảnh mặt từ thư mục.
    
    Args:
        data_dir: Đường dẫn thư mục chứa các folder identity
        max_images_per_identity: Số ảnh tối đa mỗi identity (None = lấy tất cả)
        max_identities: Số identity tối đa để load (None = lấy tất cả)
                       Ví dụ: max_identities=200, max_images_per_identity=3 → 600 samples
    
    Returns:
        faces: List các ảnh grayscale
        labels: numpy array các label tương ứng
    """
    faces = []
    labels = []
    identity_count = 0

    all_labels = sorted(os.listdir(data_dir), key=lambda x: int(x) if x.isdigit() else x)
    total_identities = len([l for l in all_labels if os.path.isdir(os.path.join(data_dir, l))])
    
    print(f"[DATA] Loading from {data_dir}")
    print(f"[DATA] Total identities available: {total_identities}")
    if max_identities:
        print(f"[DATA] Limiting to {max_identities} identities")
    if max_images_per_identity:
        print(f"[DATA] Limiting to {max_images_per_identity} images per identity")

    for label in all_labels:
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        if max_identities is not None and identity_count >= max_identities:
            break

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
        
        if img_count > 0:
            identity_count += 1

    print(f"[DATA] Loaded {len(faces)} samples from {identity_count} identities")
    return faces, np.array(labels)
