import os
import cv2
import numpy as np

def load_data_no_haar(data_dir):
    faces = []
    labels = []

    for label in sorted(os.listdir(data_dir), key=lambda x: int(x) if x.isdigit() else x):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(int(label))

    return faces, np.array(labels)
