import cv2
import numpy as np

def train_classifier(faces, labels):
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    model.train(faces, labels)
    return model
