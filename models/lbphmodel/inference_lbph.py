# DÙNG TRONG WEB – NHẬN DIỆN 1 ẢNH
import cv2

def recognize_face(model, face_img, threshold):
    pred, conf = model.predict(face_img)

    if conf < threshold:
        return {
            "label": pred,
            "confidence": conf,
            "status": "known"
        }
    else:
        return {
            "label": None,
            "confidence": conf,
            "status": "unknown"
        }
