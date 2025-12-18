"""
Flask Web Application for Face Recognition
Tich hop ArcFace, FaceNet va LBPH models
Demo cho mon Xu ly anh so
"""

from flask import Flask, render_template, request, session, Response, jsonify
import threading
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import tempfile
import uuid
import atexit
import shutil
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["TEMP_FOLDER"] = os.path.join(tempfile.gettempdir(), "face_recognition_temp")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["GRADCAM_FOLDER"] = "static/gradcam"
app.config["TEST_DATA_DIR"] = os.path.join(ROOT_DIR, "data/CelebA_Aligned/test")

os.makedirs(app.config["TEMP_FOLDER"], exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["GRADCAM_FOLDER"], exist_ok=True)
os.makedirs("static/detection_bbox", exist_ok=True)

MAX_FILE_AGE_SECONDS = 3600


def draw_face_bbox(image_path, bbox, output_folder="static/detection_bbox"):
    """
    Vẽ bounding box màu xanh lá lên ảnh gốc
    
    Args:
        image_path: Đường dẫn ảnh gốc
        bbox: List [x1, y1, x2, y2]
        output_folder: Thư mục lưu ảnh output
        
    Returns:
        Path relative của ảnh đã vẽ bbox (để hiển thị trong HTML)
    """
    try:
        if bbox is None:
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Vẽ hình chữ nhật màu xanh lá #10b981 (RGB: 16, 185, 129)
        # OpenCV dùng BGR nên đảo ngược: (129, 185, 16)
        color = (129, 185, 16)
        thickness = 3
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Tạo filename unique
        unique_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(image_path)[1]
        output_filename = f"bbox_{unique_id}{ext}"
        output_path = os.path.join(output_folder, output_filename)
        
        cv2.imwrite(output_path, image)
        
        # Return relative path cho HTML
        return f"detection_bbox/{output_filename}"
        
    except Exception as e:
        print(f"Draw bbox error: {e}")
        return None


def cleanup_temp_folder():
    """Xoa thu muc temp khi app shutdown"""
    if os.path.exists(app.config["TEMP_FOLDER"]):
        shutil.rmtree(app.config["TEMP_FOLDER"], ignore_errors=True)


def cleanup_old_files(folder: str, max_age_seconds: int = MAX_FILE_AGE_SECONDS):
    """Xoa cac file cu hon max_age_seconds trong folder"""
    if not os.path.exists(folder):
        return
    
    import time
    current_time = time.time()
    
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                except Exception:
                    pass


@app.before_request
def before_request():
    """Cleanup old files truoc moi request"""
    cleanup_old_files(app.config["UPLOAD_FOLDER"])
    cleanup_old_files(app.config["GRADCAM_FOLDER"])


atexit.register(cleanup_temp_folder)

arcface_engine = None
realtime_arcface_engine = None
facenet_engine = None
lbph_model = None
lbph_label_map = None
explainability_engine = None


def get_arcface_engine():
    global arcface_engine
    if arcface_engine is None:
        try:
            from inference.recognition_engine import RecognitionEngine
            model_path = os.path.join(ROOT_DIR, "models/checkpoints/arcface/arcface_best.pth")
            db_path = os.path.join(ROOT_DIR, "data/arcface_embeddings_db.npy")
            
            if os.path.exists(model_path):
                arcface_engine = RecognitionEngine(
                    model_path=model_path,
                    db_path=db_path if os.path.exists(db_path) else None,
                    threshold=0.65
                )
                print("ArcFace engine loaded")
        except Exception as e:
            print(f"ArcFace error: {e}")
    return arcface_engine


def get_realtime_arcface_engine():
    global realtime_arcface_engine
    if realtime_arcface_engine is None:
        try:
            from inference.recognition_engine import RecognitionEngine
            model_path = os.path.join(ROOT_DIR, "models/checkpoints/arcface/arcface_best.pth")
            db_path = os.path.join(ROOT_DIR, "data/arcface_embeddings_db.npy")
            
            if os.path.exists(model_path):
                realtime_arcface_engine = RecognitionEngine(
                    model_path=model_path,
                    db_path=db_path if os.path.exists(db_path) else None,
                    threshold=0.5,
                    use_face_detection=False
                )
                print("Realtime ArcFace engine loaded (no face detection)")
        except Exception as e:
            print(f"Realtime ArcFace error: {e}")
    return realtime_arcface_engine


def get_facenet_engine():
    global facenet_engine
    if facenet_engine is None:
        try:
            from models.facenet.facenet_model import FaceNetModel
            from torchvision import transforms
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = os.path.join(ROOT_DIR, "models/checkpoints/facenet/facenet_best.pth")
            db_path = os.path.join(ROOT_DIR, "data/facenet_embeddings_db.npy")
            
            model = FaceNetModel(embedding_size=512, pretrained='vggface2', device=device)
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Load state dict, bo qua logits layer
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = {}
                
                # Filter out logits weights
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                      if 'logits' not in k}
                
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded {len(filtered_state_dict)} weights, skipped {len(state_dict) - len(filtered_state_dict)} logits weights")
            model.eval()
            
            transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            db = None
            if os.path.exists(db_path):
                db = np.load(db_path, allow_pickle=True).item()
            
            facenet_engine = {'model': model, 'transform': transform, 'device': device, 'db': db}
            print("FaceNet engine loaded")
        except Exception as e:
            print(f"FaceNet error: {e}")
    return facenet_engine


def get_lbph_model():
    global lbph_model, lbph_label_map
    if lbph_model is None:
        try:
            lbph_path = os.path.join(ROOT_DIR, "models/checkpoints/LBHP/lbph_full_celeba.xml")
            if os.path.exists(lbph_path):
                lbph_model = cv2.face.LBPHFaceRecognizer_create()
                lbph_model.read(lbph_path)
                
                label_map_path = os.path.join(ROOT_DIR, "models/checkpoints/LBHP/label_map.npy")
                if os.path.exists(label_map_path):
                    lbph_label_map = np.load(label_map_path, allow_pickle=True).item()
                else:
                    metadata_path = os.path.join(ROOT_DIR, "data/metadata/train_labels.csv")
                    if os.path.exists(metadata_path):
                        import pandas as pd
                        df = pd.read_csv(metadata_path)
                        mapping_df = df[['label', 'identity_id']].drop_duplicates()
                        lbph_label_map = dict(zip(mapping_df['label'], mapping_df['identity_id']))
                        np.save(label_map_path, lbph_label_map)
                        print(f"Created label_map with {len(lbph_label_map)} identities")
                
                print("LBPH model loaded")
        except Exception as e:
            print(f"LBPH error: {e}")
    return lbph_model


def get_explainability_engine():
    global explainability_engine
    if explainability_engine is None:
        try:
            engine = get_arcface_engine()
            if engine and engine.model:
                from inference.explainability import ExplainabilityEngine
                explainability_engine = ExplainabilityEngine(engine.model, engine.transform, engine.device)
                print("Explainability engine loaded")
        except Exception as e:
            print(f"Explainability error: {e}")
    return explainability_engine


def extract_face_detection_info(image_path):
    """
    Extract face detection info từ ảnh
    
    Returns:
        Dict chứa bbox, confidence, landmarks, num_faces, face_size
    """
    try:
        from preprocessing.face_detector import FaceDetector
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        detector = FaceDetector(
            backend='mtcnn',
            device='cpu',
            confidence_threshold=0.9,
            select_largest=True
        )
        
        detection = detector.detect(image)
        
        if detection is None:
            return {
                "num_faces": 0,
                "bbox": None,
                "confidence": 0.0,
                "landmarks": None,
                "face_size": None
            }
        
        bbox = detection['bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        return {
            "num_faces": 1,
            "bbox": bbox,
            "confidence": detection['confidence'],
            "landmarks": detection['landmarks'],
            "face_size": [face_width, face_height]
        }
    except Exception as e:
        print(f"Face detection error: {e}")
        return None


def preprocess_face_for_model(image_path, target_size=(160, 160)):
    """
    Detect, align va resize face cho model (FaceNet)
    
    Args:
        image_path: Duong dan anh
        target_size: Kich thuoc output (W, H)
        
    Returns:
        PIL Image da preprocessed hoac None
    """
    try:
        from preprocessing.face_detector import FaceDetector
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        detector = FaceDetector(
            backend='mtcnn',
            device='cpu',
            confidence_threshold=0.9,
            select_largest=True
        )
        
        detection = detector.detect(image)
        
        if detection is None:
            # Fallback: resize anh goc
            pil_img = Image.open(image_path).convert('RGB')
            return pil_img.resize(target_size)
        
        # Crop face voi margin
        cropped = detector.crop_face(image, margin=0.2, target_size=target_size)
        if cropped is not None:
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        
        # Fallback
        pil_img = Image.open(image_path).convert('RGB')
        return pil_img.resize(target_size)
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Fallback: return resized original
        try:
            pil_img = Image.open(image_path).convert('RGB')
            return pil_img.resize(target_size)
        except:
            return None


def detect_and_crop_face(image_path, target_size=(100, 100), grayscale=False):
    """
    Detect va crop face cho LBPH
    
    Args:
        image_path: Duong dan anh
        target_size: Kich thuoc output (W, H)
        grayscale: Convert to grayscale
        
    Returns:
        numpy array (grayscale neu specified)
    """
    try:
        from preprocessing.face_detector import FaceDetector
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        detector = FaceDetector(
            backend='mtcnn',
            device='cpu',
            confidence_threshold=0.9,
            select_largest=True
        )
        
        cropped = detector.crop_face(image, margin=0.2, target_size=target_size)
        
        if cropped is None:
            # Fallback: resize anh goc
            cropped = cv2.resize(image, target_size)
        
        if grayscale:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        return cropped
        
    except Exception as e:
        print(f"Crop face error: {e}")
        # Fallback
        try:
            image = cv2.imread(image_path)
            cropped = cv2.resize(image, target_size)
            if grayscale:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            return cropped
        except:
            return None


def recognize_with_arcface(image_path, threshold=0.5):
    start_time = time.time()
    engine = get_arcface_engine()
    if engine is None:
        return {"identity": "Model not loaded", "confidence": 0.0, "status": "error", "time_ms": 0, "face_detection": None}
    
    face_detection = extract_face_detection_info(image_path)
    
    # Vẽ bounding box nếu có phát hiện khuôn mặt
    bbox_image = None
    if face_detection and face_detection.get("bbox") is not None:
        bbox_image = draw_face_bbox(image_path, face_detection["bbox"])
        if face_detection and bbox_image:
            face_detection["bbox_image"] = bbox_image
    
    engine.set_threshold(threshold)
    result = engine.recognize(image_path)
    result["time_ms"] = round((time.time() - start_time) * 1000, 2)
    result["face_detection"] = face_detection
    return result


def recognize_with_facenet(image_path, threshold=0.5):
    start_time = time.time()
    engine = get_facenet_engine()
    if engine is None:
        return {"identity": "Model not loaded", "confidence": 0.0, "status": "error", "time_ms": 0, "face_detection": None}
    
    face_detection = extract_face_detection_info(image_path)
    
    try:
        # Preprocess face: detect + align + resize to 160x160
        preprocessed_img = preprocess_face_for_model(image_path, target_size=(160, 160))
        
        if preprocessed_img is None:
            return {"identity": "Cannot preprocess image", "confidence": 0.0, "status": "error", "time_ms": round((time.time() - start_time) * 1000, 2), "face_detection": face_detection}
        
        input_tensor = engine['transform'](preprocessed_img).unsqueeze(0).to(engine['device'])
        
        with torch.no_grad():
            embedding = engine['model'](input_tensor).cpu().numpy().flatten()
        
        if engine['db'] is None:
            return {"identity": "No database", "confidence": 0.0, "status": "error", "time_ms": round((time.time() - start_time) * 1000, 2), "face_detection": face_detection}
        
        top_k = []
        for name, db_emb in engine['db'].items():
            score = np.dot(embedding, db_emb.flatten()) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb) + 1e-8)
            distance = np.linalg.norm(embedding - db_emb.flatten())
            top_k.append((name, float(score), float(distance)))
        top_k.sort(key=lambda x: x[1], reverse=True)
        
        best_name, best_score = top_k[0][0], top_k[0][1] if top_k else ("Unknown", 0.0)
        best_distance = top_k[0][2] if top_k else 0.0
        if best_score < threshold:
            best_name = "Unknown"
        
        elapsed = round((time.time() - start_time) * 1000, 2)
        return {"identity": best_name, "confidence": best_score, "distance": best_distance, "top_k": top_k[:5], "status": "success", "time_ms": elapsed, "face_detection": face_detection}
    except Exception as e:
        return {"identity": str(e), "confidence": 0.0, "status": "error", "time_ms": round((time.time() - start_time) * 1000, 2), "face_detection": face_detection}


def recognize_with_lbph(image_path, threshold=80):
    start_time = time.time()
    model = get_lbph_model()
    if model is None:
        return {"identity": "Model not loaded", "confidence": 0.0, "status": "error", "time_ms": 0, "face_detection": None}
    
    face_detection = extract_face_detection_info(image_path)
    
    try:
        # Detect va crop face (100x100 grayscale)
        cropped_face = detect_and_crop_face(image_path, target_size=(100, 100), grayscale=True)
        
        if cropped_face is None:
            return {"identity": "Cannot detect face", "confidence": 0.0, "status": "error", "time_ms": 0, "face_detection": face_detection}
        
        img = cropped_face  # Already resized to 100x100 grayscale
        label, distance = model.predict(img)
        
        identity = lbph_label_map.get(label, f"Person_{label}") if lbph_label_map else f"Person_{label}"
        confidence = max(0, (100 - distance) / 100)
        if distance > threshold:
            identity = "Unknown"
        
        elapsed = round((time.time() - start_time) * 1000, 2)
        return {"identity": identity, "confidence": float(confidence), "distance": float(distance), "status": "success", "time_ms": elapsed, "face_detection": face_detection}
    except Exception as e:
        return {"identity": str(e), "confidence": 0.0, "status": "error", "time_ms": round((time.time() - start_time) * 1000, 2), "face_detection": face_detection}


def get_test_data_info():
    test_dir = app.config["TEST_DATA_DIR"]
    info = {"aligned": {"classes": 0, "images": 0}, "original": {"classes": 0, "images": 0}}
    
    for key, subdir in [("aligned", "images_aligned"), ("original", "original_images")]:
        path = os.path.join(test_dir, subdir)
        if os.path.exists(path):
            classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            info[key]["classes"] = len(classes)
            for cls in classes:
                cls_path = os.path.join(path, cls)
                info[key]["images"] += len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    return info


def calculate_evaluation_metrics(y_true, y_pred, labels):
    """
    Tính toán các metrics đánh giá và confusion matrix
    
    Args:
        y_true: List các label thật
        y_pred: List các label dự đoán
        labels: List tên các classes
        
    Returns:
        Dict chứa accuracy, precision, recall, f1, confusion_matrix
    """
    from collections import defaultdict
    import numpy as np
    
    # Convert labels to indices
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Build confusion matrix
    n_classes = len(labels)
    confusion_matrix = [[0] * n_classes for _ in range(n_classes)]
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_idx and pred_label in label_to_idx:
            true_idx = label_to_idx[true_label]
            pred_idx = label_to_idx[pred_label]
            confusion_matrix[true_idx][pred_idx] += 1
    
    # Calculate overall metrics
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Calculate per-class metrics
    per_class_metrics = {}
    precision_list = []
    recall_list = []
    f1_list = []
    
    for idx, label in enumerate(labels):
        # True Positives, False Positives, False Negatives
        tp = confusion_matrix[idx][idx]
        fp = sum(confusion_matrix[i][idx] for i in range(n_classes)) - tp
        fn = sum(confusion_matrix[idx][j] for j in range(n_classes)) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(confusion_matrix[idx])
        }
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Macro average
    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    
    return {
        'accuracy': accuracy,
        'precision': macro_precision * 100,
        'recall': macro_recall * 100,
        'f1': macro_f1 * 100,
        'confusion_matrix': confusion_matrix,
        'per_class_metrics': per_class_metrics
    }


@app.route("/", methods=["GET", "POST"])
def home():
    """Trang chính: Nhận dạng với 3 models + threshold + Grad-CAM"""
    result = None
    image_name = None
    threshold_val = 0.65
    gradcam_image = None
    display_image = None

    if request.args.get("action") == "clear":
        session.pop('last_image_path', None)
        session.pop('last_image_name', None)
        return render_template("home.html", active="home", result=None, 
                              image_name=None, threshold=threshold_val, gradcam_image=None)

    if request.method == "POST":
        threshold_val = float(request.form.get("threshold", 0.65))
        file = request.files.get("image")
        
        image_to_process = None
        is_new_upload = False
        
        if file and file.filename:
            is_new_upload = True
            unique_id = str(uuid.uuid4())[:8]
            ext = os.path.splitext(file.filename)[1]
            temp_filename = f"{unique_id}{ext}"
            temp_path = os.path.join(app.config["TEMP_FOLDER"], temp_filename)
            file.save(temp_path)
            image_to_process = temp_path
            
            display_filename = f"display_{unique_id}{ext}"
            display_path = os.path.join(app.config["UPLOAD_FOLDER"], display_filename)
            shutil.copy2(temp_path, display_path)
            
            session['last_image_path'] = display_path
            session['last_image_name'] = display_filename
            image_name = display_filename
        elif session.get('last_image_path'):
            image_to_process = session['last_image_path']
            image_name = session.get('last_image_name')
        
        if image_to_process and os.path.exists(image_to_process):
            try:
                arcface_result = recognize_with_arcface(image_to_process, threshold_val)
                facenet_result = recognize_with_facenet(image_to_process, threshold_val)
                lbph_result = recognize_with_lbph(image_to_process)
                
                def fmt(res):
                    if res.get("status") == "success":
                        # Convert confidence to percentage (0-1 → 0-100%)
                        confidence_pct = res['confidence'] * 100
                        return {
                            "text": f"{res['identity']} ({confidence_pct:.2f}%)", 
                            "ok": True, 
                            "detail": res
                        }
                    return {"text": "Error", "ok": False, "detail": res}
                
                result = {
                    "arcface": fmt(arcface_result),
                    "facenet": fmt(facenet_result),
                    "lbph": fmt(lbph_result)
                }
                
                if is_new_upload:
                    exp_engine = get_explainability_engine()
                    if exp_engine:
                        try:
                            exp_result = exp_engine.explain(image_to_process)
                            unique_id = session['last_image_name'].split('_')[1].split('.')[0]
                            ext = os.path.splitext(session['last_image_name'])[1]
                            gradcam_filename = f"gradcam_{unique_id}{ext}"
                            gradcam_path = os.path.join(app.config["GRADCAM_FOLDER"], gradcam_filename)
                            cv2.imwrite(gradcam_path, cv2.cvtColor(exp_result['overlay'], cv2.COLOR_RGB2BGR))
                            gradcam_image = f"gradcam/{gradcam_filename}"
                            session['gradcam_image'] = gradcam_image
                        except Exception as e:
                            print(f"Grad-CAM error: {e}")
                else:
                    gradcam_image = session.get('gradcam_image')
                
                if is_new_upload and image_to_process.startswith(app.config["TEMP_FOLDER"]):
                    if os.path.exists(image_to_process):
                        os.remove(image_to_process)
            except Exception as e:
                print(f"Recognition error: {e}")

    return render_template("home.html", active="home", result=result, 
                          image_name=image_name, threshold=threshold_val, gradcam_image=gradcam_image)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Batch processing với cả 3 models đồng thời"""
    results = []

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            if file and file.filename:
                unique_id = str(uuid.uuid4())[:8]
                ext = os.path.splitext(file.filename)[1]
                temp_filename = f"{unique_id}{ext}"
                temp_path = os.path.join(app.config["TEMP_FOLDER"], temp_filename)
                file.save(temp_path)

                try:
                    # Chạy cả 3 models
                    arcface_result = recognize_with_arcface(temp_path, threshold=0.65)
                    facenet_result = recognize_with_facenet(temp_path, threshold=0.65)
                    lbph_result = recognize_with_lbph(temp_path)
                    
                    # Tổng hợp kết quả
                    result_item = {
                        "image": file.filename,
                        "arcface": {
                            "identity": arcface_result.get("identity", "Error"),
                            "confidence": arcface_result.get("confidence", 0.0),
                            "time_ms": arcface_result.get("time_ms", 0),
                            "status": arcface_result.get("status", "error")
                        },
                        "facenet": {
                            "identity": facenet_result.get("identity", "Error"),
                            "confidence": facenet_result.get("confidence", 0.0),
                            "distance": facenet_result.get("distance", 0.0),
                            "time_ms": facenet_result.get("time_ms", 0),
                            "status": facenet_result.get("status", "error")
                        },
                        "lbph": {
                            "identity": lbph_result.get("identity", "Error"),
                            "confidence": lbph_result.get("confidence", 0.0),
                            "distance": lbph_result.get("distance", 0.0),
                            "time_ms": lbph_result.get("time_ms", 0),
                            "status": lbph_result.get("status", "error")
                        }
                    }
                    
                    # Xác định model tốt nhất (highest confidence)
                    best_model = "arcface"
                    best_confidence = arcface_result.get("confidence", 0.0)
                    
                    if facenet_result.get("confidence", 0.0) > best_confidence:
                        best_model = "facenet"
                        best_confidence = facenet_result.get("confidence", 0.0)
                    
                    if lbph_result.get("confidence", 0.0) > best_confidence:
                        best_model = "lbph"
                    
                    result_item["best_model"] = best_model
                    results.append(result_item)
                    
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    return render_template("batch.html", active="batch", results=results)


@app.route("/evaluation", methods=["GET", "POST"])
def evaluation():
    """Đánh giá: 2 options - Kaggle results hoặc Demo từ test set"""
    mode = request.args.get("mode", "kaggle")
    demo_results = None
    
    kaggle_metrics = {
        "arcface": {"accuracy": 92.3, "top5_accuracy": 98.1, "precision": 0.91, "recall": 0.92, "f1": 0.91, "auc": 0.987, "eer": 0.043, "latency_ms": 45.2, "throughput": 22.1},
        "facenet": {"accuracy": 89.5, "top5_accuracy": 96.8, "precision": 0.88, "recall": 0.89, "f1": 0.88, "auc": 0.972, "eer": 0.058, "latency_ms": 38.5, "throughput": 26.0},
        "lbph": {"accuracy": 67.2, "top5_accuracy": 82.5, "precision": 0.65, "recall": 0.67, "f1": 0.66, "auc": 0.834, "eer": 0.185, "latency_ms": 5.2, "throughput": 192.3}
    }
    
    if request.method == "POST" and mode == "demo":
        test_folder = request.form.get("test_folder", "").strip()
        if not test_folder:
            test_folder = os.path.join(app.config["TEST_DATA_DIR"], "images_aligned")
        
        selected_model = request.form.get("model", "arcface")
        
        if os.path.exists(test_folder) and os.path.isdir(test_folder):
            y_true = []
            y_pred = []
            y_pred_top5 = []
            samples = []
            confidences = []
            processing_times = []
            
            classes = sorted([d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))])
            max_images_per_class = int(request.form.get("max_images", 5))
            
            for cls in classes:
                cls_path = os.path.join(test_folder, cls)
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:max_images_per_class]
                
                for img_name in images:
                    img_path = os.path.join(cls_path, img_name)
                    
                    if selected_model == "arcface":
                        result = recognize_with_arcface(img_path)
                    elif selected_model == "facenet":
                        result = recognize_with_facenet(img_path)
                    else:
                        result = recognize_with_lbph(img_path)
                    
                    predicted = result.get("identity", "Unknown") if result.get("status") == "success" else "Unknown"
                    confidence = result.get("confidence", 0)
                    time_ms = result.get("time_ms", 0)
                    top_k = result.get("top_k", [])
                    
                    y_true.append(cls)
                    y_pred.append(predicted)
                    confidences.append(confidence)
                    processing_times.append(time_ms)
                    
                    top5_names = [predicted] + [k[0] for k in top_k[1:5]] if top_k else [predicted]
                    y_pred_top5.append(top5_names)
                    
                    is_correct = (cls.lower() == predicted.lower())
                    samples.append({
                        "image": img_name,
                        "true_label": cls,
                        "predicted": predicted,
                        "confidence": confidence,
                        "time_ms": time_ms,
                        "correct": is_correct
                    })
            
            if y_true and y_pred:
                metrics = calculate_evaluation_metrics(y_true, y_pred, classes)
                
                top5_correct = sum(1 for t, preds in zip(y_true, y_pred_top5) if any(t.lower() == p.lower() for p in preds))
                top5_accuracy = (top5_correct / len(y_true) * 100) if y_true else 0
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
                total_time = sum(processing_times)
                
                demo_results = {
                    "accuracy": metrics["accuracy"],
                    "top5_accuracy": top5_accuracy,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "avg_confidence": avg_confidence * 100,
                    "avg_time_ms": avg_time,
                    "total_time_ms": total_time,
                    "confusion_matrix": metrics["confusion_matrix"],
                    "per_class_metrics": metrics["per_class_metrics"],
                    "classes": classes,
                    "samples": samples,
                    "total": len(y_true),
                    "correct": sum(1 for t, p in zip(y_true, y_pred) if t == p),
                    "wrong": sum(1 for t, p in zip(y_true, y_pred) if t != p),
                    "model": selected_model
                }
        else:
            demo_results = {"error": f"Thư mục không tồn tại: {test_folder}"}
    
    info = get_test_data_info()
    return render_template("evaluation.html", active="evaluation", metrics=kaggle_metrics, info=info, mode=mode, demo_results=demo_results)


# ===== REALTIME FACE RECOGNITION =====

camera = None
camera_lock = threading.Lock()
realtime_result = {"identity": None, "confidence": 0.0, "bbox": None}
realtime_running = False
realtime_detector = None
realtime_processing = False
realtime_model = "arcface"


def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera


def release_camera():
    global camera, realtime_detector
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        realtime_detector = None


def get_realtime_detector():
    global realtime_detector
    if realtime_detector is None:
        try:
            from preprocessing.face_detector import FaceDetector
            realtime_detector = FaceDetector(
                backend='opencv',
                device='cpu',
                confidence_threshold=0.5,
                select_largest=True
            )
        except Exception as e:
            print(f"Failed to init detector: {e}")
    return realtime_detector


def recognize_frame(frame):
    """Nhận diện khuôn mặt từ frame"""
    global realtime_result, realtime_processing, realtime_model
    
    if realtime_processing:
        return
    
    realtime_processing = True
    
    try:
        temp_path = os.path.join(app.config["TEMP_FOLDER"], f"realtime_{threading.current_thread().ident}.jpg")
        cv2.imwrite(temp_path, frame)
        
        result = None
        if realtime_model == "arcface":
            engine = get_realtime_arcface_engine()
            if engine is not None:
                result = engine.recognize(temp_path)
        elif realtime_model == "facenet":
            result = recognize_with_facenet(temp_path, threshold=0.5)
        elif realtime_model == "lbph":
            result = recognize_with_lbph(temp_path, threshold=80)
        
        detector = get_realtime_detector()
        detection = detector.detect(frame) if detector else None
        
        if result and result.get("status") == "success" and result.get("identity") != "Unknown":
            if detection is not None:
                bbox = [int(x) for x in detection['bbox']]
            else:
                bbox = None
            
            realtime_result = {
                "identity": result.get("identity", "Unknown"),
                "confidence": result.get("confidence", 0.0),
                "bbox": bbox
            }
        elif detection is not None:
            realtime_result = {
                "identity": None,
                "confidence": 0.0,
                "bbox": [int(x) for x in detection['bbox']]
            }
        else:
            realtime_result = {"identity": None, "confidence": 0.0, "bbox": None}
        
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"Realtime recognition error: {e}")
    finally:
        realtime_processing = False


def generate_frames():
    """Generator cho video streaming"""
    global realtime_running
    
    frame_count = 0
    realtime_running = True
    last_recognition_time = 0
    recognition_interval = 0.5
    
    while realtime_running:
        cam = get_camera()
        if cam is None:
            break
            
        success, frame = cam.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        
        current_time = time.time()
        if current_time - last_recognition_time >= recognition_interval and not realtime_processing:
            threading.Thread(target=recognize_frame, args=(frame.copy(),), daemon=True).start()
            last_recognition_time = current_time
        
        if realtime_result["bbox"] is not None:
            x1, y1, x2, y2 = realtime_result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
            
            identity = realtime_result["identity"]
            confidence = realtime_result["confidence"]
            
            if identity and identity != "Unknown":
                label = f"{identity}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2)
            else:
                cv2.putText(frame, "Unknown", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_count += 1
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route("/realtime")
def realtime():
    """Trang nhận diện khuôn mặt realtime"""
    return render_template("realtime.html", active="realtime")


@app.route("/video_feed")
def video_feed():
    """Video streaming endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/realtime_result")
def get_realtime_result():
    """API endpoint để lấy kết quả recognition hiện tại"""
    return jsonify(realtime_result)


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    """Dừng camera"""
    global realtime_running
    realtime_running = False
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/set_realtime_model", methods=["POST"])
def set_realtime_model():
    """Thay đổi model cho realtime recognition"""
    global realtime_model
    data = request.get_json()
    model = data.get("model", "arcface")
    if model in ["arcface", "facenet", "lbph"]:
        realtime_model = model
        return jsonify({"status": "success", "model": model})
    return jsonify({"status": "error", "message": "Invalid model"})


if __name__ == "__main__":
    app.run(debug=True)
