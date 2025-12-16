"""
Flask Web Application for Face Recognition
Tich hop ArcFace, FaceNet va LBPH models
Demo cho mon Xu ly anh so
"""

from flask import Flask, render_template, request, session
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

MAX_FILE_AGE_SECONDS = 3600


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
            db_path = os.path.join(ROOT_DIR, "data/embeddings_db.npy")
            
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
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
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
            lbph_path = os.path.join(ROOT_DIR, "models/checkpoints/LBHP/lbph_model.yml")
            if os.path.exists(lbph_path):
                lbph_model = cv2.face.LBPHFaceRecognizer_create()
                lbph_model.read(lbph_path)
                label_map_path = os.path.join(ROOT_DIR, "models/checkpoints/LBHP/label_map.npy")
                if os.path.exists(label_map_path):
                    lbph_label_map = np.load(label_map_path, allow_pickle=True).item()
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
                        return {"text": f"{res['identity']} ({res['confidence']:.2f})", "ok": True, "detail": res}
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
    """Batch processing với chọn model"""
    results = []
    model_type = "arcface"

    if request.method == "POST":
        files = request.files.getlist("images")
        model_type = request.form.get("model", "arcface")

        for file in files:
            if file and file.filename:
                unique_id = str(uuid.uuid4())[:8]
                ext = os.path.splitext(file.filename)[1]
                temp_filename = f"{unique_id}{ext}"
                temp_path = os.path.join(app.config["TEMP_FOLDER"], temp_filename)
                file.save(temp_path)

                try:
                    if model_type == "facenet":
                        rec = recognize_with_facenet(temp_path)
                    elif model_type == "lbph":
                        rec = recognize_with_lbph(temp_path)
                    else:
                        rec = recognize_with_arcface(temp_path)
                    
                    results.append({
                        "image": file.filename,
                        "name": rec.get("identity", "Error") if rec.get("status") == "success" else "Error",
                        "score": rec.get("confidence", 0.0)
                    })
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    return render_template("batch.html", active="batch", results=results, model_type=model_type)


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
        test_dir = os.path.join(app.config["TEST_DATA_DIR"], "images_aligned")
        if os.path.exists(test_dir):
            demo_results = {"correct": 0, "wrong": 0, "samples": []}
            classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))][:5]
            for cls in classes:
                cls_path = os.path.join(test_dir, cls)
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))][:2]
                for img_name in images:
                    result = recognize_with_arcface(os.path.join(cls_path, img_name))
                    predicted = result.get("identity", "Unknown") if result.get("status") == "success" else "Error"
                    is_correct = cls.lower() in predicted.lower() or predicted.lower() in cls.lower()
                    demo_results["correct" if is_correct else "wrong"] += 1
                    demo_results["samples"].append({"image": img_name, "true_label": cls, "predicted": predicted, "confidence": result.get("confidence", 0), "correct": is_correct})
            total = demo_results["correct"] + demo_results["wrong"]
            demo_results["accuracy"] = (demo_results["correct"] / total * 100) if total > 0 else 0
    
    info = get_test_data_info()
    return render_template("evaluation.html", active="evaluation", metrics=kaggle_metrics, info=info, mode=mode, demo_results=demo_results)





if __name__ == "__main__":
    app.run(debug=True)
