"""
Face Detection Module
Task 2.2: Face Detection cho CelebA preprocessing

Ho tro:
- MTCNN (facenet-pytorch)
- RetinaFace (optional)
- Batch processing
- Edge cases handling
"""

import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Union


class FaceDetector:
    """
    Face Detector ho tro nhieu backend
    
    Backends:
    - mtcnn: Su dung MTCNN tu facenet-pytorch (recommended)
    - retinaface: Su dung RetinaFace (do chinh xac cao hon)
    - opencv: Su dung Haar Cascade (nhanh nhung kem chinh xac)
    
    Usage:
        detector = FaceDetector(backend='mtcnn', device='cuda')
        result = detector.detect(image)
        # result = {'bbox': [x1,y1,x2,y2], 'landmarks': {...}, 'confidence': 0.99}
    """
    
    # Nguong mac dinh
    DEFAULT_CONFIDENCE_THRESHOLD = 0.9
    MIN_FACE_SIZE = 20  # pixels
    
    def __init__(
        self, 
        backend: str = 'mtcnn',
        device: str = 'cpu',
        confidence_threshold: float = 0.9,
        min_face_size: int = 20,
        select_largest: bool = True
    ):
        """
        Khoi tao Face Detector
        
        Args:
            backend: 'mtcnn', 'retinaface', hoac 'opencv'
            device: 'cuda' hoac 'cpu'
            confidence_threshold: Nguong confidence de chap nhan detection
            min_face_size: Kich thuoc face toi thieu (pixels)
            select_largest: Chon bbox lon nhat khi co nhieu faces
        """
        self.backend = backend.lower()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self.select_largest = select_largest
        
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        """Khoi tao detector theo backend"""
        if self.backend == 'mtcnn':
            self._init_mtcnn()
        elif self.backend == 'retinaface':
            self._init_retinaface()
        elif self.backend == 'opencv':
            self._init_opencv()
        else:
            raise ValueError(f"Backend khong ho tro: {self.backend}")
    
    def _init_mtcnn(self):
        """Khoi tao MTCNN detector"""
        try:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True  # Detect tat ca faces
            )
            print(f"[OK] MTCNN initialized on {self.device}")
        except ImportError:
            raise ImportError(
                "MTCNN chua duoc cai dat. "
                "Chay: pip install facenet-pytorch"
            )
    
    def _init_retinaface(self):
        """Khoi tao RetinaFace detector"""
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            print("[OK] RetinaFace initialized")
        except ImportError:
            print("[WARN] RetinaFace khong kha dung, fallback to MTCNN")
            self.backend = 'mtcnn'
            self._init_mtcnn()
    
    def _init_opencv(self):
        """Khoi tao OpenCV Haar Cascade detector"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Khong the load Haar Cascade classifier")
        print("[OK] OpenCV Haar Cascade initialized")
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face trong anh
        
        Args:
            image: numpy array (BGR hoac RGB), shape (H, W, 3)
            
        Returns:
            Dict chua:
                - bbox: [x1, y1, x2, y2]
                - landmarks: Dict voi 5 diem (left_eye, right_eye, nose, left_mouth, right_mouth)
                - confidence: float
            Hoac None neu khong detect duoc face
        """
        if image is None or image.size == 0:
            return None
        
        if self.backend == 'mtcnn':
            return self._detect_mtcnn(image)
        elif self.backend == 'retinaface':
            return self._detect_retinaface(image)
        elif self.backend == 'opencv':
            return self._detect_opencv(image)
        
        return None
    
    def _detect_mtcnn(self, image: np.ndarray) -> Optional[Dict]:
        """Detect voi MTCNN"""
        from PIL import Image
        
        # Convert BGR -> RGB neu can
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        pil_image = Image.fromarray(rgb_image)
        
        # Detect
        boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return None
        
        # Loc theo confidence threshold
        valid_indices = probs >= self.confidence_threshold
        if not np.any(valid_indices):
            return None
        
        boxes = boxes[valid_indices]
        probs = probs[valid_indices]
        landmarks = landmarks[valid_indices] if landmarks is not None else None
        
        # Loc face qua nho
        valid_faces = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if min(width, height) >= self.min_face_size:
                valid_faces.append(i)
        
        if not valid_faces:
            return None
        
        # Chon face
        if self.select_largest and len(valid_faces) > 1:
            areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) 
                     for i in valid_faces]
            best_idx = valid_faces[np.argmax(areas)]
        else:
            best_idx = valid_faces[0]
        
        box = boxes[best_idx]
        prob = probs[best_idx]
        
        result = {
            'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            'confidence': float(prob),
            'landmarks': None
        }
        
        if landmarks is not None:
            lm = landmarks[best_idx]
            result['landmarks'] = {
                'left_eye': (float(lm[0][0]), float(lm[0][1])),
                'right_eye': (float(lm[1][0]), float(lm[1][1])),
                'nose': (float(lm[2][0]), float(lm[2][1])),
                'left_mouth': (float(lm[3][0]), float(lm[3][1])),
                'right_mouth': (float(lm[4][0]), float(lm[4][1])),
            }
        
        return result
    
    def _detect_retinaface(self, image: np.ndarray) -> Optional[Dict]:
        """Detect voi RetinaFace"""
        # RetinaFace chap nhan BGR
        faces = self.detector.detect_faces(image)
        
        if not faces:
            return None
        
        # Loc theo confidence
        valid_faces = {
            k: v for k, v in faces.items() 
            if v['score'] >= self.confidence_threshold
        }
        
        if not valid_faces:
            return None
        
        # Loc face qua nho
        filtered = {}
        for k, v in valid_faces.items():
            x, y, w, h = v['facial_area']
            if min(w, h) >= self.min_face_size:
                filtered[k] = v
        
        if not filtered:
            return None
        
        # Chon face lon nhat
        if self.select_largest and len(filtered) > 1:
            best_key = max(filtered.keys(), 
                          key=lambda k: filtered[k]['facial_area'][2] * filtered[k]['facial_area'][3])
        else:
            best_key = list(filtered.keys())[0]
        
        face = filtered[best_key]
        x, y, w, h = face['facial_area']
        
        result = {
            'bbox': [x, y, x + w, y + h],
            'confidence': float(face['score']),
            'landmarks': {
                'left_eye': tuple(face['landmarks']['left_eye']),
                'right_eye': tuple(face['landmarks']['right_eye']),
                'nose': tuple(face['landmarks']['nose']),
                'left_mouth': tuple(face['landmarks']['mouth_left']),
                'right_mouth': tuple(face['landmarks']['mouth_right']),
            }
        }
        
        return result
    
    def _detect_opencv(self, image: np.ndarray) -> Optional[Dict]:
        """Detect voi OpenCV Haar Cascade (chi tra ve bbox, khong co landmarks)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        if len(faces) == 0:
            return None
        
        # Chon face lon nhat
        if self.select_largest and len(faces) > 1:
            areas = [w * h for (x, y, w, h) in faces]
            best_idx = np.argmax(areas)
        else:
            best_idx = 0
        
        x, y, w, h = faces[best_idx]
        
        return {
            'bbox': [int(x), int(y), int(x + w), int(y + h)],
            'confidence': 1.0,  # Haar khong tra ve confidence
            'landmarks': None
        }
    
    def detect_batch(
        self, 
        image_paths: List[str], 
        output_csv: Optional[str] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Batch processing cho nhieu anh
        
        Args:
            image_paths: List cac duong dan anh
            output_csv: Duong dan de luu ket qua (optional)
            show_progress: Hien thi progress bar
            
        Returns:
            DataFrame voi columns: image_path, status, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                   confidence, lm_left_eye_x, lm_left_eye_y, ...
        """
        results = []
        
        iterator = tqdm(image_paths, desc="Detecting faces") if show_progress else image_paths
        
        for img_path in iterator:
            record = {'image_path': img_path}
            
            try:
                image = cv2.imread(img_path)
                if image is None:
                    record['status'] = 'error_read'
                    results.append(record)
                    continue
                
                detection = self.detect(image)
                
                if detection is None:
                    record['status'] = 'no_face'
                else:
                    record['status'] = 'success'
                    record['bbox_x1'] = detection['bbox'][0]
                    record['bbox_y1'] = detection['bbox'][1]
                    record['bbox_x2'] = detection['bbox'][2]
                    record['bbox_y2'] = detection['bbox'][3]
                    record['confidence'] = detection['confidence']
                    
                    if detection['landmarks']:
                        for name, (x, y) in detection['landmarks'].items():
                            record[f'lm_{name}_x'] = x
                            record[f'lm_{name}_y'] = y
                            
            except Exception as e:
                record['status'] = f'error: {str(e)}'
            
            results.append(record)
        
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Saved detection results to {output_csv}")
        
        # In thong ke
        if show_progress:
            total = len(df)
            success = len(df[df['status'] == 'success'])
            no_face = len(df[df['status'] == 'no_face'])
            errors = total - success - no_face
            
            print(f"\nDetection Summary:")
            print(f"  Total: {total}")
            print(f"  Success: {success} ({100*success/total:.1f}%)")
            print(f"  No face: {no_face} ({100*no_face/total:.1f}%)")
            print(f"  Errors: {errors} ({100*errors/total:.1f}%)")
        
        return df
    
    def crop_face(
        self, 
        image: np.ndarray, 
        margin: float = 0.3,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Detect va crop face tu anh
        
        Args:
            image: Input image
            margin: Bo sung margin xung quanh face (0.3 = 30%)
            target_size: Resize output ve kích thuoc nay (width, height)
            
        Returns:
            Cropped face image hoac None
        """
        detection = self.detect(image)
        if detection is None:
            return None
        
        x1, y1, x2, y2 = detection['bbox']
        h, w = image.shape[:2]
        
        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin_w = int(face_w * margin)
        margin_h = int(face_h * margin)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        cropped = image[y1:y2, x1:x2]
        
        if target_size:
            cropped = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
        
        return cropped
    
    def visualize(
        self, 
        image: np.ndarray, 
        detection: Optional[Dict] = None,
        show_landmarks: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Ve bbox va landmarks len anh
        
        Args:
            image: Input image
            detection: Ket qua tu detect(), neu None se tu dong detect
            show_landmarks: Ve 5 diem landmark
            show_confidence: Hien thi confidence score
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        if detection is None:
            detection = self.detect(image)
        
        if detection is None:
            cv2.putText(vis_image, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_image
        
        x1, y1, x2, y2 = detection['bbox']
        
        # Ve bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Ve confidence
        if show_confidence:
            conf_text = f"{detection['confidence']:.2f}"
            cv2.putText(vis_image, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Ve landmarks
        if show_landmarks and detection['landmarks']:
            colors = {
                'left_eye': (0, 0, 255),
                'right_eye': (0, 0, 255),
                'nose': (0, 255, 255),
                'left_mouth': (255, 0, 255),
                'right_mouth': (255, 0, 255),
            }
            for name, (x, y) in detection['landmarks'].items():
                color = colors.get(name, (0, 255, 0))
                cv2.circle(vis_image, (int(x), int(y)), 3, color, -1)
        
        return vis_image


def compare_detectors(image_paths: List[str], output_dir: str = 'comparison_results'):
    """
    So sanh cac detector backends
    
    Args:
        image_paths: List anh de test
        output_dir: Thu muc luu ket qua
    """
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    backends = ['mtcnn', 'opencv']
    
    # Check RetinaFace
    try:
        import retinaface
        backends.append('retinaface')
    except ImportError:
        print("[WARN] RetinaFace not available")
    
    results = {backend: {'times': [], 'detected': 0} for backend in backends}
    
    for backend in backends:
        print(f"\n{'='*40}")
        print(f"Testing {backend.upper()}")
        print(f"{'='*40}")
        
        try:
            detector = FaceDetector(backend=backend)
            
            for img_path in tqdm(image_paths[:100], desc=f"{backend}"):
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                start = time.time()
                result = detector.detect(image)
                elapsed = time.time() - start
                
                results[backend]['times'].append(elapsed)
                if result is not None:
                    results[backend]['detected'] += 1
                    
        except Exception as e:
            print(f"[ERROR] {backend}: {e}")
            continue
    
    # In ket qua
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    comparison_data = []
    for backend in backends:
        if results[backend]['times']:
            avg_time = np.mean(results[backend]['times']) * 1000
            detection_rate = results[backend]['detected'] / len(results[backend]['times']) * 100
            comparison_data.append({
                'Backend': backend,
                'Avg Time (ms)': f"{avg_time:.1f}",
                'Detection Rate (%)': f"{detection_rate:.1f}",
                'Speed (img/s)': f"{1000/avg_time:.1f}"
            })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False)
    print(f"\nSaved to {output_dir}/comparison_results.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Detection Module")
    parser.add_argument('--backend', type=str, default='mtcnn',
                       choices=['mtcnn', 'retinaface', 'opencv'])
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--dir', type=str, help='Directory chua anh')
    parser.add_argument('--output', type=str, default='detection_results.csv')
    parser.add_argument('--compare', action='store_true', help='Compare backends')
    
    args = parser.parse_args()
    
    if args.compare and args.dir:
        image_paths = [
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        compare_detectors(image_paths)
        
    elif args.image:
        detector = FaceDetector(backend=args.backend)
        image = cv2.imread(args.image)
        result = detector.detect(image)
        
        if result:
            print(f"Detection successful!")
            print(f"  Bbox: {result['bbox']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            if result['landmarks']:
                print(f"  Landmarks: {list(result['landmarks'].keys())}")
            
            vis = detector.visualize(image, result)
            output_path = 'detection_result.jpg'
            cv2.imwrite(output_path, vis)
            print(f"Saved visualization to {output_path}")
        else:
            print("No face detected!")
            
    elif args.dir:
        detector = FaceDetector(backend=args.backend)
        image_paths = [
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        detector.detect_batch(image_paths, args.output)
        
    else:
        print("Usage:")
        print("  Single image: python face_detector.py --image path/to/image.jpg")
        print("  Batch: python face_detector.py --dir path/to/images/ --output results.csv")
        print("  Compare: python face_detector.py --compare --dir path/to/images/")
