"""
Recognition Engine
Xu ly nhan dang khuon mat su dung embeddings database
Ho tro: ArcFace model, FAISS index, threshold tuning, Face Detection, Face Alignment
"""

import os
import sys
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Union
from PIL import Image

import torch

try:
    from skimage.transform import SimilarityTransform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from inference.extract_embeddings import (
    load_arcface_model,
    get_transform,
    extract_embedding_single
)
from preprocessing.face_detector import FaceDetector

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tinh cosine similarity giua 2 vectors.
    
    NOTE: Embeddings tu extract_embedding_single da duoc L2 normalize,
    nen with normalized vectors: cos_sim = dot(a, b)
    """
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    
    # Check if vectors are normalized (optimization for pre-normalized embeddings)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # If both vectors are approximately L2-normalized (norm ≈ 1), use dot product
    if abs(norm_a - 1.0) < 1e-3 and abs(norm_b - 1.0) < 1e-3:
        return float(np.dot(a, b))
    
    # Otherwise, compute full cosine similarity
    return float(np.dot(a, b) / (norm_a * norm_b))


class RecognitionEngine:
    """
    Engine nhan dang khuon mat
    
    Ho tro 2 che do:
    1. Dict database (npy file): {'name': embedding}
    2. FAISS index + prototypes
    """
    
    def __init__(
        self,
        model_path: str = "models/checkpoints/arcface/arcface_best.pth",
        db_path: str = None,
        faiss_index_path: str = None,
        prototypes_path: str = None,
        label_mapping_path: str = None,
        device: str = None,
        threshold: float = 0.5,
        use_face_detection: bool = True
    ):
        """
        Khoi tao Recognition Engine
        
        Args:
            model_path: Duong dan den trained model
            db_path: Duong dan den dict database (.npy)
            faiss_index_path: Duong dan den FAISS index
            prototypes_path: Duong dan den prototypes (.npy)
            label_mapping_path: Duong dan den label mapping (.npy)
            device: 'cuda' hoac 'cpu'
            threshold: Nguong similarity de nhan dang
            use_face_detection: Bat/tat face detection truoc khi extract embedding
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.use_face_detection = use_face_detection
        
        # Load model
        self.model = None
        self.model_info = None
        if model_path and os.path.exists(model_path):
            self.model, self.model_info = load_arcface_model(model_path, self.device)
        
        self.transform = get_transform()
        
        # Khoi tao Face Detector
        self.face_detector = None
        if self.use_face_detection:
            try:
                self.face_detector = FaceDetector(
                    backend='mtcnn',
                    device=self.device,
                    confidence_threshold=0.9,
                    select_largest=True
                )
                print("Face detector initialized (MTCNN)")
            except Exception as e:
                print(f"Khong the khoi tao Face Detector: {e}")
                self.use_face_detection = False
        
        # Load database
        self.db = None
        self.faiss_index = None
        self.prototypes = None
        self.label_to_id = None
        self.id_to_label = None
        
        # Mode 1: Dict database
        if db_path and os.path.exists(db_path):
            self.db = np.load(db_path, allow_pickle=True).item()
            print(f"Loaded database: {len(self.db)} identities")
        
        # Mode 2: FAISS index
        if faiss_index_path and os.path.exists(faiss_index_path):
            self._load_faiss(faiss_index_path, prototypes_path, label_mapping_path)
    
    def _load_faiss(self, index_path: str, prototypes_path: str = None, mapping_path: str = None):
        """Load FAISS index va metadata"""
        try:
            import faiss
            self.faiss_index = faiss.read_index(index_path)
            print(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")
        except ImportError:
            print("FAISS chua cai dat")
            return
        except Exception as e:
            print(f"Loi load FAISS: {e}")
            return
        
        if prototypes_path and os.path.exists(prototypes_path):
            self.prototypes = np.load(prototypes_path)
            print(f"Loaded prototypes: {self.prototypes.shape}")
        
        if mapping_path and os.path.exists(mapping_path):
            mapping = np.load(mapping_path, allow_pickle=True).item()
            self.label_to_id = mapping.get('label_to_id', {})
            self.id_to_label = mapping.get('id_to_label', {})
            print(f"Loaded label mapping: {len(self.label_to_id)} classes")
    
    def set_threshold(self, threshold: float):
        """Thay doi threshold"""
        self.threshold = threshold
    
    def align_face(self, image: np.ndarray, landmarks: Dict) -> Optional[np.ndarray]:
        """
        Align face theo ArcFace template (112x112)
        
        Args:
            image: Anh BGR format (tu cv2)
            landmarks: Dict voi 5 diem (left_eye, right_eye, nose, left_mouth, right_mouth)
            
        Returns:
            Anh da align 112x112 hoac None
        """
        if not HAS_SKIMAGE:
            return None
            
        try:
            src = np.array([
                landmarks.get('left_eye', [0, 0]),
                landmarks.get('right_eye', [0, 0]),
                landmarks.get('nose', [0, 0]),
                landmarks.get('left_mouth', [0, 0]),
                landmarks.get('right_mouth', [0, 0])
            ], dtype=np.float32)
            
            if np.all(src == 0):
                return None
            
            tform = SimilarityTransform()
            tform.estimate(src, ARCFACE_TEMPLATE)
            M = tform.params[0:2, :]
            
            aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)
            return aligned
            
        except Exception as e:
            print(f"Loi align face: {e}")
            return None
    
    def detect_and_align(self, img_input: Union[str, np.ndarray]) -> Optional[Image.Image]:
        """
        Detect va align face tu anh
        
        Args:
            img_input: Duong dan anh hoac numpy array BGR
            
        Returns:
            PIL Image da align hoac None
        """
        if self.face_detector is None:
            return None
        
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                return None
        else:
            image = img_input
        
        detection = self.face_detector.detect(image)
        if detection is None:
            return None
        
        landmarks = detection.get('landmarks')
        if landmarks and HAS_SKIMAGE:
            aligned = self.align_face(image, landmarks)
            if aligned is not None:
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                return Image.fromarray(aligned_rgb)
        
        cropped = self.face_detector.crop_face(image, margin=0.2, target_size=(112, 112))
        if cropped is not None:
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        
        return None
    
    def extract_embedding(self, img_input: Union[str, Image.Image]) -> Optional[np.ndarray]:
        """
        Trich xuat embedding tu anh (co ho tro face detection + alignment)
        
        Args:
            img_input: Duong dan anh hoac PIL Image
            
        Returns:
            Embedding vector hoac None
        """
        if self.model is None:
            print("Model chua duoc load")
            return None
        
        processed_img = img_input
        
        if self.use_face_detection and self.face_detector is not None:
            aligned = self.detect_and_align(img_input)
            if aligned is not None:
                processed_img = aligned
        
        return extract_embedding_single(processed_img, self.model, self.transform, self.device)
    
    def recognize_with_db(self, embedding: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Nhan dang su dung dict database
        
        Returns:
            (best_name, best_score, top_k_results)
        """
        if self.db is None:
            return "No database", 0.0, []
        
        scores = []
        for name, vec in self.db.items():
            score = cosine_similarity(embedding, vec)
            scores.append((name, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_name, best_score = scores[0]
        
        if best_score < self.threshold:
            return "Unknown", best_score, scores[:5]
        
        return best_name, best_score, scores[:5]
    
    def recognize_with_faiss(self, embedding: np.ndarray, k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Nhan dang su dung FAISS index
        
        Returns:
            (best_name, best_score, top_k_results)
        """
        if self.faiss_index is None:
            return "No FAISS index", 0.0, []
        
        embedding = embedding.astype(np.float32).reshape(1, -1)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        scores, indices = self.faiss_index.search(embedding, k)
        scores = scores.flatten()
        indices = indices.flatten()
        
        results = []
        for idx, score in zip(indices, scores):
            if idx == -1:
                continue
            if self.id_to_label:
                name = self.id_to_label.get(idx, f"ID_{idx}")
            else:
                name = f"ID_{idx}"
            results.append((name, float(score)))
        
        if len(results) == 0:
            return "Unknown", 0.0, []
        
        best_name, best_score = results[0]
        
        if best_score < self.threshold:
            return "Unknown", best_score, results
        
        return best_name, best_score, results
    
    def recognize(
        self,
        img_input: Union[str, Image.Image],
        use_faiss: bool = None,
        k: int = 5
    ) -> Dict:
        """
        Nhan dang khuon mat tu anh
        
        Args:
            img_input: Duong dan anh hoac PIL Image
            use_faiss: Su dung FAISS (None = auto detect)
            k: So luong top-k results
            
        Returns:
            Dict chua ket qua nhan dang
        """
        result = {
            'identity': 'Unknown',
            'confidence': 0.0,
            'top_k': [],
            'embedding': None,
            'status': 'success'
        }
        
        # Extract embedding
        embedding = self.extract_embedding(img_input)
        
        if embedding is None:
            result['status'] = 'error'
            result['message'] = 'Cannot extract embedding (no face or invalid image)'
            return result
        
        result['embedding'] = embedding
        
        # Chon mode
        if use_faiss is None:
            use_faiss = self.faiss_index is not None
        
        # Nhan dang
        if use_faiss and self.faiss_index is not None:
            identity, confidence, top_k = self.recognize_with_faiss(embedding, k)
        elif self.db is not None:
            identity, confidence, top_k = self.recognize_with_db(embedding)
        else:
            result['status'] = 'error'
            result['message'] = 'No database loaded'
            return result
        
        result['identity'] = identity
        result['confidence'] = confidence
        result['top_k'] = top_k
        
        return result
    
    def recognize_batch(
        self,
        img_inputs: List[Union[str, Image.Image]],
        use_faiss: bool = None
    ) -> List[Dict]:
        """Nhan dang nhieu anh"""
        return [self.recognize(img, use_faiss) for img in img_inputs]
    
    def add_to_db(self, name: str, img_inputs: List[Union[str, Image.Image]]) -> bool:
        """
        Them identity moi vao database
        
        Args:
            name: Ten identity
            img_inputs: List anh cua identity
            
        Returns:
            True neu thanh cong
        """
        embeddings = []
        for img in img_inputs:
            emb = self.extract_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            print(f"Khong the extract embedding cho {name}")
            return False
        
        # Tinh mean embedding
        mean_emb = np.mean(np.stack(embeddings), axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        
        if self.db is None:
            self.db = {}
        
        self.db[name] = mean_emb
        print(f"Added {name} to database (from {len(embeddings)} images)")
        
        return True
    
    def save_db(self, path: str):
        """Luu database"""
        if self.db:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            np.save(path, self.db)
            print(f"Saved database: {path}")
    
    def get_db_identities(self) -> List[str]:
        """Lay danh sach identities trong database"""
        if self.db:
            return list(self.db.keys())
        return []


def create_engine_from_embeddings_dir(
    model_path: str,
    embeddings_dir: str,
    threshold: float = 0.5,
    device: str = None
) -> RecognitionEngine:
    """
    Tao RecognitionEngine tu thu muc embeddings (output cua extract_embeddings.py)
    
    Args:
        model_path: Duong dan den trained model
        embeddings_dir: Thu muc chua embeddings files
        threshold: Nguong similarity
        device: Device
    """
    faiss_path = os.path.join(embeddings_dir, "arcface_index.faiss")
    prototypes_path = os.path.join(embeddings_dir, "arcface_prototypes.npy")
    mapping_path = os.path.join(embeddings_dir, "label_mapping.npy")
    
    return RecognitionEngine(
        model_path=model_path,
        faiss_index_path=faiss_path if os.path.exists(faiss_path) else None,
        prototypes_path=prototypes_path if os.path.exists(prototypes_path) else None,
        label_mapping_path=mapping_path if os.path.exists(mapping_path) else None,
        threshold=threshold,
        device=device
    )


if __name__ == "__main__":
    print("="*60)
    print("RECOGNITION ENGINE TEST")
    print("="*60)
    
    # Test voi dummy database
    db_path = "data/arcface_embeddings_db.npy"
    model_path = "models/checkpoints/arcface/arcface_best.pth"
    
    if os.path.exists(db_path) and os.path.exists(model_path):
        engine = RecognitionEngine(
            model_path=model_path,
            db_path=db_path,
            threshold=0.5
        )
        
        print(f"\nDatabase identities: {engine.get_db_identities()}")
        
        # Test recognition
        test_image = "static/uploads/anh1.jpg"
        if os.path.exists(test_image):
            result = engine.recognize(test_image)
            print(f"\nResult: {result['identity']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Top-5: {result['top_k']}")
    else:
        print(f"Model hoac database chua ton tai")
        print(f"  - Model: {model_path} - {'OK' if os.path.exists(model_path) else 'MISSING'}")
        print(f"  - DB: {db_path} - {'OK' if os.path.exists(db_path) else 'MISSING'}")
