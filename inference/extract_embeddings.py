"""
Extract Embeddings Script
Trich xuat embeddings tu anh su dung trained ArcFace model
Ho tro: FAISS index, prototype computation, t-SNE visualization, Face Detection + Alignment
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    from skimage.transform import SimilarityTransform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from preprocessing.face_detector import FaceDetector

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def load_arcface_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, dict]:
    """
    Load trained ArcFace model
    
    Args:
        model_path: Duong dan den file checkpoint (.pth)
        device: 'cuda' hoac 'cpu'
        
    Returns:
        (model, checkpoint_info)
    """
    from models.arcface.arcface_model import ArcFaceModel
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model khong ton tai: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Lay thong tin tu checkpoint hoac config
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes', checkpoint.get('num_classes', 100))
    embedding_size = config.get('model', {}).get('embedding_size', 512)
    
    model = ArcFaceModel(num_classes=num_classes, embedding_size=embedding_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    info = {
        'num_classes': num_classes,
        'embedding_size': embedding_size,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 'N/A'),
        'best_val_acc': checkpoint.get('best_val_acc', 'N/A')
    }
    
    print(f"Loaded model from {model_path}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Embedding size: {embedding_size}")
    print(f"  - Epoch: {info['epoch']}")
    if info['val_acc'] != 'N/A':
        print(f"  - Val accuracy: {info['val_acc']:.2f}%")
    
    return model, info


def load_facenet_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, dict]:
    """
    Load trained FaceNet model
    
    Args:
        model_path: Duong dan den file checkpoint (.pth)
        device: 'cuda' hoac 'cpu'
        
    Returns:
        (model, checkpoint_info)
    """
    from models.facenet.facenet_model import FaceNetModel
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model khong ton tai: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    embedding_size = config.get('model', {}).get('embedding_size', 512)
    
    model = FaceNetModel(embedding_size=embedding_size, pretrained='vggface2', device=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.to(device)
    model.eval()
    
    info = {
        'embedding_size': embedding_size,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_val_acc': checkpoint.get('best_val_acc', 'N/A')
    }
    
    print(f"Loaded FaceNet model from {model_path}")
    print(f"  - Embedding size: {embedding_size}")
    print(f"  - Epoch: {info['epoch']}")
    
    return model, info


def get_transform(image_size: int = 112):
    """Transform cho inference"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_facenet_transform(image_size: int = 160):
    """Transform cho FaceNet inference (160x160)"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


class FacePreprocessor:
    """
    Face Preprocessor: Detect + Align face truoc khi extract embedding
    Dam bao nhat quan voi recognition_engine.py
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        try:
            self.detector = FaceDetector(
                backend='mtcnn',
                device=self.device,
                confidence_threshold=0.9,
                select_largest=True
            )
            print("[OK] FacePreprocessor initialized (MTCNN)")
        except Exception as e:
            print(f"[WARN] Khong the khoi tao Face Detector: {e}")
            self.detector = None
    
    def align_face(self, image: np.ndarray, landmarks: Dict) -> Optional[np.ndarray]:
        """
        Align face theo ArcFace template (112x112)
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
        except Exception:
            return None
    
    def process(self, img_input: Union[str, np.ndarray]) -> Optional[Image.Image]:
        """
        Detect va align face tu anh
        
        Args:
            img_input: Duong dan anh hoac numpy array BGR
            
        Returns:
            PIL Image da align (112x112) hoac None
        """
        if self.detector is None:
            if isinstance(img_input, str):
                return Image.open(img_input).convert('RGB')
            return Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                return None
        else:
            image = img_input
        
        detection = self.detector.detect(image)
        if detection is None:
            return None
        
        landmarks = detection.get('landmarks')
        if landmarks and HAS_SKIMAGE:
            aligned = self.align_face(image, landmarks)
            if aligned is not None:
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                return Image.fromarray(aligned_rgb)
        
        cropped = self.detector.crop_face(image, margin=0.2, target_size=(112, 112))
        if cropped is not None:
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        
        return None


class FaceNetPreprocessor:
    """
    Face Preprocessor cho FaceNet: Detect + Crop face (160x160)
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        try:
            self.detector = FaceDetector(
                backend='mtcnn',
                device=self.device,
                confidence_threshold=0.9,
                select_largest=True
            )
            print("[OK] FaceNetPreprocessor initialized (MTCNN)")
        except Exception as e:
            print(f"[WARN] Khong the khoi tao Face Detector: {e}")
            self.detector = None
    
    def process(self, img_input: Union[str, np.ndarray]) -> Optional[Image.Image]:
        """
        Detect va crop face tu anh cho FaceNet
        
        Args:
            img_input: Duong dan anh hoac numpy array BGR
            
        Returns:
            PIL Image da crop (160x160) hoac None
        """
        if self.detector is None:
            if isinstance(img_input, str):
                img = Image.open(img_input).convert('RGB')
                return img.resize((160, 160))
            img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            return img.resize((160, 160))
        
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                return None
        else:
            image = img_input
        
        detection = self.detector.detect(image)
        if detection is None:
            return None
        
        cropped = self.detector.crop_face(image, margin=0.2, target_size=(160, 160))
        if cropped is not None:
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cropped_rgb)
        
        return None


def extract_embedding_single(
    img_input,
    model: nn.Module,
    transform,
    device: str = 'cpu',
    model_type: str = 'arcface'
) -> Optional[np.ndarray]:
    """
    Trich xuat embedding cho 1 anh
    
    Args:
        img_input: Duong dan anh hoac PIL Image
        model: ArcFace or FaceNet model
        transform: Image transform
        device: Device
        model_type: 'arcface' or 'facenet'
        
    Returns:
        Embedding vector (numpy array) hoac None neu loi
    """
    try:
        if isinstance(img_input, str):
            img = Image.open(img_input).convert('RGB')
        else:
            img = img_input.convert('RGB')
            
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if model_type == 'facenet':
                embedding = model(img_tensor)
            else:
                embedding = model(img_tensor, labels=None)
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.cpu().numpy().flatten()
        
        return embedding
        
    except Exception as e:
        if isinstance(img_input, str):
            print(f"Loi xu ly {img_input}: {e}")
        return None


def extract_embeddings_batch(
    image_paths: List[str],
    model: nn.Module,
    transform,
    device: str = 'cpu',
    batch_size: int = 64
) -> Tuple[np.ndarray, List[str]]:
    """
    Trich xuat embeddings cho nhieu anh (batch processing)
    
    Returns:
        (embeddings array, valid_paths list)
    """
    embeddings = []
    valid_paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                batch_valid_paths.append(path)
            except Exception as e:
                print(f"Skip {path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
            
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            batch_embeddings = model(batch_tensor, labels=None)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy()
        
        embeddings.append(batch_embeddings)
        valid_paths.extend(batch_valid_paths)
    
    if len(embeddings) == 0:
        return np.array([]), []
    
    return np.vstack(embeddings), valid_paths


def extract_embeddings_from_csv(
    model_path: str,
    csv_path: str,
    data_root: str = None,
    output_dir: str = "data/embeddings",
    device: str = None,
    batch_size: int = 64
) -> Dict:
    """
    Trich xuat embeddings tu metadata CSV (training set)
    
    Args:
        model_path: Duong dan checkpoint
        csv_path: Duong dan file metadata CSV
        data_root: Thu muc goc chua anh
        output_dir: Thu muc luu output
        device: cuda/cpu
        batch_size: Batch size
        
    Returns:
        Dict chua thong tin embeddings
    """
    print("="*60)
    print("EXTRACT EMBEDDINGS FROM TRAINING SET")
    print("="*60)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    model, model_info = load_arcface_model(model_path, device)
    transform = get_transform()
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded CSV: {len(df)} samples")
    
    # Auto-detect columns
    if 'image_path' in df.columns:
        path_col = 'image_path'
    elif 'image' in df.columns:
        path_col = 'image'
    else:
        raise ValueError(f"CSV khong co cot path. Columns: {list(df.columns)}")
    
    if 'identity_name' in df.columns:
        id_col = 'identity_name'
    elif 'person_id' in df.columns:
        id_col = 'person_id'
    else:
        raise ValueError(f"CSV khong co cot identity. Columns: {list(df.columns)}")
    
    # Build full paths
    if data_root:
        image_paths = [os.path.join(data_root, p) for p in df[path_col]]
    else:
        image_paths = df[path_col].tolist()
    
    identities = df[id_col].astype(str).tolist()
    
    # Create label mapping
    unique_ids = sorted(set(identities))
    id_to_label = {id_: idx for idx, id_ in enumerate(unique_ids)}
    labels = [id_to_label[id_] for id_ in identities]
    
    print(f"Unique identities: {len(unique_ids)}")
    
    # Extract embeddings
    embeddings, valid_paths = extract_embeddings_batch(
        image_paths, model, transform, device, batch_size
    )
    
    # Filter valid samples
    valid_indices = [image_paths.index(p) for p in valid_paths]
    valid_labels = [labels[i] for i in valid_indices]
    valid_identities = [identities[i] for i in valid_indices]
    
    print(f"\nExtracted {len(embeddings)} embeddings")
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "arcface_train_embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings: {embeddings_path}")
    
    # Save metadata
    meta_df = pd.DataFrame({
        'image_path': valid_paths,
        'identity': valid_identities,
        'label': valid_labels
    })
    meta_path = os.path.join(output_dir, "embeddings_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved metadata: {meta_path}")
    
    # Save label mapping
    mapping_path = os.path.join(output_dir, "label_mapping.npy")
    np.save(mapping_path, {'id_to_label': id_to_label, 'label_to_id': {v:k for k,v in id_to_label.items()}})
    print(f"Saved label mapping: {mapping_path}")
    
    return {
        'embeddings': embeddings,
        'labels': np.array(valid_labels),
        'identities': valid_identities,
        'paths': valid_paths,
        'id_to_label': id_to_label
    }


def compute_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str = None
) -> np.ndarray:
    """
    Tinh prototype (mean embedding) cho moi identity
    
    Args:
        embeddings: (N, D) embeddings array
        labels: (N,) label array
        output_path: Duong dan luu prototypes
        
    Returns:
        (num_classes, D) prototype array
    """
    print("\n=== COMPUTING PROTOTYPES ===")
    
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    embedding_dim = embeddings.shape[1]
    
    prototypes = np.zeros((num_classes, embedding_dim), dtype=np.float32)
    
    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        prototype = class_embeddings.mean(axis=0)
        prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
        prototypes[label] = prototype
    
    print(f"Computed {num_classes} prototypes")
    
    if output_path:
        np.save(output_path, prototypes)
        print(f"Saved prototypes: {output_path}")
    
    return prototypes


def build_faiss_index(
    embeddings: np.ndarray,
    output_path: str = None,
    use_gpu: bool = False
):
    """
    Xay dung FAISS index tu embeddings
    
    Args:
        embeddings: (N, D) embeddings array (da L2 normalize)
        output_path: Duong dan luu index
        use_gpu: Su dung GPU FAISS
        
    Returns:
        FAISS index
    """
    try:
        import faiss
    except ImportError:
        print("FAISS chua duoc cai dat. Chay: pip install faiss-cpu hoac faiss-gpu")
        return None
    
    print("\n=== BUILDING FAISS INDEX ===")
    
    embeddings = embeddings.astype('float32')
    
    # L2 normalize (neu chua)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    dim = embeddings.shape[1]
    
    # IndexFlatIP cho cosine similarity (vi da normalize)
    index = faiss.IndexFlatIP(dim)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Using GPU FAISS")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings)
    
    print(f"Index built: {index.ntotal} vectors, {dim}D")
    
    if output_path:
        if use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, output_path)
        print(f"Saved FAISS index: {output_path}")
    
    return index


def visualize_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    num_samples: int = 2000,
    perplexity: int = 30
):
    """
    Visualize embedding space voi t-SNE
    
    Args:
        embeddings: (N, D) embeddings array
        labels: (N,) label array
        output_path: Duong dan luu anh
        num_samples: So samples toi da (de tang toc)
        perplexity: t-SNE perplexity
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("Thieu sklearn/matplotlib. Chay: pip install scikit-learn matplotlib")
        return
    
    print("\n=== T-SNE VISUALIZATION ===")
    
    # Subsample neu qua nhieu
    if len(embeddings) > num_samples:
        indices = np.random.choice(len(embeddings), num_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    print(f"Running t-SNE on {len(embeddings)} samples...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 12))
    
    num_classes = len(np.unique(labels))
    if num_classes <= 20:
        cmap = 'tab20'
    else:
        cmap = 'viridis'
    
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.6,
        s=15
    )
    
    plt.colorbar(scatter, label='Identity')
    plt.title(f'ArcFace Embedding Space (t-SNE)\n{len(np.unique(labels))} identities, {len(embeddings)} samples')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE visualization: {output_path}")


def extract_embedding_for_folder(
    folder: str,
    model: nn.Module,
    transform,
    device: str = 'cpu',
    preprocessor: Optional[FacePreprocessor] = None
) -> Optional[np.ndarray]:
    """
    Trich xuat va tinh trung binh embeddings cho tat ca anh trong folder
    Co ho tro face detection + alignment
    
    Args:
        folder: Thu muc chua anh
        model: ArcFace model
        transform: Image transform
        device: Device
        preprocessor: FacePreprocessor instance (optional)
    """
    if not os.path.exists(folder):
        return None
    
    embeddings = []
    
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(folder, f)
            
            processed_img = None
            if preprocessor is not None:
                processed_img = preprocessor.process(img_path)
            
            if processed_img is not None:
                emb = extract_embedding_single(processed_img, model, transform, device)
            else:
                emb = extract_embedding_single(img_path, model, transform, device)
                
            if emb is not None:
                embeddings.append(emb)
    
    if len(embeddings) == 0:
        return None
    
    stacked = np.stack(embeddings, axis=0)
    mean_emb = np.mean(stacked, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    
    return mean_emb


def build_db(
    model_path: str,
    root_folder: str = "data/celeb",
    save_path: str = "data/embeddings_db.npy",
    device: str = None,
    use_face_detection: bool = True,
    model_type: str = "arcface"
) -> None:
    """
    Build embedding database tu folder celebrities
    
    Args:
        model_path: Duong dan checkpoint
        root_folder: Thu muc chua cac folder celebrity
        save_path: Duong dan luu embeddings_db.npy
        device: 'cuda' hoac 'cpu'
        use_face_detection: Su dung face detection + alignment truoc khi extract
        model_type: 'arcface' hoac 'facenet'
    """
    print("="*60)
    print(f"EXTRACT EMBEDDINGS DATABASE ({model_type.upper()})")
    print("="*60)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nModel type: {model_type}")
    print(f"Device: {device}")
    print(f"Use face detection: {use_face_detection}")
    
    if not os.path.exists(root_folder):
        print(f"Root folder khong ton tai: {root_folder}")
        return
    
    if model_type == "facenet":
        model, _ = load_facenet_model(model_path, device)
        transform = get_facenet_transform()
        preprocessor = FaceNetPreprocessor(device=device) if use_face_detection else None
    else:
        model, _ = load_arcface_model(model_path, device)
        transform = get_transform()
        preprocessor = FacePreprocessor(device=device) if use_face_detection else None
    
    db: Dict[str, np.ndarray] = {}
    
    persons = [p for p in os.listdir(root_folder)
               if os.path.isdir(os.path.join(root_folder, p))]
    
    print(f"\nTim thay {len(persons)} celebrities")
    print("Dang extract embeddings...\n")
    
    success_count = 0
    for person in tqdm(persons, desc="Processing"):
        person_folder = os.path.join(root_folder, person)
        emb = extract_embedding_for_folder(
            person_folder, model, transform, device, preprocessor
        )
        if emb is not None:
            db[person] = emb
            success_count += 1
    
    if len(db) == 0:
        print("\nKhong co embeddings nao duoc tao!")
        return
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    np.save(save_path, db)
    
    print(f"\nDa luu {len(db)} embeddings vao {save_path}")
    print(f"Success rate: {success_count}/{len(persons)} ({100*success_count/len(persons):.1f}%)")
    print("\nDatabase ready!")


def full_pipeline(
    model_path: str,
    csv_path: str,
    data_root: str = None,
    output_dir: str = "data/embeddings",
    device: str = None,
    batch_size: int = 64
):
    """
    Chay full pipeline: extract embeddings -> prototypes -> FAISS -> t-SNE
    """
    print("="*60)
    print("FULL EMBEDDING EXTRACTION PIPELINE")
    print("="*60)
    
    # 1. Extract embeddings
    result = extract_embeddings_from_csv(
        model_path=model_path,
        csv_path=csv_path,
        data_root=data_root,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size
    )
    
    embeddings = result['embeddings']
    labels = result['labels']
    
    # 2. Compute prototypes
    prototype_path = os.path.join(output_dir, "arcface_prototypes.npy")
    prototypes = compute_prototypes(embeddings, labels, prototype_path)
    
    # 3. Build FAISS index
    faiss_path = os.path.join(output_dir, "arcface_index.faiss")
    build_faiss_index(prototypes, faiss_path)
    
    # 4. t-SNE visualization
    tsne_path = os.path.join(output_dir, "tsne_visualization.png")
    visualize_tsne(embeddings, labels, tsne_path)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print(f"  - arcface_train_embeddings.npy")
    print(f"  - embeddings_metadata.csv")
    print(f"  - label_mapping.npy")
    print(f"  - arcface_prototypes.npy")
    print(f"  - arcface_index.faiss")
    print(f"  - tsne_visualization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument('--model-path', type=str, default='models/checkpoints/arcface/arcface_best.pth')
    parser.add_argument('--mode', type=str, choices=['db', 'csv', 'full'], default='full',
                       help='db: build from folders, csv: from metadata CSV, full: full pipeline')
    parser.add_argument('--csv-path', type=str, default=None,
                       help='Path to metadata CSV (for csv/full mode)')
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root folder for images')
    parser.add_argument('--data-dir', type=str, default='data/celeb',
                       help='Folder chua anh celebrities (for db mode)')
    parser.add_argument('--output-dir', type=str, default='data/embeddings')
    parser.add_argument('--output-path', type=str, default='data/embeddings_db.npy',
                       help='Output path for db mode')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--use-face-detection', action='store_true', default=True,
                       help='Use face detection + alignment before extracting embeddings (default: True)')
    parser.add_argument('--no-face-detection', action='store_true',
                       help='Disable face detection (use raw images)')
    parser.add_argument('--model-type', type=str, choices=['arcface', 'facenet'], default='arcface',
                       help='Model type: arcface or facenet (for db mode)')
    
    args = parser.parse_args()
    
    use_fd = args.use_face_detection and not args.no_face_detection
    
    if args.mode == 'db':
        build_db(
            model_path=args.model_path,
            root_folder=args.data_dir,
            save_path=args.output_path,
            device=args.device,
            use_face_detection=use_fd,
            model_type=args.model_type
        )
    elif args.mode == 'csv':
        if args.csv_path is None:
            print("Vui long cung cap --csv-path")
        else:
            extract_embeddings_from_csv(
                model_path=args.model_path,
                csv_path=args.csv_path,
                data_root=args.data_root,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size
            )
    else:  # full
        if args.csv_path is None:
            print("Vui long cung cap --csv-path cho full pipeline")
        else:
            full_pipeline(
                model_path=args.model_path,
                csv_path=args.csv_path,
                data_root=args.data_root,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size
            )
