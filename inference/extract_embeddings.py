"""
Extract Embeddings Script
Trích xuất embeddings từ ảnh sử dụng trained ArcFace model
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def load_arcface_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load trained ArcFace model
    
    Args:
        model_path: Đường dẫn đến file checkpoint (.pth)
        device: 'cuda' hoặc 'cpu'
        
    Returns:
        Model đã load weights
    """
    from models.arcface.arcface_model import ArcFaceModel
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = checkpoint.get('num_classes', 100)
    embedding_size = checkpoint.get('embedding_size', 512)
    
    model = ArcFaceModel(num_classes=num_classes, embedding_size=embedding_size)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Embedding size: {embedding_size}")
    
    return model


def get_transform(image_size: int = 112):
    """Transform cho inference"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def extract_embedding_single_image(
    img_path: str, 
    model: nn.Module, 
    transform, 
    device: str = 'cpu'
) -> Optional[np.ndarray]:
    """
    Trích xuất embedding cho 1 ảnh
    
    Args:
        img_path: Đường dẫn ảnh
        model: ArcFace model
        transform: Image transform
        device: Device
        
    Returns:
        Embedding vector (numpy array) hoặc None nếu lỗi
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, embedding = model(img_tensor, labels=None)
            embedding = embedding.cpu().numpy().flatten()
        
        return embedding
        
    except Exception as e:
        print(f"Lỗi xử lý {img_path}: {e}")
        return None


def extract_embedding_for_folder(
    folder: str, 
    model: nn.Module, 
    transform, 
    device: str = 'cpu'
) -> Optional[np.ndarray]:
    """
    Trích xuất và tính trung bình embeddings cho tất cả ảnh trong folder
    
    Args:
        folder: Đường dẫn folder chứa ảnh
        model: ArcFace model
        transform: Image transform
        device: Device
        
    Returns:
        Mean embedding vector
    """
    if not os.path.exists(folder):
        return None
    
    embeddings = []
    
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, f)
            emb = extract_embedding_single_image(img_path, model, transform, device)
            
            if emb is not None:
                embeddings.append(emb)
    
    if len(embeddings) == 0:
        return None
    
    stacked = np.stack(embeddings, axis=0)
    mean_emb = np.mean(stacked, axis=0)
    
    return mean_emb


def build_db(
    model_path: str,
    root_folder: str = "data/celeb",
    save_path: str = "data/embeddings_db.npy",
    device: str = None
) -> None:
    """
    Build embedding database từ folder celebrities
    
    Args:
        model_path: Đường dẫn đến trained model
        root_folder: Folder chứa các subfolder của từng celebrity
        save_path: Đường dẫn lưu database
        device: 'cuda' hoặc 'cpu', None = auto detect
    """
    print("="*60)
    print("EXTRACT EMBEDDINGS DATABASE")
    print("="*60)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    
    if not os.path.exists(root_folder):
        print(f"Root folder không tồn tại: {root_folder}")
        return
    
    model = load_arcface_model(model_path, device)
    transform = get_transform()
    
    db: Dict[str, np.ndarray] = {}
    
    persons = [p for p in os.listdir(root_folder) 
               if os.path.isdir(os.path.join(root_folder, p))]
    
    print(f"\nTìm thấy {len(persons)} celebrities")
    print("Đang extract embeddings...\n")
    
    for person in tqdm(persons, desc="Processing"):
        person_folder = os.path.join(root_folder, person)
        
        emb = extract_embedding_for_folder(person_folder, model, transform, device)
        
        if emb is not None:
            db[person] = emb
    
    if len(db) == 0:
        print("\nKhông có embeddings nào được tạo!")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, db)
    
    print(f"\nĐã lưu {len(db)} embeddings vào {save_path}")
    print("\nDatabase ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings database")
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='models/checkpoints/arcface_best.pth',
        help='Đường dẫn đến trained model'
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/celeb',
        help='Folder chứa ảnh celebrities'
    )
    parser.add_argument(
        '--output-path', 
        type=str, 
        default='data/embeddings_db.npy',
        help='Đường dẫn lưu database'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='cuda hoặc cpu (default: auto)'
    )
    
    args = parser.parse_args()
    
    build_db(
        model_path=args.model_path,
        root_folder=args.data_dir,
        save_path=args.output_path,
        device=args.device
    )
