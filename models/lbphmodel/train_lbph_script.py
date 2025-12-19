"""
Script train LBPH model từ dataset thư mục
Tương tự extract_embeddings.py, hỗ trợ train lại với dataset mới
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from models.lbphmodel.train_lbph import train_lbph_model
from preprocessing.face_detector import FaceDetector


def create_label_map_from_directory(data_dir: str) -> Dict[int, str]:
    """
    Tạo label_map từ cấu trúc thư mục dataset.
    - Folder name là danh tính (có thể không phải số).
    - label_id được gán theo thứ tự sort ổn định.
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Thư mục không tồn tại: {data_dir}")
    
    identities = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: (not x.isdigit(), x if x.isdigit() else x.lower())
    )
    
    if len(identities) == 0:
        raise ValueError(f"Không tìm thấy identity nào trong {data_dir}")
    
    # Gán label_id tuần tự để dùng cho LBPH
    label_map = {idx: identity for idx, identity in enumerate(identities)}
    
    print(f"[OK] Tạo label_map từ dataset: {len(label_map)} identities")
    print(f"  Data dir: {data_dir}")
    print(f"  Sample identities: {list(identities[:5])}")
    print(f"  Sample label_map: {dict(list(label_map.items())[:5])}")
    
    return label_map


def _preprocess_image_for_lbph(
    image_path: str,
    detector: Optional[FaceDetector],
    target_size: Tuple[int, int] = (100, 100),
    grayscale: bool = True
) -> Optional[np.ndarray]:
    """
    Detect + crop + resize face cho LBPH. Nếu detector None thì chỉ resize ảnh gốc.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        if detector is not None:
            cropped = detector.crop_face(image, margin=0.2, target_size=target_size)
            if cropped is None:
                cropped = cv2.resize(image, target_size)
        else:
            cropped = cv2.resize(image, target_size)
        
        if grayscale:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        return cropped
    except Exception:
        return None


def load_faces_and_labels(
    data_dir: str,
    label_map: Dict[int, str],
    use_face_detection: bool = True,
    target_size: Tuple[int, int] = (100, 100)
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load ảnh, detect + crop (nếu bật), trả về (faces, labels) theo label_map.
    Cấu trúc: data_dir/identity_name/img1.jpg, img2.jpg, ...
    
    Args:
        data_dir: Thư mục chứa các folder identity (ví dụ: data/celeb)
        label_map: Mapping label_id -> identity_name
        use_face_detection: Có detect + crop trước khi train hay không
        target_size: Kích thước face crop
        
    Returns:
        (faces, labels) - faces là list ảnh grayscale, labels là numpy array
    """
    faces: List[np.ndarray] = []
    labels: List[int] = []
    
    identity_to_label = {v: k for k, v in label_map.items()}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    detector = FaceDetector(
        backend='mtcnn',
        device='cpu',
        confidence_threshold=0.9,
        select_largest=True
    ) if use_face_detection else None
    
    for identity in tqdm(sorted(identity_to_label.keys()), desc="Loading images"):
        label_id = identity_to_label[identity]
        identity_path = os.path.join(data_dir, identity)
        if not os.path.isdir(identity_path):
            continue
        
        img_files = [f for f in os.listdir(identity_path) 
                    if os.path.splitext(f.lower())[1] in image_extensions]
        
        for img_name in img_files:
            img_path = os.path.join(identity_path, img_name)
            img = _preprocess_image_for_lbph(
                img_path,
                detector=detector,
                target_size=target_size,
                grayscale=True
            )
            if img is None:
                continue
            faces.append(img)
            labels.append(label_id)
    
    return faces, np.array(labels, dtype=np.int32)


def train_lbph_from_directory(
    data_dir: str,
    output_dir: str = "models/checkpoints/LBHP",
    model_name: str = "lbph_model.xml",
    radius: int = 1,
    neighbors: int = 8,
    grid_x: int = 8,
    grid_y: int = 8,
    use_val_for_threshold: bool = False,
    val_dir: Optional[str] = None,
    use_face_detection: bool = True,
    target_size: Tuple[int, int] = (100, 100)
) -> Tuple[cv2.face.LBPHFaceRecognizer, Dict[int, str]]:
    """
    Train LBPH model từ thư mục dataset
    
    Args:
        data_dir: Thư mục chứa các folder identity (train data)
        output_dir: Thư mục lưu model và label_map
        model_name: Tên file model (default: lbph_model.xml)
        radius: Bán kính LBP operator
        neighbors: Số neighbors LBP operator
        grid_x: Số cells theo chiều ngang
        grid_y: Số cells theo chiều dọc
        use_val_for_threshold: Có tìm threshold tối ưu trên validation set không
        val_dir: Thư mục validation (nếu use_val_for_threshold=True)
        use_face_detection: Detect + crop face trước khi train
        target_size: Kích thước crop (W, H)
        
    Returns:
        (model, label_map)
    """
    print("="*60)
    print("TRAIN LBPH MODEL FROM DIRECTORY")
    print("="*60)
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Thư mục train không tồn tại: {data_dir}")
    
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model parameters: radius={radius}, neighbors={neighbors}, grid=({grid_x}, {grid_y})")
    print(f"Preprocess: face_detection={'on' if use_face_detection else 'off'}, target_size={target_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/4] Đang load training data...")
    label_map = create_label_map_from_directory(data_dir)
    faces, labels = load_faces_and_labels(
        data_dir,
        label_map,
        use_face_detection=use_face_detection,
        target_size=target_size
    )
    
    if len(faces) == 0:
        raise ValueError(f"Không tìm thấy ảnh nào trong {data_dir}")
    
    print(f"  Loaded: {len(faces)} faces")
    print(f"  Labels range: {labels.min()} - {labels.max()}")
    print(f"  Unique labels: {len(np.unique(labels))}")
    
    print("\n[2/4] Kiểm tra label_map...")
    unique_labels = set(np.unique(labels))
    map_labels = set(label_map.keys())
    missing_labels = unique_labels - map_labels
    if missing_labels:
        print(f"[WARNING] Có {len(missing_labels)} labels trong data không có trong label_map")
        print(f"  Missing labels: {sorted(list(missing_labels))[:10]}")
    
    extra_labels = map_labels - unique_labels
    if extra_labels:
        print(f"[INFO] Có {len(extra_labels)} identities trong label_map không có trong training data")
    
    print("\n[3/4] Đang train LBPH model...")
    model = train_lbph_model(
        faces=faces,
        labels=labels,
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y
    )
    print("[OK] Training hoàn thành!")
    
    print("\n[4/4] Đang lưu model và label_map...")
    model_path = os.path.join(output_dir, model_name)
    model.save(model_path)
    print(f"  Model saved: {model_path}")
    
    label_map_path = os.path.join(output_dir, "label_map.npy")
    np.save(label_map_path, label_map)
    print(f"  Label map saved: {label_map_path}")
    
    if use_val_for_threshold and val_dir:
        print("\n[OPTIONAL] Đang tìm threshold tối ưu trên validation set...")
        try:
            from models.lbphmodel.threshold_lbph import find_optimal_threshold
            
            # Load val với cùng label_map (bỏ qua identity không có trong train)
            identity_to_label = {v: k for k, v in label_map.items()}
            val_faces: List[np.ndarray] = []
            val_labels: List[int] = []
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            detector = FaceDetector(
                backend='mtcnn',
                device='cpu',
                confidence_threshold=0.9,
                select_largest=True
            ) if use_face_detection else None
            
            for identity, label_id in identity_to_label.items():
                identity_path = os.path.join(val_dir, identity)
                if not os.path.isdir(identity_path):
                    continue
                img_files = [f for f in os.listdir(identity_path) 
                            if os.path.splitext(f.lower())[1] in image_extensions]
                for img_name in img_files:
                    img_path = os.path.join(identity_path, img_name)
                    img = _preprocess_image_for_lbph(
                        img_path,
                        detector=detector,
                        target_size=target_size,
                        grayscale=True
                    )
                    if img is None:
                        continue
                    val_faces.append(img)
                    val_labels.append(label_id)
            
            val_labels_np = np.array(val_labels, dtype=np.int32)
            print(f"  Validation samples: {len(val_faces)}")
            
            best_threshold, best_score, threshold_results = find_optimal_threshold(
                model, val_faces, val_labels_np
            )
            
            print(f"\n  Best threshold: {best_threshold}")
            print(f"  Best score (acc * coverage): {best_score:.4f}")
            
            threshold_path = os.path.join(output_dir, "optimal_threshold.txt")
            with open(threshold_path, 'w') as f:
                f.write(f"Optimal threshold: {best_threshold}\n")
                f.write(f"Score (accuracy * coverage): {best_score:.4f}\n\n")
                f.write("Threshold analysis:\n")
                f.write("Threshold | Accuracy | Coverage | Score\n")
                f.write("-" * 50 + "\n")
                for thr, acc, cov, score in threshold_results:
                    f.write(f"{thr:9d} | {acc:8.4f} | {cov:8.4f} | {score:.4f}\n")
            
            print(f"  Threshold analysis saved: {threshold_path}")
            
            # Cập nhật configs/lbph_config.yaml để web_app tự dùng best_threshold
            try:
                import yaml
                config_dir = os.path.join(ROOT_DIR, "configs")
                os.makedirs(config_dir, exist_ok=True)
                config_path = os.path.join(config_dir, "lbph_config.yaml")
                
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                else:
                    cfg = {}
                
                if "lbph" not in cfg:
                    cfg["lbph"] = {}
                cfg["lbph"]["default_threshold"] = int(best_threshold)
                
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
                
                print(f"[CONFIG] Updated LBPH default_threshold={best_threshold} in {config_path}")
            except Exception as e:
                print(f"[CONFIG WARNING] Không thể cập nhật lbph_config.yaml: {e}")
            
        except Exception as e:
            print(f"[WARNING] Không thể tìm threshold tối ưu: {e}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Label map: {label_map_path}")
    print(f"Total identities: {len(label_map)}")
    print(f"Total training samples: {len(faces)}")
    
    return model, label_map


def main():
    parser = argparse.ArgumentParser(
        description="Train LBPH model từ dataset thư mục",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Thư mục chứa các folder identity (train data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/checkpoints/LBHP',
        help='Thư mục lưu model và label_map'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='lbph_model.xml',
        help='Tên file model output'
    )
    
    parser.add_argument(
        '--radius',
        type=int,
        default=1,
        help='Bán kính LBP operator'
    )
    
    parser.add_argument(
        '--neighbors',
        type=int,
        default=8,
        help='Số neighbors LBP operator'
    )
    
    parser.add_argument(
        '--grid-x',
        type=int,
        default=8,
        help='Số cells theo chiều ngang'
    )
    
    parser.add_argument(
        '--grid-y',
        type=int,
        default=8,
        help='Số cells theo chiều dọc'
    )
    
    parser.add_argument(
        '--face-detection',
        action='store_true',
        default=True,
        help='Bật detect + crop face (default: bật)'
    )
    
    parser.add_argument(
        '--no-face-detection',
        action='store_true',
        help='Tắt detect + crop, dùng ảnh gốc (đã crop sẵn)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=[100, 100],
        metavar=('W', 'H'),
        help='Kích thước crop đầu vào cho LBPH'
    )
    
    parser.add_argument(
        '--find-threshold',
        action='store_true',
        help='Tìm threshold tối ưu trên validation set'
    )
    
    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help='Thư mục validation (cần thiết nếu --find-threshold)'
    )
    
    args = parser.parse_args()
    
    if args.find_threshold and not args.val_dir:
        print("[ERROR] Cần cung cấp --val-dir khi sử dụng --find-threshold")
        return 1
    
    # Resolve face detection flag
    use_fd = args.face_detection and not args.no_face_detection
    target_size = tuple(args.target_size)
    
    try:
        model, label_map = train_lbph_from_directory(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            radius=args.radius,
            neighbors=args.neighbors,
            grid_x=args.grid_x,
            grid_y=args.grid_y,
            use_val_for_threshold=args.find_threshold,
            val_dir=args.val_dir,
            use_face_detection=use_fd,
            target_size=target_size
        )
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Lỗi khi train: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

