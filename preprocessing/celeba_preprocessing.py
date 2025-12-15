"""
CelebA Dataset Preprocessing Pipeline
Su dung cac file metadata goc:
- identity_CelebA.txt: Mapping anh -> identity (QUAN TRONG)
- list_landmarks_align_celeba.csv: 5 diem landmark
- list_attr_celeba.csv: 40 thuoc tinh khuon mat
- list_bbox_celeba.csv: Bounding box

Modules:
- FaceDetector: Detect face tu anh moi (MTCNN/RetinaFace/OpenCV)
- CelebAPreprocessor: Pipeline xu ly CelebA dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import json
import random
import shutil
from collections import defaultdict

try:
    from skimage.transform import SimilarityTransform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# ArcFace template 112x112
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],   # left_eye
    [73.5318, 51.5014],   # right_eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left_mouth
    [70.7299, 92.2041],   # right_mouth
], dtype=np.float32)


class CelebAPreprocessor:
    """
    Xu ly CelebA dataset voi cac buoc:
    1. Load metadata goc (identity, landmarks, attributes)
    2. Loc identity co it anh (< min_images)
    3. Augmentation cho identity it anh
    4. Chia train/val/test (theo identity hoac theo anh)
    5. Alignment theo ArcFace template (112x112)
    6. Tao metadata moi cho training
    """
    
    def __init__(
        self,
        images_dir="data/img_align_celeba",
        meta_dir="data/meta_origin",
        output_dir="data/CelebA_Aligned_Balanced",
        min_images_per_identity=5,
        augment_threshold=10,
        target_min_images=10,
        target_size=112,
        split_method='by_image',  # 'by_image' hoac 'by_identity'
        seed=42
    ):
        self.images_dir = Path(images_dir)
        self.meta_dir = Path(meta_dir)
        self.output_dir = Path(output_dir)
        self.min_images = min_images_per_identity
        self.augment_threshold = augment_threshold
        self.target_min_images = target_min_images
        self.target_size = target_size
        self.split_method = split_method
        self.seed = seed
        
        # Metadata files
        self.identity_file = self.meta_dir / "identity_CelebA.txt"
        self.landmarks_file = self.meta_dir / "list_landmarks_align_celeba.csv"
        self.attributes_file = self.meta_dir / "list_attr_celeba.csv"
        self.bbox_file = self.meta_dir / "list_bbox_celeba.csv"
        
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Temp directories
        self.temp_by_id = Path("data/temp_celeba_by_id")
        
    def load_metadata(self):
        """Load tat ca metadata goc"""
        print("\n" + "="*60)
        print("LOAD METADATA GOC")
        print("="*60)
        
        # Load identity (QUAN TRONG NHAT)
        if not self.identity_file.exists():
            raise FileNotFoundError(f"Khong tim thay {self.identity_file}")
        
        self.identity_df = pd.read_csv(
            self.identity_file, sep=" ", header=None, 
            names=["image", "identity_id"]
        )
        print(f"Identity: {len(self.identity_df):,} anh, {self.identity_df['identity_id'].nunique():,} nguoi")
        
        # Load landmarks
        if self.landmarks_file.exists():
            self.landmarks_df = pd.read_csv(self.landmarks_file)
            self.landmarks = {}
            for _, row in self.landmarks_df.iterrows():
                self.landmarks[row['image_id']] = {
                    "left_eye": (row['lefteye_x'], row['lefteye_y']),
                    "right_eye": (row['righteye_x'], row['righteye_y']),
                    "nose": (row['nose_x'], row['nose_y']),
                    "left_mouth": (row['leftmouth_x'], row['leftmouth_y']),
                    "right_mouth": (row['rightmouth_x'], row['rightmouth_y']),
                }
            print(f"Landmarks: {len(self.landmarks):,} records")
        else:
            self.landmarks = {}
            print("WARNING: Landmarks file not found")
        
        # Load attributes (optional)
        if self.attributes_file.exists():
            self.attributes_df = pd.read_csv(self.attributes_file)
            print(f"Attributes: {len(self.attributes_df):,} records, {len(self.attributes_df.columns)-1} features")
        else:
            self.attributes_df = None
            print("WARNING: Attributes file not found")
        
        # Load bbox (optional)
        if self.bbox_file.exists():
            self.bbox_df = pd.read_csv(self.bbox_file)
            print(f"Bbox: {len(self.bbox_df):,} records")
        else:
            self.bbox_df = None
            print("WARNING: Bbox file not found")
    
    def analyze_and_filter(self):
        """Phan tich va loc identity"""
        print("\n" + "="*60)
        print("PHAN TICH VA LOC IDENTITY")
        print("="*60)
        
        # Thong ke so anh moi identity
        self.identity_counts = self.identity_df['identity_id'].value_counts()
        
        # Phan loai
        self.ids_to_remove = self.identity_counts[
            self.identity_counts < self.min_images
        ].index.tolist()
        
        self.ids_to_augment = self.identity_counts[
            (self.identity_counts >= self.min_images) & 
            (self.identity_counts < self.augment_threshold)
        ].index.tolist()
        
        self.ids_normal = self.identity_counts[
            self.identity_counts >= self.augment_threshold
        ].index.tolist()
        
        print(f"LOAI BO (< {self.min_images} anh): {len(self.ids_to_remove):,} identity")
        print(f"CAN AUGMENT ({self.min_images}-{self.augment_threshold-1} anh): {len(self.ids_to_augment):,} identity")
        print(f"BINH THUONG (>= {self.augment_threshold} anh): {len(self.ids_normal):,} identity")
        
        # Loc
        valid_ids = set(self.ids_to_augment + self.ids_normal)
        self.filtered_df = self.identity_df[
            self.identity_df['identity_id'].isin(valid_ids)
        ].copy()
        
        removed_count = self.identity_counts[self.ids_to_remove].sum()
        print(f"\nSo anh bi loai: {removed_count:,} ({100*removed_count/len(self.identity_df):.1f}%)")
        print(f"Con lai: {len(self.filtered_df):,} anh, {len(valid_ids):,} identity")
    
    def organize_by_identity(self):
        """Gom anh theo identity"""
        print("\n" + "="*60)
        print("GOM ANH THEO IDENTITY")
        print("="*60)
        
        # Xoa temp folder cu
        if self.temp_by_id.exists():
            shutil.rmtree(self.temp_by_id)
        self.temp_by_id.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        missing = 0
        
        for _, row in tqdm(self.filtered_df.iterrows(), total=len(self.filtered_df), desc="Gom anh"):
            img_file = row['image']
            pid = str(row['identity_id'])
            
            src_path = self.images_dir / img_file
            dst_dir = self.temp_by_id / pid
            dst_dir.mkdir(exist_ok=True)
            dst_path = dst_dir / img_file
            
            if src_path.exists():
                shutil.copy(src_path, dst_path)
                copied += 1
            else:
                missing += 1
        
        print(f"Copied: {copied:,}, Missing: {missing}")
    
    def align_all_faces_first(self):
        """
        BUOC 1: Alignment TRUOC augmentation
        Ap dung landmarks goc cho tat ca anh goc
        """
        print("\n" + "="*60)
        print("ALIGNMENT TRUOC (su dung landmarks goc)")
        print("="*60)
        
        # Tao thu muc aligned tam
        self.temp_aligned = Path("data/temp_celeba_aligned")
        if self.temp_aligned.exists():
            shutil.rmtree(self.temp_aligned)
        self.temp_aligned.mkdir(parents=True, exist_ok=True)
        
        align_stats = {'aligned': 0, 'center_crop': 0, 'failed': 0}
        
        all_ids = [d.name for d in self.temp_by_id.iterdir() if d.is_dir()]
        
        for pid in tqdm(all_ids, desc="Aligning"):
            src_dir = self.temp_by_id / pid
            dst_dir = self.temp_aligned / pid
            dst_dir.mkdir(exist_ok=True)
            
            for img_path in src_dir.iterdir():
                if img_path.suffix.lower() != '.jpg':
                    continue
                
                img = cv2.imread(str(img_path))
                if img is None:
                    align_stats['failed'] += 1
                    continue
                
                # Dung landmarks goc
                if img_path.name in self.landmarks:
                    aligned = self.align_face(img, self.landmarks[img_path.name])
                    align_stats['aligned'] += 1
                else:
                    aligned = self.align_face_center_crop(img)
                    align_stats['center_crop'] += 1
                
                cv2.imwrite(str(dst_dir / img_path.name), aligned)
        
        print(f"Aligned with landmarks: {align_stats['aligned']:,}")
        print(f"Center crop: {align_stats['center_crop']:,}")
        print(f"Failed: {align_stats['failed']}")
    
    def augment_aligned_images(self):
        """
        BUOC 2: Augmentation SAU alignment
        Ap dung tren anh da align (112x112) -> landmarks da chuan hoa
        """
        print("\n" + "="*60)
        print("AUGMENTATION SAU (tren anh da align 112x112)")
        print("="*60)
        
        try:
            import albumentations as A
            # Augmentation phu hop cho anh da align 112x112
            augment_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REPLICATE),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.8),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.2),
            ])
        except ImportError:
            print("WARNING: albumentations not installed, using simple augmentation")
            augment_transform = None
        
        total_augmented = 0
        
        for pid in tqdm(self.ids_to_augment, desc="Augmenting aligned"):
            person_dir = self.temp_aligned / str(pid)
            if not person_dir.exists():
                continue
            
            images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            current_count = len(images)
            
            if current_count >= self.target_min_images:
                continue
            
            needed = self.target_min_images - current_count
            augmented = 0
            
            while augmented < needed:
                src_img_name = random.choice(images)
                src_path = person_dir / src_img_name
                
                img = cv2.imread(str(src_path))
                if img is None:
                    continue
                
                if augment_transform:
                    aug_img = augment_transform(image=img)['image']
                else:
                    aug_img = img.copy()
                    if random.random() > 0.5:
                        aug_img = cv2.flip(aug_img, 1)
                
                base_name = os.path.splitext(src_img_name)[0]
                new_name = f"{base_name}_aug{augmented+1}.jpg"
                cv2.imwrite(str(person_dir / new_name), aug_img)
                augmented += 1
            
            total_augmented += augmented
        
        print(f"Tong anh augmented: {total_augmented:,}")
    
    def split_dataset(self):
        """Chia train/val/test tu anh da align + augment"""
        print("\n" + "="*60)
        print(f"CHIA DATASET ({self.split_method.upper()})")
        print("="*60)
        
        # Xoa output folder cu
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Doc tu temp_aligned (da align + augment)
        all_ids = sorted([d.name for d in self.temp_aligned.iterdir() if d.is_dir()])
        print(f"Tong so identity: {len(all_ids)}")
        
        stats = {'train': 0, 'val': 0, 'test': 0}
        identity_stats = {'train': set(), 'val': set(), 'test': set()}
        
        if self.split_method == 'by_image':
            # Chia anh trong moi identity vao ca 3 tap
            for pid in tqdm(all_ids, desc="Chia theo anh"):
                src_dir = self.temp_aligned / pid
                images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
                
                if len(images) < 3:
                    # Qua it anh -> cho vao train
                    dst_dir = self.output_dir / "train" / pid
                    dst_dir.mkdir(exist_ok=True)
                    for img in images:
                        shutil.copy(src_dir / img, dst_dir / img)
                    stats['train'] += len(images)
                    identity_stats['train'].add(pid)
                    continue
                
                random.shuffle(images)
                n = len(images)
                n_val = max(1, int(0.1 * n))
                n_test = max(1, int(0.1 * n))
                n_train = n - n_val - n_test
                
                if n_train < 1:
                    n_train = 1
                    remaining = n - n_train
                    n_val = remaining // 2
                    n_test = remaining - n_val
                
                splits_imgs = {
                    'train': images[:n_train],
                    'val': images[n_train:n_train + n_val],
                    'test': images[n_train + n_val:]
                }
                
                for split, split_imgs in splits_imgs.items():
                    if len(split_imgs) == 0:
                        continue
                    dst_dir = self.output_dir / split / pid
                    dst_dir.mkdir(exist_ok=True)
                    for img in split_imgs:
                        shutil.copy(src_dir / img, dst_dir / img)
                    stats[split] += len(split_imgs)
                    identity_stats[split].add(pid)
        
        else:  # by_identity
            random.shuffle(all_ids)
            n_total = len(all_ids)
            n_val = int(0.1 * n_total)
            n_test = int(0.1 * n_total)
            
            split_ids = {
                'train': all_ids[:n_total - n_val - n_test],
                'val': all_ids[n_total - n_val - n_test:n_total - n_test],
                'test': all_ids[n_total - n_test:]
            }
            
            for split, ids_list in split_ids.items():
                for pid in tqdm(ids_list, desc=f"Copying {split}"):
                    src_dir = self.temp_aligned / pid
                    dst_dir = self.output_dir / split / pid
                    if src_dir.exists():
                        shutil.copytree(src_dir, dst_dir)
                        stats[split] += len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')])
                        identity_stats[split].add(pid)
        
        # In ket qua (bao ve chia 0)
        total = sum(stats.values())
        if total == 0:
            print("  [WARNING] Khong co anh nao duoc copy vao cac split.")
            for split in ['train', 'val', 'test']:
                print(f"  {split:5}: 0 anh (0.0%), 0 identities")
        else:
            for split in ['train', 'val', 'test']:
                n_imgs = stats[split]
                n_ids = len(identity_stats[split])
                pct = 100 * n_imgs / total if total > 0 else 0.0
                print(f"  {split:5}: {n_imgs:,} anh ({pct:.1f}%), {n_ids} identities")
        
        # Kiem tra overlap
        if self.split_method == 'by_image':
            overlap_all = identity_stats['train'] & identity_stats['val'] & identity_stats['test']
            print(f"\nIdentity co anh trong CA 3 tap: {len(overlap_all)}")
        else:
            overlap = identity_stats['train'] & identity_stats['val']
            print(f"\nTrain & Val overlap: {len(overlap)}")
        
        self.identity_stats = identity_stats
    
    def align_face(self, img, landmark):
        """Align face theo ArcFace template"""
        if not HAS_SKIMAGE:
            return self.align_face_center_crop(img)
        
        src = np.array([
            landmark["left_eye"],
            landmark["right_eye"],
            landmark["nose"],
            landmark["left_mouth"],
            landmark["right_mouth"]
        ], dtype=np.float32)
        
        tform = SimilarityTransform()
        tform.estimate(src, ARCFACE_TEMPLATE)
        M = tform.params[0:2, :]
        return cv2.warpAffine(img, M, (self.target_size, self.target_size), borderValue=0)
    
    def align_face_center_crop(self, img):
        """Fallback: center crop + resize"""
        h, w = img.shape[:2]
        if h > w:
            start = (h - w) // 2
            img = img[start:start+w, :]
        elif w > h:
            start = (w - h) // 2
            img = img[:, start:start+h]
        return cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
    
    
    def create_metadata(self):
        """Tao metadata cho training"""
        print("\n" + "="*60)
        print("TAO METADATA")
        print("="*60)
        
        meta_dir = self.output_dir / "metadata"
        meta_dir.mkdir(exist_ok=True)
        
        # Thu thap TAT CA identity tu ca 3 split (quan trong cho by_image)
        all_identities = set()
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            for d in split_dir.iterdir():
                if d.is_dir():
                    all_identities.add(d.name)
        
        # Tao GLOBAL label mapping cho TAT CA identity
        all_identities_sorted = sorted(all_identities)
        global_id_to_label = {pid: idx for idx, pid in enumerate(all_identities_sorted)}
        
        print(f"Total identities (all splits): {len(global_id_to_label)}")
        
        # Luu global mapping
        pd.DataFrame([
            {"identity_id": pid, "label": label}
            for pid, label in global_id_to_label.items()
        ]).to_csv(meta_dir / "global_id_mapping.csv", index=False)
        
        # Tao labels file cho moi split
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            records = []
            
            for person_dir in split_dir.iterdir():
                if not person_dir.is_dir():
                    continue
                
                pid = person_dir.name
                label = global_id_to_label.get(pid, -1)
                
                for img_path in person_dir.iterdir():
                    if img_path.suffix.lower() == '.jpg':
                        records.append({
                            "image": f"{pid}/{img_path.name}",
                            "identity_id": pid,
                            "label": label,
                            "is_augmented": 1 if '_aug' in img_path.name else 0
                        })
            
            # Neu khong co anh, van luu file rong voi headers de tranh loi read_csv
            df = pd.DataFrame(records, columns=["image", "identity_id", "label", "is_augmented"])
            df.to_csv(meta_dir / f"{split}_labels.csv", index=False)
            
            n_ids = df['identity_id'].nunique()
            n_aug = df['is_augmented'].sum()
            print(f"{split}: {len(df):,} images, {n_ids} identities, {n_aug:,} augmented")
        
        # Tao dataset config
        config = {
            "dataset_name": "CelebA_Aligned_Balanced",
            "preprocessing": {
                "min_images_per_identity": self.min_images,
                "augment_threshold": self.augment_threshold,
                "target_min_images": self.target_min_images,
                "split_method": self.split_method
            },
            "image_size": [self.target_size, self.target_size],
            "arcface_landmarks": {
                "left_eye": ARCFACE_TEMPLATE[0].tolist(),
                "right_eye": ARCFACE_TEMPLATE[1].tolist(),
                "nose": ARCFACE_TEMPLATE[2].tolist(),
                "left_mouth": ARCFACE_TEMPLATE[3].tolist(),
                "right_mouth": ARCFACE_TEMPLATE[4].tolist()
            },
            "splits": {}
        }
        
        for split in ["train", "val", "test"]:
            csv_path = meta_dir / f"{split}_labels.csv"
            if not csv_path.exists():
                config["splits"][split] = {
                    "num_identities": 0,
                    "num_images": 0,
                    "num_augmented": 0
                }
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                config["splits"][split] = {
                    "num_identities": 0,
                    "num_images": 0,
                    "num_augmented": 0
                }
            else:
                config["splits"][split] = {
                    "num_identities": int(df['identity_id'].nunique()),
                    "num_images": int(len(df)),
                    "num_augmented": int(df['is_augmented'].sum())
                }
        
        with open(meta_dir / "dataset_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"\nConfig saved to {meta_dir / 'dataset_config.json'}")
    
    def cleanup(self):
        """Xoa temp folders"""
        if self.temp_by_id.exists():
            shutil.rmtree(self.temp_by_id)
        if hasattr(self, 'temp_aligned') and self.temp_aligned.exists():
            shutil.rmtree(self.temp_aligned)
        print("Cleaned up temp folders")
    
    def run(self):
        """
        Chay toan bo pipeline
        
        THU TU DUNG:
        1. Load metadata
        2. Loc identity it anh
        3. Gom anh theo identity
        4. ALIGNMENT TRUOC (dung landmarks goc)
        5. AUGMENTATION SAU (tren anh da align 112x112)
        6. Chia dataset
        7. Tao metadata
        """
        print("="*60)
        print("CELEBA PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Images dir: {self.images_dir}")
        print(f"Meta dir: {self.meta_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Split method: {self.split_method}")
        
        self.load_metadata()
        self.analyze_and_filter()
        self.organize_by_identity()
        
        # ALIGNMENT TRUOC - su dung landmarks goc
        self.align_all_faces_first()
        
        # AUGMENTATION SAU - tren anh da align
        self.augment_aligned_images()
        
        self.split_dataset()
        self.create_metadata()
        self.cleanup()
        
        print("\n" + "="*60)
        print("HOAN THANH!")
        print("="*60)
        print(f"Output: {self.output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CelebA dataset")
    parser.add_argument('--images-dir', type=str, default='data/img_align_celeba',
                        help='Directory chua anh goc')
    parser.add_argument('--meta-dir', type=str, default='data/meta_origin',
                        help='Directory chua metadata goc')
    parser.add_argument('--output-dir', type=str, default='data/CelebA_Aligned_Balanced',
                        help='Output directory')
    parser.add_argument('--min-images', type=int, default=5,
                        help='Loai bo identity co it hon X anh')
    parser.add_argument('--augment-threshold', type=int, default=10,
                        help='Augment identity co it hon X anh')
    parser.add_argument('--target-min', type=int, default=10,
                        help='Muc tieu toi thieu sau augment')
    parser.add_argument('--split-method', type=str, default='by_image',
                        choices=['by_image', 'by_identity'],
                        help='Phuong phap chia: by_image (moi identity o ca 3 tap) hoac by_identity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    preprocessor = CelebAPreprocessor(
        images_dir=args.images_dir,
        meta_dir=args.meta_dir,
        output_dir=args.output_dir,
        min_images_per_identity=args.min_images,
        augment_threshold=args.augment_threshold,
        target_min_images=args.target_min,
        split_method=args.split_method,
        seed=args.seed
    )
    
    preprocessor.run()
