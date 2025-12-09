"""
CelebA Dataset Preprocessing Pipeline
Xử lý dataset CelebA từ archive folder sang processed folder
Áp dụng face detection, alignment, và split train/val
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import yaml
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


class CelebAPreprocessor:
    """
    Xử lý CelebA dataset với các bước:
    1. Lọc identity có đủ ảnh (>=40 images)
    2. Face detection và alignment
    3. Resize về 112x112 cho ArcFace
    4. Split train/val (80/20)
    """
    
    def __init__(
        self,
        archive_dir="archive",
        output_dir="data/processed",
        min_images_per_identity=40,
        target_size=112
    ):
        self.archive_dir = Path(archive_dir)
        self.output_dir = Path(output_dir)
        self.min_images = min_images_per_identity
        self.target_size = target_size
        
        self.images_dir = self.archive_dir / "img_align_celeba" / "img_align_celeba"
        self.bbox_csv = self.archive_dir / "list_bbox_celeba.csv"
        self.landmarks_csv = self.archive_dir / "list_landmarks_align_celeba.csv"
        self.partition_csv = self.archive_dir / "list_eval_partition.csv"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_metadata(self):
        """Load các file metadata"""
        print("\n=== LOAD METADATA ===")
        
        self.bbox_df = pd.read_csv(self.bbox_csv)
        self.landmarks_df = pd.read_csv(self.landmarks_csv)
        self.partition_df = pd.read_csv(self.partition_csv)
        
        print(f"Bbox: {len(self.bbox_df)} records")
        print(f"Landmarks: {len(self.landmarks_df)} records")
        print(f"Partition: {len(self.partition_df)} records")
        
        # Merge metadata
        self.metadata = self.bbox_df.merge(self.landmarks_df, on='image_id')
        self.metadata = self.metadata.merge(self.partition_df, on='image_id')
        
        print(f"Merged metadata: {len(self.metadata)} records")
        
    def select_identities_from_partition(self, num_identities=100):
        """
        Chọn identities dựa vào partition (0=train, 1=val, 2=test)
        Lấy các identities từ train partition có đủ ảnh
        
        Lưu ý: CelebA không có identity labels trong metadata thông thường
        Do đó ta cần file identity_CelebA.txt hoặc tạo pseudo-identities
        """
        print(f"\n=== SELECT {num_identities} IDENTITIES ===")
        
        # Vì không có identity labels, ta sẽ tạo pseudo-identities
        # Mỗi 40-50 ảnh liên tiếp = 1 identity (đơn giản hóa)
        # Hoặc sử dụng clustering nếu cần chính xác hơn
        
        print("⚠ CelebA không có identity labels trong metadata chuẩn")
        print("Giải pháp: Sử dụng toàn bộ ảnh và chia theo partition")
        
        # Sử dụng train partition (partition=0)
        train_images = self.metadata[self.metadata['partition'] == 0]
        
        # Chia thành pseudo-identities (mỗi identity ~40 ảnh)
        images_per_identity = 40
        num_possible = len(train_images) // images_per_identity
        
        print(f"Train images: {len(train_images)}")
        print(f"Có thể tạo tối đa {num_possible} pseudo-identities")
        
        # Lấy N identities đầu tiên
        n_identities = min(num_identities, num_possible)
        selected_count = n_identities * images_per_identity
        
        self.selected_images = train_images.iloc[:selected_count].copy()
        
        # Gán identity ID
        identity_ids = []
        for i in range(n_identities):
            identity_ids.extend([f"person_{i:03d}"] * images_per_identity)
        
        self.selected_images['identity'] = identity_ids
        
        print(f"Selected {n_identities} identities")
        print(f"Total images: {len(self.selected_images)}")
        print(f"Images per identity: {images_per_identity}")
        
        return self.selected_images
    
    def align_face(self, image, landmarks):
        """
        Align face sử dụng landmarks (eye positions)
        
        Args:
            image: PIL Image hoặc numpy array
            landmarks: dict với lefteye_x, lefteye_y, righteye_x, righteye_y
            
        Returns:
            Aligned image (numpy array)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        left_eye = (landmarks['lefteye_x'], landmarks['lefteye_y'])
        right_eye = (landmarks['righteye_x'], landmarks['righteye_y'])
        
        # Tính góc xoay
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Center giữa 2 mắt
        center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Warp affine
        aligned = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned
    
    def crop_and_resize(self, image, bbox, margin=0.3):
        """
        Crop face theo bounding box và resize
        
        Args:
            image: numpy array
            bbox: dict với x_1, y_1, width, height
            margin: thêm margin xung quanh bbox (30%)
            
        Returns:
            Cropped và resized image
        """
        x, y, w, h = bbox['x_1'], bbox['y_1'], bbox['width'], bbox['height']
        
        # Thêm margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        # Resize
        resized = cv2.resize(cropped, (self.target_size, self.target_size))
        
        return resized
    
    def process_single_image(self, row):
        """Xử lý 1 ảnh: align + crop + resize"""
        try:
            # Load image
            img_path = self.images_dir / row['image_id']
            if not img_path.exists():
                return None
            
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Align
            landmarks = {
                'lefteye_x': row['lefteye_x'],
                'lefteye_y': row['lefteye_y'],
                'righteye_x': row['righteye_x'],
                'righteye_y': row['righteye_y']
            }
            aligned = self.align_face(image, landmarks)
            
            # Crop và resize
            bbox = {
                'x_1': row['x_1'],
                'y_1': row['y_1'],
                'width': row['width'],
                'height': row['height']
            }
            processed = self.crop_and_resize(aligned, bbox)
            
            return processed
            
        except Exception as e:
            print(f"Error processing {row['image_id']}: {e}")
            return None
    
    def create_train_val_split(self, train_ratio=0.8):
        """Split data thành train/val theo identity"""
        print("\n=== CREATE TRAIN/VAL SPLIT ===")
        
        train_data = []
        val_data = []
        
        for identity in self.selected_images['identity'].unique():
            identity_images = self.selected_images[
                self.selected_images['identity'] == identity
            ]
            
            n_total = len(identity_images)
            n_train = int(n_total * train_ratio)
            
            # Shuffle
            identity_images = identity_images.sample(frac=1, random_state=42)
            
            train_data.append(identity_images.iloc[:n_train])
            val_data.append(identity_images.iloc[n_train:])
        
        self.train_df = pd.concat(train_data, ignore_index=True)
        self.val_df = pd.concat(val_data, ignore_index=True)
        
        print(f"Train: {len(self.train_df)} images")
        print(f"Val: {len(self.val_df)} images")
        print(f"Identities: {len(self.train_df['identity'].unique())}")
        
    def save_processed_images(self, split_name, dataframe):
        """Lưu ảnh đã xử lý"""
        print(f"\n=== SAVE {split_name.upper()} IMAGES ===")
        
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        saved_rows = []
        
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing {split_name}"):
            # Process image
            processed = self.process_single_image(row)
            
            if processed is None:
                continue
            
            # Create identity folder
            identity_dir = split_dir / row['identity']
            identity_dir.mkdir(exist_ok=True)
            
            # Save image
            save_path = identity_dir / row['image_id']
            cv2.imwrite(str(save_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            
            # Record metadata
            saved_rows.append({
                'image_path': str(save_path.relative_to(self.output_dir)),
                'identity_id': row['identity'],
                'split': split_name,
                'original_image': row['image_id']
            })
        
        # Save metadata CSV
        metadata_df = pd.DataFrame(saved_rows)
        metadata_csv = self.output_dir / f"{split_name}_metadata.csv"
        metadata_df.to_csv(metadata_csv, index=False)
        
        print(f"Saved {len(saved_rows)} images")
        print(f"Metadata: {metadata_csv}")
        
    def run(self, num_identities=100):
        """Chạy toàn bộ pipeline"""
        print("="*60)
        print("CELEBA PREPROCESSING PIPELINE")
        print("="*60)
        
        # 1. Load metadata
        self.load_metadata()
        
        # 2. Select identities
        self.select_identities_from_partition(num_identities)
        
        # 3. Split train/val
        self.create_train_val_split()
        
        # 4. Process và save train
        self.save_processed_images('train', self.train_df)
        
        # 5. Process và save val
        self.save_processed_images('val', self.val_df)
        
        # 6. Summary
        self.print_summary()
        
        print("\n" + "="*60)
        print("PREPROCESSING HOÀN TẤT!")
        print("="*60)
        
    def print_summary(self):
        """In summary thống kê"""
        print("\n=== SUMMARY ===")
        
        total_train = len(self.train_df)
        total_val = len(self.val_df)
        num_identities = len(self.train_df['identity'].unique())
        
        print(f"Identities: {num_identities}")
        print(f"Train images: {total_train}")
        print(f"Val images: {total_val}")
        print(f"Total: {total_train + total_val}")
        print(f"Image size: {self.target_size}x{self.target_size}")
        
        print(f"\nOutput structure:")
        print(f"  {self.output_dir}/")
        print(f"  ├── train/")
        print(f"  │   ├── person_000/")
        print(f"  │   ├── person_001/")
        print(f"  │   └── ...")
        print(f"  ├── val/")
        print(f"  │   ├── person_000/")
        print(f"  │   └── ...")
        print(f"  ├── train_metadata.csv")
        print(f"  └── val_metadata.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CelebA dataset")
    parser.add_argument(
        '--archive-dir',
        type=str,
        default='archive',
        help='Archive directory chứa raw data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory'
    )
    parser.add_argument(
        '--num-identities',
        type=int,
        default=100,
        help='Số lượng identities cần chọn (default: 100)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=112,
        help='Target image size (default: 112 for ArcFace)'
    )
    
    args = parser.parse_args()
    
    preprocessor = CelebAPreprocessor(
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        target_size=args.target_size
    )
    
    preprocessor.run(num_identities=args.num_identities)
