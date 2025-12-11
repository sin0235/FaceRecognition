"""
CelebA Dataset Preprocessing - Xu ly mat can bang
Chay tren Google Colab

Quy trinh:
1. Loc bo identity co 1-4 anh
2. Augmentation offline cho identity co 5-9 anh (tang len it nhat 10 anh)
3. Chia dataset theo IDENTITY (khong phai theo anh)
4. Alignment va tao metadata
"""

import os
import json
import random
import shutil
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("WARNING: albumentations not installed, augmentation will be limited")

try:
    from skimage.transform import SimilarityTransform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: skimage not installed, alignment will use center crop")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths (thay doi khi chay tren Colab)
    CELEBA_DIR = "/content/celeba"
    IMG_DIR = "/content/celeba/img_align_celeba/img_align_celeba"
    IDENTITY_FILE = "/content/celeba/identity_CelebA.txt"
    LANDMARK_FILE = "/content/celeba/list_landmarks_align_celeba.csv"
    
    # Output
    DRIVE_OUTPUT = "/content/drive/MyDrive/FaceRecognition"
    TEMP_BY_ID = "/content/celeba_by_id"
    TEMP_SPLIT = "/content/celeba_split"
    FINAL_OUTPUT = None  # Set trong main()
    
    # Filtering
    MIN_IMAGES = 5  # Loai bo identity < 5 anh
    AUGMENT_THRESHOLD = 10  # Augment identity co 5-9 anh
    TARGET_MIN_IMAGES = 10  # Muc tieu toi thieu sau augment
    
    # Split ratio
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Random seed
    SEED = 42

# ArcFace template 112x112
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# =============================================================================
# AUGMENTATION
# =============================================================================

def get_augmentation_transform():
    """Augmentation pipeline cho offline augmentation"""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REPLICATE),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.05,
                p=0.8
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
        ])
    return None

def augment_image_simple(img):
    """Augmentation don gian khi khong co albumentations"""
    augmented = img.copy()
    
    # Random horizontal flip
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    # Random brightness/contrast
    alpha = 1.0 + random.uniform(-0.2, 0.2)  # contrast
    beta = random.randint(-20, 20)  # brightness
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    
    return augmented

def augment_identity_images(person_dir, target_count, transform=None):
    """Augment anh de dat duoc target_count anh"""
    images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    current_count = len(images)
    
    if current_count >= target_count:
        return 0
    
    needed = target_count - current_count
    augmented = 0
    
    while augmented < needed:
        src_img_name = random.choice(images)
        src_path = os.path.join(person_dir, src_img_name)
        
        img = cv2.imread(src_path)
        if img is None:
            continue
        
        # Apply augmentation
        if transform is not None:
            augmented_img = transform(image=img)['image']
        else:
            augmented_img = augment_image_simple(img)
        
        # Tao ten file moi
        base_name = os.path.splitext(src_img_name)[0]
        new_name = f"{base_name}_aug{augmented+1}.jpg"
        new_path = os.path.join(person_dir, new_name)
        
        cv2.imwrite(new_path, augmented_img)
        augmented += 1
    
    return augmented

# =============================================================================
# ALIGNMENT
# =============================================================================

def load_landmarks(landmark_file):
    """Load landmarks tu CSV"""
    df = pd.read_csv(landmark_file)
    landmarks = {}
    for _, row in df.iterrows():
        img = row['image_id']
        landmarks[img] = {
            "left_eye": (row['lefteye_x'], row['lefteye_y']),
            "right_eye": (row['righteye_x'], row['righteye_y']),
            "nose": (row['nose_x'], row['nose_y']),
            "left_mouth": (row['leftmouth_x'], row['leftmouth_y']),
            "right_mouth": (row['rightmouth_x'], row['rightmouth_y']),
        }
    return landmarks

def align_face(img, landmark):
    """Align face theo ArcFace template"""
    if not HAS_SKIMAGE:
        return align_face_center_crop(img)
    
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
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)
    
    return aligned

def align_face_center_crop(img):
    """Align bang center crop va resize (cho anh khong co landmark)"""
    h, w = img.shape[:2]
    # Crop center square
    if h > w:
        start = (h - w) // 2
        img = img[start:start+w, :]
    elif w > h:
        start = (w - h) // 2
        img = img[:, start:start+h]
    
    # Resize to 112x112
    aligned = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    return aligned

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def step1_analyze_and_filter(config):
    """Buoc 1: Phan tich va loc identity"""
    print("\n" + "="*60)
    print("BUOC 1: PHAN TICH VA LOC IDENTITY")
    print("="*60)
    
    # Load identity file
    identity_df = pd.read_csv(config.IDENTITY_FILE, sep=" ", header=None, 
                              names=["image", "identity_id"])
    print(f"Tong so anh: {len(identity_df):,}")
    print(f"Tong so identity: {identity_df['identity_id'].nunique():,}")
    
    # Thong ke so anh moi identity
    identity_counts = identity_df['identity_id'].value_counts()
    
    # Phan loai identity
    ids_to_remove = identity_counts[identity_counts < config.MIN_IMAGES].index.tolist()
    ids_to_augment = identity_counts[
        (identity_counts >= config.MIN_IMAGES) & 
        (identity_counts < config.AUGMENT_THRESHOLD)
    ].index.tolist()
    ids_normal = identity_counts[identity_counts >= config.AUGMENT_THRESHOLD].index.tolist()
    
    print(f"\nPhan loai identity:")
    print(f"  LOAI BO (< {config.MIN_IMAGES} anh): {len(ids_to_remove):,}")
    print(f"  CAN AUGMENT ({config.MIN_IMAGES}-{config.AUGMENT_THRESHOLD-1} anh): {len(ids_to_augment):,}")
    print(f"  BINH THUONG (>= {config.AUGMENT_THRESHOLD} anh): {len(ids_normal):,}")
    
    # So anh bi loai
    removed_images = identity_counts[ids_to_remove].sum()
    print(f"\nSo anh se bi loai: {removed_images:,} ({100*removed_images/len(identity_df):.1f}%)")
    
    # Loc dataset
    valid_ids = set(ids_to_augment + ids_normal)
    filtered_df = identity_df[identity_df['identity_id'].isin(valid_ids)].copy()
    
    print(f"\nSau khi loc:")
    print(f"  - So anh: {len(filtered_df):,}")
    print(f"  - So identity: {filtered_df['identity_id'].nunique():,}")
    
    return filtered_df, ids_to_augment, ids_normal

def step2_organize_by_identity(config, filtered_df):
    """Buoc 2: Gom anh theo identity"""
    print("\n" + "="*60)
    print("BUOC 2: GOM ANH THEO IDENTITY")
    print("="*60)
    
    os.makedirs(config.TEMP_BY_ID, exist_ok=True)
    
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Gom anh"):
        img_file = row['image']
        pid = str(row['identity_id'])
        
        dst_dir = f"{config.TEMP_BY_ID}/{pid}"
        os.makedirs(dst_dir, exist_ok=True)
        
        src_path = f"{config.IMG_DIR}/{img_file}"
        dst_path = f"{dst_dir}/{img_file}"
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

def step3_augment_small_identities(config, ids_to_augment):
    """Buoc 3: Augment identity co it anh"""
    print("\n" + "="*60)
    print("BUOC 3: AUGMENTATION CHO IDENTITY IT ANH")
    print("="*60)
    
    transform = get_augmentation_transform()
    total_augmented = 0
    
    for pid in tqdm(ids_to_augment, desc="Augmenting"):
        person_dir = os.path.join(config.TEMP_BY_ID, str(pid))
        if not os.path.exists(person_dir):
            continue
        
        added = augment_identity_images(person_dir, config.TARGET_MIN_IMAGES, transform)
        total_augmented += added
    
    print(f"\nTong so anh da augment: {total_augmented:,}")
    return total_augmented

def step4_split_by_identity(config):
    """Buoc 4: Chia train/val/test theo IDENTITY"""
    print("\n" + "="*60)
    print("BUOC 4: CHIA DATASET THEO IDENTITY")
    print("="*60)
    
    random.seed(config.SEED)
    
    # Lay danh sach identity
    all_ids = sorted([d for d in os.listdir(config.TEMP_BY_ID) 
                      if os.path.isdir(os.path.join(config.TEMP_BY_ID, d))])
    print(f"Tong so identity: {len(all_ids)}")
    
    # Shuffle va chia
    random.shuffle(all_ids)
    
    n_total = len(all_ids)
    n_val = int(config.VAL_RATIO * n_total)
    n_test = int(config.TEST_RATIO * n_total)
    n_train = n_total - n_val - n_test
    
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]
    
    print(f"\nChia theo IDENTITY:")
    print(f"  Train: {len(train_ids)} identities")
    print(f"  Val: {len(val_ids)} identities")
    print(f"  Test: {len(test_ids)} identities")
    
    # Copy vao cac thu muc split
    os.makedirs(config.TEMP_SPLIT, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(config.TEMP_SPLIT, split), exist_ok=True)
    
    split_mapping = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for split, ids_list in split_mapping.items():
        for pid in tqdm(ids_list, desc=f"Copying {split}"):
            src_dir = os.path.join(config.TEMP_BY_ID, pid)
            dst_dir = os.path.join(config.TEMP_SPLIT, split, pid)
            
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir)
                stats[split] += len([f for f in os.listdir(dst_dir) if f.endswith('.jpg')])
    
    print(f"\nKet qua:")
    total = sum(stats.values())
    for split, count in stats.items():
        print(f"  {split}: {count:,} anh ({100*count/total:.1f}%)")
    
    return train_ids, val_ids, test_ids

def step5_align_faces(config, landmarks):
    """Buoc 5: Alignment theo ArcFace"""
    print("\n" + "="*60)
    print("BUOC 5: ALIGNMENT THEO ARCFACE")
    print("="*60)
    
    os.makedirs(config.FINAL_OUTPUT, exist_ok=True)
    
    align_stats = {'aligned': 0, 'center_crop': 0, 'failed': 0}
    
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(config.TEMP_SPLIT, split)
        out_split_dir = os.path.join(config.FINAL_OUTPUT, split)
        os.makedirs(out_split_dir, exist_ok=True)
        
        persons = os.listdir(split_dir)
        
        for person in tqdm(persons, desc=f"Aligning {split}"):
            src_person_dir = os.path.join(split_dir, person)
            dst_person_dir = os.path.join(out_split_dir, person)
            os.makedirs(dst_person_dir, exist_ok=True)
            
            for img_name in os.listdir(src_person_dir):
                if not img_name.endswith('.jpg'):
                    continue
                
                img_path = os.path.join(src_person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    align_stats['failed'] += 1
                    continue
                
                # Xac dinh anh goc hay augmented
                original_name = img_name.split('_aug')[0] + '.jpg' if '_aug' in img_name else img_name
                
                if original_name in landmarks:
                    lm = landmarks[original_name]
                    aligned = align_face(img, lm)
                    align_stats['aligned'] += 1
                else:
                    aligned = align_face_center_crop(img)
                    align_stats['center_crop'] += 1
                
                save_path = os.path.join(dst_person_dir, img_name)
                cv2.imwrite(save_path, aligned)
    
    print(f"\nAlignment stats:")
    print(f"  Aligned with landmarks: {align_stats['aligned']:,}")
    print(f"  Center crop (augmented): {align_stats['center_crop']:,}")
    print(f"  Failed: {align_stats['failed']}")

def step6_create_metadata(config):
    """Buoc 6: Tao metadata cho training"""
    print("\n" + "="*60)
    print("BUOC 6: TAO METADATA")
    print("="*60)
    
    META_OUTPUT = os.path.join(config.FINAL_OUTPUT, "metadata")
    os.makedirs(META_OUTPUT, exist_ok=True)
    
    # Tao GLOBAL label mapping tu train set
    train_dir = os.path.join(config.FINAL_OUTPUT, "train")
    all_train_ids = sorted(os.listdir(train_dir))
    
    global_id_to_label = {pid: idx for idx, pid in enumerate(all_train_ids)}
    print(f"Total training identities: {len(global_id_to_label)}")
    
    # Luu global mapping
    global_mapping_df = pd.DataFrame([
        {"identity_id": pid, "label": label}
        for pid, label in global_id_to_label.items()
    ])
    global_mapping_df.to_csv(os.path.join(META_OUTPUT, "global_id_mapping.csv"), index=False)
    
    # Tao labels file cho moi split
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(config.FINAL_OUTPUT, split)
        records = []
        
        for pid in os.listdir(split_dir):
            person_dir = os.path.join(split_dir, pid)
            if not os.path.isdir(person_dir):
                continue
            
            label = global_id_to_label.get(pid, -1)
            
            for img_name in os.listdir(person_dir):
                if img_name.endswith('.jpg'):
                    records.append({
                        "image": f"{pid}/{img_name}",
                        "identity_id": pid,
                        "label": label,
                        "is_augmented": 1 if '_aug' in img_name else 0
                    })
        
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(META_OUTPUT, f"{split}_labels.csv"), index=False)
        
        n_ids = df['identity_id'].nunique()
        n_imgs = len(df)
        n_aug = df['is_augmented'].sum()
        print(f"{split}: {n_imgs:,} images, {n_ids} identities, {n_aug:,} augmented")
    
    # Tao dataset config
    dataset_config = {
        "dataset_name": "CelebA_Aligned_Balanced",
        "preprocessing": {
            "min_images_per_identity": config.MIN_IMAGES,
            "augment_threshold": config.AUGMENT_THRESHOLD,
            "target_min_images": config.TARGET_MIN_IMAGES
        },
        "image_size": [112, 112],
        "arcface_landmarks": {
            "left_eye": [38.2946, 51.6963],
            "right_eye": [73.5318, 51.5014],
            "nose": [56.0252, 71.7366],
            "left_mouth": [41.5493, 92.3655],
            "right_mouth": [70.7299, 92.2041]
        },
        "split_method": "by_identity",
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "splits": {}
    }
    
    for split in ["train", "val", "test"]:
        labels_df = pd.read_csv(os.path.join(META_OUTPUT, f"{split}_labels.csv"))
        dataset_config["splits"][split] = {
            "num_identities": int(labels_df['identity_id'].nunique()),
            "num_images": int(len(labels_df)),
            "num_augmented": int(labels_df['is_augmented'].sum())
        }
    
    with open(os.path.join(META_OUTPUT, "dataset_config.json"), "w") as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"\nDataset config saved to {META_OUTPUT}/dataset_config.json")

def verify_no_overlap(config):
    """Kiem tra khong co chong cheo identity"""
    print("\n" + "="*60)
    print("KIEM TRA CHONG CHEO IDENTITY")
    print("="*60)
    
    train_ids = set(os.listdir(os.path.join(config.FINAL_OUTPUT, "train")))
    val_ids = set(os.listdir(os.path.join(config.FINAL_OUTPUT, "val")))
    test_ids = set(os.listdir(os.path.join(config.FINAL_OUTPUT, "test")))
    
    print(f"Train & Val overlap: {len(train_ids & val_ids)}")
    print(f"Train & Test overlap: {len(train_ids & test_ids)}")
    print(f"Val & Test overlap: {len(val_ids & test_ids)}")
    
    if len(train_ids & val_ids) == 0 and len(train_ids & test_ids) == 0:
        print("\n[OK] KHONG CO CHONG CHEO - Dataset da chia dung!")
        return True
    else:
        print("\n[ERROR] CO CHONG CHEO - Can kiem tra lai!")
        return False

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Chay toan bo pipeline"""
    config = Config()
    config.FINAL_OUTPUT = f"{config.DRIVE_OUTPUT}/CelebA_Aligned_Balanced"
    
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    
    print("="*60)
    print("CELEBA PREPROCESSING - XU LY MAT CAN BANG")
    print("="*60)
    print(f"\nConfig:")
    print(f"  MIN_IMAGES: {config.MIN_IMAGES}")
    print(f"  AUGMENT_THRESHOLD: {config.AUGMENT_THRESHOLD}")
    print(f"  TARGET_MIN_IMAGES: {config.TARGET_MIN_IMAGES}")
    print(f"  Output: {config.FINAL_OUTPUT}")
    
    # Step 1: Analyze and filter
    filtered_df, ids_to_augment, ids_normal = step1_analyze_and_filter(config)
    
    # Step 2: Organize by identity
    step2_organize_by_identity(config, filtered_df)
    
    # Step 3: Augment small identities
    step3_augment_small_identities(config, ids_to_augment)
    
    # Step 4: Split by identity
    train_ids, val_ids, test_ids = step4_split_by_identity(config)
    
    # Step 5: Alignment
    print("\nLoading landmarks...")
    landmarks = load_landmarks(config.LANDMARK_FILE)
    print(f"Loaded {len(landmarks)} landmarks")
    step5_align_faces(config, landmarks)
    
    # Step 6: Create metadata
    step6_create_metadata(config)
    
    # Verify
    verify_no_overlap(config)
    
    print("\n" + "="*60)
    print("HOAN THANH!")
    print("="*60)
    print(f"Output: {config.FINAL_OUTPUT}")

if __name__ == "__main__":
    main()

