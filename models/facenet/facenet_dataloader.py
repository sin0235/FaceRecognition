"""
FaceNet DataLoader với Semi-Hard Negative Mining

Thay thế random triplet mining bằng online semi-hard mining để model học tốt hơn.
Semi-hard triplets: d(a,p) < d(a,n) < d(a,p) + margin
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class FaceNetTripletDataset(Dataset):
    """Dataset cho FaceNet triplet training với random mining (baseline)."""
    
    def __init__(self, root_dir, image_size=160, augment=False):
        """
        Dataset cho FaceNet triplet training.
        
        Args:
            root_dir: Path đến thư mục data
            image_size: Kích thước ảnh output (FaceNet yêu cầu 160x160)
            augment: Bật data augmentation cho training
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.identities = os.listdir(root_dir)

        self.id_to_images = {
            pid: os.listdir(os.path.join(root_dir, pid))
            for pid in self.identities
            if len(os.listdir(os.path.join(root_dir, pid))) >= 2
        }

        self.identities = list(self.id_to_images.keys())

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.identities)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, index):
        anchor_id = self.identities[index]
        imgs = self.id_to_images[anchor_id]
        a_name, p_name = random.sample(imgs, 2)

        neg_id = random.choice([i for i in self.identities if i != anchor_id])
        n_name = random.choice(self.id_to_images[neg_id])

        anchor = self._load_image(os.path.join(self.root_dir, anchor_id, a_name))
        positive = self._load_image(os.path.join(self.root_dir, anchor_id, p_name))
        negative = self._load_image(os.path.join(self.root_dir, neg_id, n_name))

        return anchor, positive, negative


class OnlineTripletDataset(Dataset):
    """
    Dataset với Online Triplet Mining.
    
    Mỗi epoch, tải tất cả ảnh của một batch identities và model sẽ
    tính embeddings để chọn semi-hard negatives trong training loop.
    """
    
    def __init__(self, root_dir, image_size=160, augment=False, 
                 min_images_per_identity=2, images_per_identity=4):
        """
        Args:
            root_dir: Path đến thư mục data
            image_size: Kích thước ảnh output
            augment: Bật data augmentation
            min_images_per_identity: Tối thiểu ảnh/identity để lọc
            images_per_identity: Số ảnh mỗi identity trong 1 batch
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.images_per_identity = images_per_identity
        
        # Scan và lọc identities
        self.id_to_images = {}
        for pid in os.listdir(root_dir):
            pid_path = os.path.join(root_dir, pid)
            if os.path.isdir(pid_path):
                images = [f for f in os.listdir(pid_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) >= min_images_per_identity:
                    self.id_to_images[pid] = images
        
        self.identities = list(self.id_to_images.keys())
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        print(f"OnlineTripletDataset: {len(self.identities)} identities")
    
    def __len__(self):
        return len(self.identities)
    
    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def __getitem__(self, index):
        """
        Trả về batch ảnh của 1 identity để online mining.
        
        Returns:
            images: Tensor (images_per_identity, 3, H, W)
            label: identity index
        """
        identity = self.identities[index]
        all_images = self.id_to_images[identity]
        
        # Sample images_per_identity ảnh (có lặp nếu không đủ)
        if len(all_images) >= self.images_per_identity:
            selected = random.sample(all_images, self.images_per_identity)
        else:
            selected = random.choices(all_images, k=self.images_per_identity)
        
        images = []
        for img_name in selected:
            img_path = os.path.join(self.root_dir, identity, img_name)
            images.append(self._load_image(img_path))
        
        images = torch.stack(images)  # (K, 3, H, W)
        label = index  # Identity index
        
        return images, label


def mine_semi_hard_triplets(embeddings, labels, margin=0.2):
    """
    Online semi-hard triplet mining từ batch embeddings.
    
    Semi-hard triplets: d(a,p) < d(a,n) < d(a,p) + margin
    
    Args:
        embeddings: Tensor (B, D) - B embeddings, D dimensions
        labels: Tensor (B,) - identity labels
        margin: Triplet margin
        
    Returns:
        anchor_idx, positive_idx, negative_idx: Lists of indices
    """
    device = embeddings.device
    B = embeddings.size(0)
    
    # Tính pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
    
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    
    for i in range(B):
        anchor_label = labels[i]
        
        # Tìm positives (cùng identity, khác index)
        positive_mask = (labels == anchor_label) & (torch.arange(B, device=device) != i)
        positive_indices_i = torch.where(positive_mask)[0]
        
        if len(positive_indices_i) == 0:
            continue
        
        # Tìm negatives (khác identity)
        negative_mask = labels != anchor_label
        negative_indices_i = torch.where(negative_mask)[0]
        
        if len(negative_indices_i) == 0:
            continue
        
        # Với mỗi positive, tìm semi-hard negative
        for p_idx in positive_indices_i:
            ap_dist = dist_matrix[i, p_idx]
            
            # Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
            neg_dists = dist_matrix[i, negative_indices_i]
            semi_hard_mask = (neg_dists > ap_dist) & (neg_dists < ap_dist + margin)
            
            if semi_hard_mask.any():
                # Chọn negative khó nhất trong semi-hard
                semi_hard_negs = negative_indices_i[semi_hard_mask]
                semi_hard_dists = neg_dists[semi_hard_mask]
                hardest_idx = semi_hard_negs[semi_hard_dists.argmin()]
                
                anchor_indices.append(i)
                positive_indices.append(p_idx.item())
                negative_indices.append(hardest_idx.item())
            else:
                # Fallback: chọn hard negative (gần nhất)
                hardest_neg_idx = negative_indices_i[neg_dists.argmin()]
                
                anchor_indices.append(i)
                positive_indices.append(p_idx.item())
                negative_indices.append(hardest_neg_idx.item())
    
    return anchor_indices, positive_indices, negative_indices


def mine_batch_hard_triplets(embeddings, labels):
    """
    Batch hard triplet mining: chọn hardest positive và hardest negative.
    
    Args:
        embeddings: Tensor (B, D)
        labels: Tensor (B,)
        
    Returns:
        anchor_idx, positive_idx, negative_idx
    """
    device = embeddings.device
    B = embeddings.size(0)
    
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    anchor_indices = []
    positive_indices = []
    negative_indices = []
    
    for i in range(B):
        anchor_label = labels[i]
        
        # Positives: cùng identity
        positive_mask = (labels == anchor_label) & (torch.arange(B, device=device) != i)
        positive_indices_i = torch.where(positive_mask)[0]
        
        # Negatives: khác identity
        negative_mask = labels != anchor_label
        negative_indices_i = torch.where(negative_mask)[0]
        
        if len(positive_indices_i) == 0 or len(negative_indices_i) == 0:
            continue
        
        # Hardest positive: xa nhất
        pos_dists = dist_matrix[i, positive_indices_i]
        hardest_pos = positive_indices_i[pos_dists.argmax()]
        
        # Hardest negative: gần nhất
        neg_dists = dist_matrix[i, negative_indices_i]
        hardest_neg = negative_indices_i[neg_dists.argmin()]
        
        anchor_indices.append(i)
        positive_indices.append(hardest_pos.item())
        negative_indices.append(hardest_neg.item())
    
    return anchor_indices, positive_indices, negative_indices


def check_identity_overlap(train_dir, val_dir):
    """
    Kiểm tra xem train và val có identities trùng nhau không.
    
    Args:
        train_dir: Thư mục train data
        val_dir: Thư mục val data
        
    Raises:
        ValueError: Nếu phát hiện identity overlap
    """
    # Lấy danh sách identities từ train
    train_identities = set()
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path):
            train_identities.add(item)
    
    # Lấy danh sách identities từ val
    val_identities = set()
    for item in os.listdir(val_dir):
        item_path = os.path.join(val_dir, item)
        if os.path.isdir(item_path):
            val_identities.add(item)
    
    # Kiểm tra overlap
    overlap = train_identities.intersection(val_identities)
    
    print(f"\n{'='*60}")
    print("IDENTITY SPLIT VALIDATION")
    print(f"{'='*60}")
    print(f"Train identities: {len(train_identities)}")
    print(f"Val identities: {len(val_identities)}")
    print(f"Overlap identities: {len(overlap)}")
    
    if overlap:
        print(f"\n{'!'*60}")
        print("ERROR: DATA LEAKAGE DETECTED")
        print(f"{'!'*60}")
        print(f"Found {len(overlap)} identities in both train and val sets!")
        print(f"Sample overlapping identities: {list(overlap)[:10]}")
        print("\nThis indicates that train and val are NOT split by identity (by_id).")
        print("This will lead to inflated validation accuracy and poor generalization.")
        print(f"{'!'*60}\n")
        raise ValueError(
            f"Data leakage detected: {len(overlap)} identities appear in both "
            f"train and val sets. Please re-split your dataset using 'by_id' strategy."
        )
    else:
        print("✓ No identity overlap detected (split_strategy: by_id)")
        print(f"{'='*60}\n")
    
    return True


def create_online_dataloaders(train_dir, val_dir, batch_size=32, 
                              image_size=160, num_workers=4,
                              images_per_identity=4):
    """
    Tạo DataLoaders cho online triplet mining.
    
    Args:
        train_dir: Thư mục train data
        val_dir: Thư mục val data
        batch_size: Số identities mỗi batch 
        image_size: Kích thước ảnh
        num_workers: Số workers
        images_per_identity: Số ảnh mỗi identity
        
    Returns:
        train_loader, val_loader
    """
    # Validate no identity overlap between train and val
    check_identity_overlap(train_dir, val_dir)
    
    train_dataset = OnlineTripletDataset(
        root_dir=train_dir,
        image_size=image_size,
        augment=True,
        images_per_identity=images_per_identity
    )
    
    val_dataset = OnlineTripletDataset(
        root_dir=val_dir,
        image_size=image_size,
        augment=False,
        images_per_identity=images_per_identity
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Train: {len(train_dataset)} identities, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} identities, {len(val_loader)} batches")
    print(f"Images per identity: {images_per_identity}")
    print(f"Effective batch size: {batch_size * images_per_identity} images")
    
    return train_loader, val_loader


def get_val_transforms(image_size=160):
    """Transforms cho validation/test set (không augmentation) cho FaceNet"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
