"""
ArcFace DataLoader
Xử lý tải dữ liệu và augmentation cho training/validation
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Optional: albumentations (nếu không có sẽ dùng torchvision)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ albumentations không có, sử dụng torchvision transforms")


class ArcFaceDataset(Dataset):
    """
    Dataset cho ArcFace training
    """
    def __init__(self, csv_path, transform=None, use_albumentations=False):
        """
        Args:
            csv_path: Đường dẫn đến file metadata CSV
            transform: Transforms để áp dụng
            use_albumentations: Sử dụng albumentations thay vì torchvision
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.use_albumentations = use_albumentations
        
        # Ánh xạ identity name sang integer labels
        self.identity_to_label = {identity: idx for idx, identity in 
                                  enumerate(self.df['identity_name'].unique())}
        self.label_to_identity = {v: k for k, v in self.identity_to_label.items()}
        
        self.num_classes = len(self.identity_to_label)
        
        print(f"Loaded {len(self.df)} ảnh với {self.num_classes} identities")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        identity_name = row['identity_name']
        label = self.identity_to_label[identity_name]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                if self.use_albumentations:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    image = Image.fromarray(image)
                    image = self.transform(image)
            
            return image, label, image_path
            
        except Exception as e:
            print(f"Lỗi khi load ảnh {image_path}: {e}")
            # Return dummy data nếu có lỗi
            dummy_image = torch.zeros(3, 112, 112)
            return dummy_image, label, image_path


def get_train_transforms(image_size=112, use_albumentations=False):
    """
    Transforms cho training set (có augmentation)
    """
    if use_albumentations and HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:
        # Fallback to torchvision
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def get_val_transforms(image_size=112, use_albumentations=False):
    """
    Transforms cho validation/test set (không có augmentation)
    """
    if use_albumentations and HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:
        # Fallback to torchvision
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def create_dataloaders(train_csv, val_csv, batch_size=64, num_workers=4, 
                      image_size=112, use_albumentations=False):
    """
    Tạo DataLoaders cho training và validation
    
    Args:
        train_csv: Đường dẫn đến train metadata CSV
        val_csv: Đường dẫn đến validation metadata CSV
        batch_size: Batch size
        num_workers: Số workers cho parallel data loading
        image_size: Kích thước ảnh
        use_albumentations: Sử dụng albumentations
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_transform = get_train_transforms(image_size, use_albumentations)
    val_transform = get_val_transforms(image_size, use_albumentations)
    
    train_dataset = ArcFaceDataset(
        csv_path=train_csv,
        transform=train_transform,
        use_albumentations=use_albumentations
    )
    
    val_dataset = ArcFaceDataset(
        csv_path=val_csv,
        transform=val_transform,
        use_albumentations=use_albumentations
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
    
    num_classes = train_dataset.num_classes
    
    print(f"\nDataLoader Summary:")
    print(f"Train: {len(train_dataset)} ảnh, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} ảnh, {len(val_loader)} batches")
    print(f"Num classes: {num_classes}")
    
    return train_loader, val_loader, num_classes


def visualize_batch(dataloader, num_images=16, save_path=None):
    """
    Visualize một batch ảnh để kiểm tra augmentation
    """
    import matplotlib.pyplot as plt
    
    images, labels, paths = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Denormalize
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx >= len(axes):
            break
        
        img = img.permute(1, 2, 0).numpy()
        axes[idx].imshow(img)
        axes[idx].set_title(f"Label: {label.item()}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def benchmark_dataloader(dataloader, num_iterations=100):
    """
    Đo tốc độ loading của DataLoader
    """
    import time
    
    print(f"Benchmarking DataLoader với {num_iterations} iterations...")
    
    start_time = time.time()
    total_images = 0
    
    for i, (images, labels, _) in enumerate(dataloader):
        if i >= num_iterations:
            break
        total_images += images.size(0)
    
    elapsed_time = time.time() - start_time
    images_per_second = total_images / elapsed_time
    
    print(f"Loaded {total_images} ảnh trong {elapsed_time:.2f}s")
    print(f"Tốc độ: {images_per_second:.1f} ảnh/giây")
    
    if images_per_second < 100:
        print("Tốc độ thấp hơn mục tiêu (100 ảnh/s)")
        print("Khuyến nghị: Tăng num_workers hoặc sử dụng SSD")
    else:
        print("Tốc độ đạt yêu cầu!")
    
    return images_per_second


def test_dataloader():
    """
    Test DataLoader với dummy metadata
    """
    print("Testing DataLoader...")
    
    # Tạo dummy metadata CSV
    dummy_data = {
        'image_path': ['data/test_img.jpg'] * 100,
        'identity_name': [f'person_{i%10}' for i in range(100)],
        'identity_id': [i%10 for i in range(100)]
    }
    
    df = pd.DataFrame(dummy_data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/test_metadata.csv', index=False)
    
    try:
        train_transform = get_train_transforms(image_size=112)
        dataset = ArcFaceDataset('data/test_metadata.csv', transform=train_transform)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Num classes: {dataset.num_classes}")
        print(f"Identity mapping: {dataset.identity_to_label}")
        
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        
        print("\nDataLoader test thành công!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        if os.path.exists('data/test_metadata.csv'):
            os.remove('data/test_metadata.csv')


if __name__ == "__main__":
    test_dataloader()
