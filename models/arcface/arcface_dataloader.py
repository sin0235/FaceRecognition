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
    print("albumentations không có, sử dụng torchvision transforms")


class ArcFaceDataset(Dataset):
    """
    Dataset cho ArcFace training
    Ho tro 2 format CSV:
    1. Format moi: image_path (full path), identity_name
    2. Format cu: image (relative path), person_id, label
    """
    def __init__(self, csv_path, data_root=None, transform=None, use_albumentations=False):
        """
        Args:
            csv_path: Duong dan den file metadata CSV
            data_root: Thu muc goc chua anh (chi can khi dung relative path)
            transform: Transforms de ap dung
            use_albumentations: Su dung albumentations thay vi torchvision
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.use_albumentations = use_albumentations
        
        # Auto-detect format
        if 'image_path' in self.df.columns and 'identity_name' in self.df.columns:
            # Format moi
            self.path_col = 'image_path'
            self.identity_col = 'identity_name'
        elif 'image' in self.df.columns and 'person_id' in self.df.columns:
            # Format cu - can data_root
            self.path_col = 'image'
            self.identity_col = 'person_id'
            if data_root is None:
                csv_dir = os.path.dirname(csv_path)
                # Tim data_root tu csv_path: .../metadata/ -> .../
                self.data_root = os.path.dirname(csv_dir)
                print(f"Auto-detected data_root: {self.data_root}")
        else:
            raise ValueError(f"CSV khong co columns can thiet. Columns: {list(self.df.columns)}")
        
        # Anh xa identity sang integer labels
        unique_identities = sorted(self.df[self.identity_col].unique())
        self.identity_to_label = {str(identity): idx for idx, identity in enumerate(unique_identities)}
        self.label_to_identity = {v: k for k, v in self.identity_to_label.items()}
        
        self.num_classes = len(self.identity_to_label)
        
        print(f"Loaded {len(self.df)} anh voi {self.num_classes} identities")
        print(f"Format detected: path_col='{self.path_col}', identity_col='{self.identity_col}'")
    
    def __len__(self):
        return len(self.df)
    
    def _get_image_path(self, row):
        """Lay full path tu row"""
        path = row[self.path_col]
        if self.data_root and not os.path.isabs(path):
            # Relative path - prepend data_root va split folder
            split = os.path.basename(os.path.dirname(os.path.dirname(path))) or 'train'
            # Path format: person_id/image.jpg
            # Full path: data_root/split/person_id/image.jpg
            return os.path.join(self.data_root, split, path)
        return path
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Lay path
        if self.path_col == 'image' and self.data_root:
            # Format cu: image = "person_id/img.jpg"
            # Can xac dinh split tu csv_path
            image_path = os.path.join(self.data_root, row[self.path_col])
        else:
            image_path = row[self.path_col]
        
        identity = str(row[self.identity_col])
        label = self.identity_to_label[identity]
        
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
            print(f"Loi khi load anh {image_path}: {e}")
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
                      image_size=112, use_albumentations=False,
                      train_data_root=None, val_data_root=None):
    """
    Tao DataLoaders cho training va validation
    
    Args:
        train_csv: Duong dan den train metadata CSV
        val_csv: Duong dan den validation metadata CSV
        batch_size: Batch size
        num_workers: So workers cho parallel data loading
        image_size: Kich thuoc anh
        use_albumentations: Su dung albumentations
        train_data_root: Thu muc goc chua anh train (neu dung relative path)
        val_data_root: Thu muc goc chua anh val (neu dung relative path)
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_transform = get_train_transforms(image_size, use_albumentations)
    val_transform = get_val_transforms(image_size, use_albumentations)
    
    train_dataset = ArcFaceDataset(
        csv_path=train_csv,
        data_root=train_data_root,
        transform=train_transform,
        use_albumentations=use_albumentations
    )
    
    val_dataset = ArcFaceDataset(
        csv_path=val_csv,
        data_root=val_data_root,
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
    Test DataLoader voi ca 2 format metadata
    """
    print("Testing DataLoader...")
    
    os.makedirs('data/test_split', exist_ok=True)
    
    # Test format moi (full path)
    print("\n=== Test format moi (image_path, identity_name) ===")
    dummy_data_new = {
        'image_path': ['data/test_split/test_img.jpg'] * 100,
        'identity_name': [f'person_{i%10}' for i in range(100)],
    }
    df_new = pd.DataFrame(dummy_data_new)
    df_new.to_csv('data/test_metadata_new.csv', index=False)
    
    # Test format cu (relative path)
    print("\n=== Test format cu (image, person_id, label) ===")
    dummy_data_old = {
        'image': [f'{i%10}/test_img.jpg' for i in range(100)],
        'person_id': [i%10 for i in range(100)],
        'label': [i%10 for i in range(100)]
    }
    df_old = pd.DataFrame(dummy_data_old)
    df_old.to_csv('data/test_metadata_old.csv', index=False)
    
    try:
        train_transform = get_train_transforms(image_size=112)
        
        # Test format moi
        dataset_new = ArcFaceDataset('data/test_metadata_new.csv', transform=train_transform)
        print(f"Format moi - Num classes: {dataset_new.num_classes}")
        
        # Test format cu voi data_root
        dataset_old = ArcFaceDataset('data/test_metadata_old.csv', 
                                     data_root='data/test_split',
                                     transform=train_transform)
        print(f"Format cu - Num classes: {dataset_old.num_classes}")
        
        print("\nDataLoader test thanh cong!")
        
    except Exception as e:
        print(f"Loi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        for f in ['data/test_metadata_new.csv', 'data/test_metadata_old.csv']:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists('data/test_split'):
            shutil.rmtree('data/test_split')


if __name__ == "__main__":
    test_dataloader()
