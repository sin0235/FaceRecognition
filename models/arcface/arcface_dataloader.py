"""
ArcFace DataLoader
Xu ly tai du lieu va augmentation cho training/validation
Ho tro nhieu format: CSV metadata hoac folder-based (khong can metadata)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import defaultdict

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class FolderBasedDataset(Dataset):
    """
    Dataset dua tren cau truc thu muc - KHONG CAN FILE METADATA
    
    Cau truc thu muc:
        data_root/
            identity_1/
                img1.jpg
                img2.jpg
            identity_2/
                img1.jpg
                ...
    """
    def __init__(self, data_root, transform=None, use_albumentations=False,
                 min_images_per_identity=5, extensions=('.jpg', '.jpeg', '.png')):
        """
        Args:
            data_root: Thu muc goc chua cac folder identity
            transform: Transforms de ap dung
            use_albumentations: Su dung albumentations thay vi torchvision
            min_images_per_identity: Loc bo identity co it hon X anh
            extensions: Cac dinh dang file anh ho tro
        """
        self.data_root = data_root
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.extensions = extensions
        
        # Scan thu muc va tao mapping
        self.samples = []  # List of (image_path, label)
        self.identity_to_label = {}
        self.label_to_identity = {}
        self.class_counts = {}  # So anh moi class
        
        self._scan_folder(min_images_per_identity)
        
        self.num_classes = len(self.identity_to_label)
        print(f"Loaded {len(self.samples)} images from {self.num_classes} identities")
        print(f"Min images/identity filter: {min_images_per_identity}")
    
    def _scan_folder(self, min_images):
        """Scan thu muc de tao danh sach samples"""
        identity_images = defaultdict(list)
        
        # Scan tat ca cac folder
        for identity_name in os.listdir(self.data_root):
            identity_path = os.path.join(self.data_root, identity_name)
            if not os.path.isdir(identity_path):
                continue
            
            # Scan tat ca anh trong folder
            for img_name in os.listdir(identity_path):
                if img_name.lower().endswith(self.extensions):
                    img_path = os.path.join(identity_path, img_name)
                    identity_images[identity_name].append(img_path)
        
        # Loc identity co du anh va tao mapping
        label = 0
        for identity_name, images in sorted(identity_images.items()):
            if len(images) >= min_images:
                self.identity_to_label[identity_name] = label
                self.label_to_identity[label] = identity_name
                self.class_counts[label] = len(images)
                
                for img_path in images:
                    self.samples.append((img_path, label))
                
                label += 1
        
        # In thong ke
        filtered_count = len(identity_images) - len(self.identity_to_label)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} identities with < {min_images} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                if self.use_albumentations:
                    image = np.array(image)
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            return image, label, image_path
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            dummy_image = torch.zeros(3, 112, 112)
            return dummy_image, label, image_path
    
    def get_identity_name(self, label):
        return self.label_to_identity.get(label, f"ID_{label}")
    
    def get_sample_weights(self):
        """Tinh trong so cho moi sample de class-balanced sampling"""
        # Trong so = 1 / so_anh_cua_class
        weights = []
        for _, label in self.samples:
            weights.append(1.0 / self.class_counts[label])
        return weights
    
    def get_class_weights(self):
        """Tinh trong so cho loss function (inverse frequency)"""
        total_samples = len(self.samples)
        weights = []
        for label in range(self.num_classes):
            count = self.class_counts[label]
            # Inverse frequency weighting
            weight = total_samples / (self.num_classes * count)
            weights.append(weight)
        return torch.FloatTensor(weights)


class ArcFaceDataset(Dataset):
    """
    Dataset cho ArcFace training
    
    Ho tro 3 format CSV:
    1. CelebA format: image, identity_id, label
    2. Full path format: image_path, identity_name
    3. Legacy format: image, person_id
    """
    def __init__(self, csv_path, data_root=None, transform=None, use_albumentations=False):
        """
        Args:
            csv_path: Duong dan den file metadata CSV
            data_root: Thu muc goc chua anh (vd: data/CelebA_Aligned/train)
            transform: Transforms de ap dung
            use_albumentations: Su dung albumentations thay vi torchvision
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.csv_path = csv_path
        
        # Auto-detect format va setup columns
        self._detect_format()
        
        # Setup label mapping
        self._setup_labels()
        
        print(f"Loaded {len(self.df)} images with {self.num_classes} identities")
        print(f"Format: path_col='{self.path_col}', label_col='{self.label_col}'")
    
    def _detect_format(self):
        """Auto-detect CSV format"""
        cols = set(self.df.columns)
        
        # Format 1: CelebA format (image, identity_id, label)
        if 'image' in cols and 'identity_id' in cols and 'label' in cols:
            self.path_col = 'image'
            self.identity_col = 'identity_id'
            self.label_col = 'label'
            self.has_precomputed_labels = True
            print("Detected CelebA format (image, identity_id, label)")
            
        # Format 2: Full path format (image_path, identity_name)
        elif 'image_path' in cols and 'identity_name' in cols:
            self.path_col = 'image_path'
            self.identity_col = 'identity_name'
            self.label_col = None
            self.has_precomputed_labels = False
            print("Detected full path format (image_path, identity_name)")
            
        # Format 3: Legacy format (image, person_id)
        elif 'image' in cols and 'person_id' in cols:
            self.path_col = 'image'
            self.identity_col = 'person_id'
            self.label_col = 'label' if 'label' in cols else None
            self.has_precomputed_labels = self.label_col is not None
            print("Detected legacy format (image, person_id)")
            
        else:
            raise ValueError(f"Unsupported CSV format. Columns: {list(cols)}")
        
        # Auto-detect data_root neu chua co
        if self.data_root is None and self.path_col == 'image':
            csv_dir = os.path.dirname(self.csv_path)
            parent_dir = os.path.dirname(csv_dir)
            
            # Detect split from csv filename
            csv_name = os.path.basename(self.csv_path).lower()
            if 'train' in csv_name:
                split = 'train'
            elif 'val' in csv_name:
                split = 'val'
            elif 'test' in csv_name:
                split = 'test'
            else:
                split = 'train'
            
            potential_root = os.path.join(parent_dir, split)
            if os.path.exists(potential_root):
                self.data_root = potential_root
                print(f"Auto-detected data_root: {self.data_root}")
    
    def _setup_labels(self):
        """Setup label mapping"""
        if self.has_precomputed_labels and self.label_col:
            # Su dung labels da co san trong CSV
            self.num_classes = self.df[self.label_col].nunique()
            
            # Tao mapping de tra ve identity name
            if self.identity_col in self.df.columns:
                label_to_id = self.df.groupby(self.label_col)[self.identity_col].first().to_dict()
                self.label_to_identity = {int(k): str(v) for k, v in label_to_id.items()}
                self.identity_to_label = {v: k for k, v in self.label_to_identity.items()}
            else:
                self.label_to_identity = {}
                self.identity_to_label = {}
        else:
            # Tao labels tu identity column
            unique_identities = sorted(self.df[self.identity_col].unique())
            self.identity_to_label = {str(identity): idx for idx, identity in enumerate(unique_identities)}
            self.label_to_identity = {v: k for k, v in self.identity_to_label.items()}
            self.num_classes = len(self.identity_to_label)
    
    def __len__(self):
        return len(self.df)
    
    def get_image_path(self, idx):
        """Lay full path cua anh tai index"""
        row = self.df.iloc[idx]
        path = row[self.path_col]
        
        if self.data_root and not os.path.isabs(path):
            return os.path.join(self.data_root, path)
        return path
    
    def get_label(self, idx):
        """Lay label tai index"""
        row = self.df.iloc[idx]
        
        if self.has_precomputed_labels and self.label_col:
            return int(row[self.label_col])
        else:
            identity = str(row[self.identity_col])
            return self.identity_to_label[identity]
    
    def __getitem__(self, idx):
        image_path = self.get_image_path(idx)
        label = self.get_label(idx)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                if self.use_albumentations:
                    image = np.array(image)
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            return image, label, image_path
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            dummy_image = torch.zeros(3, 112, 112)
            return dummy_image, label, image_path
    
    def get_identity_name(self, label):
        """Lay ten identity tu label"""
        return self.label_to_identity.get(label, f"ID_{label}")


def get_train_transforms(image_size=112, use_albumentations=False, augment_strength='normal'):
    """
    Transforms cho training set (co augmentation)
    
    Args:
        image_size: Kich thuoc anh dau ra
        use_albumentations: Su dung albumentations
        augment_strength: 'light', 'normal', 'strong', hoac 'heavy'
    """
    if use_albumentations and HAS_ALBUMENTATIONS:
        if augment_strength == 'heavy':
            # HEAVY: Augmentation manh nhat cho dataset it anh/class
            return A.Compose([
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.5),
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.6, border_mode=0),
                A.Affine(
                    scale=(0.85, 1.15),
                    translate_percent=(-0.1, 0.1),
                    shear=(-8, 8),
                    rotate=(-15, 15),
                    p=0.6
                ),
                A.Perspective(scale=(0.02, 0.05), p=0.3),
                A.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.1,
                    p=0.8
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 100.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.Defocus(radius=(2, 4), p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.Equalize(p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.ChannelShuffle(p=1.0),
                ], p=0.3),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=0.2),
                A.CoarseDropout(
                    max_holes=6,
                    max_height=int(image_size * 0.2),
                    max_width=int(image_size * 0.2),
                    min_holes=2,
                    fill_value=0,
                    p=0.4
                ),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        elif augment_strength == 'strong':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1),
                    shear=(-5, 5),
                    p=0.5
                ),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1,
                    p=0.7
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.CLAHE(clip_limit=2.0, p=1.0),
                    A.Equalize(p=1.0),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=int(image_size * 0.15),
                    max_width=int(image_size * 0.15),
                    min_holes=1,
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        elif augment_strength == 'light':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:  # normal
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
        # Torchvision transforms
        if augment_strength == 'heavy':
            # HEAVY: Augmentation manh nhat voi torchvision
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.85, 1.15),
                    shear=8
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.15),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=0.4, scale=(0.05, 0.2), ratio=(0.3, 3.3))
            ])
        elif augment_strength == 'strong':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            ])
        elif augment_strength == 'light':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:  # normal
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
    """Transforms cho validation/test set (khong augmentation)"""
    if use_albumentations and HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def create_dataloaders(train_csv, val_csv, batch_size=64, num_workers=4,
                      image_size=112, use_albumentations=False,
                      train_data_root=None, val_data_root=None):
    """
    Tao DataLoaders cho training va validation (tu CSV metadata)
    
    Args:
        train_csv: Duong dan den train metadata CSV
        val_csv: Duong dan den validation metadata CSV
        batch_size: Batch size
        num_workers: So workers cho parallel data loading
        image_size: Kich thuoc anh (default 112 cho ArcFace)
        use_albumentations: Su dung albumentations
        train_data_root: Thu muc goc chua anh train
        val_data_root: Thu muc goc chua anh val
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_transform = get_train_transforms(image_size, use_albumentations)
    val_transform = get_val_transforms(image_size, use_albumentations)
    
    print("\n=== Creating Train Dataset ===")
    train_dataset = ArcFaceDataset(
        csv_path=train_csv,
        data_root=train_data_root,
        transform=train_transform,
        use_albumentations=use_albumentations
    )
    
    print("\n=== Creating Val Dataset ===")
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
    
    print(f"\n=== DataLoader Summary ===")
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, num_classes


def create_folder_dataloaders(train_root, val_root, batch_size=64, num_workers=4,
                              image_size=112, use_albumentations=False,
                              min_images_per_identity=5,
                              class_balanced_sampling=True,
                              augment_strength='normal'):
    """
    Tao DataLoaders tu cau truc thu muc - KHONG CAN FILE METADATA
    
    Cau truc thu muc:
        train_root/
            identity_1/
                img1.jpg
                img2.jpg
            identity_2/
                img1.jpg
                ...
    
    Args:
        train_root: Thu muc chua anh train (co cac folder identity)
        val_root: Thu muc chua anh val
        batch_size: Batch size
        num_workers: So workers cho parallel data loading
        image_size: Kich thuoc anh (default 112 cho ArcFace)
        use_albumentations: Su dung albumentations
        min_images_per_identity: Loc bo identity co it hon X anh
        class_balanced_sampling: Su dung weighted sampling de can bang class
        augment_strength: 'light', 'normal', 'strong' - muc do augmentation
    
    Returns:
        train_loader, val_loader, num_classes, class_weights
    """
    train_transform = get_train_transforms(image_size, use_albumentations, augment_strength)
    val_transform = get_val_transforms(image_size, use_albumentations)
    
    print("\n=== Creating Train Dataset (Folder-based) ===")
    train_dataset = FolderBasedDataset(
        data_root=train_root,
        transform=train_transform,
        use_albumentations=use_albumentations,
        min_images_per_identity=min_images_per_identity
    )
    
    print("\n=== Creating Val Dataset (Folder-based) ===")
    val_dataset = FolderBasedDataset(
        data_root=val_root,
        transform=val_transform,
        use_albumentations=use_albumentations,
        min_images_per_identity=1  # Val khong loc
    )
    
    # Class-balanced sampling cho training
    if class_balanced_sampling:
        print("\n=== Using Class-Balanced Sampling ===")
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Su dung sampler thay vi shuffle
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        print(f"Sample weights range: [{min(sample_weights):.6f}, {max(sample_weights):.6f}]")
    else:
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
    class_weights = train_dataset.get_class_weights()
    
    print(f"\n=== DataLoader Summary ===")
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Augmentation: {augment_strength}")
    print(f"Class-balanced: {class_balanced_sampling}")
    
    # In thong ke class distribution
    counts = list(train_dataset.class_counts.values())
    print(f"\nClass distribution:")
    print(f"  Min images/class: {min(counts)}")
    print(f"  Max images/class: {max(counts)}")
    print(f"  Mean images/class: {np.mean(counts):.1f}")
    print(f"  Median images/class: {np.median(counts):.1f}")
    
    return train_loader, val_loader, num_classes, class_weights


def visualize_batch(dataloader, num_images=16, save_path=None):
    """Visualize mot batch anh de kiem tra augmentation"""
    import matplotlib.pyplot as plt
    
    images, labels, paths = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    paths = paths[:num_images]
    
    # Denormalize
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    nrows = int(np.ceil(np.sqrt(num_images)))
    ncols = int(np.ceil(num_images / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, (img, label, path) in enumerate(zip(images, labels, paths)):
        if idx >= len(axes):
            break
        
        img = img.permute(1, 2, 0).numpy()
        axes[idx].imshow(img)
        axes[idx].set_title(f"Label: {label.item()}", fontsize=8)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def benchmark_dataloader(dataloader, num_iterations=100):
    """Do toc do loading cua DataLoader"""
    import time
    
    print(f"Benchmarking DataLoader ({num_iterations} iterations)...")
    
    start_time = time.time()
    total_images = 0
    
    for i, (images, labels, _) in enumerate(dataloader):
        if i >= num_iterations:
            break
        total_images += images.size(0)
    
    elapsed_time = time.time() - start_time
    images_per_second = total_images / elapsed_time
    
    print(f"Loaded {total_images} images in {elapsed_time:.2f}s")
    print(f"Speed: {images_per_second:.1f} images/second")
    
    if images_per_second < 100:
        print("Speed is below target (100 img/s). Consider increasing num_workers.")
    else:
        print("Speed OK!")
    
    return images_per_second


def test_with_celeba_data():
    """Test DataLoader voi CelebA metadata thuc te"""
    print("="*60)
    print("TEST DATALOADER WITH CELEBA METADATA")
    print("="*60)
    
    train_csv = "data/CelebA_Aligned_Balanced/metadata/train_labels.csv"
    val_csv = "data/CelebA_Aligned_Balanced/metadata/val_labels.csv"
    train_root = "data/CelebA_Aligned_Balanced/train"
    val_root = "data/CelebA_Aligned_Balanced/val"
    
    # Check files exist
    for name, path in [("Train CSV", train_csv), ("Val CSV", val_csv),
                       ("Train dir", train_root), ("Val dir", val_root)]:
        if os.path.exists(path):
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            return
    
    # Create dataloaders
    train_loader, val_loader, num_classes = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        batch_size=32,
        num_workers=0,
        image_size=112,
        train_data_root=train_root,
        val_data_root=val_root
    )
    
    # Test loading
    print("\n=== Test Loading ===")
    images, labels, paths = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels[:5].tolist()}")
    print(f"Sample paths: {paths[:2]}")
    
    # Check image exists
    for path in paths[:3]:
        print(f"  {path}: {'exists' if os.path.exists(path) else 'NOT FOUND'}")
    
    print("\nTest completed!")


def test_folder_based():
    """Test Folder-based DataLoader - KHONG CAN METADATA"""
    print("="*60)
    print("TEST FOLDER-BASED DATALOADER (NO METADATA REQUIRED)")
    print("="*60)
    
    # Gia su cau truc thu muc:
    # data/celeba_by_id/train/
    #     1/
    #         000001.jpg
    #         000002.jpg
    #     2/
    #         000003.jpg
    #     ...
    
    train_root = "data/celeba_by_id/train"
    val_root = "data/celeba_by_id/val"
    
    # Check directories
    for name, path in [("Train dir", train_root), ("Val dir", val_root)]:
        if os.path.exists(path):
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            print("Hay chay notebook data_preprocessing.ipynb de tao du lieu")
            return
    
    # Tao dataloader voi class-balanced sampling va strong augmentation
    train_loader, val_loader, num_classes, class_weights = create_folder_dataloaders(
        train_root=train_root,
        val_root=val_root,
        batch_size=32,
        num_workers=0,
        image_size=112,
        use_albumentations=HAS_ALBUMENTATIONS,
        min_images_per_identity=5,
        class_balanced_sampling=True,
        augment_strength='strong'
    )
    
    # Test loading
    print("\n=== Test Loading ===")
    images, labels, paths = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels[:10].tolist()}")
    
    # Kiem tra class weights
    print(f"\nClass weights shape: {class_weights.shape}")
    print(f"Class weights range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
    
    print("\nTest completed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'folder':
        test_folder_based()
    else:
        test_with_celeba_data()
