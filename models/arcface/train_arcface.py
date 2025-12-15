"""
ArcFace Training Script
Script chinh de train mo hinh ArcFace
Ho tro: Early Stopping + Learning Rate Scheduling
Compatible with Kaggle/Colab environments
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import time
from datetime import datetime
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

TENSORBOARD_AVAILABLE = False
SummaryWriter = None
try:
    from torch.utils.tensorboard import SummaryWriter
    test_writer = SummaryWriter(log_dir='/tmp/test_tb', comment='test')
    test_writer.close()
    import shutil
    shutil.rmtree('/tmp/test_tb', ignore_errors=True)
    TENSORBOARD_AVAILABLE = True
    print("[INFO] TensorBoard available")
except Exception as e:
    print(f"[WARN] TensorBoard not available: {type(e).__name__}")
    TENSORBOARD_AVAILABLE = False

TSNE_AVAILABLE = False
TSNE = None
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except Exception:
    pass

PLT_AVAILABLE = False
plt = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except Exception:
    pass

def setup_project_path():
    """Setup sys.path cho import modules"""
    try:
        # Thu dung __file__ truoc
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        # __file__ khong kha dung (notebook/interactive)
        project_root = os.getcwd()
    
    # Kiem tra xem project_root co chua models/ khong
    if not os.path.exists(os.path.join(project_root, 'models')):
        # Thu tim trong cac vi tri pho bien (Kaggle/Colab)
        possible_roots = [
            '/kaggle/working/FaceRecognition',
            '/content/FaceRecognition',
            os.path.join(os.getcwd(), 'FaceRecognition'),
        ]
        for root in possible_roots:
            if os.path.exists(os.path.join(root, 'models')):
                project_root = root
                break
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

PROJECT_ROOT = setup_project_path()
print(f"Project root: {PROJECT_ROOT}")

from models.arcface.arcface_model import ArcFaceModel, freeze_layers, load_pretrained_backbone
from models.arcface.arcface_dataloader import create_dataloaders, create_folder_dataloaders, visualize_batch, benchmark_dataloader


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: tron 2 anh va labels de tao data ao"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: ket hop loss cua 2 labels theo ti le lam"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_embedding_metrics(embeddings, labels, num_pairs=5000):
    """
    Tinh embedding quality metrics cho validation (open-set evaluation)
    
    Khi train/val co cac identity khac nhau, classification accuracy khong co y nghia.
    Thay vao do, danh gia chat luong embedding thong qua:
    - Intra-class distance: khoang cach giua embeddings cung identity (can nho)
    - Inter-class distance: khoang cach giua embeddings khac identity (can lon)
    - Silhouette score: do phan tach clustering (-1 den 1, cang cao cang tot)
    - Verification accuracy: do chinh xac voi pairs same/different
    
    Args:
        embeddings: numpy array (N, embedding_dim)
        labels: numpy array (N,)
        num_pairs: so cap de tinh verification accuracy
        
    Returns:
        dict voi cac metrics
    """
    from scipy.spatial.distance import cosine
    
    metrics = {}
    unique_labels = np.unique(labels)
    
    # 1. Intra-class distance (cung identity - can nho)
    intra_dists = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            class_embeds = embeddings[mask]
            n = min(len(class_embeds), 20)
            for i in range(n):
                for j in range(i+1, n):
                    dist = cosine(class_embeds[i], class_embeds[j])
                    intra_dists.append(dist)
    metrics['intra_class_dist'] = np.mean(intra_dists) if intra_dists else 0
    
    # 2. Inter-class distance (khac identity - can lon)
    inter_dists = []
    sample_size = min(num_pairs, len(labels) * 2)
    for _ in range(sample_size):
        idx1, idx2 = np.random.choice(len(labels), 2, replace=False)
        if labels[idx1] != labels[idx2]:
            dist = cosine(embeddings[idx1], embeddings[idx2])
            inter_dists.append(dist)
    metrics['inter_class_dist'] = np.mean(inter_dists) if inter_dists else 0
    
    # 3. Silhouette score
    try:
        from sklearn.metrics import silhouette_score
        if len(unique_labels) > 1:
            if len(labels) > 3000:
                indices = np.random.choice(len(labels), 3000, replace=False)
                metrics['silhouette'] = silhouette_score(
                    embeddings[indices], labels[indices], metric='cosine'
                )
            else:
                metrics['silhouette'] = silhouette_score(
                    embeddings, labels, metric='cosine'
                )
        else:
            metrics['silhouette'] = 0
    except Exception:
        metrics['silhouette'] = 0
    
    # 4. Verification accuracy
    positive_pairs = []
    negative_pairs = []
    
    for _ in range(num_pairs):
        # Positive pair (cung identity)
        label = np.random.choice(unique_labels)
        mask = labels == label
        if mask.sum() >= 2:
            indices = np.where(mask)[0]
            i, j = np.random.choice(indices, 2, replace=False)
            dist = cosine(embeddings[i], embeddings[j])
            positive_pairs.append(dist)
        
        # Negative pair (khac identity)
        idx1, idx2 = np.random.choice(len(labels), 2, replace=False)
        if labels[idx1] != labels[idx2]:
            dist = cosine(embeddings[idx1], embeddings[idx2])
            negative_pairs.append(dist)
    
    # Tim threshold toi uu
    if positive_pairs and negative_pairs:
        all_dists = positive_pairs + negative_pairs
        all_binary = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        
        best_acc = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 1.0, 0.02):
            predictions = [1 if d < threshold else 0 for d in all_dists]
            acc = sum(p == l for p, l in zip(predictions, all_binary)) / len(all_binary)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        metrics['verification_acc'] = best_acc * 100
        metrics['best_threshold'] = best_threshold
    else:
        metrics['verification_acc'] = 50.0
        metrics['best_threshold'] = 0.5
    
    return metrics


class EarlyStopping:
    """
    Early Stopping ket hop voi Learning Rate Scheduling
    
    Features:
    - Dung training khi metric khong cai thien sau patience epochs
    - Ho tro min_delta de tranh dung som vi nhung thay doi nho
    - Co the theo doi loss hoac accuracy
    """
    def __init__(self, patience=10, min_delta=0.001, mode='max', verbose=True):
        """
        Args:
            patience: So epochs cho phep khong cai thien
            min_delta: Nguong toi thieu de coi la cai thien
            mode: 'min' cho loss, 'max' cho accuracy
            verbose: In thong bao
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        """
        Kiem tra xem co nen dung training khong
        
        Args:
            score: Metric hien tai (val_loss hoac val_acc)
            epoch: Epoch hien tai
            
        Returns:
            True neu can dung training
        """
        if self.mode == 'max':
            improved = self.best_score is None or score > (self.best_score + self.min_delta)
        else:
            improved = self.best_score is None or score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} (best: {self.best_score:.4f} @ epoch {self.best_epoch+1})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ArcFaceTrainer:
    """
    Class chinh de quan ly qua trinh training
    """
    def __init__(self, config_path, pretrained_path=None, data_dir=None, checkpoint_dir=None):
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = pretrained_path
        
        print(f"Su dung device: {self.device}")
        
        # Override config neu co data_dir
        if data_dir:
            # Kiem tra cau truc thu muc de tu dong detect
            # Truong hop 1: data_dir chua truc tiep train/, val/, metadata/
            # Truong hop 2: data_dir chua CelebA_Aligned_Balanced/train/, ...
            if os.path.exists(os.path.join(data_dir, 'train')):
                base_dir = data_dir
            elif os.path.exists(os.path.join(data_dir, 'CelebA_Aligned_Balanced', 'train')):
                base_dir = os.path.join(data_dir, 'CelebA_Aligned_Balanced')
            else:
                base_dir = data_dir
                print(f"[WARN] Khong tim thay thu muc 'train' trong {data_dir}")
            
            self.config['data']['train_csv'] = os.path.join(base_dir, 'metadata', 'train_labels.csv')
            self.config['data']['val_csv'] = os.path.join(base_dir, 'metadata', 'val_labels.csv')
            self.config['data']['train_data_root'] = os.path.join(base_dir, 'train')
            self.config['data']['val_data_root'] = os.path.join(base_dir, 'val')
            
            print(f"Data directory: {base_dir}")
        
        if checkpoint_dir:
            self.config['checkpoint']['save_dir'] = checkpoint_dir
        
        # Khoi tao cac thanh phan
        self.setup_dirs()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_early_stopping()
        self.setup_logging()
        self.setup_mixed_precision()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Training history for plotting
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'verification_acc': [],
            'silhouette': [],
            'intra_class_dist': [],
            'inter_class_dist': [],
            'learning_rate': [],
            'epoch': []
        }
    
    def load_config(self, config_path):
        """Load config tu YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_dirs(self):
        """Tao cac thu muc can thiet"""
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Log dir: {self.log_dir}")
    
    def setup_data(self):
        """Khoi tao DataLoaders - ho tro ca mode 'csv' va 'folder'"""
        print("\n=== Setup Data ===")
        
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        image_size = self.config['data']['image_size']
        data_mode = self.config['data'].get('mode', 'csv')
        
        train_data_root = self.config['data'].get('train_data_root', None)
        val_data_root = self.config['data'].get('val_data_root', None)
        
        print(f"Data mode: {data_mode}")
        
        if data_mode == 'folder':
            min_images = self.config['data'].get('min_images_per_identity', 5)
            class_balanced = self.config['data'].get('class_balanced_sampling', True)
            augment_strength = self.config['data'].get('augment_strength', 'normal')
            
            self.train_loader, self.val_loader, self.num_classes, self.class_weights = create_folder_dataloaders(
                train_root=train_data_root,
                val_root=val_data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size,
                use_albumentations=True,  # Bat albumentations de dung strong/heavy augmentation
                min_images_per_identity=min_images,
                class_balanced_sampling=class_balanced,
                augment_strength=augment_strength
            )
        else:
            train_csv = self.config['data']['train_csv']
            val_csv = self.config['data']['val_csv']
            
            self.train_loader, self.val_loader, self.num_classes = create_dataloaders(
                train_csv=train_csv,
                val_csv=val_csv,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size,
                use_albumentations=True,  # Bat albumentations
                train_data_root=train_data_root,
                val_data_root=val_data_root
            )
            self.class_weights = None
        
        print(f"So classes: {self.num_classes}")
    
    def setup_model(self):
        """Khoi tao model"""
        print("\n=== Setup Model ===")
        
        self.model = ArcFaceModel(
            num_classes=self.num_classes,
            embedding_size=self.config['model']['embedding_size'],
            pretrained=self.config['model']['pretrained'],
            scale=self.config['arcface']['scale'],
            margin=self.config['arcface']['margin'],
            easy_margin=self.config['arcface']['easy_margin']
        )
        
        # Load pretrained backbone neu co
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            load_pretrained_backbone(self.model, self.pretrained_path)
        
        # Freeze layers
        if self.config['model']['freeze_ratio'] > 0:
            self.model = freeze_layers(self.model, self.config['model']['freeze_ratio'])
        
        self.model = self.model.to(self.device)
        
        # Loss function with Label Smoothing
        label_smoothing = self.config['training'].get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            print(f"Label Smoothing: {label_smoothing}")
        
        # Mixup config
        mixup_config = self.config['training'].get('mixup', {})
        self.use_mixup = mixup_config.get('enabled', False)
        self.mixup_alpha = mixup_config.get('alpha', 0.2)
        if self.use_mixup:
            print(f"Mixup: enabled (alpha={self.mixup_alpha})")
        
        # Tinh tong parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Tong parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def setup_optimizer(self):
        """Khoi tao optimizer va scheduler"""
        print("\n=== Setup Optimizer ===")
        
        opt_config = self.config['training']['optimizer']
        self.target_lr = opt_config['lr']
        
        # Warmup config
        warmup_config = self.config['training'].get('warmup', {})
        self.warmup_enabled = warmup_config.get('enabled', False)
        self.warmup_epochs = warmup_config.get('epochs', 5)
        self.warmup_start_lr = warmup_config.get('start_lr', 0.0001)
        
        # Set initial LR (warmup start or target)
        initial_lr = self.warmup_start_lr if self.warmup_enabled else self.target_lr
        
        # Optimizer
        if opt_config['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=initial_lr,
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=initial_lr,
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=initial_lr,
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        
        # Learning rate scheduler
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type'].lower()
        
        if sched_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
            self.scheduler_type = 'epoch'
        elif sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'] - self.warmup_epochs if self.warmup_enabled else self.config['training']['num_epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
            self.scheduler_type = 'epoch'
        elif sched_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.get('mode', 'min'),
                factor=sched_config.get('factor', 0.1),
                patience=sched_config.get('patience', 5),
                min_lr=sched_config.get('min_lr', 1e-7),
                verbose=True
            )
            self.scheduler_type = 'metric'
        else:
            self.scheduler = None
            self.scheduler_type = None
        
        print(f"Optimizer: {opt_config['type']}")
        print(f"Target learning rate: {self.target_lr}")
        if self.warmup_enabled:
            print(f"Warmup: {self.warmup_epochs} epochs ({self.warmup_start_lr} -> {self.target_lr})")
        print(f"Scheduler: {sched_type}")
    
    def setup_early_stopping(self):
        """Khoi tao Early Stopping"""
        es_config = self.config['training']['early_stopping']
        
        if es_config['enabled']:
            # mode: 'max' cho accuracy, 'min' cho loss
            es_mode = es_config.get('mode', 'min')  # Mac dinh theo doi loss
            self.early_stopping = EarlyStopping(
                patience=es_config['patience'],
                min_delta=es_config.get('min_delta', 0.001),
                mode=es_mode,
                verbose=True
            )
            self.es_mode = es_mode
            metric_name = 'loss' if es_mode == 'min' else 'accuracy'
            print(f"\nEarly Stopping: patience={es_config['patience']}, min_delta={es_config.get('min_delta', 0.001)}, metric={metric_name}")
        else:
            self.early_stopping = None
            self.es_mode = None
    
    def setup_logging(self):
        """Khoi tao TensorBoard logging"""
        if self.config['logging']['use_tensorboard'] and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir)
            print(f"\nTensorBoard logging enabled: {self.log_dir}")
        else:
            self.writer = None
            if self.config['logging']['use_tensorboard'] and not TENSORBOARD_AVAILABLE:
                print("\nTensorBoard requested but not available - logging disabled")
    
    def setup_mixed_precision(self):
        """Khoi tao Mixed Precision Training (FP16)"""
        mp_config = self.config.get('mixed_precision', {})
        self.use_amp = mp_config.get('enabled', False) and torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            print(f"\nMixed Precision (FP16): ENABLED - Tang toc 2-3x")
        else:
            self.scaler = None
            if mp_config.get('enabled', False):
                print(f"\nMixed Precision: DISABLED (CUDA khong kha dung)")
            else:
                print(f"\nMixed Precision: DISABLED")
    
    def get_lr(self):
        """Lay learning rate hien tai"""
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self):
        """Train mot epoch (ho tro Mixed Precision)"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply Mixup if enabled
            if self.use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            
            # Forward pass voi Mixed Precision
            if self.use_amp:
                with autocast('cuda'):
                    # ArcFace can label de tinh margin - dung labels_a
                    outputs, embeddings = self.model(images, labels_a)
                    if self.use_mixup:
                        loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = self.criterion(outputs, labels)
                
                # Backward pass voi GradScaler
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip']['enabled']:
                    self.scaler.unscale_(self.optimizer)
                    max_norm = self.config['training']['grad_clip']['max_norm']
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass binh thuong
                outputs, embeddings = self.model(images, labels_a)
                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip']['enabled']:
                    max_norm = self.config['training']['grad_clip']['max_norm']
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Logging
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/Loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy_step', 100.*correct/total, self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'lr': f"{self.get_lr():.2e}"
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validation su dung embedding-based metrics (open-set evaluation)
        
        Khi train/val co cac identity khac nhau, classification accuracy khong co y nghia.
        Thay vao do, danh gia chat luong embedding thong qua:
        - Intra-class distance: khoang cach giua embeddings cung identity
        - Inter-class distance: khoang cach giua embeddings khac identity  
        - Silhouette score: do phan tach clustering
        - Verification accuracy: do chinh xac voi pairs same/different
        """
        self.model.eval()
        
        running_loss = 0.0
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with autocast('cuda'):
                        outputs, embeddings = self.model(images, labels)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs, embeddings = self.model(images, labels)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                all_embeddings.append(embeddings.float().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Tinh embedding-based metrics
        embedding_metrics = compute_embedding_metrics(all_embeddings, all_labels)
        
        return val_loss, embedding_metrics, all_embeddings, all_labels
    
    def visualize_embeddings(self, embeddings, labels, save_path):
        """Visualize embeddings voi t-SNE"""
        if not TSNE_AVAILABLE or not PLT_AVAILABLE:
            print("Skipping t-SNE visualization (sklearn/matplotlib not available)")
            return
        
        print("Tao t-SNE visualization...")
        
        max_samples = self.config['logging']['embedding_vis']['num_samples']
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='tab20', alpha=0.6, s=10)
        plt.colorbar(scatter)
        plt.title(f'Embedding Space - Epoch {self.current_epoch+1}')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved t-SNE visualization: {save_path}")
    
    def save_checkpoint(self, val_acc, val_loss, is_best=False):
        """Luu checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'num_classes': self.num_classes,
            'global_step': self.global_step,
            'warmup_enabled': getattr(self, 'warmup_enabled', False),
            'warmup_epochs': getattr(self, 'warmup_epochs', 0),
            'target_lr': getattr(self, 'target_lr', self.config['training']['optimizer']['lr']),
            'use_amp': self.use_amp,
            'history': self.history
        }
        
        if is_best:
            save_path = self.checkpoint_dir / 'arcface_best.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved best model: {save_path}")
        
        # Luon luu checkpoint moi nhat (arcface_last.pth) de resume
        last_path = self.checkpoint_dir / 'arcface_last.pth'
        torch.save(checkpoint, last_path)
        
        # Luu checkpoint dinh ky
        save_interval = self.config['checkpoint'].get('save_interval', 1)
        if (self.current_epoch + 1) % save_interval == 0:
            save_path = self.checkpoint_dir / f'arcface_epoch_{self.current_epoch+1}.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")
            
            # Xoa checkpoint cu neu can
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Xoa cac checkpoint cu, chi giu lai N checkpoints gan nhat"""
        keep_last_n = self.config['checkpoint'].get('keep_last_n', 5)
        
        checkpoint_files = sorted(
            self.checkpoint_dir.glob('arcface_epoch_*.pth'),
            key=lambda x: int(x.stem.split('_')[-1]),
            reverse=True
        )
        
        for old_ckpt in checkpoint_files[keep_last_n:]:
            old_ckpt.unlink()
            print(f"Removed old checkpoint: {old_ckpt.name}")
    
    def save_training_history(self):
        """Luu training history ra file JSON de download va phan tich"""
        import json
        
        def convert_to_native(obj):
            """Convert numpy types sang Python native types de JSON serialize"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        history_path = self.checkpoint_dir / 'training_history.json'
        
        history_data = {
            'history': convert_to_native(self.history),
            'best_val_acc': convert_to_native(self.best_val_acc),
            'best_val_loss': convert_to_native(self.best_val_loss),
            'total_epochs': self.current_epoch + 1,
            'num_classes': self.num_classes,
            'config': {
                'model': self.config.get('model', {}),
                'training': self.config.get('training', {}),
                'arcface': self.config.get('arcface', {})
            }
        }
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved training history: {history_path}")
    
    def adjust_warmup_lr(self, epoch):
        """Dieu chinh LR trong giai doan warmup"""
        if not self.warmup_enabled or epoch >= self.warmup_epochs:
            return False
        
        # Linear warmup
        warmup_factor = (epoch + 1) / self.warmup_epochs
        new_lr = self.warmup_start_lr + (self.target_lr - self.warmup_start_lr) * warmup_factor
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return True
    
    def train(self):
        """Main training loop voi Early Stopping + LR Scheduling + Warmup"""
        print("\n" + "="*60)
        print("BAT DAU TRAINING")
        print("="*60)
        
        num_epochs = self.config['training']['num_epochs']
        start_time = time.time()
        start_epoch = self.current_epoch  # Resume tu epoch da luu
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Warmup LR adjustment
            is_warmup = self.adjust_warmup_lr(epoch)
            current_lr = self.get_lr()
            
            warmup_str = " [WARMUP]" if is_warmup else ""
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}{warmup_str}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate (su dung embedding-based metrics)
            val_loss, embedding_metrics, embeddings, labels = self.validate()
            
            # Lay cac metrics tu embedding_metrics
            verification_acc = embedding_metrics.get('verification_acc', 0)
            silhouette = embedding_metrics.get('silhouette', 0)
            intra_dist = embedding_metrics.get('intra_class_dist', 0)
            inter_dist = embedding_metrics.get('inter_class_dist', 0)
            best_threshold = embedding_metrics.get('best_threshold', 0.5)
            
            # Update scheduler (chi khi khong con warmup)
            if self.scheduler and not is_warmup:
                if self.scheduler_type == 'metric':
                    # ReduceLROnPlateau - can metric (dung verification_acc)
                    self.scheduler.step(-verification_acc)  # negative vi scheduler mode='min'
                else:
                    # StepLR, CosineAnnealingLR - theo epoch
                    self.scheduler.step()
            
            # Logging
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Embedding Metrics:")
            print(f"  Verification Acc: {verification_acc:.2f}% (threshold: {best_threshold:.3f})")
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Intra-class Dist: {intra_dist:.4f} | Inter-class Dist: {inter_dist:.4f}")
            print(f"Learning Rate: {self.get_lr():.2e}")
            
            # Save to history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['verification_acc'].append(verification_acc)
            self.history['silhouette'].append(silhouette)
            self.history['intra_class_dist'].append(intra_dist)
            self.history['inter_class_dist'].append(inter_dist)
            self.history['learning_rate'].append(self.get_lr())
            
            # Save history to JSON file
            self.save_training_history()
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Verification_Acc', verification_acc, epoch)
                self.writer.add_scalar('Val/Silhouette', silhouette, epoch)
                self.writer.add_scalar('Val/Intra_Dist', intra_dist, epoch)
                self.writer.add_scalar('Val/Inter_Dist', inter_dist, epoch)
                self.writer.add_scalar('Learning_Rate', self.get_lr(), epoch)
            
            # Visualize embeddings
            if self.config['logging']['embedding_vis']['enabled']:
                if (epoch + 1) % self.config['logging']['embedding_vis']['interval'] == 0:
                    vis_path = self.log_dir / f'embeddings_epoch_{epoch+1}.png'
                    self.visualize_embeddings(embeddings, labels, vis_path)
            
            # Check best model (dung verification_acc thay vi classification acc)
            is_best = verification_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = verification_acc
                self.best_val_loss = val_loss
                print(f"New best model! Verification Acc: {verification_acc:.2f}%")
            
            self.save_checkpoint(verification_acc, val_loss, is_best)
            
            # Early Stopping check - dung verification_acc
            if self.early_stopping:
                # Luon dung verification_acc cho open-set evaluation
                es_metric = verification_acc
                # Early stopping theo max vi verification_acc cang cao cang tot
                if self.es_mode == 'min':
                    es_metric = -verification_acc  # convert sang min mode
                if self.early_stopping(es_metric, epoch):
                    print(f"\n{'='*60}")
                    print(f"EARLY STOPPING!")
                    print(f"Best epoch: {self.early_stopping.best_epoch + 1}")
                    print(f"Best verification accuracy: {self.best_val_acc:.2f}%")
                    print(f"Best val loss: {self.best_val_loss:.4f}")
                    print(f"{'='*60}")
                    break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING HOAN TAT")
        print("="*60)
        print(f"Thoi gian training: {total_time/3600:.2f} gio")
        print(f"Best verification accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final learning rate: {self.get_lr():.2e}")
        
        if self.writer:
            self.writer.close()
        
        return self.best_val_acc
    
    def resume_training(self, checkpoint_path, reset_optimizer=False):
        """Resume training tu checkpoint
        
        Args:
            checkpoint_path: Duong dan den file checkpoint
            reset_optimizer: Neu True, khong load optimizer/scheduler state ma dung LR moi tu config
        """
        import json
        
        print(f"\nResume training tu: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if reset_optimizer:
            print("[RESET] Khong load optimizer/scheduler state - dung LR moi tu config")
            new_lr = self.config['training']['optimizer']['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"[RESET] Set LR = {new_lr}")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict') and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore GradScaler state cho Mixed Precision
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        # Restore history - uu tien tu file JSON, sau do tu checkpoint
        history_loaded = False
        history_path = self.checkpoint_dir / 'training_history.json'
        
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                self.history = history_data.get('history', self.history)
                history_loaded = True
                print(f"Restored training history from JSON ({len(self.history['epoch'])} epochs)")
            except Exception as e:
                print(f"Warning: Could not load history from JSON: {e}")
        
        if not history_loaded and 'history' in checkpoint:
            self.history = checkpoint['history']
            print(f"Restored training history from checkpoint ({len(self.history['epoch'])} epochs)")
        
        # Restore warmup state neu can (chi khi khong reset optimizer)
        if not reset_optimizer and checkpoint.get('warmup_enabled'):
            self.warmup_enabled = checkpoint['warmup_enabled']
            self.warmup_epochs = checkpoint.get('warmup_epochs', 5)
            self.target_lr = checkpoint.get('target_lr', self.config['training']['optimizer']['lr'])
        
        # Auto-extend num_epochs neu da train het
        num_epochs = self.config['training']['num_epochs']
        if self.current_epoch >= num_epochs:
            new_num_epochs = self.current_epoch + 50
            print(f"\n[AUTO-EXTEND] current_epoch ({self.current_epoch}) >= num_epochs ({num_epochs})")
            print(f"[AUTO-EXTEND] Tang num_epochs tu {num_epochs} len {new_num_epochs}")
            self.config['training']['num_epochs'] = new_num_epochs
        
        print(f"Resumed from epoch {self.current_epoch}")
        print(f"Best val acc so far: {self.best_val_acc:.2f}%")
        print(f"Global step: {self.global_step}")
        print(f"Current LR: {self.get_lr():.2e}")
        print(f"Mixed Precision: {'ENABLED' if self.use_amp else 'DISABLED'}")
        
        # Kiem tra warmup status
        if self.current_epoch < self.warmup_epochs:
            print(f"Still in warmup phase ({self.current_epoch}/{self.warmup_epochs})")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ArcFace Model')
    
    parser.add_argument('--config', type=str, default='configs/arcface_config.yaml',
                       help='Duong dan den file config')
    parser.add_argument('--pretrained_backbone', type=str, default=None,
                       help='Duong dan den pretrained backbone weights')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Thu muc chua du lieu')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Thu muc luu checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume tu checkpoint')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer va scheduler khi resume (dung LR moi tu config)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("="*60)
    print("ARCFACE TRAINING")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Pretrained backbone: {args.pretrained_backbone}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Resume: {args.resume}")
    print(f"Reset optimizer: {args.reset_optimizer}")
    print("="*60)
    
    # Khoi tao trainer
    trainer = ArcFaceTrainer(
        config_path=args.config,
        pretrained_path=args.pretrained_backbone,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume neu co
    if args.resume and os.path.exists(args.resume):
        trainer.resume_training(args.resume, reset_optimizer=args.reset_optimizer)
    
    # Bat dau training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining bi ngat boi user")
        print(f"Best validation accuracy dat duoc: {trainer.best_val_acc:.2f}%")
    except Exception as e:
        print(f"\nLoi trong qua trinh training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
