"""
ArcFace Training Script
Script chinh de train mo hinh ArcFace
Ho tro: Early Stopping + Learning Rate Scheduling
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.arcface.arcface_model import ArcFaceModel, freeze_layers, load_pretrained_backbone
from models.arcface.arcface_dataloader import create_dataloaders, visualize_batch, benchmark_dataloader


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
        
        # Override config neu co
        if data_dir:
            self.config['data']['train_csv'] = os.path.join(data_dir, 'processed', 'train_metadata.csv')
            self.config['data']['val_csv'] = os.path.join(data_dir, 'processed', 'val_metadata.csv')
        
        if checkpoint_dir:
            self.config['checkpoint']['save_dir'] = checkpoint_dir
        
        # Khoi tao cac thanh phan
        self.setup_dirs()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_early_stopping()
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.global_step = 0
    
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
        """Khoi tao DataLoaders"""
        print("\n=== Setup Data ===")
        
        train_csv = self.config['data']['train_csv']
        val_csv = self.config['data']['val_csv']
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        image_size = self.config['data']['image_size']
        
        train_data_root = self.config['data'].get('train_data_root', None)
        val_data_root = self.config['data'].get('val_data_root', None)
        
        self.train_loader, self.val_loader, self.num_classes = create_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            use_albumentations=False,
            train_data_root=train_data_root,
            val_data_root=val_data_root
        )
        
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tinh tong parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Tong parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def setup_optimizer(self):
        """Khoi tao optimizer va scheduler"""
        print("\n=== Setup Optimizer ===")
        
        opt_config = self.config['training']['optimizer']
        
        # Optimizer
        if opt_config['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
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
                T_max=self.config['training']['num_epochs'],
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
        print(f"Learning rate: {opt_config['lr']}")
        print(f"Scheduler: {sched_type}")
    
    def setup_early_stopping(self):
        """Khoi tao Early Stopping"""
        es_config = self.config['training']['early_stopping']
        
        if es_config['enabled']:
            self.early_stopping = EarlyStopping(
                patience=es_config['patience'],
                min_delta=es_config.get('min_delta', 0.001),
                mode='max',  # Theo doi accuracy
                verbose=True
            )
            print(f"\nEarly Stopping: patience={es_config['patience']}, min_delta={es_config.get('min_delta', 0.001)}")
        else:
            self.early_stopping = None
    
    def setup_logging(self):
        """Khoi tao TensorBoard logging"""
        if self.config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
            print(f"\nTensorBoard logging enabled: {self.log_dir}")
        else:
            self.writer = None
    
    def get_lr(self):
        """Lay learning rate hien tai"""
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self):
        """Train mot epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, embeddings = self.model(images, labels)
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
        """Validation"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs, embeddings = self.model(images, labels)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        return val_loss, val_acc, all_embeddings, all_labels
    
    def visualize_embeddings(self, embeddings, labels, save_path):
        """Visualize embeddings voi t-SNE"""
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
            'val_acc': val_acc,
            'val_loss': val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'num_classes': self.num_classes
        }
        
        if is_best:
            save_path = self.checkpoint_dir / 'arcface_best.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved best model: {save_path}")
        
        # Luu checkpoint dinh ky
        if (self.current_epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
            save_path = self.checkpoint_dir / f'arcface_epoch_{self.current_epoch+1}.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")
    
    def train(self):
        """Main training loop voi Early Stopping + LR Scheduling"""
        print("\n" + "="*60)
        print("BAT DAU TRAINING")
        print("="*60)
        
        num_epochs = self.config['training']['num_epochs']
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            current_lr = self.get_lr()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, embeddings, labels = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if self.scheduler_type == 'metric':
                    # ReduceLROnPlateau - can metric
                    self.scheduler.step(val_loss)
                else:
                    # StepLR, CosineAnnealingLR - theo epoch
                    self.scheduler.step()
            
            # Logging
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.get_lr():.2e}")
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', self.get_lr(), epoch)
            
            # Visualize embeddings
            if self.config['logging']['embedding_vis']['enabled']:
                if (epoch + 1) % self.config['logging']['embedding_vis']['interval'] == 0:
                    vis_path = self.log_dir / f'embeddings_epoch_{epoch+1}.png'
                    self.visualize_embeddings(embeddings, labels, vis_path)
            
            # Check best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                print(f"New best model! Val Acc: {val_acc:.2f}%")
            
            self.save_checkpoint(val_acc, val_loss, is_best)
            
            # Early Stopping check
            if self.early_stopping:
                if self.early_stopping(val_acc, epoch):
                    print(f"\n{'='*60}")
                    print(f"EARLY STOPPING!")
                    print(f"Best epoch: {self.early_stopping.best_epoch + 1}")
                    print(f"Best val accuracy: {self.best_val_acc:.2f}%")
                    print(f"{'='*60}")
                    break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING HOAN TAT")
        print("="*60)
        print(f"Thoi gian training: {total_time/3600:.2f} gio")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final learning rate: {self.get_lr():.2e}")
        
        if self.writer:
            self.writer.close()
        
        return self.best_val_acc
    
    def resume_training(self, checkpoint_path):
        """Resume training tu checkpoint"""
        print(f"\nResume training tu: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Resumed from epoch {self.current_epoch}")
        print(f"Best val acc so far: {self.best_val_acc:.2f}%")


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
        trainer.resume_training(args.resume)
    
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
