"""
ArcFace Training Script
Script chính để train mô hình ArcFace
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
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Import các module của project
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.arcface.arcface_model import ArcFaceModel, freeze_layers, load_pretrained_backbone
from models.arcface.arcface_dataloader import create_dataloaders, visualize_batch, benchmark_dataloader


class ArcFaceTrainer:
    """
    Class chính để quản lý quá trình training
    """
    def __init__(self, config_path, pretrained_path=None, data_dir=None, checkpoint_dir=None):
        self.config = self.load_config(config_path)
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = pretrained_path
        
        print(f"Sử dụng device: {self.device}")
        
        # Override config nếu có
        if data_dir:
            self.config['data']['train_csv'] = os.path.join(data_dir, 'processed', 'train_metadata.csv')
            self.config['data']['val_csv'] = os.path.join(data_dir, 'processed', 'val_metadata.csv')
        
        if checkpoint_dir:
            self.config['checkpoint']['save_dir'] = checkpoint_dir
        
        # Khởi tạo các thành phần
        self.setup_dirs()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.global_step = 0
    
    def load_config(self, config_path):
        """Load config từ YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_dirs(self):
        """Tạo các thư mục cần thiết"""
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Log dir: {self.log_dir}")
    
    def setup_data(self):
        """Khởi tạo DataLoaders"""
        print("\n=== Setup Data ===")
        
        train_csv = self.config['data']['train_csv']
        val_csv = self.config['data']['val_csv']
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training']['num_workers']
        image_size = self.config['data']['image_size']
        
        self.train_loader, self.val_loader, self.num_classes = create_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            use_albumentations=False
        )
        
        print(f"Số classes: {self.num_classes}")
    
    def setup_model(self):
        """Khởi tạo model"""
        print("\n=== Setup Model ===")
        
        self.model = ArcFaceModel(
            num_classes=self.num_classes,
            embedding_size=self.config['model']['embedding_size'],
            pretrained=self.config['model']['pretrained'],
            scale=self.config['arcface']['scale'],
            margin=self.config['arcface']['margin'],
            easy_margin=self.config['arcface']['easy_margin']
        )
        
        # Load pretrained backbone nếu có
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            load_pretrained_backbone(self.model, self.pretrained_path)
        
        # Freeze layers
        if self.config['model']['freeze_ratio'] > 0:
            self.model = freeze_layers(self.model, self.config['model']['freeze_ratio'])
        
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tính tổng parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Tổng parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def setup_optimizer(self):
        """Khởi tạo optimizer và scheduler"""
        print("\n=== Setup Optimizer ===")
        
        opt_config = self.config['training']['optimizer']
        
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
        
        # Learning rate scheduler
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'].lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['type'].lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        
        print(f"Optimizer: {opt_config['type']}")
        print(f"Learning rate: {opt_config['lr']}")
    
    def setup_logging(self):
        """Khởi tạo TensorBoard logging"""
        if self.config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
            print(f"\nTensorBoard logging enabled: {self.log_dir}")
        else:
            self.writer = None
    
    def train_epoch(self):
        """Train một epoch"""
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
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy', 100.*correct/total, self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
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
        """Visualize embeddings với t-SNE"""
        print("Tạo t-SNE visualization...")
        
        # Lấy subset để tăng tốc
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
    
    def save_checkpoint(self, val_acc, is_best=False):
        """Lưu checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if is_best:
            save_path = self.checkpoint_dir / 'arcface_best.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved best model: {save_path}")
        
        # Lưu checkpoint định kỳ
        if (self.current_epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
            save_path = self.checkpoint_dir / f'arcface_epoch_{self.current_epoch+1}.pth'
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("BẮT ĐẦU TRAINING")
        print("="*50)
        
        num_epochs = self.config['training']['num_epochs']
        early_stop_patience = self.config['training']['early_stopping']['patience']
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, embeddings, labels = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if self.writer:
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            
            # Visualize embeddings
            if self.config['logging']['embedding_vis']['enabled']:
                if (epoch + 1) % self.config['logging']['embedding_vis']['interval'] == 0:
                    vis_path = self.log_dir / f'embeddings_epoch_{epoch+1}.png'
                    self.visualize_embeddings(embeddings, labels, vis_path)
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(val_acc, is_best)
            
            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                if self.patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping sau {epoch+1} epochs")
                    print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                    break
        
        total_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("TRAINING HOÀN TẤT")
        print("="*50)
        print(f"Thời gian training: {total_time/3600:.2f} giờ")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        if self.writer:
            self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ArcFace Model')
    
    parser.add_argument('--config', type=str, default='configs/arcface_config.yaml',
                       help='Đường dẫn đến file config')
    parser.add_argument('--pretrained_backbone', type=str, default=None,
                       help='Đường dẫn đến pretrained backbone weights')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Thư mục chứa dữ liệu')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Thư mục lưu checkpoints')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("="*50)
    print("ARCFACE TRAINING")
    print("="*50)
    print(f"Config: {args.config}")
    print(f"Pretrained backbone: {args.pretrained_backbone}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*50)
    
    # Khởi tạo trainer với các override parameters
    trainer = ArcFaceTrainer(
        config_path=args.config,
        pretrained_path=args.pretrained_backbone,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Bắt đầu training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining bị ngắt bởi user")
        print(f"Best validation accuracy đạt được: {trainer.best_val_acc:.2f}%")
    except Exception as e:
        print(f"\nLỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
