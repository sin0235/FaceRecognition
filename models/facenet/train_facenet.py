"""
FaceNet Training Script với Semi-Hard Negative Mining

Script này train FaceNet model với Triplet Loss và online semi-hard mining.
"""

import os
import sys
import yaml
import time
import json
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from models.facenet.facenet_model import FaceNetModel, TripletLoss
from models.facenet.facenet_dataloader import (
    FaceNetTripletDataset, 
    OnlineTripletDataset,
    mine_semi_hard_triplets,
    mine_batch_hard_triplets,
    create_online_dataloaders
)


def load_config(config_path: str) -> dict:
    """Load config từ YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_triplet_metrics(anchor_emb, positive_emb, negative_emb):
    """
    Tính metrics dựa trên triplet constraint.
    """
    pos_dist = torch.norm(anchor_emb - positive_emb, p=2, dim=1)
    neg_dist = torch.norm(anchor_emb - negative_emb, p=2, dim=1)
    correct = (pos_dist < neg_dist).float().sum()
    accuracy = correct / anchor_emb.size(0)
    
    return {
        'accuracy': accuracy.item(),
        'pos_dist': pos_dist.mean().item(),
        'neg_dist': neg_dist.mean().item()
    }


def get_gpu_memory_mb():
    """Lấy memory GPU đang sử dụng (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def train_one_epoch_online(model, loader, criterion, optimizer, device, epoch, 
                           margin=0.2, mining='semi_hard', grad_clip=None):
    """
    Train một epoch với online triplet mining.
    
    Args:
        model: FaceNet model
        loader: OnlineTripletDataset loader
        criterion: TripletLoss
        optimizer: optimizer
        device: cuda/cpu
        epoch: epoch number
        margin: triplet margin cho mining
        mining: 'semi_hard' hoặc 'hard'
        grad_clip: gradient clipping value
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_pos_dist = 0.0
    total_neg_dist = 0.0
    total_triplets = 0
    num_batches = 0

    epoch_start = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_images, batch_labels in pbar:
        # batch_images: (B, K, 3, H, W) - B identities, K images each
        # batch_labels: (B,) - identity indices
        
        B, K, C, H, W = batch_images.shape
        
        # Flatten: (B*K, 3, H, W)
        images = batch_images.view(B * K, C, H, W).to(device)
        
        # Expand labels: mỗi identity có K ảnh
        labels = batch_labels.unsqueeze(1).expand(B, K).reshape(-1).to(device)
        
        # Forward pass để lấy embeddings
        with torch.no_grad():
            embeddings = model(images)
        
        # Online triplet mining
        if mining == 'semi_hard':
            a_idx, p_idx, n_idx = mine_semi_hard_triplets(embeddings, labels, margin)
        else:
            a_idx, p_idx, n_idx = mine_batch_hard_triplets(embeddings, labels)
        
        if len(a_idx) == 0:
            continue
        
        # Lấy embeddings theo indices (cần forward lại với grad)
        optimizer.zero_grad()
        embeddings = model(images)
        
        anchor_emb = embeddings[a_idx]
        positive_emb = embeddings[p_idx]
        negative_emb = embeddings[n_idx]
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Metrics
        metrics = compute_triplet_metrics(anchor_emb.detach(), positive_emb.detach(), negative_emb.detach())
        
        total_loss += loss.item()
        total_acc += metrics['accuracy']
        total_pos_dist += metrics['pos_dist']
        total_neg_dist += metrics['neg_dist']
        total_triplets += len(a_idx)
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["accuracy"]:.4f}',
            'triplets': len(a_idx),
            'p_d': f'{metrics["pos_dist"]:.2f}',
            'n_d': f'{metrics["neg_dist"]:.2f}'
        })

    epoch_time = time.time() - epoch_start
    
    if num_batches == 0:
        return {'loss': 0, 'accuracy': 0, 'pos_dist': 0, 'neg_dist': 0, 
                'time': epoch_time, 'triplets': 0}
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'pos_dist': total_pos_dist / num_batches,
        'neg_dist': total_neg_dist / num_batches,
        'time': epoch_time,
        'triplets': total_triplets
    }


def train_one_epoch_random(model, loader, criterion, optimizer, device, epoch, grad_clip=None):
    """Train một epoch với random triplet mining (baseline)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_pos_dist = 0.0
    total_neg_dist = 0.0
    num_batches = 0

    epoch_start = time.time()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for anchor, positive, negative in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        metrics = compute_triplet_metrics(emb_a, emb_p, emb_n)

        total_loss += loss.item()
        total_acc += metrics['accuracy']
        total_pos_dist += metrics['pos_dist']
        total_neg_dist += metrics['neg_dist']
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{metrics["accuracy"]:.4f}',
            'p_d': f'{metrics["pos_dist"]:.2f}',
            'n_d': f'{metrics["neg_dist"]:.2f}'
        })

    epoch_time = time.time() - epoch_start
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'pos_dist': total_pos_dist / num_batches,
        'neg_dist': total_neg_dist / num_batches,
        'time': epoch_time
    }


def validate_online(model, loader, criterion, device, epoch, margin=0.2):
    """Validate với online mining."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pos_dist = 0.0
    total_neg_dist = 0.0
    total_triplets = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for batch_images, batch_labels in pbar:
            B, K, C, H, W = batch_images.shape
            images = batch_images.view(B * K, C, H, W).to(device)
            labels = batch_labels.unsqueeze(1).expand(B, K).reshape(-1).to(device)
            
            embeddings = model(images)
            
            a_idx, p_idx, n_idx = mine_semi_hard_triplets(embeddings, labels, margin)
            
            if len(a_idx) == 0:
                continue
            
            anchor_emb = embeddings[a_idx]
            positive_emb = embeddings[p_idx]
            negative_emb = embeddings[n_idx]
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            metrics = compute_triplet_metrics(anchor_emb, positive_emb, negative_emb)
            
            total_loss += loss.item()
            total_acc += metrics['accuracy']
            total_pos_dist += metrics['pos_dist']
            total_neg_dist += metrics['neg_dist']
            total_triplets += len(a_idx)
            num_batches += 1
            
            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_acc': f'{metrics["accuracy"]:.4f}'
            })

    if num_batches == 0:
        return {'loss': 0, 'accuracy': 0, 'pos_dist': 0, 'neg_dist': 0, 'triplets': 0}
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'pos_dist': total_pos_dist / num_batches,
        'neg_dist': total_neg_dist / num_batches,
        'triplets': total_triplets
    }


def validate_random(model, loader, criterion, device, epoch):
    """Validate với random triplets (baseline)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pos_dist = 0.0
    total_neg_dist = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            metrics = compute_triplet_metrics(emb_a, emb_p, emb_n)

            total_loss += loss.item()
            total_acc += metrics['accuracy']
            total_pos_dist += metrics['pos_dist']
            total_neg_dist += metrics['neg_dist']
            num_batches += 1

            pbar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_acc': f'{metrics["accuracy"]:.4f}'
            })

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'pos_dist': total_pos_dist / num_batches,
        'neg_dist': total_neg_dist / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='FaceNet Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--mining', type=str, default='semi_hard', 
                        choices=['random', 'semi_hard', 'hard'],
                        help='Triplet mining strategy')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FACENET TRAINING")
    print("=" * 60)
    
    # Load config
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(ROOT_DIR, "configs/facenet_config.yaml")
    
    config = load_config(config_path)
    print(f"Config: {config_path}")
    print(f"Mining strategy: {args.mining}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_cfg = config['model']
    train_cfg = config['training']
    data_cfg = config['dataset']
    path_cfg = config['path']

    checkpoint_dir = os.path.join(ROOT_DIR, path_cfg['checkpoint_dir'])
    logs_dir = os.path.join(ROOT_DIR, path_cfg['logs_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Data paths
    if args.data_dir:
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
    else:
        train_dir = os.path.join(ROOT_DIR, data_cfg['train_data_root'])
        val_dir = os.path.join(ROOT_DIR, data_cfg['val_data_root'])

    image_size = data_cfg.get('image_size', 160)
    num_workers = train_cfg.get('num_workers', 4)
    batch_size = train_cfg['batch_size']
    margin = model_cfg.get('margin', 0.2)
    images_per_identity = train_cfg.get('images_per_identity', 4)

    print(f"\n=== Dataset ===")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}")
    print(f"Image size: {image_size}x{image_size}")
    
    # Tạo DataLoader dựa trên mining strategy
    if args.mining in ['semi_hard', 'hard']:
        train_loader, val_loader = create_online_dataloaders(
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            images_per_identity=images_per_identity
        )
        use_online_mining = True
    else:
        train_dataset = FaceNetTripletDataset(
            root_dir=train_dir,
            image_size=image_size,
            augment=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_dataset = FaceNetTripletDataset(
            root_dir=val_dir,
            image_size=image_size,
            augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        use_online_mining = False
        print(f"Train identities: {len(train_dataset.identities)}")
        print(f"Val identities: {len(val_dataset.identities)}")

    print(f"\n=== Model ===")
    pretrained = model_cfg.get('pretrained', 'vggface2')
    model = FaceNetModel(
        embedding_size=model_cfg['embedding_size'],
        pretrained=pretrained,
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Pretrained: {pretrained}")
    print(f"Embedding size: {model_cfg['embedding_size']}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    criterion = TripletLoss(margin=margin)
    print(f"Triplet margin: {margin}")

    print(f"\n=== Training Config ===")
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.0001)
    )
    print(f"Optimizer: Adam")
    print(f"Learning rate: {train_cfg['learning_rate']}")
    print(f"Weight decay: {train_cfg.get('weight_decay', 0.0001)}")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg.get('scheduler_step', 10),
        gamma=train_cfg.get('scheduler_gamma', 0.1)
    )
    print(f"Scheduler: StepLR (step={train_cfg.get('scheduler_step', 10)}, gamma={train_cfg.get('scheduler_gamma', 0.1)})")

    num_epochs = train_cfg['num_epochs']
    patience = train_cfg.get('patience', 7)
    grad_clip = train_cfg.get('grad_clip', None)
    
    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_pos_dist': [],
        'train_neg_dist': [],
        'val_pos_dist': [],
        'val_neg_dist': [],
        'lr': [],
        'epoch_time': [],
        'gpu_memory_mb': [],
        'mining_strategy': args.mining
    }

    print(f"\nEpochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Patience: {patience}")
    print(f"Mining: {args.mining}")
    print("=" * 60)

    training_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        if use_online_mining:
            train_metrics = train_one_epoch_online(
                model, train_loader, criterion, optimizer, device, epoch,
                margin=margin, mining=args.mining, grad_clip=grad_clip
            )
        else:
            train_metrics = train_one_epoch_random(
                model, train_loader, criterion, optimizer, device, epoch, grad_clip
            )

        # Validate
        if use_online_mining:
            val_metrics = validate_online(
                model, val_loader, criterion, device, epoch, margin=margin
            )
        else:
            val_metrics = validate_random(
                model, val_loader, criterion, device, epoch
            )

        current_lr = optimizer.param_groups[0]['lr']
        gpu_mem = get_gpu_memory_mb()
        epoch_total_time = time.time() - epoch_start

        history['train_loss'].append(float(train_metrics['loss']))
        history['train_acc'].append(float(train_metrics['accuracy']))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_acc'].append(float(val_metrics['accuracy']))
        history['train_pos_dist'].append(float(train_metrics['pos_dist']))
        history['train_neg_dist'].append(float(train_metrics['neg_dist']))
        history['val_pos_dist'].append(float(val_metrics['pos_dist']))
        history['val_neg_dist'].append(float(val_metrics['neg_dist']))
        history['lr'].append(float(current_lr))
        history['epoch_time'].append(float(epoch_total_time))
        history['gpu_memory_mb'].append(float(gpu_mem))

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_total_time:.1f}s | GPU: {gpu_mem:.0f}MB")
        print(f"-" * 60)
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        print(f"        | Pos Dist: {train_metrics['pos_dist']:.4f} | Neg Dist: {train_metrics['neg_dist']:.4f}")
        if 'triplets' in train_metrics:
            print(f"        | Triplets mined: {train_metrics['triplets']}")
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"        | Pos Dist: {val_metrics['pos_dist']:.4f} | Neg Dist: {val_metrics['neg_dist']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model (dựa trên val_loss thay vì accuracy)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            best_path = os.path.join(checkpoint_dir, "facenet_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'config': config,
                'mining': args.mining
            }, best_path)
            print(f"  [SAVED] Best model (val_loss: {val_metrics['loss']:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}")
            break

        scheduler.step()
        
        # Save history mỗi epoch
        history_path = os.path.join(logs_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - training_start
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final epoch: {epoch}")

    history['total_time_minutes'] = total_time / 60
    history['best_val_loss'] = best_val_loss
    history['final_epoch'] = epoch
    
    history_path = os.path.join(logs_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history: {history_path}")

    last_path = os.path.join(checkpoint_dir, "facenet_last.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'config': config,
        'mining': args.mining
    }, last_path)
    print(f"Last checkpoint: {last_path}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'facenet_best.pth')}")


if __name__ == "__main__":
    main()
