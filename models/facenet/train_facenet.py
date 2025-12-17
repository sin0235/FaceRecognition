"""
FaceNet Training Script

Script này train FaceNet model với Triplet Loss, sử dụng config từ facenet_config.yaml.
Bao gồm: validation loop, learning rate scheduler, early stopping, checkpoint saving.
"""

import os
import sys
import yaml
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from models.facenet.facenet_model import FaceNetModel, TripletLoss
from models.facenet.facenet_dataloader import FaceNetTripletDataset


def load_config(config_path: str) -> dict:
    """Load config từ YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_triplet_metrics(anchor_emb, positive_emb, negative_emb):
    """
    Tính metrics dựa trên triplet constraint:
    d(anchor, positive) < d(anchor, negative)
    
    Returns:
        accuracy: tỷ lệ triplet thỏa mãn constraint
        avg_pos_dist: khoảng cách trung bình anchor-positive
        avg_neg_dist: khoảng cách trung bình anchor-negative
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train một epoch."""
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


def validate(model, loader, criterion, device, epoch):
    """Validate trên validation set."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pos_dist = 0.0
    total_neg_dist = 0.0
    num_batches = 0

    epoch_start = time.time()
    
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

    epoch_time = time.time() - epoch_start
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_acc / num_batches,
        'pos_dist': total_pos_dist / num_batches,
        'neg_dist': total_neg_dist / num_batches,
        'time': epoch_time
    }


def main():
    print("=" * 60)
    print("FACENET TRAINING")
    print("=" * 60)
    
    config_path = os.path.join(ROOT_DIR, "configs/facenet_config.yaml")
    config = load_config(config_path)
    print(f"Config: {config_path}")

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

    train_dir = os.path.join(ROOT_DIR, data_cfg['train_data_root'])
    val_dir = os.path.join(ROOT_DIR, data_cfg['val_data_root'])

    image_size = data_cfg.get('image_size', 160)
    num_workers = train_cfg.get('num_workers', 4)

    print(f"\n=== Dataset ===")
    print(f"Train: {train_dir}")
    print(f"Val: {val_dir}")
    print(f"Image size: {image_size}x{image_size}")
    
    train_dataset = FaceNetTripletDataset(
        root_dir=train_dir,
        image_size=image_size,
        augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
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
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train identities: {len(train_dataset.identities)}")
    print(f"Val identities: {len(val_dataset.identities)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

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

    criterion = TripletLoss(margin=model_cfg['margin'])
    print(f"Triplet margin: {model_cfg['margin']}")

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
    warmup_epochs = train_cfg.get('warmup_epochs', 0)
    
    best_val_acc = 0.0
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
        'gpu_memory_mb': []
    }

    print(f"\nEpochs: {num_epochs}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Patience: {patience}")
    print(f"Warmup epochs: {warmup_epochs}")
    print("=" * 60)

    training_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_metrics = validate(
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
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"        | Pos Dist: {val_metrics['pos_dist']:.4f} | Neg Dist: {val_metrics['neg_dist']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0

            best_path = os.path.join(checkpoint_dir, "facenet_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'config': config
            }, best_path)
            print(f"  [SAVED] Best model (val_acc: {val_metrics['accuracy']:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[EARLY STOPPING] Triggered at epoch {epoch}")
            break

        scheduler.step()

    total_time = time.time() - training_start
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final epoch: {epoch}")

    history['total_time_minutes'] = total_time / 60
    history['best_val_acc'] = best_val_acc
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
        'config': config
    }, last_path)
    print(f"Last checkpoint: {last_path}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'facenet_best.pth')}")


if __name__ == "__main__":
    main()
