"""
Script Resume Training ArcFace - Fix Overfitting
Script nay se:
1. Load checkpoint hien tai (100 epochs, val_acc 77.5%)
2. Reset optimizer state de escape local minimum
3. Ap dung config moi voi regularization manh hon
4. Tiep tuc training them 30 epochs

Usage:
    python resume_arcface_fix_overfit.py --config configs/arcface_resume_fix_overfit.yaml
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.arcface.train_arcface import ArcFaceTrainer

def main():
    parser = argparse.ArgumentParser(description='Resume ArcFace Training - Fix Overfitting')
    parser.add_argument('--config', type=str, 
                       default='configs/arcface_resume_fix_overfit.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       default='/kaggle/working/checkpoints/arcface/arcface_best.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, 
                       default='/kaggle/input/celeba-aligned-balanced',
                       help='Path to data directory')
    parser.add_argument('--reset_optimizer', action='store_true',
                       help='Reset optimizer state (recommended for escaping local minimum)')
    parser.add_argument('--reset_scheduler', action='store_true',
                       help='Reset scheduler state')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RESUME ARCFACE TRAINING - FIX OVERFITTING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Reset optimizer: {args.reset_optimizer}")
    print(f"Reset scheduler: {args.reset_scheduler}")
    print("=" * 80)
    
    # Kiem tra checkpoint ton tai
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        print("Please make sure checkpoint exists before running this script.")
        return
    
    # Khoi tao trainer voi config moi
    print("\nInitializing trainer with new config...")
    trainer = ArcFaceTrainer(
        config_path=args.config,
        data_dir=args.data_dir
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    
    # Load model weights
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded model weights")
    
    # Load optimizer state (neu khong reset)
    if not args.reset_optimizer:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Loaded optimizer state")
    else:
        print("✓ Reset optimizer state (new learning rate will be applied)")
    
    # Load scheduler state (neu khong reset)
    if not args.reset_scheduler and checkpoint.get('scheduler_state_dict') and trainer.scheduler:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Loaded scheduler state")
    else:
        print("✓ Reset scheduler state (new schedule will be applied)")
    
    # Restore training state
    trainer.current_epoch = checkpoint['epoch'] + 1
    trainer.best_val_acc = checkpoint['best_val_acc']
    trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    trainer.global_step = checkpoint.get('global_step', 0)
    
    # Load training history
    import json
    history_path = Path(trainer.checkpoint_dir) / 'training_history.json'
    if history_path.exists():
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            trainer.history = history_data.get('history', trainer.history)
            print(f"✓ Loaded training history ({len(trainer.history['epoch'])} epochs)")
        except Exception as e:
            print(f"Warning: Could not load history: {e}")
    
    # Print status
    print("\n" + "=" * 80)
    print("TRAINING STATUS")
    print("=" * 80)
    print(f"Resume from epoch: {trainer.current_epoch}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Global step: {trainer.global_step}")
    print(f"Current learning rate: {trainer.get_lr():.2e}")
    print("=" * 80)
    
    # Start training
    print("\nStarting training with new config...")
    print(f"Will train until epoch {trainer.config['training']['num_epochs']}")
    print("\nNew config highlights:")
    print(f"  - Margin: {trainer.config['arcface']['margin']}")
    print(f"  - Scale: {trainer.config['arcface']['scale']}")
    print(f"  - Learning rate: {trainer.config['training']['optimizer']['lr']}")
    print(f"  - Weight decay: {trainer.config['training']['optimizer']['weight_decay']}")
    print(f"  - Label smoothing: {trainer.config['training'].get('label_smoothing', 0.0)}")
    print("=" * 80)
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Final best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Final best validation loss: {trainer.best_val_loss:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
