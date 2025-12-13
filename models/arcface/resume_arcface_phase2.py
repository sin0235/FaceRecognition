"""
Script Resume Training cho arcface_config.yaml - Phase 2
Resume từ checkpoint epoch 50 (val_acc 80.55%) với reset optimizer/scheduler
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.arcface.train_arcface import ArcFaceTrainer

def main():
    parser = argparse.ArgumentParser(description='Resume ArcFace Training - Phase 2')
    parser.add_argument('--config', type=str, 
                       default='configs/arcface_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       default='/kaggle/working/checkpoints/arcface/arcface_last.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, 
                       default='/kaggle/input/celeba-aligned-balanced',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RESUME ARCFACE TRAINING - PHASE 2 FINE-TUNE")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print("=" * 80)
    
    # Kiem tra checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Khoi tao trainer voi config moi
    print("\nInitializing trainer with Phase 2 config...")
    trainer = ArcFaceTrainer(
        config_path=args.config,
        data_dir=args.data_dir
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    
    # Load ONLY model weights
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded model weights")
    
    # Reset optimizer (giữ config mới)
    print("✓ Reset optimizer state (Phase 2 LR will be applied)")
    
    # Reset scheduler (giữ config mới)
    print("✓ Reset scheduler state (Phase 2 schedule will be applied)")
    
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
    print(f"Current learning rate: {trainer.get_lr():.2e} (NEW - RESET)")
    print("=" * 80)
    
    # Print config highlights
    print("\nPhase 2 config highlights:")
    print(f"  - Margin: {trainer.config['arcface']['margin']}")
    print(f"  - Easy Margin: {trainer.config['arcface']['easy_margin']}")
    print(f"  - Learning rate: {trainer.config['training']['optimizer']['lr']}")
    print(f"  - Weight decay: {trainer.config['training']['optimizer']['weight_decay']}")
    print(f"  - Label smoothing: {trainer.config['training'].get('label_smoothing', 0.0)}")
    print(f"  - Augmentation: {trainer.config['data']['augment_strength']}")
    print(f"  - Target epochs: {trainer.config['training']['num_epochs']}")
    print("=" * 80)
    
    # Train
    print("\nStarting Phase 2 fine-tune training...")
    trainer.train()
    
    print("\n" + "=" * 80)
    print("PHASE 2 TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Final best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Final best validation loss: {trainer.best_val_loss:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
