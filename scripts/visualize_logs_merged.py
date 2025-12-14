"""
Visualize ArcFace Training Logs - Merged Version
Gop data tu logs_v1.json (epoch 1-50) va logs_v2.json (epoch 51-110)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_and_merge_logs(logs_v1_path, logs_v2_path):
    with open(logs_v1_path, 'r') as f:
        data_v1 = json.load(f)
    with open(logs_v2_path, 'r') as f:
        data_v2 = json.load(f)
    
    epochs = data_v1['history']['epoch'] + data_v2['history']['epoch']
    train_loss = data_v1['history']['train_loss'] + data_v2['history']['train_loss']
    train_acc = data_v1['history']['train_acc'] + data_v2['history']['train_acc']
    val_loss = data_v1['history']['val_loss'] + data_v2['history']['val_loss']
    val_acc = data_v1['history']['val_acc'] + data_v2['history']['val_acc']
    
    summary = {
        'best_val_acc': data_v2.get('best_val_acc', max(val_acc)),
        'best_val_loss': data_v2.get('best_val_loss'),
        'total_epochs': data_v2.get('total_epochs', len(epochs)),
        'v1_best_val_acc': data_v1.get('best_val_acc'),
        'v1_epochs': len(data_v1['history']['epoch']),
        'v2_epochs': len(data_v2['history']['epoch'])
    }
    
    return epochs, train_loss, train_acc, val_loss, val_acc, summary


def main():
    script_dir = Path(__file__).parent
    logs_v1_path = script_dir / '../logs/arcface/logs_v1.json'
    logs_v2_path = script_dir / '../logs/arcface/logs_v2.json'
    output_path = script_dir / '../logs/arcface/training_visualization_merged.png'

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 12

    epochs, train_loss, train_acc, val_loss, val_acc, summary = load_and_merge_logs(logs_v1_path, logs_v2_path)

    print("=== MERGED TRAINING LOGS ===")
    print(f"V1 epochs: 1-{summary['v1_epochs']}")
    print(f"V2 epochs: {summary['v1_epochs']+1}-{summary['total_epochs']}")
    print(f"Total epochs: {len(epochs)}")
    print(f"\n=== SUMMARY ===")
    print(f"V1 Best val accuracy: {summary['v1_best_val_acc']}%")
    print(f"Final Best val accuracy: {summary['best_val_acc']:.2f}%")
    print(f"Final Best val loss: {summary['best_val_loss']:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    v1_end = summary['v1_epochs']

    # 1. Train vs Val Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=2)
    ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=2)
    ax1.axvline(x=v1_end, color='green', linestyle='--', alpha=0.7, label=f'V1/V2 boundary (epoch {v1_end})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss (V1 + V2)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Train vs Val Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc', marker='o', markersize=2)
    ax2.plot(epochs, val_acc, 'r-', linewidth=2, label='Val Acc', marker='s', markersize=2)
    ax2.axvline(x=v1_end, color='green', linestyle='--', alpha=0.7, label=f'V1/V2 boundary (epoch {v1_end})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training vs Validation Accuracy (V1 + V2)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy Gap (Overfitting indicator)
    ax3 = axes[1, 0]
    acc_gap = np.array(train_acc) - np.array(val_acc)
    ax3.fill_between(epochs, acc_gap, alpha=0.5, color='orange')
    ax3.plot(epochs, acc_gap, 'orange', linewidth=2)
    ax3.axvline(x=v1_end, color='green', linestyle='--', alpha=0.7, label=f'V1/V2 boundary (epoch {v1_end})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Gap (%)')
    ax3.set_title('Train-Val Accuracy Gap (Overfitting Indicator)')
    ax3.axhline(y=10, color='r', linestyle='--', label='Warning threshold (10%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Loss in Log Scale
    ax4 = axes[1, 1]
    ax4.semilogy(epochs, train_loss, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=2)
    ax4.semilogy(epochs, val_loss, 'r-', linewidth=2, label='Val Loss', marker='s', markersize=2)
    ax4.axvline(x=v1_end, color='green', linestyle='--', alpha=0.7, label=f'V1/V2 boundary (epoch {v1_end})')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Loss (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nDa luu bieu do vao: {output_path}")

    # Analysis
    print("\n=== PHAN TICH KET QUA TRAINING ===")
    print(f"\nFinal Train Accuracy: {train_acc[-1]:.2f}%")
    print(f"Final Val Accuracy: {val_acc[-1]:.2f}%")
    print(f"Accuracy Gap: {train_acc[-1] - val_acc[-1]:.2f}%")

    print(f"\nFinal Train Loss: {train_loss[-1]:.4f}")
    print(f"Final Val Loss: {val_loss[-1]:.4f}")

    best_val_acc = summary['best_val_acc']
    best_epoch = epochs[val_acc.index(max(val_acc))]
    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

    gap = train_acc[-1] - val_acc[-1]
    if gap > 15:
        print("\n[CANH BAO] Mo hinh co dau hieu overfit nghiem trong (gap > 15%)")
    elif gap > 10:
        print("\n[LUU Y] Mo hinh co dau hieu overfit nhe (gap > 10%)")
    else:
        print("\n[OK] Mo hinh khong bi overfit nghiem trong")


if __name__ == '__main__':
    main()
