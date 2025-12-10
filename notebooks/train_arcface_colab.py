"""
ArcFace Training Script for Google Colab
Chay script nay trong Colab notebook

Usage trong Colab:
    1. Mount Drive
    2. Clone repo
    3. Chay: !python notebooks/train_arcface_colab.py --drive-root /content/drive/MyDrive/FaceRecognition
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

def setup_paths(drive_root=None, local_root=None):
    """Setup paths cho Colab hoac Local"""
    
    # Detect environment
    try:
        from google.colab import drive
        IS_COLAB = True
    except ImportError:
        IS_COLAB = False
    
    if IS_COLAB and drive_root:
        ROOT = "/content/FaceRecognition"
        DATA_DIR = os.path.join(drive_root, "CelebA_Aligned")
        CHECKPOINT_DIR = os.path.join(drive_root, "models", "checkpoints")
        LOG_DIR = os.path.join(drive_root, "logs", "arcface")
        EMBEDDINGS_DIR = os.path.join(drive_root, "data", "embeddings")
    else:
        ROOT = local_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_DIR = os.path.join(ROOT, "data", "CelebA_Aligned")
        CHECKPOINT_DIR = os.path.join(ROOT, "models", "checkpoints")
        LOG_DIR = os.path.join(ROOT, "logs", "arcface")
        EMBEDDINGS_DIR = os.path.join(ROOT, "data", "embeddings")
    
    # Tao thu muc
    for d in [CHECKPOINT_DIR, LOG_DIR, EMBEDDINGS_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Add to path
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    
    return {
        'root': ROOT,
        'data_dir': DATA_DIR,
        'checkpoint_dir': CHECKPOINT_DIR,
        'log_dir': LOG_DIR,
        'embeddings_dir': EMBEDDINGS_DIR,
        'is_colab': IS_COLAB,
        'train_csv': os.path.join(DATA_DIR, "metadata", "train_labels_filtered.csv"),
        'val_csv': os.path.join(DATA_DIR, "metadata", "val_labels_filtered.csv"),
        'train_img_dir': os.path.join(DATA_DIR, "train"),
        'val_img_dir': os.path.join(DATA_DIR, "val"),
    }


def check_data(paths):
    """Kiem tra du lieu"""
    print("=== KIEM TRA DU LIEU ===")
    data_ready = True
    
    checks = [
        ("Train CSV", paths['train_csv']),
        ("Val CSV", paths['val_csv']),
        ("Train images", paths['train_img_dir']),
        ("Val images", paths['val_img_dir']),
    ]
    
    for name, path in checks:
        if os.path.exists(path):
            print(f"[OK] {name}")
        else:
            print(f"[THIEU] {name}: {path}")
            data_ready = False
    
    return data_ready


def create_runtime_config(paths, base_config_path):
    """Tao config voi paths da override"""
    import yaml
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override paths
    config['data']['train_csv'] = paths['train_csv']
    config['data']['val_csv'] = paths['val_csv']
    config['data']['train_data_root'] = paths['train_img_dir']
    config['data']['val_data_root'] = paths['val_img_dir']
    config['checkpoint']['save_dir'] = paths['checkpoint_dir']
    config['logging']['log_dir'] = paths['log_dir']
    
    # Luu runtime config
    runtime_config = os.path.join(paths['root'], "configs", "arcface_runtime.yaml")
    with open(runtime_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Saved runtime config: {runtime_config}")
    return runtime_config, config


def run_training(config_path, checkpoint_dir):
    """Chay training"""
    from models.arcface.train_arcface import ArcFaceTrainer
    
    print("="*60)
    print("BAT DAU TRAINING")
    print("="*60)
    
    trainer = ArcFaceTrainer(
        config_path=config_path,
        pretrained_path=None,
        checkpoint_dir=checkpoint_dir
    )
    
    trainer.train()
    
    return os.path.join(checkpoint_dir, "arcface_best.pth")


def run_embedding_extraction(model_path, paths):
    """Trich xuat embeddings"""
    from inference.extract_embeddings import full_pipeline
    import torch
    
    print("="*60)
    print("EXTRACT EMBEDDINGS")
    print("="*60)
    
    full_pipeline(
        model_path=model_path,
        csv_path=paths['train_csv'],
        data_root=paths['train_img_dir'],
        output_dir=paths['embeddings_dir'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=64
    )


def test_model(model_path, paths):
    """Test model sau training"""
    import torch
    from models.arcface.arcface_model import ArcFaceModel
    
    print("="*60)
    print("TEST MODEL")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Model khong ton tai: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val accuracy: {checkpoint.get('val_acc', 'N/A')}")
    print(f"Best val accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    
    # Test embedding
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes', 100)
    
    model = ArcFaceModel(num_classes=num_classes, embedding_size=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        emb = model.extract_features(dummy)
    
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding normalized: {torch.allclose(emb.norm(), torch.ones(1))}")
    print("\nModel san sang!")


def main():
    parser = argparse.ArgumentParser(description="ArcFace Training Script")
    parser.add_argument('--drive-root', type=str, default=None,
                       help='Google Drive root (e.g., /content/drive/MyDrive/FaceRecognition)')
    parser.add_argument('--local-root', type=str, default=None,
                       help='Local project root')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, only extract embeddings')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip embedding extraction')
    
    args = parser.parse_args()
    
    # Setup
    paths = setup_paths(args.drive_root, args.local_root)
    
    print("="*60)
    print("ARCFACE TRAINING PIPELINE")
    print("="*60)
    print(f"Environment: {'Colab' if paths['is_colab'] else 'Local'}")
    print(f"Root: {paths['root']}")
    print(f"Data: {paths['data_dir']}")
    print(f"Checkpoints: {paths['checkpoint_dir']}")
    
    # Check data
    if not check_data(paths):
        print("\nThieu du lieu. Vui long upload data len Drive truoc.")
        return
    
    # Create config
    base_config = os.path.join(paths['root'], "configs", "arcface_config.yaml")
    if not os.path.exists(base_config):
        print(f"Thieu config: {base_config}")
        return
    
    runtime_config, config = create_runtime_config(paths, base_config)
    
    # Training
    best_model = os.path.join(paths['checkpoint_dir'], "arcface_best.pth")
    
    if not args.skip_training:
        best_model = run_training(runtime_config, paths['checkpoint_dir'])
    
    # Test model
    test_model(best_model, paths)
    
    # Extract embeddings
    if not args.skip_extraction and os.path.exists(best_model):
        run_embedding_extraction(best_model, paths)
    
    # Summary
    print("\n" + "="*60)
    print("HOAN TAT")
    print("="*60)
    print(f"Best model: {best_model}")
    print(f"Embeddings: {paths['embeddings_dir']}")
    print(f"Logs: {paths['log_dir']}")


if __name__ == "__main__":
    main()

