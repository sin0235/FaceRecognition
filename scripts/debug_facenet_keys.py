"""Debug script: Inspect FaceNet checkpoint keys"""
import torch
import sys
from pathlib import Path

# Add root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from models.facenet.facenet_model import FaceNetModel

# Path tới checkpoint
checkpoint_path = root / "models" / "checkpoints" / "facenet" / "facenet_best.pth"

if checkpoint_path.exists():
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\n=== CHECKPOINT KEYS (first 10) ===")
    checkpoint_keys = list(checkpoint['model_state_dict'].keys())
    for key in checkpoint_keys[:10]:
        print(f"  {key}")
    
    print(f"\n...total {len(checkpoint_keys)} keys")
    
    # Check prefixes
    prefixes = set()
    for key in checkpoint_keys:
        if '.' in key:
            prefix = key.split('.')[0]
            prefixes.add(prefix)
    
    print(f"\nKey prefixes in checkpoint: {sorted(prefixes)}")
    
    # Create model và check expected keys
    print("\n=== MODEL EXPECTED KEYS (first 10) ===")
    model = FaceNetModel(embedding_size=128, pretrained=None)
    model_keys = list(model.state_dict().keys())
    for key in model_keys[:10]:
        print(f"  {key}")
    
    print(f"\n...total {len(model_keys)} keys")
    
    # Check prefixes
    model_prefixes = set()
    for key in model_keys:
        if '.' in key:
            prefix = key.split('.')[0]
            model_prefixes.add(prefix)
    
    print(f"\nKey prefixes in model: {sorted(model_prefixes)}")
    
else:
    print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
