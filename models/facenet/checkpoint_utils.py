"""Helper function để load FaceNet checkpoint với key remapping tự động"""
import torch
import os

def load_facenet_checkpoint_flexible(model, checkpoint_path):
    """
    Load checkpoint với automatic key remapping nếu cần.
    
    Handle các trường hợp:
    - Keys mismatch giữa 'model.*' và 'backbone.*'
    - Missing projection layer trong checkpoint
    
    Args:
        model: FaceNetModel instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        model: Model với weights đã load
        checkpoint_info: Dict chứa thông tin checkpoint (epoch, metrics, etc)
    """
    # Validate checkpoint file
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file không tồn tại: {checkpoint_path}")
    
    file_size = os.path.getsize(checkpoint_path)
    if file_size == 0:
        raise ValueError(f"Checkpoint file rỗng: {checkpoint_path}")
    
    if file_size < 1024:  # File quá nhỏ (< 1KB) có thể bị hỏng
        raise ValueError(f"Checkpoint file có vẻ bị hỏng (kích thước: {file_size} bytes): {checkpoint_path}")
    
    print(f"[INFO] Loading checkpoint: {checkpoint_path} ({file_size / 1024 / 1024:.2f} MB)")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if "failed finding central directory" in str(e) or "zip archive" in str(e):
            raise RuntimeError(
                f"Checkpoint file bị hỏng hoặc không đầy đủ: {checkpoint_path}\n"
                f"Lỗi: {e}\n"
                f"Vui lòng kiểm tra lại file checkpoint hoặc tải lại từ nguồn."
            ) from e
        raise
    state_dict = checkpoint['model_state_dict']
    
    # Get current model keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # Check for key prefix mismatch
    needs_remap = False
    if checkpoint_keys and model_keys:
        # Get first key prefix from each
        ckpt_prefix = list(checkpoint_keys)[0].split('.')[0]
        model_prefix = list(model_keys)[0].split('.')[0]
        
        if ckpt_prefix != model_prefix:
            print(f"[INFO] Key mismatch detected: checkpoint uses '{ckpt_prefix}.*' but model uses '{model_prefix}.*'")
            needs_remap = True
    
    # Remap keys if needed
    if needs_remap:
        new_state_dict = {}
        for key, value in state_dict.items():
            # Replace first prefix
            parts = key.split('.', 1)
            if len(parts) == 2:
                new_key = f"{model_prefix}.{parts[1]}"
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        state_dict = new_state_dict
        print(f"[OK] Remapped {len(state_dict)} keys")
    
    # Load với strict=False để ignore missing projection layer nếu cần
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Report loading status
    if missing_keys:
        # Filter out projection layer (OK to be missing if embedding_size=512)
        important_missing = [k for k in missing_keys if not k.startswith('projection')]
        if important_missing:
            print(f"[WARNING] Missing keys ({len(important_missing)}): {important_missing[:5]}...")
    
    if unexpected_keys:
        # Filter out logits layer (thường có trong checkpoint nhưng không cần cho inference)
        logits_keys = [k for k in unexpected_keys if 'logits' in k]
        other_unexpected = [k for k in unexpected_keys if 'logits' not in k]
        
        if logits_keys:
            print(f"[INFO] Ignored logits layer keys ({len(logits_keys)}): {logits_keys[:3]}...")
        if other_unexpected:
            print(f"[WARNING] Unexpected keys ({len(other_unexpected)}): {other_unexpected[:5]}...")
    
    if not missing_keys and not unexpected_keys:
        print("[OK] Checkpoint loaded perfectly")
    elif not important_missing:
        print("[OK] Checkpoint loaded successfully")
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_triplet_acc': checkpoint.get('val_triplet_acc', 0),
        'val_ver_acc': checkpoint.get('val_ver_acc', 0),
        'val_loss': checkpoint.get('val_loss', 0),
        'mining': checkpoint.get('mining', 'unknown')
    }
    
    return model, checkpoint_info
