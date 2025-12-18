"""Script update evaluate_facenet_kaggle.ipynb để dùng checkpoint_utils"""
import json
import sys
from pathlib import Path

def update_load_model_cell(notebook_path):
    """Update cell load model để dùng checkpoint_utils với flexible loading."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Tìm cell load FaceNet model
    fixed = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            source_str = ''.join(source)
            
            # Tìm cell có FaceNetModel initialization và load_state_dict
            if 'FaceNetModel(' in source_str and 'load_state_dict' in source_str:
                # Replace với code mới
                new_source = [
                    "from models.facenet.facenet_model import FaceNetModel\n",
                    "from models.facenet.checkpoint_utils import load_facenet_checkpoint_flexible\n",
                    "\n",
                    "checkpoint_path = os.path.join(CHECKPOINT_DIR, \"facenet_best.pth\")\n",
                    "\n",
                    "# Initialize model\n",
                    "model = FaceNetModel(embedding_size=128, pretrained=None)\n",
                    "\n",
                    "# Load checkpoint với automatic key remapping\n",
                    "model, ckpt_info = load_facenet_checkpoint_flexible(model, checkpoint_path)\n",
                    "model.to(device).eval()\n",
                    "\n",
                    "print(f\"Model loaded: embedding_size={128}\")\n",
                    "print(f\"Training epochs: {ckpt_info['epoch'] + 1}\")\n",
                    "print(f\"Val triplet acc: {ckpt_info.get('val_triplet_acc', 0):.2f}%\")\n",
                    "print(f\"Val verification acc: {ckpt_info.get('val_ver_acc', 0):.2f}%\")"
                ]
                
                cell['source'] = new_source
                fixed = True
                print(f"[OK] Đã update cell load model")
                break
    
    if not fixed:
        print("[WARNING] Không tìm thấy cell load model")
        return False
    
    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Đã lưu notebook: {notebook_path}")
    return True

if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    notebook_path = repo_root / "notebooks" / "evaluate_facenet_kaggle.ipynb"
    
    if not notebook_path.exists():
        print(f"[ERROR] Không tìm thấy notebook: {notebook_path}")
        sys.exit(1)
    
    success = update_load_model_cell(notebook_path)
    sys.exit(0 if success else 1)
