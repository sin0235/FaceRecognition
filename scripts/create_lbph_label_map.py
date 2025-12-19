"""
Script tạo label_map.npy cho LBPH model
Mapping từ label ID (số) sang identity name
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


def create_label_map_from_csv():
    """Tạo label_map từ CSV metadata"""
    metadata_path = os.path.join(ROOT_DIR, "data/metadata/train_labels.csv")
    
    if not os.path.exists(metadata_path):
        print(f"[INFO] Không tìm thấy CSV: {metadata_path}")
        return None
    
    try:
        df = pd.read_csv(metadata_path)
        
        if 'label' not in df.columns or 'identity_id' not in df.columns:
            print("[ERROR] CSV không có đúng columns: 'label' và 'identity_id'")
            return None
        
        # Tạo mapping: label -> identity_id
        mapping_df = df[['label', 'identity_id']].drop_duplicates()
        label_map = dict(zip(mapping_df['label'], mapping_df['identity_id']))
        
        print(f"[OK] Tạo label_map từ CSV: {len(label_map)} identities")
        return label_map
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc CSV: {e}")
        return None


def create_label_map_from_dataset():
    """Tạo label_map từ cấu trúc dataset (identity folders)"""
    train_dirs = [
        os.path.join(ROOT_DIR, "data/CelebA_Aligned_Balanced/train"),
        os.path.join(ROOT_DIR, "data/CelebA_Aligned/train"),
        os.path.join(ROOT_DIR, "data/train")
    ]
    
    train_dir = None
    for td in train_dirs:
        if os.path.exists(td):
            train_dir = td
            break
    
    if train_dir is None:
        print("[ERROR] Không tìm thấy train directory")
        print("Đã thử:")
        for td in train_dirs:
            print(f"  - {td}")
        return None
    
    try:
        identities = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
        
        # Tạo mapping: label_id (0, 1, 2...) -> identity_name
        # LBPH model dùng label là số tương ứng với thứ tự identity
        label_map = {i: identity for i, identity in enumerate(identities)}
        
        print(f"[OK] Tạo label_map từ dataset: {len(label_map)} identities")
        print(f"  Train dir: {train_dir}")
        print(f"  Sample identities: {list(identities[:5])}")
        
        return label_map
        
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Tạo label_map.npy"""
    print("="*60)
    print("CREATE LBPH LABEL MAP")
    print("="*60)
    
    output_path = os.path.join(ROOT_DIR, "models/checkpoints/LBHP/label_map.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Thử tạo từ CSV trước
    label_map = create_label_map_from_csv()
    
    # Nếu không có CSV, tạo từ dataset structure
    if label_map is None:
        print("\n[INFO] Thử tạo từ dataset structure...")
        label_map = create_label_map_from_dataset()
    
    if label_map is None:
        print("\n[ERROR] Không thể tạo label_map!")
        return 1
    
    # Lưu file
    np.save(output_path, label_map)
    print(f"\n[OK] Đã lưu label_map: {output_path}")
    print(f"  Số lượng identities: {len(label_map)}")
    print(f"  Sample mapping:")
    for i, (label_id, identity) in enumerate(list(label_map.items())[:5]):
        print(f"    Label {label_id} -> {identity}")
    
    # Verify file
    if os.path.exists(output_path):
        loaded = np.load(output_path, allow_pickle=True).item()
        if len(loaded) == len(label_map):
            print(f"\n[OK] Verify thành công: {len(loaded)} identities")
        else:
            print(f"\n[WARNING] Verify không khớp: {len(loaded)} vs {len(label_map)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
