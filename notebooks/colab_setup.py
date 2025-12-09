"""
Colab Setup Helper Script
Chạy script này đầu tiên khi mở notebook trên Colab để setup môi trường
"""

import os
import sys
import subprocess
from pathlib import Path


def check_colab():
    """Kiểm tra có đang chạy trên Colab không"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_colab_environment():
    """Setup môi trường Colab"""
    print("="*60)
    print("SETUP GOOGLE COLAB ENVIRONMENT")
    print("="*60)
    
    IS_COLAB = check_colab()
    
    if not IS_COLAB:
        print("Không phải môi trường Colab!")
        print("Script này chỉ dùng trên Google Colab")
        return False
    
    print("Đang chạy trên Google Colab")
    
    # Mount Google Drive
    print("\n[1] Mount Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("Google Drive đã mount")
    except Exception as e:
        print(f"Lỗi mount Drive: {e}")
        return False
    
    # Setup paths
    print("\n[2] Setup paths...")
    ROOT = "/content/FaceRecognition"
    DRIVE_ROOT = "/content/drive/MyDrive/FaceRecognition"
    
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    os.makedirs(f"{DRIVE_ROOT}/checkpoints", exist_ok=True)
    os.makedirs(f"{DRIVE_ROOT}/logs", exist_ok=True)
    os.makedirs(f"{DRIVE_ROOT}/data", exist_ok=True)
    
    print(f"ROOT: {ROOT}")
    print(f"DRIVE_ROOT: {DRIVE_ROOT}")
    
    # Clone hoặc update repo
    print("\n[3] Clone/Update repository...")
    REPO_URL = "https://github.com/sin0235/FaceRecognition.git"
    
    if os.path.exists(ROOT) and os.path.exists(f"{ROOT}/.git"):
        print("Repo đã tồn tại, đang pull updates...")
        os.chdir(ROOT)
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Updated repository")
        else:
            print(f"Pull warning: {result.stderr}")
    else:
        print("Đang clone repository...")
        if os.path.exists(ROOT):
            import shutil
            shutil.rmtree(ROOT)
        
        result = subprocess.run(['git', 'clone', REPO_URL, ROOT], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Cloned repository")
            os.chdir(ROOT)
        else:
            print(f"Clone failed: {result.stderr}")
            return False
    
    # Add to Python path
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    print(f"Added {ROOT} to Python path")
    
    # Install requirements
    print("\n[4] Install dependencies...")
    req_file = f"{ROOT}/requirements-colab.txt"
    
    if os.path.exists(req_file):
        print(f"Installing from {req_file}...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', req_file],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Dependencies installed")
        else:
            print(f"Installation warning: {result.stderr}")
    else:
        print(f"File {req_file} không tồn tại!")
        return False
    
    # Verify key imports
    print("\n[5] Verify imports...")
    try:
        import torch
        import torchvision
        import insightface
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"InsightFace: {insightface.__version__}")
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    print("\n" + "="*60)
    print("SETUP HOÀN TẤT!")
    print("="*60)
    print(f"\nWorking directory: {os.getcwd()}")
    print("\nBước tiếp theo:")
    print("1. Upload dataset lên Google Drive")
    print("2. Kiểm tra config trong configs/arcface_config.yaml")
    print("3. Chạy training script")
    
    return True


def get_colab_info():
    """Lấy thông tin về Colab runtime"""
    import torch
    
    info = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else 'N/A'
    }
    
    print("\n=== COLAB RUNTIME INFO ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    return info


if __name__ == "__main__":
    success = setup_colab_environment()
    
    if success:
        get_colab_info()
    else:
        print("\nSetup không thành công!")
        print("Vui lòng kiểm tra lỗi ở trên và thử lại")
