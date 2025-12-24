# Hướng Dẫn Tạo Embeddings cho ArcFace và FaceNet

## Windows PowerShell Commands

### 1. Tạo ArcFace Embeddings Database

```powershell
# Cú pháp 1 dòng (Đơn giản nhất - Khuyên dùng)
╰─ python inference/extract_embeddings.py --mode db --model-path models/checkpoints/arcface/arcface_best.pth --data-dir data/celeb --output-path data/arcface_embeddings_db.npy --model-type arcface --use-face-detection
============================================================
EXTRACT EMBEDDINGS DATABASE (ARCFACE)
============================================================

Model type: arcface
Device: cpu
Use face detection: True
[WARN] TensorBoard not available: ModuleNotFoundError
Project root: D:\HCMUTE_project\DIP\FaceRecognition
Loaded ImageNet pretrained weights
Loaded model from models/checkpoints/arcface/arcface_best.pth
  - Num classes: 9343
  - Embedding size: 512
  - Epoch: 104
  - Val accuracy: 81.53%
[OK] MTCNN initialized on cpu
[OK] FacePreprocessor initialized (MTCNN)

Tim thay 18 celebrities
Dang extract embeddings...

Processing:  83%|███████████████   | 15/18 [00:32<00:06,  2.05s/it][ WARN:0@56.757] global loadsave.cpp:241 cv::findDecoder imread_('data/celeb\Truc_Nhan\tải xuống.jpg'): can't open/read file: check file path/integrity
Processing: 100%|██████████████████| 18/18 [00:40<00:00,  2.27s/it]

Da luu 17 embeddings vao data/arcface_embeddings_db.npy
Success rate: 17/18 (94.4%)

# Hoặc cú pháp nhiều dòng (dùng backtick ` để xuống dòng)
python inference/extract_embeddings.py `
    --mode db `
    --model-path models/checkpoints/arcface/arcface_best.pth `
    --data-dir data/celeb `
    --output-path data/arcface_embeddings_db.npy `
    --model-type arcface `
    --use-face-detection
```

### 2. Tạo FaceNet Embeddings Database

```powershell
# Cú pháp 1 dòng
python inference/extract_embeddings.py --mode db --model-path models/checkpoints/facenet/facenet_best.pth --data-dir data/celeb --output-path data/facenet_embeddings_db.npy --model-type facenet --use-face-detection

# Hoặc nhiều dòng
python inference/extract_embeddings.py `
    --mode db `
    --model-path models/checkpoints/facenet/facenet_best.pth `
    --data-dir data/celeb `
    --output-path data/facenet_embeddings_db.npy `
    --model-type facenet `
    --use-face-detection
```

## Giải Thích Arguments

- `--mode db`: Chế độ build database từ folder celebrities
- `--model-path`: Đường dẫn đến checkpoint (.pth)
- `--data-dir`: Thư mục chứa các folder celebrity (mỗi folder = 1 người)
- `--output-path`: File .npy output chứa embeddings
- `--model-type`: Loại model (`arcface` hoặc `facenet`)
- `--use-face-detection`: Tự động detect và align face trước khi extract
