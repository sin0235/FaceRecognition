# Hướng Dẫn Tạo Embeddings cho ArcFace và FaceNet

## Windows PowerShell Commands

### 1. Tạo ArcFace Embeddings Database

```powershell
# Cú pháp 1 dòng (Đơn giản nhất - Khuyên dùng)
python inference/extract_embeddings.py --mode db --model-path models/checkpoints/arcface/arcface_best.pth --data-dir data/celeb --output-path data/arcface_embeddings_db.npy --model-type arcface --use-face-detection

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
