# Hướng Dẫn Cài Đặt Dependencies

## 🖥️ Cài Đặt Trên Máy Local (Windows/Linux/Mac)

### Bước 1: Cài đặt PyTorch

**Truy cập**: https://pytorch.org/get-started/locally/

**Chọn cấu hình phù hợp:**

#### Windows với CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Windows với CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only (không có GPU):
```bash
pip install torch torchvision
```

### Bước 2: Cài đặt dependencies cơ bản
```bash
pip install -r requirements-local.txt
```

### Bước 3: Cài đặt InsightFace và MXNet
```bash
# InsightFace
pip install insightface

# MXNet (chọn phù hợp với CUDA)
# CUDA 11.8:
pip install mxnet-cu118

# Hoặc CPU:
pip install mxnet
```

### Bước 4: Cài đặt ONNX Runtime
```bash
# GPU version:
pip install onnxruntime-gpu

# Hoặc CPU version:
pip install onnxruntime
```

### Bước 5: (Optional) FAISS và TensorFlow
```bash
# FAISS cho inference nhanh
pip install faiss-cpu  # hoặc faiss-gpu

# TensorFlow cho FaceNet (nếu cần)
pip install tensorflow keras
```

---

## ☁️ Cài Đặt Trên Google Colab

**Đơn giản hơn nhiều!** Colab đã có sẵn CUDA 11.8.

### Trong notebook, thêm cell:

```python
# Cài PyTorch với CUDA 11.8
!pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Cài MXNet và InsightFace
!pip install -q mxnet-cu118==1.9.1 onnxruntime-gpu==1.16.0 insightface==0.7.3

# Cài các dependencies còn lại
!pip install -q -r requirements-colab.txt
```

Hoặc sử dụng cell đã có sẵn trong `arcface_colab.ipynb` (cell 6).

---

## 🔍 Kiểm Tra Cài Đặt

### Kiểm tra PyTorch và CUDA:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Kiểm tra InsightFace:
```python
import insightface
print(f"InsightFace version: {insightface.__version__}")
```

### Kiểm tra MXNet:
```python
import mxnet as mx
print(f"MXNet version: {mx.__version__}")
print(f"MXNet GPUs: {mx.context.num_gpus()}")
```

---

## ❗ Xử Lý Lỗi Thường Gặp

### 1. "Could not find a version that satisfies the requirement torch"
**Nguyên nhân**: Đang cài version có tag `+cu118` trên máy local

**Giải pháp**: Cài PyTorch từ link chính thức với `--index-url`
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. "No module named 'torch'"
**Giải pháp**: Cài PyTorch trước các thư viện khác

### 3. "CUDA error: no kernel image is available"
**Nguyên nhân**: Version CUDA không khớp với GPU

**Giải pháp**: Kiểm tra CUDA version của bạn:
```bash
nvidia-smi
```
Sau đó cài PyTorch phù hợp.

### 4. "ImportError: libmxnet.so: cannot open shared object file"
**Giải pháp**: 
```bash
pip uninstall mxnet mxnet-cu118
pip install mxnet-cu118==1.9.1
```

### 5. Lỗi với albumentations
**Giải pháp**:
```bash
pip install albumentations --no-deps
pip install opencv-python-headless
```

---

## 📝 Tóm Tắt

| Platform | Command |
|----------|---------|
| **Local (GPU)** | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`<br>`pip install -r requirements-local.txt`<br>`pip install insightface mxnet-cu118` |
| **Local (CPU)** | `pip install torch torchvision`<br>`pip install -r requirements-local.txt`<br>`pip install insightface mxnet` |
| **Colab** | Sử dụng cell 6 trong `arcface_colab.ipynb` |

---

## 🚀 Next Steps

Sau khi cài đặt xong:

1. **Kiểm tra**: Chạy các đoạn code kiểm tra ở trên
2. **Test model**: 
   ```bash
   python models/arcface/arcface_model.py
   ```
3. **Test dataloader**:
   ```bash
   python models/arcface/arcface_dataloader.py
   ```
