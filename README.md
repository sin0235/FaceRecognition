# Face Recognition System

Hệ thống nghiên cứu nhận dạng khuôn mặt với ba hướng tiếp cận:

- **ArcFace**: embedding 512 chiều, nhận dạng bằng cosine similarity.
- **FaceNet**: embedding 512 chiều, huấn luyện với triplet loss.
- **LBPH**: mô hình OpenCV truyền thống, dùng khoảng cách LBPH.

Ứng dụng Flask tại `web_app.py` là entry point chính. Giao diện hỗ trợ nhận dạng một ảnh, xử lý theo lô, webcam và tạo dữ liệu nhận dạng từ thư mục ảnh.

## Thành phần chính

```text
face-recognition-system/
├── web_app.py                         # Ứng dụng Flask chính
├── app/
│   └── app.py                         # Demo Streamlit độc lập
├── configs/
│   ├── arcface_config.yaml             # Cấu hình huấn luyện ArcFace
│   ├── arcface_kaggle.yaml             # Cấu hình ArcFace cho Kaggle
│   ├── facenet_config.yaml             # Cấu hình huấn luyện FaceNet
│   ├── facenet_kaggle.yaml             # Cấu hình FaceNet cho Kaggle
│   └── lbph_config.yaml                # Ngưỡng và kích thước đầu vào LBPH
├── preprocessing/
│   ├── face_detector.py                # MTCNN, RetinaFace hoặc OpenCV Haar Cascade
│   └── celeba_preprocessing.py         # Tiền xử lý dữ liệu CelebA
├── inference/
│   ├── recognition_engine.py           # Suy luận ArcFace từ embedding database hoặc FAISS
│   ├── extract_embeddings.py           # Tạo embedding database ArcFace/FaceNet
│   ├── database_builder.py              # Job nền tạo database qua giao diện Flask
│   ├── explainability.py                # Grad-CAM cho ArcFace và FaceNet
│   └── evaluate.py                      # Metric, ROC, confusion matrix, threshold sweep
├── models/
│   ├── arcface/                         # Model, dataloader và script huấn luyện ArcFace
│   ├── facenet/                         # Model, dataloader và script huấn luyện FaceNet
│   └── lbphmodel/                       # Huấn luyện, suy luận, đánh giá LBPH
├── scripts/                             # Tiền xử lý, tạo label map, trực quan hóa log
├── notebooks/                           # Notebook train, đánh giá và phân tích
├── templates/                           # Template Flask
├── static/                              # CSS, logo và file giao diện sinh lúc chạy
├── docs/
│   └── extract_embeddings.md            # Hướng dẫn chi tiết tạo embedding database
├── requirements.txt                     # Phụ thuộc cho môi trường cục bộ
├── requirements-colab.txt               # Phụ thuộc cho Colab/Kaggle
└── LICENSE                              # MIT License
```

## Cài đặt

Cài môi trường và phụ thuộc từ thư mục gốc dự án:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` bao gồm PyTorch, OpenCV, Flask, MTCNN, scikit-learn và các thư viện xử lý ảnh. Với GPU, cài bản `torch` và `torchvision` phù hợp CUDA trước hoặc thay cho bản do `requirements.txt` cài đặt.

## Tài nguyên không có trong repository

`.gitignore` loại trừ `data/`, checkpoint mô hình và các file embedding. Muốn nhận dạng có kết quả, chuẩn bị các tài nguyên sau:

```text
models/checkpoints/
├── arcface/arcface_best.pth
├── facenet/facenet_best.pth
└── LBHP/
    ├── lbph_model.xml
    └── label_map.npy

data/
├── arcface_embeddings_db.npy
├── facenet_embeddings_db.npy
└── celeb/
    └── <identity>/
        └── <image>.jpg
```

`data/celeb/<identity>/` là cấu trúc đầu vào để tạo database. Mỗi thư mục con tương ứng một danh tính. ArcFace và FaceNet dùng các file `.npy`; LBPH dùng `lbph_model.xml` cùng `label_map.npy`.

## Chạy ứng dụng Flask

```bash
python web_app.py
```

Truy cập `http://127.0.0.1:5000`.

Khi chạy, ứng dụng tự tạo các thư mục tạm:

- `static/uploads/`: ảnh đã tải lên.
- `static/gradcam/`: ảnh Grad-CAM.
- `static/detection_bbox/`: ảnh có bounding box.
- Thư mục tạm hệ điều hành: ảnh dùng trong lúc suy luận.

Các mô hình được nạp lười. Thiếu checkpoint hoặc embedding database làm model tương ứng trả lỗi hoặc không có kết quả nhận dạng.

## Chức năng giao diện

| Đường dẫn | Chức năng |
| --- | --- |
| `/` | Tải một ảnh, chạy đồng thời ArcFace, FaceNet và LBPH; hiển thị bounding box và Grad-CAM khi khả dụng. |
| `/batch` | Tải nhiều ảnh và so sánh kết quả từ ba model. |
| `/realtime` | Nhận dạng webcam. Webcam được mở từ máy chủ chạy Flask. |
| `/database-builder` | Tạo embedding database ArcFace/FaceNet hoặc train LBPH trong background thread. |

Các endpoint nội bộ dùng bởi giao diện realtime và database builder:

```text
GET  /video_feed
GET  /realtime_result
POST /stop_camera
POST /set_realtime_model
POST /database-builder/build
GET  /database-builder/status/<job_id>
GET  /database-builder/download/<path:filename>
```

Đây là endpoint phục vụ giao diện hiện tại, chưa phải REST API có phiên bản hoặc cơ chế xác thực cho môi trường production.

## Tạo embedding database

Tạo database từ cấu trúc `data/celeb/<identity>/<image>`.

### ArcFace

```bash
python inference/extract_embeddings.py \
  --mode db \
  --model-type arcface \
  --model-path models/checkpoints/arcface/arcface_best.pth \
  --data-dir data/celeb \
  --output-path data/arcface_embeddings_db.npy \
  --use-face-detection
```

### FaceNet

```bash
python inference/extract_embeddings.py \
  --mode db \
  --model-type facenet \
  --model-path models/checkpoints/facenet/facenet_best.pth \
  --data-dir data/celeb \
  --output-path data/facenet_embeddings_db.npy \
  --use-face-detection
```

Tham số `--no-face-detection` tắt detect và align trước khi tạo embedding. Hướng dẫn bổ sung có tại [`docs/extract_embeddings.md`](docs/extract_embeddings.md).

## Huấn luyện

### ArcFace

```bash
python models/arcface/train_arcface.py \
  --config configs/arcface_config.yaml \
  --data_dir data/CelebA_Aligned_Balanced \
  --checkpoint_dir models/checkpoints/arcface
```

Script nhận thêm `--pretrained_backbone`, `--resume` và `--reset_optimizer`.

### FaceNet

```bash
python models/facenet/train_facenet.py \
  --config configs/facenet_config.yaml
```

Đường dẫn dữ liệu, checkpoint và log mặc định nằm trong `configs/facenet_config.yaml`.

### LBPH

```bash
python models/lbphmodel/train_lbph_script.py \
  --data-dir data/celeb \
  --output-dir models/checkpoints/LBHP \
  --find-threshold \
  --val-dir data/celeb_val
```

Bỏ `--find-threshold` và `--val-dir` khi không có tập validation. LBPH mặc định detect, crop mặt về `100x100` rồi chuyển grayscale. Cấu hình ngưỡng mặc định nằm trong `configs/lbph_config.yaml`.

## Tiền xử lý và đánh giá

- `preprocessing/celeba_preprocessing.py`: chuẩn bị dữ liệu CelebA.
- `scripts/celeba_balanced_preprocessing.py`: tạo dữ liệu CelebA cân bằng.
- `scripts/create_lbph_label_map.py`: tạo label map cho LBPH.
- `inference/evaluate.py`: các hàm tính metric, ROC, confusion matrix và threshold sweep.
- `notebooks/`: các workflow huấn luyện, đánh giá và phân tích trên Kaggle/Colab.

Dùng `requirements-colab.txt` trong Colab hoặc Kaggle khi cần `insightface`, `albumentations`, TensorBoard và FAISS GPU.

## Demo Streamlit

`app/app.py` là demo Streamlit độc lập dùng `RecognitionEngine`. `streamlit` không có trong `requirements.txt`; cài riêng khi cần:

```bash
python -m pip install streamlit
streamlit run app/app.py
```

## License

Dự án phát hành theo [MIT License](LICENSE).
