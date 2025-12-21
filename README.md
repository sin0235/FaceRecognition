<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Face%20Recognition%20System&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Enterprise-Grade%20Multi-Model%20Face%20Recognition%20Platform&descAlignY=52&descSize=18" width="100%"/>

<!-- Badges Row 1 -->
<p>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
</p>

<!-- Badges Row 2 -->
<p>
  <img src="https://img.shields.io/badge/CUDA-11.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status"/>
</p>

<!-- Navigation -->
<p>
  <a href="#-overview">Overview</a> &nbsp;&bull;&nbsp;
  <a href="#-features">Features</a> &nbsp;&bull;&nbsp;
  <a href="#-models">Models</a> &nbsp;&bull;&nbsp;
  <a href="#-installation">Installation</a> &nbsp;&bull;&nbsp;
  <a href="#-usage">Usage</a> &nbsp;&bull;&nbsp;
  <a href="#-api">API</a> &nbsp;&bull;&nbsp;
  <a href="#-benchmarks">Benchmarks</a>
</p>

</div>

---

## Overview

<img align="right" width="300" src="https://raw.githubusercontent.com/gist/patevs/b007a0e98fb216438d4cbf559fac4166/raw/88f20c9d749d756be63f22b09f3c4ac570bc5101/programming.gif" alt="coding"/>

**Face Recognition System** la nen tang nhan dang khuon mat da mo hinh duoc thiet ke cho cac ung dung doanh nghiep.

He thong tich hop **3 phuong phap** nhan dang tien tien:

- **ArcFace** - Do chinh xac cao nhat
- **FaceNet** - Can bang giua toc do va chinh xac
- **LBPH** - Nhe, nhanh, khong can GPU

### Tai sao chon he thong nay?

|       | Feature          | Description                            |
| :---: | :--------------- | :------------------------------------- |
| **1** | Multi-Model      | So sanh ket qua tu 3 mo hinh dong thoi |
| **2** | Production Ready | Kien truc microservices, de scale      |
| **3** | Explainable AI   | Grad-CAM visualization                 |
| **4** | Real-time        | Video streaming do tre thap            |

<br clear="right"/>

---

## Features

<div align="center">

### Tech Stack

<p>
  <img src="https://skillicons.dev/icons?i=python,pytorch,flask,opencv,docker,git,github,vscode&theme=dark&perline=8" alt="Tech Stack"/>
</p>

</div>

### Core Capabilities

```
                              +------------------+
                              |   Face Detection |
                              |  RetinaFace/MTCNN|
                              +--------+---------+
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
+--------v---------+         +---------v--------+         +----------v-------+
|     ArcFace      |         |     FaceNet      |         |       LBPH       |
|  IResNet100      |         | InceptionResNetV1|         |  Local Binary    |
|  Embedding: 512  |         |  Embedding: 512  |         |  Pattern Hist    |
|  Best Accuracy   |         |  Balanced        |         |  Lightweight     |
+--------+---------+         +---------+--------+         +----------+-------+
         |                             |                             |
         +-----------------------------+-----------------------------+
                                       |
                              +--------v---------+
                              | Unified Engine   |
                              | + Grad-CAM       |
                              +--------+---------+
                                       |
         +-----------------------------+-----------------------------+
         |                             |                             |
+--------v---------+         +---------v--------+         +----------v-------+
|  Single Image    |         |  Batch Process   |         |   Real-time      |
|  Recognition     |         |  Multi-image     |         |   Video Stream   |
+------------------+         +------------------+         +------------------+
```

### Feature Matrix

| Feature                  | Description                      |                                Status                                 |
| :----------------------- | :------------------------------- | :-------------------------------------------------------------------: |
| Single Image Recognition | Upload va nhan dang anh don le   | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| Batch Processing         | Xu ly hang loat nhieu anh        | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| Real-time Recognition    | Nhan dang tu webcam/video stream | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| Multi-Model Comparison   | So sanh ket qua tu nhieu model   | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| Grad-CAM Visualization   | Truc quan hoa vung attention     | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| Face Detection           | RetinaFace + MTCNN fallback      | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| Database Management      | Xay dung va quan ly embedding DB | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| REST API                 | API endpoints cho tich hop       | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |
| GPU Acceleration         | CUDA support                     | ![Done](https://img.shields.io/badge/-Done-success?style=flat-square) |

---

## Models

<div align="center">

### Model Comparison

</div>

|                |                                           ArcFace                                            |                                           FaceNet                                            |                                        LBPH                                        |
| :------------- | :------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| **Type**       | ![Deep Learning](https://img.shields.io/badge/-Deep%20Learning-blueviolet?style=flat-square) | ![Deep Learning](https://img.shields.io/badge/-Deep%20Learning-blueviolet?style=flat-square) | ![Traditional](https://img.shields.io/badge/-Traditional-orange?style=flat-square) |
| **Backbone**   |                                          IResNet100                                          |                                      InceptionResNetV1                                       |                                   LBP Histogram                                    |
| **Embedding**  |                                           512-dim                                            |                                           512-dim                                            |                                        N/A                                         |
| **Input Size** |                                           112x112                                            |                                           160x160                                            |                                      100x100                                       |
| **GPU**        |                                         Recommended                                          |                                         Recommended                                          |                                    Not Required                                    |
| **Speed**      |                                            ~15ms                                             |                                            ~12ms                                             |                                        ~2ms                                        |
| **Use Case**   |                                        High Security                                         |                                         General Use                                          |                                    Edge Devices                                    |

---

## Installation

### Prerequisites

| Requirement                                                                                         | Minimum | Recommended |
| :-------------------------------------------------------------------------------------------------- | :-----: | :---------: |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |   3.8   |    3.10+    |
| ![RAM](https://img.shields.io/badge/RAM-grey?style=flat-square)                                     |   8GB   |    16GB+    |
| ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)     |    -    |    11.x     |
| ![Storage](https://img.shields.io/badge/Storage-grey?style=flat-square)                             |   2GB   |    5GB+     |

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/sin0235/FaceRecognition.git
cd FaceRecognition

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch application
python web_app.py
```

<details>
<summary><b>Docker Deployment</b></summary>

```bash
# Build image
docker build -t face-recognition .

# Run container
docker run -p 5000:5000 --gpus all face-recognition
```

</details>

---

## Usage

### Web Interface

> Access `http://localhost:5000` after launching

| Endpoint    | Function     | Description                             |
| :---------- | :----------- | :-------------------------------------- |
| `/`         | **Home**     | Upload anh don, chon model va threshold |
| `/batch`    | **Batch**    | Upload nhieu anh de nhan dang dong thoi |
| `/realtime` | **Realtime** | Nhan dang tu webcam                     |

### Project Structure

```
FaceRecognition/
│
├── web_app.py                 # Main Flask Application
│
├── configs/                   # Configuration Files
│   ├── arcface_config.yaml
│   ├── facenet_config.yaml
│   └── lbph_config.yaml
│
├── inference/                 # Core Recognition Modules
│   ├── recognition_engine.py      # Unified Recognition Engine
│   ├── explainability.py          # Grad-CAM Implementation
│   ├── extract_embeddings.py      # Embedding Extraction
│   ├── database_builder.py        # Database Builder
│   └── evaluate.py                # Model Evaluation
│
├── models/                    # Model Weights & Checkpoints
│   ├── arcface/
│   ├── facenet/
│   └── lbphmodel/
│
├── notebooks/                 # Training & Analysis
│   ├── *_kaggle.ipynb             # Training notebooks
│   ├── evaluate_*.ipynb           # Evaluation notebooks
│   └── analysis_*.ipynb           # Analysis notebooks
│
├── preprocessing/             # Data Preprocessing
├── templates/                 # HTML Templates
├── static/                    # Static Assets
└── data/                      # Datasets & Embeddings
```

---

## API

<details open>
<summary><b>Recognition Engine</b></summary>

```python
from inference.recognition_engine import FaceRecognitionEngine

# Initialize engine
engine = FaceRecognitionEngine(
    model_type='arcface',     # 'arcface', 'facenet', or 'lbph'
    device='cuda'             # 'cuda' or 'cpu'
)

# Single recognition
result = engine.recognize(
    image_path='path/to/image.jpg',
    threshold=0.5
)

# Result structure
{
    'identity': str,          # Predicted identity
    'confidence': float,      # Confidence score [0, 1]
    'distance': float,        # Embedding distance
    'bbox': [x1, y1, x2, y2], # Face bounding box
    'top_k': [...]            # Top K predictions
}
```

</details>

<details>
<summary><b>Explainability Engine</b></summary>

```python
from inference.explainability import ArcFaceExplainabilityEngine

# Initialize explainer
explainer = ArcFaceExplainabilityEngine()

# Generate Grad-CAM visualization
heatmap = explainer.generate_gradcam(
    image_path='path/to/image.jpg',
    target_identity='person_name'
)
```

</details>

<details>
<summary><b>Database Builder</b></summary>

```python
from inference.database_builder import DatabaseBuilder

# Build embedding database
builder = DatabaseBuilder(model_type='arcface')
builder.build_from_folder('path/to/identities/')
builder.save('data/embeddings.pkl')
```

</details>

---

## Benchmarks

<div align="center">

### Performance Metrics

_Evaluated on CelebA Dataset (500 identities, 10 images/identity)_

</div>

| Metric        |                                    ArcFace                                    |                                 FaceNet                                 |                                   LBPH                                   |
| :------------ | :---------------------------------------------------------------------------: | :---------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| **Accuracy**  | ![98.2%](https://img.shields.io/badge/-98.2%25-brightgreen?style=flat-square) | ![95.6%](https://img.shields.io/badge/-95.6%25-green?style=flat-square) | ![78.4%](https://img.shields.io/badge/-78.4%25-yellow?style=flat-square) |
| **Precision** |                                     97.8%                                     |                                  94.9%                                  |                                  76.2%                                   |
| **Recall**    |                                     98.1%                                     |                                  95.3%                                  |                                  77.8%                                   |
| **F1-Score**  |                                     97.9%                                     |                                  95.1%                                  |                                  76.9%                                   |
| **AUC-ROC**   |                                     0.995                                     |                                  0.982                                  |                                  0.856                                   |

### Inference Speed

| Model   |  CPU   |  GPU  | Throughput |
| :------ | :----: | :---: | :--------: |
| ArcFace | ~120ms | ~15ms |  66 img/s  |
| FaceNet | ~100ms | ~12ms |  83 img/s  |
| LBPH    |  ~2ms  |  N/A  | 500+ img/s |

---

## Configuration

<details>
<summary><b>Model Configuration</b></summary>

```yaml
# configs/arcface_config.yaml
model:
  backbone: iresnet100
  embedding_size: 512
  pretrained: true

inference:
  threshold: 0.5
  top_k: 5

device: cuda
```

</details>

<details>
<summary><b>Environment Variables</b></summary>

| Variable               | Default    | Description             |
| :--------------------- | :--------- | :---------------------- |
| `FLASK_ENV`            | production | Flask environment       |
| `CUDA_VISIBLE_DEVICES` | 0          | GPU device ID           |
| `MODEL_PATH`           | ./models   | Model weights directory |
| `LOG_LEVEL`            | INFO       | Logging level           |

</details>

---

## Troubleshooting

| Issue                                                                               | Solution                                      |
| :---------------------------------------------------------------------------------- | :-------------------------------------------- |
| ![Error](https://img.shields.io/badge/-CUDA%20OOM-red?style=flat-square)            | Giam batch size hoac su dung CPU mode         |
| ![Error](https://img.shields.io/badge/-Model%20Load%20Failed-red?style=flat-square) | Kiem tra duong dan trong config               |
| ![Error](https://img.shields.io/badge/-No%20Face-red?style=flat-square)             | Dam bao anh co khuon mat ro rang, min 80x80px |
| ![Error](https://img.shields.io/badge/-Slow%20Inference-red?style=flat-square)      | Su dung GPU hoac giam resolution              |

---

## Contributing

<div align="center">

Contributions are welcome!

</div>

```
1. Fork the repository
2. Create feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open Pull Request
```

---

## License

<div align="center">

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

</div>

---

## References

<div align="center">

| Paper                                                            |  Conference   |
| :--------------------------------------------------------------- | :-----------: |
| ArcFace: Additive Angular Margin Loss for Deep Face Recognition  |   CVPR 2019   |
| FaceNet: A Unified Embedding for Face Recognition and Clustering |   CVPR 2015   |
| Face Description with Local Binary Patterns                      |  TPAMI 2006   |
| Joint Face Detection and Alignment Using MTCNN                   | IEEE SPL 2016 |

</div>

---

<div align="center">

<!-- Footer Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

**Developed with passion by [Tran Phuc Toan](https://github.com/sin0235)**

_HCMUTE - Digital Image Processing - 2025_

</div>
