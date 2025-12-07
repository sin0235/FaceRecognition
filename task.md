Chia Vai Trò (4 Thành Viên)
Thành viên	Vai trò	Nhiệm vụ
A	Data Pipeline Engineer	Xử lý dữ liệu, preprocessing, augmentation
B	ArcFace Engineer	Triển khai và fine-tune ArcFace model (model chính)
C	FaceNet Engineer	Triển khai và fine-tune FaceNet model (model phụ)
D	Integration & Interface Engineer	Xây dựng inference engine, Streamlit, đánh giá


 
Hình 1. Cấu trúc project (Tạm, có thể tuỳ chỉnh theo thực tế triển khai)
Task 1.1: Khởi Tạo Project
Phân công cụ thể:
Bước	Nội dung	Dependencies
A	Setup môi trường Python, tạo requirements.txt với dependencies preprocessing	opencv, mtcnn, pillow, numpy, pandas
B	Thêm dependencies cho ArcFace	insightface, mxnet, torch, torchvision
C	Thêm dependencies cho FaceNet	facenet-pytorch, tensorflow nếu cần
D	Thêm dependencies cho Streamlit	streamlit, plotly, scikit-learn, matplotlib

GIAI ĐOẠN 2: XỬ LÝ DỮ LIỆU 
Task 2.1: Download và Tổ Chức Dataset [Thành viên A]
Module: preprocessing/data_preparation.py
Nhiệm vụ chi tiết:
1.	Download CelebA dataset từ Kaggle (202,599 ảnh)
2.	Download files metadata: identity_CelebA.txt, list_attr_celeba.txt
3.	Viết script download_celeba.py:
•	Function download_from_kaggle(): tự động download qua Kaggle API
•	Function verify_dataset(): check integrity (số file, format)
•	Function parse_identity_file(): đọc file identity, đếm số ảnh per person
4.	Phân tích và chọn subset:
•	Lọc các identity có >= 40 ảnh
•	Chọn 80-100 identities (cân bằng giữa đủ data và khả thi)
•	Lưu danh sách: data/selected_identities.txt
5.	Tạo Jupyter notebook EDA_CelebA.ipynb:
•	Visualize phân bố số ảnh per identity (histogram)
•	Hiển thị sample images (grid 10x10)
•	Phân tích attributes (pose, lighting, occlusion)
Output:
•	Dataset organized trong data/raw/
•	File selected_identities.txt với 80-100 identities
•	Notebook EDA với visualizations
Task 2.2: Face Detection Module [Thành viên A]
Module: preprocessing/face_detector.py
Nhiệm vụ chi tiết:
1.	Implement class FaceDetector
2.	So sánh 2 phương pháp:
•	MTCNN: install mtcnn, test trên 100 ảnh sample
•	RetinaFace: install retinaface, test trên cùng 100 ảnh
•	Tạo bảng so sánh: detection rate, confidence, speed
•	Chọn method tốt nhất (recommend MTCNN vì dễ dùng)
3.	Handle edge cases:
•	No face detected → return None + error message
•	Multiple faces → chọn bounding box lớn nhất
•	Face quá nhỏ  → reject
4.	Viết unit tests trong tests/test_face_detector.py:
•	Test với ảnh có face rõ
•	Test với ảnh không có face
•	Test với ảnh có nhiều faces
•	Test với ảnh blur, low quality
5.	Batch processing function:
•	detect_batch(image_paths, output_csv)
•	Chạy toàn bộ selected dataset
•	Lưu results: data/detection_results.csv (image_path, bbox, landmarks, confidence, status)
Output:
•	Module face_detector.py hoàn chỉnh
•	Detection results CSV cho toàn bộ dataset
•	Unit tests pass
Task 2.3: Face Alignment & Preprocessing [Thành viên A]
Module: preprocessing/face_preprocessing.py
Nhiệm vụ	Chi tiết
Implement class	FacePreprocessor
Face alignment algorithm	Tính eye centers từ landmarks, Compute rotation angle để eyes nằm ngang, Apply affine transformation với cv2.warpAffine, Expand bounding box thêm 30% margin, Crop aligned face
ArcFace pipeline	Resize to 112x112 pixels, Normalize, Convert BGR → RGB
FaceNet pipeline	Resize to 160x160 pixels, Normalize to [-1, 1], Specific preprocessing theo pretrained model
Quality enhancement	Apply CLAHE cho contrast normalization, Gaussian blur (kernel 3x3) để reduce noise, Brightness adjustment nếu cần
Data augmentation cho training	Horizontal flip (p=0.5), Random rotation (±10 degrees), Color jitter (brightness ±20%, contrast ±20%), chỉ augment training set, không augment val/test
Batch processing	Function process_dataset(input_dir, output_dir, model_type), Progress bar với tqdm, Xử lý song song với multiprocessing (4-8 workers), Lưu preprocessed images vào data/processed/arcface/ và data/processed/facenet/
Output:
•	Module face_preprocessing.py hoàn chỉnh
•	Preprocessed dataset cho cả ArcFace và FaceNet
•	Before/after alignment visualization
Task 2.4: Dataset Split và Metadata [Thành viên A]
Module: preprocessing/create_splits.py
Nhiệm vụ chi tiết:
Nội dung	Chi tiết
Stratified split	Đảm bảo mỗi identity có ảnh trong cả 3 sets
Train	70% (minimum 20 ảnh/identity)
Validation	15% (minimum 5 ảnh/identity)
Test	15% (minimum 5 ảnh/identity)
Script	create_splits.py
Metadata CSV files	train_metadata.csv, val_metadata.csv, test_metadata.csv (columns: image_path, identity_id, identity_name, split)
Tổ chức file	Copy/symlink files vào data/processed/train/, val/, test/; mỗi folder có subfolders theo identity
Statistics report	Total identities: X; Train: Y images, Z identities; Val: Y images, Z identities; Test: Y images, Z identities; Class balance metrics (std deviation của số ảnh per identity)
Validation	Verify không có duplicate images giữa splits; Check data leakage; Visualize distribution với bar charts
Output:
•	Organized dataset với train/val/test splits
•	3 metadata CSV files
•	Split statistics document
GIAI ĐOẠN 3: TRAINING MODELS
Task 3.1: ArcFace Model Implementation [Thành viên B]
Module: models/arcface/arcface_model.py
Nhiệm vụ chi tiết:
Step	Details
Install InsightFace library	
Implement class ArcFaceModel	
Download pretrained weights	ResNet50 backbone trained trên MS1MV2 (3.8M identities), Từ InsightFace model zoo, Lưu vào models/checkpoints/arcface_pretrained.pth
Setup ArcFace Loss	
Freezing strategy	Freeze 80% đầu của backbone (early layers), Unfreeze last few blocks + classifier head
Implement function freeze_layers(model, freeze_ratio=0.8)	
Tạo config file configs/arcface_config.yaml	
Implement forward pass testing	Test với 1 batch ảnh, Verify output shape (batch_size, 512), Check gradients flow correctly
Output:
•	arcface_model.py với đầy đủ functions
•	Pretrained weights downloaded
•	Config file documented
•	Forward pass tested
Task 3.2: ArcFace Data Loader [Thành viên B]
Module: models/arcface/arcface_dataloader.py
Nhiệm vụ chi tiết:
1.	Implement PyTorch Dataset class:
2.	Load data từ metadata CSV:
•	Parse CSV với pandas
Step	Details
Map identity names to integer labels	0 to N-1
Store image paths và labels	
Training transforms	RandomHorizontalFlip(p=0.5), RandomRotation(degrees=10), ColorJitter(brightness=0.2, contrast=0.2), ToTensor(), Normalize(mean=[0.5], std=[0.5])
Validation/Test transforms	ToTensor(), Normalize(mean=[0.5], std=[0.5])
Create DataLoaders	
Test DataLoader	Iterate qua 1 epoch, Measure loading speed (images/second), Verify batch shapes, Check augmentation working (visualize augmented samples)
Implement caching	Option để cache preprocessed images vào RAM, Tăng tốc độ training nếu có đủ memory
Output:
•	arcface_dataloader.py hoàn chỉnh
•	Tested DataLoader với visualizations
•	Loading speed benchmarked (>100 img/s)
Task 3.3: ArcFace Training Loop [Thành viên B]
Module: models/arcface/train_arcface.py
Nhiệm vụ chi tiết:
Step	Details
Load config	từ YAML
Initialize	model, loss, optimizer, scheduler
Load pretrained weights	
Freeze layers	theo strategy
Training loop	Forward pass, Compute ArcFace loss, Backward pass, Gradient clipping (max_norm=5.0), Optimizer step, Scheduler step
Logging với TensorBoard	Training loss per iteration, Validation loss per epoch, Validation accuracy per epoch, Learning rate per epoch, Embedding visualization mỗi 5 epochs (t-SNE)
Model checkpointing	Save best model dựa trên validation accuracy, Save checkpoint mỗi 10 epochs, Lưu vào models/checkpoints/arcface_best.pth
Early stopping	Patience = 10 epochs, Stop nếu validation loss không improve
Run training	Train trên GPU nếu có, Monitor training progress
Target	>85% validation accuracy
Thời gian dự kiến	4-8 giờ tùy hardware

Output:
•	train_arcface.py script hoàn chỉnh
•	Trained model checkpoint
•	TensorBoard logs
•	Training report (loss curves, accuracy curves)
Task 3.4: FaceNet Model Implementation [Thành viên C]
Module: models/facenet/facenet_model.py
Nhiệm vụ chi tiết:
Step	Details
Install facenet-pytorch	
Implement class	FaceNetModel
Download pretrained model	InceptionResNetV1 trained trên VGGFace2, từ facenet-pytorch library
Embedding options	512-dim hoặc 128-dim embeddings
Triplet Loss implementation	
Triplet mining strategy	Semi-hard negative mining, Online mining trong mỗi batch
Function	mine_triplets(embeddings, labels, margin)
Create config file	configs/facenet_config.yaml
Forward pass testing	Test với 1 batch triplets, Verify embedding shapes, Check triplet loss computation

Output:
•	facenet_model.py hoàn chỉnh
•	Pretrained weights loaded
•	Config file
•	Forward pass tested
Task 3.5: FaceNet Data Loader với Triplet Sampling [Thành viên C]
Module: models/facenet/facenet_dataloader.py
Nhiệm vụ chi tiết:
Task	Description/Steps
Triplet sampling logic	Cho mỗi anchor image (identity A): Sample 1 positive: cùng identity A, ảnh khác; Sample 1 negative: khác identity A
Efficient sampling	Pre-build dict: {identity: [list of image paths]}; Random sample từ dict
Hard negative mining (optional)	Compute embeddings cho toàn bộ training set; Build kNN index với FAISS; Sample hard negatives (gần anchor nhưng khác identity)
Transforms cho FaceNet - Training	Resize(160, 160); RandomHorizontalFlip(); ToTensor(); Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
Transforms cho FaceNet - Validation/Test	Không flip, chỉ normalize
Create TripletDataLoader	
Validation	Visualize sample triplets; Verify anchor-positive cùng identity; Verify anchor-negative khác identity; Check data loading speed

Output:
•	facenet_dataloader.py với triplet sampling
•	Tested DataLoader
•	Triplet visualizations
Task 3.6: FaceNet Training Loop [Thành viên C]
Module: models/facenet/train_facenet.py
Nhiệm vụ chi tiết:
Step	Details
Training loop	Load anchors, positives, negatives; Extract embeddings for all 3; Compute triplet loss; Backward và optimize; Log: triplet loss, hard negative ratio
Validation strategy	Validation không dùng triplet loss trực tiếp; Tính accuracy trên validation set; Extract embeddings cho tất cả val images; Nearest neighbor classification; Compute top-1 accuracy
Logging với TensorBoard	Triplet loss per iteration; Validation accuracy per epoch; Hard negative percentage; Embedding space visualization (t-SNE)
Model checkpointing	Save best model dựa trên validation accuracy; Early stopping patience=10
Run training	Target: >80-85% validation accuracy; Thời gian: 4-8 giờ

Output:
•	train_facenet.py hoàn chỉnh
•	Trained FaceNet checkpoint
•	TensorBoard logs
•	Training report
GIAI ĐOẠN 4: INFERENCE VÀ EVALUATION 
Task 4.1: Embedding Extraction [Thành viên D - với support từ B, C]
Module: inference/extract_embeddings.py
Nhiệm vụ chi tiết:
Step	Description	Output/Files
Load checkpoints	Load best checkpoints từ B và C	
Extract embeddings	Iterate qua training set, extract embeddings, lưu vào numpy arrays	data/embeddings/arcface_train_embeddings.npy, data/embeddings/facenet_train_embeddings.npy
Lưu metadata	Lưu image_path, identity, embedding_index	data/embeddings/train_metadata.csv
Compute reference embeddings	Average embeddings per identity, tạo "prototype" cho mỗi người nổi tiếng	data/embeddings/arcface_prototypes.npy
Build FAISS index	Index ArcFace embeddings, Index FaceNet embeddings	arcface_index.faiss, facenet_index.faiss
Visualization	t-SNE projection 512D → 2D, plot colored by identity, verify clusters rõ ràng	

Output:
•	Embedding files cho cả 2 models
•	FAISS indexes
•	t-SNE visualization
Task 4.2: Recognition Engine [Thành viên D]
Module: inference/recognition_engine.py
Nhiệm vụ chi tiết:
Component	Description
Class	RecognitionEngine
Matching pipeline	Input: query image path; Preprocessing (gọi module của A); Extract embedding (load model của B hoặc C); FAISS search để tìm k-nearest neighbors; Compute cosine similarity scores; Rank results
Distance metrics	ArcFace: Cosine similarity = 1 - cosine_distance; FaceNet: Euclidean distance, có thể convert sang similarity
Threshold-based decision	If max_similarity > threshold → return identity; Else → return "Unknown"; Threshold sẽ tune ở Task 4.3
Ensemble method	Combine predictions từ ArcFace và FaceNet; Weighted voting: ArcFace weight=0.6, FaceNet=0.4; Average similarity scores
Optimization	Cache loaded models trong memory; Batch processing cho multiple images; Target latency: <500ms per image

Output:
•	recognition_engine.py hoàn chỉnh
•	Support cả ArcFace, FaceNet, Ensemble
•	Latency benchmarked
Task 4.3: Threshold Tuning & Evaluation [Thành viên D]
Module: inference/evaluate.py
Nhiệm vụ chi tiết:
Task	Details
Implement threshold sweep	Threshold optimization trên validation set, test thresholds từ 0.3 đến 0.9, step 0.05, tính metrics: TPR, FPR, Accuracy, F1-score, plot ROC curve, chọn threshold tại EER point (TPR = 1-FPR) hoặc maximize F1-score
Evaluate trên test set với optimal threshold	Metrics: Top-1 Accuracy, Top-5 Accuracy, Precision, Recall, F1 per identity, Confusion Matrix, Average Inference Time
Compare ArcFace vs FaceNet	Side-by-side metrics table, statistical significance test (paired t-test), win/tie/loss count
Error analysis	Identify misclassified images, categorize errors: lighting issues, pose variations, occlusions (glasses, masks), similar-looking identities, visualize failure cases (grid)
Performance breakdown	Accuracy by quality bins (low/medium/high), accuracy by attributes (smiling, young/old, male/female)
Generate report	results/evaluation_report.md, tables, charts, analysis, recommendations

Output:
•	Optimal thresholds cho cả 2 models
•	Complete evaluation report
•	ROC curves, confusion matrices
•	Error analysis document
Task 4.4: Explainability - Grad-CAM [Thành viên D]
Module: inference/explainability.py
Nhiệm vụ chi tiết:
Feature	Details
Grad-CAM Visualization Process	Forward pass để get prediction, Backward từ predicted class, Compute gradients w.r.t. target layer, Generate heatmap, Overlay lên original image
Visualization	Heatmap màu (red=high importance, blue=low), Overlay với alpha blending, Side-by-side: original | Grad-CAM | top-3 predictions
Batch Generation	Generate Grad-CAM cho 50-100 test images, Lưu vào results/gradcam/, Tạo HTML gallery để browse
Embedding Space Visualization	Interactive plot với Plotly, t-SNE projection, Click vào point → show image, Highlight query và k-nearest neighbors
Integration vào Recognition Pipeline	Option explain=True trong recognize() function, Return both prediction và explanation

Output:
•	explainability.py module
•	Grad-CAM visualizations
•	Interactive embedding space plot
•	HTML gallery
GIAI ĐOẠN 5: STREAMLIT INTERFACE 
Task 5.1: Basic App Structure [Thành viên D]
Module: app/streamlit_app.py
Nhiệm vụ chi tiết:
Feature	Description
Page navigation	Sidebar radio: "Home", "Recognition", "Batch Processing", "Evaluation", "About"
Multi-page app structure	Implemented
Session state initialization	Cache loaded models với @st.cache_resource, Cache embeddings database, Store user settings
Sidebar settings	Model selection: ArcFace / FaceNet / Ensemble, Threshold adjustment slider, Display options checkboxes
Styling	Custom CSS cho professional look, Color scheme consistent, Responsive layout
Test	Test basic navigation

Output:
•	App structure functional
•	Navigation working
•	Settings persistent
Task 5.2: Input Interface [Thành viên D]
Module: app/pages/recognition.py
Nhiệm vụ chi tiết:
Feature	Description
Upload	st.file_uploader(type=['jpg', 'jpeg', 'png'])
Webcam	st.camera_input("Take a photo")
URL	st.text_input("Image URL")
Sample gallery	dropdown với test images
Input validation	Check file size (< 10MB), Validate image format, Preview uploaded image
Processing workflow	Display uploaded image, Button "Recognize Face", Progress spinner during processing, Display results
Error handling	No face detected → friendly error message, Invalid format → helpful guidance, Network error → retry option

Output:
•	Input interface functional
•	All input methods working
•	Validation robust
Task 5.3: Recognition Results Display [Thành viên D]
Module: app/components/results_display.py
Nhiệm vụ chi tiết:
Section	Details
Results layout (2 columns)	Left: Input image, preprocessing info; Right: Predictions, references
Prediction display	Large card: Top-1 identity + confidence; Reference images từ database (5 ảnh); Similarity score bar chart
Top-K results table Columns	Rank, Identity, Confidence, Similarity; Thumbnail của reference images; Expandable để see details
Model comparison view	Side-by-side: ArcFace | FaceNet; Highlight nếu predictions khác nhau; Explain why có sự khác biệt
Explainability section (expandable)	Grad-CAM heatmap; Embedding space visualization; "Why this prediction?" explanation
Download options	Button download JSON results; Button download annotated image

Output:
•	Results display polished
•	Multiple visualizations
•	User-friendly
Task 5.4: Batch Processing Page [Thành viên D]
Module: app/pages/batch_processing.py
Nhiệm vụ chi tiết:
Feature	Description
Batch upload	st.file_uploader(accept_multiple_files=True)
Display thumbnail grid	Display thumbnail grid
Parallel processing	Process multiple images với threading
Progress bar + status updates	Progress bar + status updates
Results summary	Table: Filename, Predicted Identity, Confidence, Status
Success rate	Success rate
Average processing time	Average processing time
Export functionality	Download results CSV, Download ZIP với annotated images

Output:
•	Batch processing functional
•	Can handle 50+ images
Task 5.5: Evaluation Dashboard [Thành viên D]
Module: app/pages/evaluation_dashboard.py
Nhiệm vụ chi tiết:
Section	Details
Metric cards	Top-1 Accuracy, Top-5 Accuracy, Avg Speed
Model comparison charts	Bar charts: ArcFace vs FaceNet
Performance by quality bins	
Confusion matrix	Interactive heatmap với Plotly, Hover tooltips
Error gallery	Grid of misclassified examples, Filter by error type
ROC curves	Interactive plot, Threshold slider
Per-identity performance	Searchable table, Sort by accuracy

Output:
•	Dashboard comprehensive
•	Interactive visualizations
Task 5.6: Final Polish [Thành viên D]
Thời gian: 1 ngày
Nhiệm vụ:
Category	Details
Performance optimization	Model caching với @st.cache_resource, Embedding caching, Lazy loading components
UI polish	Consistent styling, Loading animations, Error messages friendly
About page	Project description, Team members, Dataset info, Model architectures, GitHub link
Testing	Test all features, Fix bugs, Performance profiling
Output:
•	App production-ready
•	All features working smoothly
GIAI ĐOẠN 6: TESTING & DOCUMENTATION 
Task 6.1: Unit Testing [Thành viên A]
Module: tests/
Nhiệm vụ:
•	Write unit tests cho preprocessing module
•	Test face detection, alignment, preprocessing
•	Use pytest, achieve >80% coverage
Task 6.2: Integration Testing [Thành viên B]
Thời gian: 2 ngày
Nhiệm vụ:
•	Test end-to-end pipeline
•	Test model switching
•	Performance testing
Task 6.3: Code Documentation [Thành viên C]
Thời gian: 2 ngày
Nhiệm vụ:
•	Write docstrings cho all functions
•	Generate API documentation
•	Write README.md
•	Deployment guide

