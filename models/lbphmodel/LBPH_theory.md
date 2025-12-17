# Local Binary Patterns Histograms (LBPH)

## 1. Introduction

**Local Binary Patterns Histograms (LBPH)** is a traditional face recognition method based on **local texture descriptors**. Unlike deep learning approaches, LBPH does not learn embeddings from large datasets but instead encodes local intensity patterns in grayscale images.

LBPH is widely used as a **baseline method** in face recognition research, especially when comparing classical approaches with modern deep learning models such as **ArcFace** and **FaceNet**.

---

## 2. Local Binary Pattern (LBP)

### 2.1 Concept

Local Binary Pattern (LBP) was first introduced by **Ojala et al. (1996)** as a texture descriptor. The core idea is to describe the local neighborhood of a pixel by comparing it with surrounding pixels.

---

### 2.2 LBP Operator

For a center pixel ( g_c ), consider ( P ) neighboring pixels ( g_p ) on a circle of radius ( R ).

Each neighbor is thresholded against the center pixel:

* If ( g_p \ge g_c ) → assign **1**
* Otherwise → assign **0**

The resulting binary pattern is converted into a decimal number:

```
LBP(P, R) = Σ s(g_p − g_c) · 2^p

s(x) = 1 if x ≥ 0, else 0
```

---

### 2.3 Properties of LBP

* ✔ Invariant to monotonic gray-scale changes
* ✔ Computationally efficient
* ❌ Sensitive to noise
* ❌ Captures only local texture information

---

## 3. LBPH for Face Recognition

### 3.1 From LBP to LBPH

LBPH extends the LBP operator by incorporating spatial information:

1. The face image is divided into a grid of cells
2. An LBP histogram is computed for each cell
3. All histograms are concatenated into a single feature vector

This representation encodes the **distribution of local texture patterns** across the face.

---

### 3.2 LBPH Pipeline

1. Face detection (e.g., Haar Cascade)
2. Grayscale conversion
3. Face alignment and resizing
4. LBP feature extraction
5. Histogram comparison using distance metrics

---

## 4. LBPH Parameters

| Parameter       | Description                     |
| --------------- | ------------------------------- |
| `radius (R)`    | Radius of circular neighborhood |
| `neighbors (P)` | Number of surrounding pixels    |
| `grid_x`        | Number of horizontal grid cells |
| `grid_y`        | Number of vertical grid cells   |

Typical configuration:

```python
radius = 1
neighbors = 8
grid_x = 8
grid_y = 8
```

---

## 5. Confidence Score in LBPH

LBPH does not output probabilities. Instead, it returns a **distance value**:

* Smaller distance → higher similarity
* Larger distance → lower similarity

A **threshold** is required to decide whether a prediction is accepted.

This leads to a trade-off between:

* **Accuracy**: correctness of accepted predictions
* **Coverage**: proportion of images accepted

---

## 6. Advantages and Limitations

### 6.1 Advantages

* ✔ No GPU required
* ✔ Fast training and inference
* ✔ Simple implementation (OpenCV support)
* ✔ Suitable as a baseline model

---

### 6.2 Limitations

* ❌ Sensitive to pose, illumination, and occlusion
* ❌ Poor generalization on unconstrained datasets
* ❌ Not scalable to large-scale face recognition tasks

---

## 7. LBPH in Modern Face Recognition Research

In recent studies, LBPH is mainly used as:

* A **traditional baseline**
* A comparison point for deep learning models such as:

  * ArcFace
  * FaceNet
  * CosFace

LBPH highlights the performance gap between handcrafted features and deep learned embeddings.

---

## 8. Comparison with Deep Learning Methods

| Criterion     | LBPH                | ArcFace / FaceNet |
| ------------- | ------------------- | ----------------- |
| Feature type  | Handcrafted texture | Deep embeddings   |
| Training cost | Low                 | High              |
| Robustness    | Low                 | High              |
| Scalability   | Limited             | Excellent         |
| Typical role  | Baseline            | State-of-the-art  |

---

## 9. Conclusion

LBPH is a simple yet effective classical face recognition approach for small-scale problems. In this project, LBPH is used as a **baseline method** to compare against deep learning-based face recognition systems.

---

## 10. References

1. Ojala, T., Pietikäinen, M., & Harwood, D. (1996). *A comparative study of texture measures with classification based on featured distributions*. Pattern Recognition.
2. Ahonen, T., Hadid, A., & Pietikäinen, M. (2006). *Face description with local binary patterns: Application to face recognition*. IEEE TPAMI.
3. OpenCV Documentation – LBPHFaceRecognizer.
4. Taigman et al. (2014). *DeepFace: Closing the Gap to Human-Level Performance in Face Verification*.
