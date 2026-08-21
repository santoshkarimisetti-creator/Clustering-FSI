# Indic Handwritten Character Recognition & Clustering Benchmark

A modular, high-performance machine learning framework evaluating supervised classification (**K-Nearest Neighbors**) and unsupervised clustering (**K-Means**, **Gaussian Mixture Models (GMM)**, **Agglomerative Clustering**) on Indic script handwritten character datasets (**Devanagari** and **Gujarati**) using **Histogram of Oriented Gradients (HOG)** and **Principal Component Analysis (PCA)**.

---

## 📌 Project Overview

Processing complex Indic scripts requires robust preprocessing, feature extraction, and dimensionality reduction. This repository provides reproducible, leak-free machine learning benchmarks for two prominent Indic scripts:

1. **Devanagari Script** (dataset: [`rishianand/devanagari-character-set`](https://www.kaggle.com/datasets/rishianand/devanagari-character-set))
2. **Gujarati Script** (dataset: [`meet1265/gujarati-handwritten-characters-dataset`](https://www.kaggle.com/datasets/meet1265/gujarati-handwritten-characters-dataset))

Each pipeline executes automated dataset retrieval, **Otsu's adaptive binarization**, multi-threaded parallel image loading, **HOG stroke-direction feature extraction**, data-leak-free train/test splitting, dynamic PCA component selection, hyperparameter tuning via cross-validation, multi-algorithm clustering, and **t-SNE non-linear manifold visualizations**.

---

## 🛠️ Modular Pipeline Architecture

```
                 [ Kaggle Dataset via kagglehub ]
                                │
                                ▼
           [ Multi-Threaded Parallel Data Loading ]
            (ThreadPoolExecutor + Dataset Caching)
                                │
                                ▼
              [ Image Preprocessing & Segmentation ]
             • Resizing to 32x32 grayscale
             • Otsu Adaptive Thresholding (cv2.THRESH_OTSU)
             • Stroke polarity normalization (Background ~0, Ink ~1)
             • Class imbalance filtering (min samples/class = 15)
                                │
                                ▼
           [ Feature Extraction: HOG / Raw Pixels ]
            • Histogram of Oriented Gradients (skimage.feature.hog)
                                │
                                ▼
                   [ 80/20 Stratified Split ]
                     (Prevents Data Leakage)
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
               [ Standard Scaler ]   [ Standard Scaler ]
               (Fit on Train)        (Transform Test)
                       │                 │
                       ▼                 ▼
                 [ PCA Fit ]       [ PCA Transform ]
               (Target: 95% Var)     (Projected)
                       │                 │
                 ┌─────┴────────┐        │
                 ▼              ▼        ▼
            [ Supervised ]  [ Unsupervised Benchmarks ]
              KNN + CV        • K-Means Clustering
              (GridSearch)    • Gaussian Mixture Models (GMM)
                              • Agglomerative Hierarchical
                                │
                                ▼
                     [ Visual Manifold Projections ]
                      • 2D PCA Linear Scatter Plots
                      • 2D t-SNE Non-Linear Projections
```

---

## ✨ Key Features & Technical Highlights

- **Modular Architecture**: Centralized core engine in [`indic_pipeline.py`](file:///d:/Projects/Clustering/indic_pipeline.py) powering clean entry points [`Devanagari.py`](file:///d:/Projects/Clustering/Devanagari.py) and [`Gujarathi.py`](file:///d:/Projects/Clustering/Gujarathi.py).
- **High-Speed Parallel Preprocessing**: Multi-threaded image reading via `ThreadPoolExecutor` and automatic disk caching with `joblib`.
- **Otsu Adaptive Binarization**: Replaces simple intensity thresholding with `cv2.THRESH_OTSU` to segment handwritten strokes clean from background noise and lighting variations.
- **HOG Feature Extraction**: Extracts **Histogram of Oriented Gradients** features (`skimage.feature.hog`) to represent character stroke orientations and edge shapes.
- **Data-Leak-Free Pipeline**: Scaling and PCA transformation matrices are fitted strictly on training splits before transforming test splits.
- **Adaptive PCA Dimensionality Reduction**: Dynamically selects the minimum number of principal components needed to capture **95% cumulative variance** (capped at 150 components).
- **GridSearch Hyperparameter-Tuned KNN**: Searches optimal neighbor counts ($k$) and distance weighting mechanisms (`uniform` vs. `distance`) using `StratifiedKFold` cross-validation.
- **Multi-Algorithm Clustering Benchmark**: Evaluates three distinct clustering algorithms:
  - **K-Means Clustering**
  - **Gaussian Mixture Models (GMM)**
  - **Agglomerative Hierarchical Clustering**
  Evaluating intrinsic Silhouette scores, **Adjusted Rand Index (ARI)**, and **Normalized Mutual Information (NMI)** against true character labels.
- **Non-Linear Manifold Visualizations**: Renders side-by-side **2D PCA** linear projections and **2D t-SNE** manifold visual clusters.

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed along with the required libraries:

```bash
pip install numpy matplotlib pillow scikit-learn scikit-image opencv-python joblib kagglehub
```

### Running the Pipelines

Execute the benchmark scripts directly:

**1. Devanagari Character Recognition & Clustering Pipeline:**
```bash
python Devanagari.py
```

**2. Gujarati Character Recognition & Clustering Pipeline:**
```bash
python Gujarathi.py
```

---

## 📁 Repository Structure

```
.
├── indic_pipeline.py # Shared core engine (Data loading, HOG, PCA, KNN, Clustering, t-SNE)
├── Devanagari.py     # Devanagari script benchmark runner
├── Gujarathi.py      # Gujarati script benchmark runner
└── README.md         # Project documentation & benchmark overview
```

---

## 📊 Evaluation & Metrics Summary

| Evaluation Paradigm | Model / Method | Metrics Evaluated |
| :--- | :--- | :--- |
| **Feature Extraction** | **HOG vs. Raw Pixels** | Gradient Orientations ($4 \times 4$ cells, $2 \times 2$ blocks) |
| **Supervised Learning** | **KNN + GridSearch** | Accuracy, Precision, Recall, F1-Score, Normalized Confusion Matrix |
| **Unsupervised Learning** | **K-Means, GMM, Agglomerative** | Silhouette Score, Adjusted Rand Index (ARI), Normalized Mutual Info (NMI) |
| **Manifold Projection** | **PCA & t-SNE** | 2D Linear Projection vs. Non-linear Perplexity Manifolds |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
