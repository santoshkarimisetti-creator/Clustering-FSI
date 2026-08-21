# Indic Handwritten Character Recognition & Clustering Benchmark

A machine learning pipeline evaluating supervised classification (**K-Nearest Neighbors**) and unsupervised clustering (**K-Means**) on Indic script handwritten character datasets (**Devanagari** and **Gujarati**) reduced via **Principal Component Analysis (PCA)**.

---

## 📌 Project Overview

This repository provides reproducible benchmarks for handwritten Indic character recognition and clustering. Processing complex Indic scripts requires robust preprocessing, feature extraction, and dimensionality reduction.

The project evaluates two distinct Indic scripts:
1. **Devanagari Script** (dataset: [`rishianand/devanagari-character-set`](https://www.kaggle.com/datasets/rishianand/devanagari-character-set))
2. **Gujarati Script** (dataset: [`meet1265/gujarati-handwritten-characters-dataset`](https://www.kaggle.com/datasets/meet1265/gujarati-handwritten-characters-dataset))

Each pipeline performs automated dataset retrieval, background polarity normalization, data-leak-free train/test splitting, dynamic PCA component selection, hyperparameter tuning via cross-validation, supervised classification, and unsupervised cluster quality evaluation.

---

## 🛠️ Pipeline Architecture

```
     [ Kaggle Dataset via kagglehub ]
                    │
                    ▼
     [ Preprocessing & Normalization ]
    • Resizing to 32x32 grayscale
    • Stroke polarity correction (Background ~0, Ink ~1)
    • Class imbalance filtering (min samples/class = 15)
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
[ Supervised ]  [ Unsupervised ]
  KNN + CV        K-Means (k=n_classes)
  (GridSearch)    (Silhouette, ARI, NMI)
```

---

## ✨ Key Features

- **Automated Data Retrieval**: Integrates `kagglehub` to directly download and manage Kaggle datasets.
- **Polarity-Aware Preprocessing**: Automatically detects dark vs. light backgrounds and aligns image polarities so stroke pixels consistently evaluate as high intensity ($\sim 1.0$) and background pixels as zero ($\sim 0.0$).
- **Strict Leakage Prevention**: Scaling and PCA transformation matrices are fitted strictly on training splits before transforming test splits.
- **Adaptive PCA Dimensionality Reduction**: Dynamically selects the minimum number of principal components needed to capture **95% cumulative variance** (capped at 150 components).
- **Hyperparameter-Tuned KNN**: Employs `GridSearchCV` with `StratifiedKFold` cross-validation to search optimal neighbor counts ($k$) and distance weighting mechanisms (`uniform` vs. `distance`).
- **Cluster Diagnostics & Evaluation**: Evaluates K-Means clustering using intrinsic metric sweeps (Elbow method inertia & Silhouette score) alongside ground-truth metrics:
  - **Silhouette Score** (cluster compactness and separation)
  - **Adjusted Rand Index (ARI)** (similarity to true class assignments)
  - **Normalized Mutual Information (NMI)** (mutual information shared with true labels)
- **Rich Visual Diagnostics**:
  - Sample character grid displaying normalized characters
  - PCA Cumulative Explained Variance scree plot
  - 2D PCA scatter projection of true labels
  - Row-normalized confusion matrices for classification analysis
  - K-Means Elbow & Silhouette score curves
  - 2D PCA cluster distribution with mapped cluster centroids

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed along with the required libraries:

```bash
pip install numpy matplotlib pillow scikit-learn kagglehub
```

> **Note**: `kagglehub` requires standard Kaggle API authentication credentials (`kaggle.json` or standard environment variables) if accessing restricted datasets.

### Running the Pipelines

Run the scripts directly from the terminal:

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
├── Devanagari.py   # Full pipeline for Devanagari character recognition & clustering
├── Gujarathi.py    # Full pipeline for Gujarati character recognition & clustering
└── README.md       # Comprehensive project documentation
```

---

## 📊 Evaluation & Metrics

The scripts output performance summaries across supervised and unsupervised paradigms:

| Evaluation Paradigm | Model | Primary Metrics Evaluated |
| :--- | :--- | :--- |
| **Supervised** | **KNN** | Accuracy, Precision, Recall, F1-Score, Normalized Confusion Matrix |
| **Unsupervised** | **K-Means** | Silhouette Score, Adjusted Rand Index (ARI), Normalized Mutual Info (NMI), Inertia |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more details.
