import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import kagglehub
import joblib
from skimage.feature import hog

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                               confusion_matrix, ConfusionMatrixDisplay,
                               silhouette_score, adjusted_rand_score,
                               normalized_mutual_info_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image_files(root):
    """Recursively search for valid image files."""
    files = []
    for ext in VALID_EXT:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(files))


def infer_label(filepath, root):
    """Extract class label from immediate parent directory."""
    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)
    return parts[-2] if len(parts) >= 2 else "unknown"


def load_single_image(fp, img_size=32):
    """
    Load, resize, and apply Otsu thresholding & polarity normalization.
    Ensures stroke ink pixels are high intensity (~1.0) and background is ~0.0.
    """
    try:
        # Load image as grayscale numpy array
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL if OpenCV read fails
            pil_img = Image.open(fp).convert("L")
            img = np.array(pil_img)

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Otsu thresholding for sharp binarization invariant to lighting/scan contrast
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr = thresh.astype(np.float32) / 255.0

        # Standardize stroke polarity (Background ~0, Ink strokes ~1)
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        return arr
    except Exception:
        return None


def extract_hog_features(img, img_size=32):
    """
    Extract Histogram of Oriented Gradients (HOG) features for stroke directionality.
    """
    features = hog(
        img,
        orientations=8,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )
    return features


def load_dataset_parallel(dataset_name, kaggle_handle, max_samples=8000, img_size=32, min_per_class=15):
    """
    Download dataset via kagglehub and load images in parallel using ThreadPoolExecutor.
    """
    cache_dir = os.path.join(".cache", dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"dataset_{max_samples}_{img_size}.joblib")

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        return joblib.load(cache_file)

    print(f"Downloading dataset '{kaggle_handle}' via kagglehub...")
    path = kagglehub.dataset_download(kaggle_handle)
    print("Path to dataset files:", path)

    print("Scanning for image files...")
    all_files = find_image_files(path)
    if len(all_files) == 0:
        raise RuntimeError(f"No image files found under {path}.")
    print(f"Found {len(all_files)} total image files.")

    labels_raw = [infer_label(f, path) for f in all_files]

    # Filter classes with insufficient samples
    counts = Counter(labels_raw)
    keep_classes = {c for c, n in counts.items() if n >= min_per_class}
    filtered_pairs = [(f, l) for f, l in zip(all_files, labels_raw) if l in keep_classes]
    all_files, labels_raw = zip(*filtered_pairs)
    all_files, labels_raw = list(all_files), list(labels_raw)

    print(f"Using {len(set(labels_raw))} classes after filtering, {len(all_files)} images total.")

    # Subsample if dataset exceeds max_samples
    if len(all_files) > max_samples:
        np.random.seed(RANDOM_STATE)
        idx = np.random.choice(len(all_files), max_samples, replace=False)
        all_files = [all_files[i] for i in idx]
        labels_raw = [labels_raw[i] for i in idx]
        print(f"Subsampled to {max_samples} images for computational efficiency.")

    print(f"Loading and preprocessing {len(all_files)} images using multi-threading...")
    t0 = time.time()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_image, fp, img_size) for fp in all_files]
        loaded_images = [f.result() for f in futures]

    valid_pairs = [(im, l) for im, l in zip(loaded_images, labels_raw) if im is not None]
    print(f"Loaded {len(valid_pairs)} images in {time.time() - t0:.2f} seconds.")

    X_images = np.array([im for im, _ in valid_pairs])
    labels = [l for _, l in valid_pairs]

    # Re-verify minimum class counts
    counts = Counter(labels)
    keep_classes = {c for c, n in counts.items() if n >= min_per_class}
    keep_mask = [l in keep_classes for l in labels]
    X_images = X_images[keep_mask]
    labels = [l for l, k in zip(labels, keep_mask) if k]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_

    dataset_dict = {
        "X_images": X_images,
        "y": y,
        "class_names": class_names,
        "dataset_name": dataset_name
    }

    joblib.dump(dataset_dict, cache_file)
    print(f"Saved dataset cache to {cache_file}.")
    return dataset_dict


def run_indic_benchmark(dataset_name, kaggle_handle, max_samples=8000, img_size=32, min_per_class=15, feature_type="hog"):
    """
    Main benchmark pipeline runner for Indic script character analysis.

    feature_type: 'raw' (raw pixels) or 'hog' (Histogram of Oriented Gradients)
    """
    print(f"\n============================================================")
    print(f" RUNNING BENCHMARK: {dataset_name.upper()} (Feature: {feature_type.upper()})")
    print(f"============================================================")

    # 1. Load Dataset
    data = load_dataset_parallel(dataset_name, kaggle_handle, max_samples, img_size, min_per_class)
    X_images = data["X_images"]
    y = data["y"]
    class_names = data["class_names"]

    # 2. Visualize Sample Images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    sample_idx = np.random.choice(len(X_images), 10, replace=False)
    for ax, i in zip(axes.ravel(), sample_idx):
        ax.imshow(X_images[i], cmap="gray")
        ax.set_title(str(class_names[y[i]]), fontsize=9)
        ax.axis("off")
    plt.suptitle(f"Sample {dataset_name} Characters (Otsu Preprocessed)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 3. Extract Features
    if feature_type == "hog":
        print("Extracting HOG (Histogram of Oriented Gradients) features...")
        X_features = np.array([extract_hog_features(img, img_size) for img in X_images])
    else:
        print("Using raw pixel intensity features...")
        X_features = X_images.reshape(len(X_images), -1)

    print(f"Feature matrix shape: {X_features.shape}, Total Classes: {len(class_names)}")

    # 4. Train / Test Split (BEFORE scaler and PCA -> no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    min_class_in_train = min(Counter(y_train).values())
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Min samples/class in train: {min_class_in_train}")

    # 5. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. PCA Component Selection (95% Cumulative Variance Target)
    variance_target = 0.95
    max_components_cap = min(150, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])

    pca_probe = PCA(n_components=max_components_cap, svd_solver="randomized", random_state=RANDOM_STATE)
    pca_probe.fit(X_train_scaled)
    cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, variance_target) + 1)
    n_components = int(np.clip(n_components, 15, max_components_cap))

    print(f"PCA components required for {variance_target*100:.0f}% variance: {n_components} (Cap={max_components_cap})")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Actual explained variance ratio retained: {np.sum(pca.explained_variance_ratio_):.4f}")

    # PCA Scree Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, max_components_cap + 1), cum_var, marker="o", markersize=2)
    plt.axhline(variance_target, color="red", linestyle="--", label=f"{variance_target*100:.0f}% Variance")
    plt.axvline(n_components, color="green", linestyle="--", label=f"Chosen k = {n_components}")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Cumulative Explained Variance — {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Supervised Classification: KNN with GridSearchCV
    print("\nTuning K-Nearest Neighbors (KNN) via GridSearchCV...")
    cv_splits = min(5, min_class_in_train)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    k_candidates = sorted(set([1, 3, 5, 7, 9, 11, 15] + [int(round(np.sqrt(len(X_train_pca))))]))
    k_candidates = [k for k in k_candidates if k < len(X_train_pca)]
    param_grid = {"n_neighbors": k_candidates, "weights": ["uniform", "distance"]}

    knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    knn_gs.fit(X_train_pca, y_train)

    best_k = knn_gs.best_params_["n_neighbors"]
    best_weights = knn_gs.best_params_["weights"]
    print(f"Best KNN Hyperparameters: k={best_k}, weights='{best_weights}' (CV Accuracy: {knn_gs.best_score_:.4f})")

    best_knn = knn_gs.best_estimator_
    y_pred = best_knn.predict(X_test_pca)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n>>> KNN Test Accuracy: {test_acc:.4f} <<<\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))

    # Normalized Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    plt.figure(figsize=(9, 8))
    if len(class_names) <= 40:
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
            cmap="Blues", xticks_rotation=90, values_format=".2f", colorbar=True
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
    plt.title(f"KNN Confusion Matrix (Row-Normalized) — {dataset_name}")
    plt.tight_layout()
    plt.show()

    # 8. Unsupervised Clustering Benchmarks
    n_clusters = len(class_names)
    print(f"\nRunning Unsupervised Clustering Benchmarks (k = {n_clusters} matching ground-truth classes)...")

    # 8a. K-Means Sweep
    k_range = range(2, min(30, n_clusters + 6), 2)
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        labels_k = km.fit_predict(X_train_pca)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_train_pca, labels_k))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(list(k_range), inertias, marker="o", color="tab:blue", label="Inertia")
    ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(list(k_range), sil_scores, marker="s", color="tab:orange", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="tab:orange")
    plt.title(f"K-Means Diagnostics: Elbow & Silhouette vs k — {dataset_name}")
    fig.tight_layout()
    plt.show()

    # 8b. Final K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    km_labels = kmeans.fit_predict(X_train_pca)
    km_sil = silhouette_score(X_train_pca, km_labels)
    km_ari = adjusted_rand_score(y_train, km_labels)
    km_nmi = normalized_mutual_info_score(y_train, km_labels)

    # 8c. Gaussian Mixture Model (GMM)
    print("Fitting Gaussian Mixture Model (GMM)...")
    try:
        gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE, covariance_type="diag", reg_covar=1e-2)
        gmm_labels = gmm.fit_predict(X_train_pca)
        gmm_sil = silhouette_score(X_train_pca, gmm_labels)
        gmm_ari = adjusted_rand_score(y_train, gmm_labels)
        gmm_nmi = normalized_mutual_info_score(y_train, gmm_labels)
    except Exception as e:
        print(f"Warning: GMM fitting skipped for {dataset_name} due to high component count / covariance ill-conditioning: {e}")
        gmm_sil, gmm_ari, gmm_nmi = 0.0, 0.0, 0.0

    # 8d. Agglomerative Hierarchical Clustering
    print("Fitting Agglomerative Hierarchical Clustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(X_train_pca)
    agg_sil = silhouette_score(X_train_pca, agg_labels)
    agg_ari = adjusted_rand_score(y_train, agg_labels)
    agg_nmi = normalized_mutual_info_score(y_train, agg_labels)

    print("\n------------------------------------------------------------")
    print(f" CLUSTERING BENCHMARK RESULTS — {dataset_name.upper()}")
    print("------------------------------------------------------------")
    print(f"{'Algorithm':<25} | {'Silhouette':<12} | {'ARI':<12} | {'NMI':<12}")
    print("-" * 68)
    print(f"{'K-Means':<25} | {km_sil:<12.4f} | {km_ari:<12.4f} | {km_nmi:<12.4f}")
    print(f"{'Gaussian Mixture (GMM)':<25} | {gmm_sil:<12.4f} | {gmm_ari:<12.4f} | {gmm_nmi:<12.4f}")
    print(f"{'Agglomerative':<25} | {agg_sil:<12.4f} | {agg_ari:<12.4f} | {agg_nmi:<12.4f}")
    print("------------------------------------------------------------\n")

    # 9. 2D Visual Projections: PCA vs t-SNE
    print("Generating 2D Visual Projections (PCA & t-SNE)...")
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_pca2d = pca_2d.fit_transform(X_train_scaled)

    # Subsample for fast t-SNE plot if train set is large
    tsne_sample_size = min(2000, len(X_train_pca))
    tsne_idx = np.random.choice(len(X_train_pca), tsne_sample_size, replace=False)

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_train_tsne = tsne.fit_transform(X_train_pca[tsne_idx])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # PCA 2D Plot
    scatter1 = axes[0].scatter(X_train_pca2d[:, 0], X_train_pca2d[:, 1], c=y_train, cmap="tab20", s=8, alpha=0.7)
    axes[0].set_title(f"2D PCA Projection (True Labels) — {dataset_name}")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(scatter1, ax=axes[0], label="Class ID")

    # t-SNE 2D Plot
    scatter2 = axes[1].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train[tsne_idx], cmap="tab20", s=10, alpha=0.8)
    axes[1].set_title(f"2D t-SNE Projection (Non-linear Manifold) — {dataset_name}")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
    plt.colorbar(scatter2, ax=axes[1], label="Class ID")

    plt.tight_layout()
    plt.show()

    print(f"Benchmark completed for {dataset_name}.\n")

    return best_knn, X_test_pca, y_test, y_pred, class_names