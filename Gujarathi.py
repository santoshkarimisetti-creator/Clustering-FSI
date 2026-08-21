# ============================================================
# Gujarati Handwritten Characters — PCA, KNN, K-Means (v2)
# ============================================================
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

import kagglehub
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from sklearn.cluster import KMeans

RANDOM_STATE = 42
IMG_SIZE = 32
MAX_COMPONENTS_CAP = 150
VARIANCE_TARGET = 0.95
MIN_PER_CLASS = 15          # must be >= n_splits used in CV after train split
MAX_SAMPLES = 6000
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------
# 1. Download dataset
# ------------------------------------------------------------
print("Downloading dataset...")
path = kagglehub.dataset_download("meet1265/gujarati-handwritten-characters-dataset")
print("Path to dataset files:", path)

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def find_image_files(root):
    files = []
    for ext in VALID_EXT:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(files))

def infer_label(filepath, root):
    # Label = immediate parent folder name (standard Kaggle "root/<class>/<image>" layout)
    rel = os.path.relpath(filepath, root)
    parts = rel.split(os.sep)
    return parts[-2] if len(parts) >= 2 else "unknown"

print("Scanning for image files...")
all_files = find_image_files(path)
if len(all_files) == 0:
    raise RuntimeError(f"No image files found under {path}. Check dataset structure.")
print(f"Found {len(all_files)} image files.")

labels_raw = [infer_label(f, path) for f in all_files]

# Keep only classes with enough samples for a safe stratified split + CV
counts = Counter(labels_raw)
keep_classes = {c for c, n in counts.items() if n >= MIN_PER_CLASS}
all_files, labels_raw = zip(*[(f, l) for f, l in zip(all_files, labels_raw) if l in keep_classes])
all_files, labels_raw = list(all_files), list(labels_raw)
print(f"Using {len(set(labels_raw))} classes after filtering, {len(all_files)} images total.")

# Subsample for tractability (stratified-ish via random choice is fine pre-split)
if len(all_files) > MAX_SAMPLES:
    idx = np.random.choice(len(all_files), MAX_SAMPLES, replace=False)
    all_files = [all_files[i] for i in idx]
    labels_raw = [labels_raw[i] for i in idx]
    print(f"Subsampled to {MAX_SAMPLES} images for tractability.")

# ------------------------------------------------------------
# 2. Load, preprocess, normalize images (robust to bad files)
# ------------------------------------------------------------
print("Loading and preprocessing images...")

def load_image(fp):
    try:
        img = Image.open(fp).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Normalize polarity: force background ~0 (dark), strokes ~1 (bright).
        # Handwriting datasets are inconsistent about ink/background convention;
        # this makes PCA/KNN see a uniform representation.
        if arr.mean() > 0.5:
            arr = 1.0 - arr
        return arr
    except Exception:
        return None

loaded = [(load_image(f), l) for f, l in zip(all_files, labels_raw)]
loaded = [(im, l) for im, l in loaded if im is not None]
dropped = len(all_files) - len(loaded)
if dropped:
    print(f"Skipped {dropped} unreadable/corrupt files.")

X_images = np.array([im for im, _ in loaded])
labels_raw = [l for _, l in loaded]

# Re-filter classes in case dropped files pushed any class below MIN_PER_CLASS
counts = Counter(labels_raw)
keep_classes = {c for c, n in counts.items() if n >= MIN_PER_CLASS}
keep_mask = [l in keep_classes for l in labels_raw]
X_images = X_images[keep_mask]
labels_raw = [l for l, k in zip(labels_raw, keep_mask) if k]

X = X_images.reshape(len(X_images), -1)
le = LabelEncoder()
y = le.fit_transform(labels_raw)
class_names = le.classes_
print(f"Final feature matrix: {X.shape}, Classes: {len(class_names)}")

# ------------------------------------------------------------
# 3. Visualize sample images
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
sample_idx = np.random.choice(len(X_images), 10, replace=False)
for ax, i in zip(axes.ravel(), sample_idx):
    ax.imshow(X_images[i], cmap="gray")
    ax.set_title(str(class_names[y[i]]), fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Gujarati Characters (normalized polarity)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 4. Train/test split BEFORE fitting scaler/PCA -> no leakage
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
min_class_in_train = min(Counter(y_train).values())
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, "
      f"min samples/class in train: {min_class_in_train}")

# ------------------------------------------------------------
# 5. Standardize (fit on train only)
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 6. PCA — select components by explained-variance target (fit on train only)
# ------------------------------------------------------------
max_components = min(MAX_COMPONENTS_CAP, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])

pca_probe = PCA(n_components=max_components, svd_solver="randomized", random_state=RANDOM_STATE)
pca_probe.fit(X_train_scaled)
cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
n_components = int(np.searchsorted(cum_var, VARIANCE_TARGET) + 1)
n_components = int(np.clip(n_components, 20, max_components))
print(f"Components needed for {VARIANCE_TARGET*100:.0f}% variance: {n_components} "
      f"(cap={max_components})")

pca = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Actual cumulative explained variance retained: {np.sum(pca.explained_variance_ratio_):.4f}")

plt.figure(figsize=(7, 5))
plt.plot(range(1, max_components + 1), cum_var, marker="o", markersize=2)
plt.axhline(VARIANCE_TARGET, color="red", linestyle="--", label=f"{VARIANCE_TARGET*100:.0f}% variance")
plt.axvline(n_components, color="green", linestyle="--", label=f"chosen k = {n_components}")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.tight_layout()
plt.show()

# 2D PCA visualization
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca_2d.fit_transform(X_train_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap="tab20", s=8, alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("2D PCA Projection (Train Set, true labels)")
plt.colorbar(scatter, label="Class index")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. KNN on PCA features — tune k AND weighting scheme
# ------------------------------------------------------------
print("Tuning KNN (k, weights) via GridSearchCV...")
cv_splits = min(5, min_class_in_train)
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

k_candidates = sorted(set(list(range(1, 16, 2)) + [int(round(np.sqrt(len(X_train_pca))))]))
k_candidates = [k for k in k_candidates if k < len(X_train_pca)]
param_grid = {"n_neighbors": k_candidates, "weights": ["uniform", "distance"]}

knn_gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
knn_gs.fit(X_train_pca, y_train)
best_k = knn_gs.best_params_["n_neighbors"]
best_weights = knn_gs.best_params_["weights"]
print(f"Best params: k={best_k}, weights={best_weights} (CV accuracy: {knn_gs.best_score_:.4f})")

knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
print(f"\nKNN Test Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))

# Confusion matrix — normalized (row-wise %) for readability with many classes
cm = confusion_matrix(y_test, y_pred, normalize="true")
plt.figure(figsize=(10, 9))
if len(class_names) <= 40:
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        cmap="Blues", xticks_rotation=90, values_format=".2f", colorbar=True)
else:
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix (normalized, labels hidden - too many classes)")
plt.title("KNN Confusion Matrix (row-normalized)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. K-Means clustering on PCA features
# ------------------------------------------------------------
# 8a. Justify cluster count with an elbow/silhouette sweep
print("\nSweeping k for K-Means (elbow + silhouette)...")
k_range = range(2, min(30, len(class_names) + 10), 2)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
    labels_k = km.fit_predict(X_train_pca)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_train_pca, labels_k))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(list(k_range), inertias, marker="o", color="tab:blue", label="Inertia")
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(list(k_range), sil_scores, marker="s", color="tab:orange", label="Silhouette")
ax2.set_ylabel("Silhouette Score", color="tab:orange")
plt.title("K-Means: Elbow (inertia) & Silhouette vs k")
fig.tight_layout()
plt.show()

best_sil_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"k with best silhouette score in sweep: {best_sil_k}")

# 8b. Final K-Means at k = number of true classes (for direct comparison to labels)
n_clusters = len(class_names)
print(f"Running final K-Means with k={n_clusters} (matches true class count)...")
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_pca)

sil = silhouette_score(X_train_pca, cluster_labels)
ari = adjusted_rand_score(y_train, cluster_labels)
nmi = normalized_mutual_info_score(y_train, cluster_labels)
print(f"Silhouette Score: {sil:.4f}")
print(f"Adjusted Rand Index (vs true labels): {ari:.4f}")
print(f"Normalized Mutual Info (vs true labels): {nmi:.4f}")

# Correctly project cluster centers into the same 2D PCA space as the scatter plot
centers_scaled_space = pca.inverse_transform(kmeans.cluster_centers_)
centers_2d = pca_2d.transform(centers_scaled_space)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=cluster_labels, cmap="tab20", s=8, alpha=0.7)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=120,
            edgecolors="white", linewidths=1.2, label="Cluster centers")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("K-Means Clusters (2D PCA Projection)")
plt.colorbar(scatter, label="Cluster ID")
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")