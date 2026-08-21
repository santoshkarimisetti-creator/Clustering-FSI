# ============================================================
# Devanagari Handwritten Character Recognition & Clustering
# Benchmark Runner using Modular indic_pipeline Engine
# ============================================================
from indic_pipeline import run_indic_benchmark

if __name__ == "__main__":
    run_indic_benchmark(
        dataset_name="Devanagari",
        kaggle_handle="rishianand/devanagari-character-set",
        max_samples=8000,
        img_size=32,
        min_per_class=15,
        feature_type="hog"  # Options: 'hog' (Histogram of Oriented Gradients) or 'raw'
    )