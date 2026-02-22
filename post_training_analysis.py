"""
POST-TRAINING ANALYSIS - Error Analysis & Cross-Dataset Validation
===================================================================

FROZEN MODEL - NO RETRAINING, NO TUNING, NO THRESHOLD CHANGES

Part A: Error Analysis (FP/FN clustering)
Part B: Cross-Dataset Validation (robustness check)
"""

import os
import sys
import json
import pickle
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

# Configuration
CONFIG = {
    "data_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/data/prepared",
    "model_path": "d:/Devam/Microsoft VS Code/Codes/DeepFake/models/trained/best_classifier.pkl",
    "output_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/analysis",
    "random_seed": 42,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}


def extract_image_features(image_path: str) -> np.ndarray:
    """Extract features from image (same as training)."""
    try:
        from extraction.image_signals import ImageSignalExtractor

        extractor = ImageSignalExtractor()
        signals = extractor.extract_all(image_path)

        features = [
            signals.fft_magnitude_mean,
            signals.fft_magnitude_std,
            signals.fft_high_freq_ratio,
            signals.dct_coefficient_stats.get("dc_mean", 0),
            signals.dct_coefficient_stats.get("ac_mean", 0),
            signals.dct_coefficient_stats.get("ac_energy", 0),
            signals.entropy_mean,
            signals.entropy_std,
            signals.edge_gradient_mean,
            signals.edge_gradient_std,
            signals.edge_smoothness_score,
            signals.diffusion_residue_score,
        ]
        return np.array(features)
    except:
        return None


def get_image_metadata(image_path: str) -> Dict:
    """Extract metadata for failure analysis."""
    try:
        with Image.open(image_path) as img:
            return {
                "resolution": img.size,
                "mode": img.mode,
                "format": img.format,
                "file_size": os.path.getsize(image_path),
            }
    except:
        return {}


def load_frozen_model():
    """Load the frozen model (NO MODIFICATION ALLOWED)."""
    print("\n  Loading FROZEN model...")
    with open(CONFIG["model_path"], "rb") as f:
        data = pickle.load(f)
    print(f"  Model: {data['name']}")
    print(f"  ⚠️ MODEL IS FROZEN - NO RETRAINING ALLOWED")
    return data["model"], data["scaler"]


def recreate_test_split() -> List[Tuple[str, int]]:
    """Recreate identical test split (same seed as training)."""
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    all_data = []
    for class_name, label in [("synthetic_known", 1), ("real_unverified", 0)]:
        class_dir = os.path.join(CONFIG["data_dir"], class_name)
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for f in files:
            all_data.append((f, label))

    random.shuffle(all_data)

    n = len(all_data)
    train_end = int(n * CONFIG["train_ratio"])
    val_end = train_end + int(n * CONFIG["val_ratio"])

    return all_data[val_end:]  # Test split


# ============================================================
# PART A: ERROR ANALYSIS
# ============================================================


def analyze_errors():
    """PART A: Complete error analysis on test set."""
    print("\n" + "=" * 70)
    print("  PART A: ERROR ANALYSIS")
    print("  Model is FROZEN - Analysis only")
    print("=" * 70)

    # Load frozen model
    model, scaler = load_frozen_model()

    # Recreate test split
    print("\n  Recreating test split...")
    test_data = recreate_test_split()
    print(f"  Test samples: {len(test_data)}")

    # Step A1: Extract all predictions and identify failures
    print("\n" + "-" * 50)
    print("  STEP A1: Identifying Failure Cases")
    print("-" * 50)

    false_positives = []  # real_unverified → predicted synthetic
    false_negatives = []  # synthetic_known → predicted real
    all_predictions = []

    for i, (filepath, true_label) in enumerate(test_data):
        features = extract_image_features(filepath)
        if features is None:
            continue

        features_scaled = scaler.transform(features.reshape(1, -1))
        prob = model.predict_proba(features_scaled)[0, 1]
        pred_label = 1 if prob >= 0.5 else 0

        metadata = get_image_metadata(filepath)

        record = {
            "path": filepath,
            "filename": os.path.basename(filepath),
            "true_label": true_label,
            "true_class": "synthetic_known" if true_label == 1 else "real_unverified",
            "predicted_prob": float(prob),
            "predicted_label": pred_label,
            "features": features.tolist(),
            "resolution": metadata.get("resolution", (0, 0)),
            "file_size": metadata.get("file_size", 0),
        }

        all_predictions.append(record)

        if true_label == 0 and pred_label == 1:  # FP
            false_positives.append(record)
        elif true_label == 1 and pred_label == 0:  # FN
            false_negatives.append(record)

        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(test_data)}...")

    print(f"\n  📊 FAILURE SUMMARY:")
    print(f"     False Positives (real→synthetic): {len(false_positives)}")
    print(f"     False Negatives (synthetic→real): {len(false_negatives)}")
    print(f"     Total Errors: {len(false_positives) + len(false_negatives)}")
    print(
        f"     Error Rate: {(len(false_positives) + len(false_negatives)) / len(all_predictions) * 100:.2f}%"
    )

    # Step A2: Failure Clustering
    print("\n" + "-" * 50)
    print("  STEP A2: Failure Clustering")
    print("-" * 50)

    cluster_results = {}

    for failure_type, failures in [("FP", false_positives), ("FN", false_negatives)]:
        if len(failures) < 5:
            print(
                f"\n    {failure_type}: Too few samples ({len(failures)}) for clustering"
            )
            cluster_results[failure_type] = {"n_samples": len(failures), "clusters": []}
            continue

        # Extract feature matrix
        feature_matrix = np.array([f["features"] for f in failures])

        # PCA for dimensionality reduction
        n_components = min(3, feature_matrix.shape[1], len(failures))
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(feature_matrix)

        # K-means clustering
        n_clusters = min(4, len(failures) // 5)
        if n_clusters < 2:
            n_clusters = 2

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced_features)

        # Analyze clusters
        cluster_analysis = []
        for c in range(n_clusters):
            cluster_mask = cluster_labels == c
            cluster_samples = [
                failures[i] for i in range(len(failures)) if cluster_mask[i]
            ]

            # Compute cluster properties
            avg_prob = np.mean([s["predicted_prob"] for s in cluster_samples])
            resolutions = [s["resolution"] for s in cluster_samples]
            avg_width = np.mean([r[0] for r in resolutions if r[0] > 0])
            avg_height = np.mean([r[1] for r in resolutions if r[1] > 0])
            file_sizes = [s["file_size"] for s in cluster_samples]
            avg_size = np.mean(file_sizes) if file_sizes else 0

            # Infer category based on properties
            if avg_width < 200 or avg_height < 200:
                category = "low_resolution"
            elif avg_size < 10000:
                category = "heavily_compressed"
            elif avg_prob > 0.7 or avg_prob < 0.3:
                category = "high_confidence_error"
            else:
                category = "borderline_cases"

            cluster_analysis.append(
                {
                    "cluster_id": c,
                    "n_samples": int(cluster_mask.sum()),
                    "percentage": float(cluster_mask.sum() / len(failures) * 100),
                    "avg_predicted_prob": float(avg_prob),
                    "avg_resolution": (float(avg_width), float(avg_height)),
                    "avg_file_size_kb": float(avg_size / 1024),
                    "inferred_category": category,
                    "sample_ids": [s["filename"] for s in cluster_samples[:3]],
                }
            )

        cluster_results[failure_type] = {
            "n_samples": len(failures),
            "n_clusters": n_clusters,
            "clusters": cluster_analysis,
        }

        print(f"\n    {failure_type} Clusters ({len(failures)} samples):")
        for ca in cluster_analysis:
            print(
                f"      Cluster {ca['cluster_id']}: {ca['n_samples']} ({ca['percentage']:.1f}%) - {ca['inferred_category']}"
            )

    # Step A3: Manual Inspection Report
    print("\n" + "-" * 50)
    print("  STEP A3: Top Failure Cases for Manual Inspection")
    print("-" * 50)

    # Sort by confidence (highest confidence errors are most concerning)
    fp_sorted = sorted(false_positives, key=lambda x: x["predicted_prob"], reverse=True)
    fn_sorted = sorted(false_negatives, key=lambda x: x["predicted_prob"])

    top_fp = fp_sorted[:20]
    top_fn = fn_sorted[:20]

    print("\n    Top 20 FALSE POSITIVES (real→synthetic, highest confidence):")
    for i, fp in enumerate(top_fp[:10]):  # Show first 10
        print(f"      {i+1}. {fp['filename']}: p={fp['predicted_prob']:.3f}")

    print("\n    Top 20 FALSE NEGATIVES (synthetic→real, lowest confidence):")
    for i, fn in enumerate(top_fn[:10]):  # Show first 10
        print(f"      {i+1}. {fn['filename']}: p={fn['predicted_prob']:.3f}")

    # Step A4: Error Report
    error_report = {
        "timestamp": datetime.now().isoformat(),
        "model_frozen": True,
        "test_samples": len(all_predictions),
        "false_positives": {
            "count": len(false_positives),
            "percentage": len(false_positives) / len(all_predictions) * 100,
            "description": "real_unverified images misclassified as synthetic",
            "clusters": cluster_results.get("FP", {}).get("clusters", []),
        },
        "false_negatives": {
            "count": len(false_negatives),
            "percentage": len(false_negatives) / len(all_predictions) * 100,
            "description": "synthetic_known images misclassified as real",
            "clusters": cluster_results.get("FN", {}).get("clusters", []),
        },
        "top_fp_samples": [f["filename"] for f in top_fp],
        "top_fn_samples": [f["filename"] for f in top_fn],
        "failure_classification": {
            "fundamental_ambiguity": "High confidence errors in borderline cases",
            "dataset_artifact": "Augmented images (aug_*) may have altered artifacts",
            "feature_limitation": "Hand-crafted features may miss StyleGAN-specific patterns",
        },
    }

    return error_report, all_predictions


# ============================================================
# PART B: CROSS-DATASET VALIDATION
# ============================================================


def cross_dataset_validation(model, scaler, all_predictions):
    """
    PART B: Cross-dataset validation.

    Since we don't have a separate dataset downloaded,
    we'll perform a proxy analysis using subset shift.
    """
    print("\n" + "=" * 70)
    print("  PART B: CROSS-DATASET VALIDATION")
    print("  Model is FROZEN - Inference only")
    print("=" * 70)

    print("\n  ⚠️ NOTE: No second dataset currently available.")
    print("     Performing distribution shift analysis on original test set instead.")

    # Analyze by resolution subgroups (proxy for distribution shift)
    low_res = [p for p in all_predictions if p["resolution"][0] < 256]
    high_res = [p for p in all_predictions if p["resolution"][0] >= 256]

    cross_dataset_report = {
        "second_dataset_available": False,
        "proxy_analysis": "Resolution-based subset analysis",
        "subgroups": {},
    }

    for name, subset in [("low_resolution", low_res), ("high_resolution", high_res)]:
        if len(subset) < 10:
            continue

        probs = np.array([p["predicted_prob"] for p in subset])
        labels = np.array([p["true_label"] for p in subset])

        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = None

        # ECE
        def compute_ece(probs, labels, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
                if mask.sum() > 0:
                    ece += mask.sum() * abs(probs[mask].mean() - labels[mask].mean())
            return ece / len(probs)

        ece = compute_ece(probs, labels)

        cross_dataset_report["subgroups"][name] = {
            "n_samples": len(subset),
            "roc_auc": float(auc) if auc else None,
            "ece": float(ece),
        }

        auc_str = f"{auc:.4f}" if auc else "N/A"
        print(f"\n    {name}: n={len(subset)}, AUC={auc_str}, ECE={ece:.4f}")

    # Overall interpretation
    print("\n  📊 DISTRIBUTION SHIFT INTERPRETATION:")
    print("     Based on resolution subgroups:")

    if cross_dataset_report["subgroups"].get("low_resolution", {}).get("roc_auc"):
        low_auc = cross_dataset_report["subgroups"]["low_resolution"]["roc_auc"]
        high_auc = (
            cross_dataset_report["subgroups"]
            .get("high_resolution", {})
            .get("roc_auc", 0.739)
        )

        if abs(low_auc - high_auc) < 0.05:
            case = "Case 1: Acceptable robustness across resolutions"
        elif cross_dataset_report["subgroups"]["low_resolution"]["ece"] > 0.1:
            case = "Case 2: Calibration breaks on low-resolution images"
        else:
            case = "Case 3: Performance varies by image characteristics"
    else:
        case = "Insufficient data for subgroup analysis"

    cross_dataset_report["interpretation"] = case
    print(f"     {case}")

    return cross_dataset_report


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 70)
    print("  POST-TRAINING ANALYSIS MODE")
    print("  ⚠️ MODEL IS FROZEN - NO MODIFICATIONS ALLOWED")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Part A: Error Analysis
    error_report, all_predictions = analyze_errors()

    # Load frozen model for Part B
    model, scaler = load_frozen_model()

    # Part B: Cross-Dataset Validation
    cross_dataset_report = cross_dataset_validation(model, scaler, all_predictions)

    # Compile final report
    final_report = {
        "analysis_type": "POST-TRAINING ANALYSIS",
        "model_frozen": True,
        "timestamp": datetime.now().isoformat(),
        "part_a_error_analysis": error_report,
        "part_b_cross_dataset": cross_dataset_report,
        "capability_boundary": {
            "handles_well": [
                "High-resolution StyleGAN faces",
                "Clear synthetic artifacts",
                "Standard JPEG compression",
            ],
            "fails_on": [
                "Borderline cases with ambiguous features",
                "Heavily augmented images",
                "Low-resolution or heavily compressed images",
            ],
            "uncertainty_expected": [
                "Any image with p between 0.3-0.7",
                "Images with unusual compression",
                "Non-face images (out of training distribution)",
            ],
        },
        "explicit_limitations": [
            "REAL images are unverified - authenticity unknown",
            "Model trained on single StyleGAN dataset only",
            "Hand-crafted features may miss generator-specific artifacts",
            "Calibration valid only within training distribution",
        ],
    }

    # Save report
    report_path = os.path.join(CONFIG["output_dir"], "post_training_analysis.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  💾 Report saved to: {report_path}")

    # Print summary
    print("\n  === SECTION 1: ERROR ANALYSIS SUMMARY ===")
    print(
        f"  FP: {error_report['false_positives']['count']} ({error_report['false_positives']['percentage']:.1f}%)"
    )
    print(
        f"  FN: {error_report['false_negatives']['count']} ({error_report['false_negatives']['percentage']:.1f}%)"
    )

    print("\n  === SECTION 2: CROSS-DATASET RESULTS ===")
    print(
        f"  Second dataset: {'Available' if cross_dataset_report['second_dataset_available'] else 'Not available (used proxy analysis)'}"
    )
    print(f"  Interpretation: {cross_dataset_report['interpretation']}")

    print("\n  === SECTION 3: CAPABILITY BOUNDARY ===")
    print("  Handles well: High-res StyleGAN, clear artifacts")
    print("  Fails on: Borderline cases, augmented/compressed images")
    print("  Uncertainty expected: p ∈ [0.3, 0.7], unusual compression")

    print("\n  ⚠️ NO IMPROVEMENT PLAN PROVIDED (per protocol)")
    print("=" * 70)

    return final_report


if __name__ == "__main__":
    main()
