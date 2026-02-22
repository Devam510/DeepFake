"""
Improved AI Detector Training with sklearn classifiers
Uses Random Forest, SVM, and Gradient Boosting on image features
"""

import os
import sys
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
import pickle

# Configuration
CONFIG = {
    "data_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/data/prepared",
    "output_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/models/trained",
    "random_seed": 42,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}


def extract_image_features(image_path: str) -> np.ndarray:
    """Extract comprehensive features from image."""
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
    except Exception as e:
        return None


def load_and_split_data(data_dir: str, config: Dict) -> Tuple[Dict, Dict]:
    """Load data and create train/val/test splits."""
    print("\n" + "=" * 60)
    print("STEP 1-2: DATA LOADING AND SPLITTING")
    print("=" * 60)

    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    all_data = []

    for class_name, label in [("synthetic_known", 1), ("real_unverified", 0)]:
        class_dir = os.path.join(data_dir, class_name)
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for f in files:
            all_data.append((f, label))

        print(f"  {class_name}: {len(files)} files (label={label})")

    random.shuffle(all_data)

    n = len(all_data)
    train_end = int(n * config["train_ratio"])
    val_end = train_end + int(n * config["val_ratio"])

    splits = {
        "train": all_data[:train_end],
        "val": all_data[train_end:val_end],
        "test": all_data[val_end:],
    }

    print(f"\n  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"  Test:  {len(splits['test'])}")

    return splits


def extract_features_batch(
    file_list: List[Tuple[str, int]], max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from batch of files."""
    features = []
    labels = []

    samples = file_list[:max_samples] if max_samples else file_list

    for i, (filepath, label) in enumerate(samples):
        feat = extract_image_features(filepath)
        if feat is not None:
            features.append(feat)
            labels.append(label)

        if (i + 1) % 500 == 0:
            print(f"    Extracted {i + 1}/{len(samples)}...")

    return np.array(features), np.array(labels)


def train_and_evaluate():
    """Main training function with multiple classifiers."""
    print("\n" + "=" * 70)
    print("  IMPROVED AI DETECTOR TRAINING (sklearn)")
    print("=" * 70)

    # Load and split data
    splits = load_and_split_data(CONFIG["data_dir"], CONFIG)

    # Extract features
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE EXTRACTION")
    print("=" * 60)

    print(f"\n  Extracting training features...")
    X_train, y_train = extract_features_batch(splits["train"])
    print(f"  Train: {len(X_train)} samples, {X_train.shape[1]} features")

    print(f"\n  Extracting validation features...")
    X_val, y_val = extract_features_batch(splits["val"])
    print(f"  Val: {len(X_val)} samples")

    print(f"\n  Extracting test features...")
    X_test, y_test = extract_features_batch(splits["test"])
    print(f"  Test: {len(X_test)} samples")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple classifiers
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING CLASSIFIERS")
    print("=" * 60)

    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    best_model = None
    best_auc = 0
    best_name = None

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")

        # Train with calibration
        calibrated_clf = CalibratedClassifierCV(clf, cv=3, method="isotonic")
        calibrated_clf.fit(X_train_scaled, y_train)

        # Evaluate on validation
        val_probs = calibrated_clf.predict_proba(X_val_scaled)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)

        # Evaluate on test
        test_probs = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)

        results[name] = {
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
        }

        print(f"    Val AUC:  {val_auc:.4f}")
        print(f"    Test AUC: {test_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = calibrated_clf
            best_name = name

    print(f"\n  Best model: {best_name} (Val AUC: {best_auc:.4f})")

    # Detailed evaluation of best model
    print("\n" + "=" * 60)
    print("STEP 7: HONEST EVALUATION (Best Model)")
    print("=" * 60)

    test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, test_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # ECE calculation
    def compute_ece(probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = probs[mask].mean()
                bin_acc = labels[mask].mean()
                ece += mask.sum() * abs(bin_conf - bin_acc)
        return ece / len(probs)

    ece = compute_ece(test_probs, y_test)

    print(f"\n  📊 TEST RESULTS ({best_name}):")
    print(f"     ROC-AUC:               {test_auc:.4f}")
    print(f"     False Negative Rate:   {fnr:.4f}")
    print(f"     False Positive Rate:   {fpr:.4f}")
    print(f"     Calibration Error:     {ece:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"     TN: {tn}, FP: {fp}")
    print(f"     FN: {fn}, TP: {tp}")

    # Save model
    model_path = os.path.join(CONFIG["output_dir"], "best_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "name": best_name}, f)
    print(f"\n  💾 Model saved to: {model_path}")

    # Save evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_name,
        "all_results": results,
        "test_evaluation": {
            "roc_auc": float(test_auc),
            "false_negative_rate": float(fnr),
            "false_positive_rate": float(fpr),
            "calibration_error_ece": float(ece),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
        "data_summary": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "explicit_limitations": [
            "REAL images are unverified - source authenticity not proven",
            "Model estimates synthetic likelihood, NOT authenticity",
            "Confidence reflects model uncertainty, NOT truth",
        ],
    }

    report_path = os.path.join(CONFIG["output_dir"], "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  💾 Report saved to: {report_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Model: {best_name}")
    print(f"  ROC-AUC: {test_auc:.4f}")
    print(f"  ECE: {ece:.4f}")

    if test_auc < 0.6:
        print(f"\n  ⚠️ WARNING: AUC < 0.6 indicates weak discrimination")
        print(
            f"     The hand-crafted features may not capture StyleGAN artifacts well."
        )
        print(f"     Consider using deep learning (CNN) for better results.")

    print(f"\n  ⚠️ EXPLICIT LIMITATIONS:")
    print(f"     • REAL images are unverified")
    print(f"     • Model estimates synthetic likelihood, not authenticity")
    print(f"     • Confidence reflects model uncertainty, not truth")

    print("\n" + "=" * 70)

    return report


if __name__ == "__main__":
    train_and_evaluate()
