"""
AI-Generated Image Detector Training Pipeline
==============================================

Trains a calibrated detector estimating P(synthetic) with uncertainty.

STRICT RULES:
- 100% accuracy is NOT a goal
- REAL ≠ authentic
- Outputs must include uncertainty
- Overfitting = failure

Author: Training Pipeline
License: MIT
"""

import os
import sys
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "data_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/data/prepared",
    "output_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/models/trained",
    "random_seed": 42,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    # Statistical model
    "stat_learning_rate": 0.01,
    "stat_iterations": 2000,
    "stat_regularization": 0.01,
    # Neural model
    "neural_epochs": 20,
    "neural_batch_size": 32,
    "neural_learning_rate": 0.001,
    "early_stopping_patience": 5,
    # Calibration
    "calibration_bins": 10,
}


# ============================================================
# STEP 1: DATA AUDIT
# ============================================================


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def audit_dataset(data_dir: str) -> Dict:
    """
    STEP 1: Comprehensive data audit.

    Checks:
    - Image counts per class
    - Resolution distribution
    - Duplicate detection
    - Class imbalance validation
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA AUDIT")
    print("=" * 60)

    synthetic_dir = os.path.join(data_dir, "synthetic_known")
    real_dir = os.path.join(data_dir, "real_unverified")

    audit_result = {
        "timestamp": datetime.now().isoformat(),
        "synthetic_known": {"count": 0, "resolutions": [], "hashes": set()},
        "real_unverified": {"count": 0, "resolutions": [], "hashes": set()},
        "duplicates": [],
        "class_imbalance_ok": False,
        "audit_passed": False,
    }

    all_hashes = {}

    for class_name, class_dir in [
        ("synthetic_known", synthetic_dir),
        ("real_unverified", real_dir),
    ]:
        if not os.path.exists(class_dir):
            print(f"  ❌ ERROR: {class_dir} does not exist!")
            return audit_result

        files = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        audit_result[class_name]["count"] = len(files)

        print(f"\n  Auditing {class_name}: {len(files)} files...")

        resolution_counts = defaultdict(int)
        sample_size = min(500, len(files))  # Sample for resolution check

        for i, filename in enumerate(files[:sample_size]):
            filepath = os.path.join(class_dir, filename)

            # Check resolution
            try:
                with Image.open(filepath) as img:
                    resolution_counts[img.size] += 1
            except Exception as e:
                print(f"    ⚠️ Could not read {filename}: {e}")

            # Check for duplicates (sample)
            if i < 200:
                file_hash = compute_file_hash(filepath)
                if file_hash in all_hashes:
                    audit_result["duplicates"].append(
                        {"file1": all_hashes[file_hash], "file2": filepath}
                    )
                else:
                    all_hashes[file_hash] = filepath

        audit_result[class_name]["resolutions"] = dict(resolution_counts)

    # Report counts
    syn_count = audit_result["synthetic_known"]["count"]
    real_count = audit_result["real_unverified"]["count"]
    total = syn_count + real_count

    print(f"\n  📊 CLASS DISTRIBUTION:")
    print(f"     synthetic_known:  {syn_count:,} ({syn_count/total*100:.1f}%)")
    print(f"     real_unverified:  {real_count:,} ({real_count/total*100:.1f}%)")
    print(f"     TOTAL:            {total:,}")

    # Check class imbalance
    ratio = max(syn_count, real_count) / total
    if ratio <= 0.70:
        audit_result["class_imbalance_ok"] = True
        print(f"\n  ✅ Class imbalance OK ({ratio:.1%} <= 70%)")
    else:
        print(f"\n  ❌ Class imbalance FAILED ({ratio:.1%} > 70%)")
        print("     ABORT: Would need to correct imbalance before training.")
        return audit_result

    # Check duplicates
    if len(audit_result["duplicates"]) > 0:
        print(f"\n  ⚠️ Found {len(audit_result['duplicates'])} potential duplicates")
    else:
        print(f"\n  ✅ No duplicates found in sample")

    # Resolution summary
    print(f"\n  📐 RESOLUTION DISTRIBUTION (sampled):")
    for class_name in ["synthetic_known", "real_unverified"]:
        resolutions = audit_result[class_name]["resolutions"]
        if resolutions:
            most_common = max(resolutions.items(), key=lambda x: x[1])
            print(f"     {class_name}: Most common = {most_common[0]}")

    audit_result["audit_passed"] = True
    print(f"\n  ✅ DATA AUDIT PASSED")

    return audit_result


# ============================================================
# STEP 2: DATA SPLITTING
# ============================================================


def split_dataset(data_dir: str, config: Dict) -> Dict[str, List[Tuple[str, int]]]:
    """
    STEP 2: Split dataset into train/val/test.

    Rules:
    - 70% train, 15% val, 15% test
    - Stratified by class
    - No overlap
    - Reproducible via seed
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA SPLITTING (70/15/15)")
    print("=" * 60)

    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    splits = {"train": [], "val": [], "test": []}

    for class_name, label in [("synthetic_known", 1), ("real_unverified", 0)]:
        class_dir = os.path.join(data_dir, class_name)
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(files)

        n = len(files)
        train_end = int(n * config["train_ratio"])
        val_end = train_end + int(n * config["val_ratio"])

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        splits["train"].extend([(f, label) for f in train_files])
        splits["val"].extend([(f, label) for f in val_files])
        splits["test"].extend([(f, label) for f in test_files])

        print(f"\n  {class_name} (label={label}):")
        print(f"    Train: {len(train_files):,}")
        print(f"    Val:   {len(val_files):,}")
        print(f"    Test:  {len(test_files):,}")

    # Shuffle each split
    for split_name in splits:
        random.shuffle(splits[split_name])

    print(f"\n  📊 FINAL SPLIT SIZES:")
    print(f"     Train: {len(splits['train']):,}")
    print(f"     Val:   {len(splits['val']):,}")
    print(f"     Test:  {len(splits['test']):,}")

    # Verify no overlap
    train_set = set(f for f, _ in splits["train"])
    val_set = set(f for f, _ in splits["val"])
    test_set = set(f for f, _ in splits["test"])

    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"

    print(f"\n  ✅ No overlap between splits verified")

    return splits


# ============================================================
# STEP 3: STATISTICAL BASELINE TRAINING
# ============================================================


def train_statistical_model(splits: Dict, config: Dict) -> Tuple[object, Dict]:
    """
    STEP 3: Train StatisticalBaselineDetector.

    Uses frequency-domain and texture features.
    """
    print("\n" + "=" * 60)
    print("STEP 3: STATISTICAL BASELINE TRAINING")
    print("=" * 60)

    from modeling.statistical_detector import StatisticalBaselineDetector

    detector = StatisticalBaselineDetector()

    # Extract features from training data
    print(f"\n  Extracting features from {len(splits['train'])} training images...")

    feature_vectors = []
    labels = []
    failed_count = 0

    for i, (filepath, label) in enumerate(splits["train"]):
        try:
            features = detector.get_feature_vector(filepath)
            if features is not None and len(features) > 0:
                feature_vectors.append(features)
                labels.append(label)
        except Exception as e:
            failed_count += 1

        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{len(splits['train'])}...")

    if failed_count > 0:
        print(f"    ⚠️ Failed to extract features from {failed_count} images")

    X_train = np.array(feature_vectors)
    y_train = np.array(labels)

    print(f"\n  Training on {len(X_train)} samples with {X_train.shape[1]} features...")

    # Train model
    detector.train(
        X_train,
        y_train,
        learning_rate=config["stat_learning_rate"],
        n_iterations=config["stat_iterations"],
        regularization=config["stat_regularization"],
    )

    # Evaluate on validation set
    print(f"\n  Evaluating on validation set...")
    val_preds = []
    val_labels = []

    for filepath, label in splits["val"][:500]:  # Sample for speed
        try:
            pred = detector.predict(filepath)
            val_preds.append(pred.synthetic_probability)
            val_labels.append(label)
        except:
            pass

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    # Calculate metrics
    from sklearn.metrics import roc_auc_score

    val_auc = roc_auc_score(val_labels, val_preds)

    print(f"\n  📊 VALIDATION RESULTS:")
    print(f"     ROC-AUC: {val_auc:.4f}")

    metrics = {
        "model": "StatisticalBaselineDetector",
        "train_samples": len(X_train),
        "feature_dim": X_train.shape[1],
        "val_auc": float(val_auc),
    }

    print(f"\n  ✅ Statistical model trained")

    return detector, metrics


# ============================================================
# STEP 5: CALIBRATION
# ============================================================


def calibrate_model(detector, splits: Dict, config: Dict) -> Tuple[float, Dict]:
    """
    STEP 5: Calibrate probabilities using temperature scaling.

    Measures Expected Calibration Error (ECE).
    """
    print("\n" + "=" * 60)
    print("STEP 5: PROBABILITY CALIBRATION")
    print("=" * 60)

    # Get validation predictions
    print(f"\n  Getting validation predictions for calibration...")

    val_probs = []
    val_labels = []

    for filepath, label in splits["val"]:
        try:
            pred = detector.predict(filepath)
            val_probs.append(pred.synthetic_probability)
            val_labels.append(label)
        except:
            pass

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # Find optimal temperature using validation set
    print(f"\n  Finding optimal temperature...")

    def compute_ece(probs, labels, n_bins=10):
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = probs[mask].mean()
                bin_acc = labels[mask].mean()
                ece += mask.sum() * abs(bin_conf - bin_acc)

        return ece / len(probs)

    # Temperature scaling
    best_temp = 1.0
    best_ece = compute_ece(val_probs, val_labels)

    for temp in np.linspace(0.1, 3.0, 30):
        # Apply temperature scaling
        logits = np.log(val_probs / (1 - val_probs + 1e-10) + 1e-10)
        scaled_logits = logits / temp
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))

        ece = compute_ece(scaled_probs, val_labels)

        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    print(f"\n  📊 CALIBRATION RESULTS:")
    print(f"     Optimal Temperature: {best_temp:.3f}")
    print(f"     ECE (before): {compute_ece(val_probs, val_labels):.4f}")
    print(f"     ECE (after):  {best_ece:.4f}")

    calibration_params = {
        "temperature": float(best_temp),
        "ece_before": float(compute_ece(val_probs, val_labels)),
        "ece_after": float(best_ece),
    }

    print(f"\n  ✅ Calibration complete")

    return best_temp, calibration_params


# ============================================================
# STEP 7: EVALUATION
# ============================================================


def evaluate_model(detector, splits: Dict, temperature: float, config: Dict) -> Dict:
    """
    STEP 7: Honest evaluation on test set.

    Reports ROC-AUC, FNR, calibration curve, failure modes.
    """
    print("\n" + "=" * 60)
    print("STEP 7: HONEST EVALUATION")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

    # Get test predictions
    print(f"\n  Evaluating on {len(splits['test'])} test samples...")

    test_probs = []
    test_labels = []
    test_files = []

    for filepath, label in splits["test"]:
        try:
            pred = detector.predict(filepath)

            # Apply temperature scaling
            logit = np.log(
                pred.synthetic_probability / (1 - pred.synthetic_probability + 1e-10)
                + 1e-10
            )
            scaled_prob = 1 / (1 + np.exp(-logit / temperature))

            test_probs.append(scaled_prob)
            test_labels.append(label)
            test_files.append(filepath)
        except:
            pass

    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)

    # ROC-AUC
    test_auc = roc_auc_score(test_labels, test_probs)

    # Confusion matrix at 0.5 threshold
    pred_labels = (test_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # ECE on test set
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

    test_ece = compute_ece(test_probs, test_labels)

    # Find failure cases (high confidence wrong predictions)
    failures = []
    for i, (prob, label, filepath) in enumerate(
        zip(test_probs, test_labels, test_files)
    ):
        pred_label = 1 if prob >= 0.5 else 0
        if pred_label != label and (prob > 0.8 or prob < 0.2):
            failures.append(
                {
                    "file": os.path.basename(filepath),
                    "true_label": "synthetic" if label == 1 else "real_unverified",
                    "predicted_prob": float(prob),
                }
            )

    print(f"\n  📊 TEST RESULTS:")
    print(f"     ROC-AUC:               {test_auc:.4f}")
    print(f"     False Negative Rate:   {fnr:.4f}")
    print(f"     False Positive Rate:   {fpr:.4f}")
    print(f"     Calibration Error:     {test_ece:.4f}")
    print(f"     High-conf failures:    {len(failures)}")

    # Check for generalization issues
    if test_ece > 0.15:
        print(f"\n  ⚠️ WARNING: Calibration error is high (>{0.15})")

    evaluation = {
        "test_samples": len(test_probs),
        "roc_auc": float(test_auc),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "calibration_error_ece": float(test_ece),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "high_confidence_failures": len(failures),
        "sample_failures": failures[:10],  # First 10
    }

    print(f"\n  ✅ Evaluation complete")

    return evaluation


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("  AI-GENERATED IMAGE DETECTOR TRAINING PIPELINE")
    print("  " + "=" * 66)
    print("  Goal: Train calibrated detector with honest uncertainty")
    print("  " + "=" * 66)
    print("=" * 70)

    # STEP 1: Data Audit
    audit_result = audit_dataset(CONFIG["data_dir"])
    if not audit_result["audit_passed"]:
        print("\n❌ TRAINING ABORTED: Data audit failed")
        return

    # STEP 2: Data Splitting
    splits = split_dataset(CONFIG["data_dir"], CONFIG)

    # STEP 3: Statistical Model Training
    stat_detector, stat_metrics = train_statistical_model(splits, CONFIG)

    # STEP 5: Calibration
    temperature, calibration_params = calibrate_model(stat_detector, splits, CONFIG)

    # STEP 7: Evaluation
    evaluation = evaluate_model(stat_detector, splits, temperature, CONFIG)

    # Save model
    model_path = os.path.join(CONFIG["output_dir"], "statistical_baseline.pkl")
    stat_detector.save(model_path)
    print(f"\n  💾 Model saved to: {model_path}")

    # Save calibration params
    calib_path = os.path.join(CONFIG["output_dir"], "calibration_params.json")
    with open(calib_path, "w") as f:
        json.dump(calibration_params, f, indent=2)
    print(f"  💾 Calibration params saved to: {calib_path}")

    # Save evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "audit": {
            "synthetic_known": audit_result["synthetic_known"]["count"],
            "real_unverified": audit_result["real_unverified"]["count"],
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "statistical_model": stat_metrics,
        "calibration": calibration_params,
        "evaluation": evaluation,
        "explicit_limitations": [
            "REAL images are unverified - source authenticity not proven",
            "Model estimates synthetic likelihood, NOT authenticity",
            "Confidence reflects model uncertainty, NOT truth",
            "100% accuracy is NOT claimed and should NOT be expected",
            "High confidence does NOT guarantee correctness",
        ],
    }

    report_path = os.path.join(CONFIG["output_dir"], "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  💾 Evaluation report saved to: {report_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  📊 FINAL METRICS:")
    print(f"     ROC-AUC:           {evaluation['roc_auc']:.4f}")
    print(f"     Calibration (ECE): {evaluation['calibration_error_ece']:.4f}")
    print(f"     False Negative:    {evaluation['false_negative_rate']:.4f}")

    print(f"\n  ⚠️ EXPLICIT LIMITATIONS:")
    print(f"     • REAL images are unverified")
    print(f"     • Model estimates synthetic likelihood, not authenticity")
    print(f"     • Confidence reflects model uncertainty, not truth")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
