"""
Cross-Model Meta-Voter
======================

Trains a meta-model that learns WHEN to trust which signal.
Instead of hand-tuned if/elif logic, this model takes all signal
scores as input and outputs the optimal final AI probability.

Usage:
    # Train the meta-voter on your labeled dataset
    python meta_voter.py --train
    
    # Predict (used internally by ensemble_detector.py)
    from meta_voter import MetaVoter
    voter = MetaVoter()
    result = voter.predict(feature_dict)
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# ============================================================
# META-VOTER MODEL
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "trained", "meta_voter.pkl")


class MetaVoter:
    """
    Cross-model voting meta-learner.
    
    Takes individual signal scores and outputs optimal final prediction.
    Falls back to simple weighted average if not trained.
    """
    
    FEATURE_NAMES = [
        "eff_prob",              # EfficientNet AI probability
        "stat_prob",             # Statistical model probability
        "meta_prob",             # Metadata AI probability
        "filter_confidence",     # Filter detection confidence
        "forensic_lighting",     # Lighting consistency score
        "forensic_noise",        # Noise pattern score
        "forensic_reflection",   # Reflection analysis score
        "forensic_gan_fp",       # GAN fingerprint score
        "jpeg_quality",          # JPEG compression quality (0-100)
        "disagreement",          # Model disagreement (eff vs stat)
    ]
    
    def __init__(self):
        self._model = None
        self._scaler = None
        self._loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load trained meta-voter if available."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self._model = data["model"]
                self._scaler = data["scaler"]
                self._loaded = True
                print(f"[META-VOTER] Loaded trained model (accuracy: {data.get('accuracy', '?')})")
            except Exception as e:
                print(f"[META-VOTER] Failed to load: {e}")
                self._loaded = False
        else:
            self._loaded = False
    
    @property
    def is_trained(self) -> bool:
        return self._loaded
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict final AI probability from all signal scores.
        
        Args:
            features: dict with keys matching FEATURE_NAMES
            
        Returns:
            float: AI probability 0.0 - 1.0
        """
        # Build feature vector
        X = np.array([features.get(name, 0.5) for name in self.FEATURE_NAMES])
        
        if self._loaded:
            # Use trained model
            X_scaled = self._scaler.transform(X.reshape(1, -1))
            prob = float(self._model.predict_proba(X_scaled)[0, 1])
            return prob
        else:
            # Fallback: simple weighted average (same as current ensemble logic)
            weights = {
                "eff_prob": 0.40,
                "stat_prob": 0.20,
                "meta_prob": 0.10,
                "forensic_lighting": 0.08,
                "forensic_noise": 0.08,
                "forensic_reflection": 0.04,
                "forensic_gan_fp": 0.10,
            }
            
            weighted_sum = 0.0
            total_weight = 0.0
            for key, weight in weights.items():
                val = features.get(key, 0.5)
                weighted_sum += val * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5


# ============================================================
# TRAINING PIPELINE
# ============================================================

def extract_all_signals(image_path: str) -> Dict[str, float]:
    """Extract all signal features from a single image."""
    features = {}
    
    # 1. EfficientNet
    try:
        from ensemble_detector import get_efficientnet_prediction
        eff = get_efficientnet_prediction(image_path)
        features["eff_prob"] = eff.get("probability", 0.5)
    except Exception:
        features["eff_prob"] = 0.5
    
    # 2. Statistical model
    try:
        from ensemble_detector import get_statistical_prediction
        stat = get_statistical_prediction(image_path)
        features["stat_prob"] = stat.get("probability", 0.5)
    except Exception:
        features["stat_prob"] = 0.5
    
    # 3. Metadata
    try:
        from ensemble_detector import get_metadata_signals
        meta = get_metadata_signals(image_path)
        features["meta_prob"] = meta.get("probability", 0.5)
        features["jpeg_quality"] = (meta.get("jpeg_quality") or 0) / 100.0  # normalize to 0-1
    except Exception:
        features["meta_prob"] = 0.5
        features["jpeg_quality"] = 0.5
    
    # 4. Filter detection
    try:
        from src.extraction.filter_detector import detect_filter
        filt = detect_filter(image_path)
        features["filter_confidence"] = filt.get("confidence", 0.0)
    except Exception:
        features["filter_confidence"] = 0.0
    
    # 5. Forensic signals
    try:
        from forensic_signals import analyze_forensics
        forensics = analyze_forensics(image_path)
        features["forensic_lighting"] = forensics.get("lighting", {}).get("probability", 0.5)
        features["forensic_noise"] = forensics.get("noise", {}).get("probability", 0.5)
        features["forensic_reflection"] = forensics.get("reflection", {}).get("probability", 0.5)
        features["forensic_gan_fp"] = forensics.get("gan_fingerprint", {}).get("probability", 0.5)
    except Exception:
        features["forensic_lighting"] = 0.5
        features["forensic_noise"] = 0.5
        features["forensic_reflection"] = 0.5
        features["forensic_gan_fp"] = 0.5
    
    # 6. Disagreement
    features["disagreement"] = abs(features["eff_prob"] - features["stat_prob"])
    
    return features


def train_meta_voter():
    """
    Train the cross-model meta-voter on labeled data.
    
    Scans all available data folders:
    - data/general_detector/real/ + synthetic/ (~97K images)
    - data/prepared/real_unverified/ + synthetic_known/ (~12K images)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    import time
    
    base_dir = os.path.dirname(__file__)
    
    # Collect image paths and labels from ALL data folders
    print("\n" + "=" * 60)
    print("  CROSS-MODEL META-VOTER TRAINING (97K images)")
    print("=" * 60)
    
    images = []
    labels = []
    
    # Data sources: (folder_path, label)
    real_dirs = [
        os.path.join(base_dir, "data", "general_detector", "real"),
        os.path.join(base_dir, "data", "prepared", "real_unverified"),
    ]
    synth_dirs = [
        os.path.join(base_dir, "data", "general_detector", "synthetic"),
        os.path.join(base_dir, "data", "prepared", "synthetic_known"),
    ]
    
    # Scan real images (label = 0)
    for real_dir in real_dirs:
        if os.path.exists(real_dir):
            count = 0
            for root, _, files in os.walk(real_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        images.append(os.path.join(root, f))
                        labels.append(0)
                        count += 1
            print(f"\n  Real images from {os.path.basename(os.path.dirname(real_dir))}/{os.path.basename(real_dir)}: {count}")
    
    # Scan synthetic images (label = 1) 
    for synth_dir in synth_dirs:
        if os.path.exists(synth_dir):
            count = 0
            for root, _, files in os.walk(synth_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        images.append(os.path.join(root, f))
                        labels.append(1)
                        count += 1
            print(f"  Synthetic images from {os.path.basename(os.path.dirname(synth_dir))}/{os.path.basename(synth_dir)}: {count}")
    
    total_real = sum(1 for l in labels if l == 0)
    total_synth = sum(1 for l in labels if l == 1)
    print(f"\n  Total: {len(images)} images ({total_real} real + {total_synth} synthetic)")
    
    if len(images) < 100:
        print(f"\n  ERROR: Need at least 100 images, found {len(images)}")
        sys.exit(1)
    
    # Estimate time
    est_seconds = len(images) * 2  # ~2 seconds per image
    est_hours = est_seconds / 3600
    print(f"  Estimated time: ~{est_hours:.1f} hours ({est_seconds//60} minutes)")
    print(f"\n  Starting feature extraction...\n")
    
    # Check for existing checkpoint (resume support)
    checkpoint_path = os.path.join(base_dir, "models", "trained", "meta_voter_checkpoint.pkl")
    X_features = []
    y_labels = []
    start_idx = 0
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            X_features = ckpt["X_features"]
            y_labels = ckpt["y_labels"]
            start_idx = ckpt["next_idx"]
            print(f"  [RESUME] Found checkpoint at image {start_idx}/{len(images)}")
            print(f"           {len(X_features)} features already extracted\n")
        except Exception:
            start_idx = 0
    
    feature_names = MetaVoter.FEATURE_NAMES
    start_time = time.time()
    
    for idx in range(start_idx, len(images)):
        img_path = images[idx]
        label = labels[idx]
        
        if (idx + 1) % 100 == 0 or idx == start_idx:
            elapsed = time.time() - start_time
            processed = idx - start_idx + 1
            rate = processed / (elapsed + 1e-8)
            remaining = (len(images) - idx) / (rate + 1e-8)
            print(f"    [{idx+1}/{len(images)}] {rate:.1f} img/s | ETA: {remaining/60:.0f} min")
        
        try:
            features = extract_all_signals(img_path)
            feature_vector = [features.get(name, 0.5) for name in feature_names]
            X_features.append(feature_vector)
            y_labels.append(label)
        except Exception:
            continue
        
        # Save checkpoint every 5000 images
        if (idx + 1) % 5000 == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "X_features": X_features,
                    "y_labels": y_labels,
                    "next_idx": idx + 1,
                }, f)
            print(f"    [CHECKPOINT] Saved at image {idx+1}")
    
    print(f"\n  Feature extraction complete!")
    print(f"  Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    X = np.array(X_features)
    y = np.array(y_labels)
    
    print(f"\n  Successfully extracted features for {len(X)} images")
    print(f"  Real: {sum(y == 0)}, Synthetic: {sum(y == 1)}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models and pick the best
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }
    
    best_model = None
    best_name = ""
    best_auc = 0
    best_acc = 0
    
    print(f"\n  Training 3 models to find the best...\n")
    
    for name, clf in models.items():
        print(f"  Training {name}...")
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"    Accuracy: {acc:.1%} | AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            best_model = clf
            best_name = name
    
    print(f"\n  Winner: {best_name} (AUC: {best_auc:.4f})")
    
    model = best_model
    accuracy = best_acc
    auc = best_auc
    
    # Final evaluation with best model
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n  {'='*50}")
    print(f"  META-VOTER RESULTS")
    print(f"  {'='*50}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  {classification_report(y_test, y_pred, target_names=['Real', 'AI'])}")
    
    # Show feature importance
    print(f"  Feature Importance ({best_name}):")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        for name, imp in importance:
            bar = "█" * int(imp * 50)
            print(f"    {name:<25s}: {imp:.3f} {bar}")
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        for name, coef in importance:
            direction = "AI" if coef > 0 else "REAL"
            print(f"    {name:<25s}: {coef:>+.3f} ({direction})")
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    save_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "accuracy": f"{accuracy:.1%}",
        "auc": f"{auc:.4f}",
        "n_samples": len(X),
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)
    
    print(f"\n  Model saved to: {MODEL_PATH}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Checkpoint cleaned up.")
    
    print(f"  {'='*50}\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if "--train" in sys.argv:
        train_meta_voter()
    else:
        print("Usage:")
        print("  python meta_voter.py --train     # Train the meta-voter")
        print()
        print("After training, the meta-voter is automatically used by ensemble_detector.py")
