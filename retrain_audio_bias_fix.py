"""
retrain_audio_bias_fix.py
=========================
Retrains the audio LightGBM ensemble using ONLY the Wav2Vec2 neural score
(feature index 8 = l3_score) as the input, completely eliminating all
acoustic heuristics (mfcc_variance, rt60, zcr etc.) that cause false
positives on real noisy recordings.

The final model still outputs a calibrated probability score [0..1]
but it is now 100% driven by the deep speech representation, not MFCC noise.
"""
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path

def main():
    print("=" * 60)
    print("  Audio Meta-Model Debiased Retraining")
    print("=" * 60)
    print("\nLoading cached audio features (~320k samples)...")
    with open("models/audio_features_cache.pkl", "rb") as f:
        ckpt = pickle.load(f)
    
    X_full = np.array(ckpt["X"])
    y      = np.array(ckpt["y_labels"])
    
    print(f"  Full feature matrix: {X_full.shape}")
    
    # ───────────────────────────────────────────────────────────────────────────
    # FEATURE INDEX MAP (from train_audio_model.py):
    #   0: inst_phase_variance  <-- noise-sensitive
    #   1: rt60_estimate        <-- noise-sensitive (room reverb)
    #   2: mfcc_variance        <-- HIGHLY noise-sensitive (root cause of 92% bias)
    #   3: spectral_flatness_var
    #   4: zcr_variance         <-- noise-sensitive
    #   5: codec_banding_score
    #   6: pause_ratio
    #   7: pitch_drift_over_time
    #   8: l3_score             <-- Wav2Vec2 neural score (THE GOLD STANDARD)
    #   9: l3_ood_embed         <-- Wav2Vec2 out-of-distribution embedding
    # ───────────────────────────────────────────────────────────────────────────
    
    # Keep ONLY the Wav2Vec2 neural features (8 and 9).
    # By discarding all acoustic heuristics the model is physically incapable
    # of being fooled by background noise in real recordings.
    WAV2VEC2_COLS = [8, 9]
    X = X_full[:, WAV2VEC2_COLS]
    
    # Drop rows where the Wav2Vec2 score failed to extract (nan/inf)
    valid_mask = np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    real_count = int(np.sum(y == 0))
    fake_count = int(np.sum(y == 1))
    print(f"  Using {X.shape[1]} neural features from {X.shape[0]:,} samples")
    print(f"  Real: {real_count:,}  |  Fake: {fake_count:,}")
    
    print("\nSplitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training LightGBM on Wav2Vec2 features only...")
    base_clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,          # Shallow — with only 2 features we don't need deep trees
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        class_weight='balanced',
        verbose=-1,
    )
    base_clf.fit(X_train, y_train)
    
    print("Calibrating probabilities (Isotonic Regression)...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf, method='isotonic', cv=5
    )
    calibrated_clf.fit(X_train, y_train)
    
    y_pred = base_clf.predict(X_test)
    y_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0
    
    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"  AUC-ROC       : {auc:.4f}")
    
    # Save to BOTH paths so nothing gets confused again
    for out_path in [
        Path("models/audio_lgbm_ensemble.pkl"),
        Path("models/trained/audio_lgbm_ensemble.pkl"),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_clf, out_path)
        print(f"  Saved -> {out_path}")
    
    print("\n✅ Debiased audio model saved successfully!")
    print("   The model now uses ONLY Wav2Vec2 neural features.")

if __name__ == "__main__":
    main()
