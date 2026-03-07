"""
Audio Forensics Meta-Ensemble Training Pipeline
===============================================
Extracts features from any audio available in `data/audio_forensics/raw/`
and trains the LightGBM production meta-classifier.

Handles:
 - Layer 1: Signal Forensics (MFCC, Spectral, Zero-Crossing)
 - Layer 2: Speech Behavior (Ratios, Pauses)
 - Layer 3: Neural Backbone (Wav2Vec2 embedding scoring)

Outputs: `models/audio_lgbm_ensemble.pkl`
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm

try:
    import lightgbm as lgb
    import librosa
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    print("[!] Missing dependencies. Run: pip install lightgbm scikit-learn librosa torch")
    sys.exit(1)

# Import our custom feature extractors
try:
    from audio_forensics import AdvancedAudioForensics
    from audio_neural_model import AudioNeuralDetector
except ImportError:
    print("[!] Unable to import core modules. Make sure 'audio_forensics.py' and 'audio_neural_model.py' are in the same folder.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# DATA SCANNER
# ══════════════════════════════════════════════════════════════════════════════
def scan_datasets(base_dir: str):
    """Scans raw data directory for any available real and fake audio."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found.")
        return [], []
        
    real_files = []
    fake_files = []

    # Map all supported extensions
    exts = ["*.wav", "*.flac", "*.mp3", "*.ogg"]

    print("[*] Scanning for REAL audio...")
    real_dir = base_path / "real"
    if real_dir.exists():
        for ext in exts:
            found = list(real_dir.rglob(ext))
            real_files.extend(found)
            if found: print(f"  -> Found {len(found)} REAL files matching {ext}")

    print("[*] Scanning for FAKE audio...")
    fake_dir = base_path / "fake"
    if fake_dir.exists():
        for ext in exts:
            found = list(fake_dir.rglob(ext))
            fake_files.extend(found)
            if found: print(f"  -> Found {len(found)} FAKE files matching {ext}")

    if not real_files and not fake_files:
        print("\n[!] CRITICAL: No audio files found to train on!")
        print(f"Please put some audio files in {base_dir}/real/ and {base_dir}/fake/")

    return real_files, fake_files

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def extract_all_features(real_paths, fake_paths, max_samples=None):
    """
    Runs files through Layer 1, Layer 2, and Layer 3 extractors.
    Constructs the X (features) and Y (labels) matrices.
    """
    audio_system = AdvancedAudioForensics()
    neural_system = AudioNeuralDetector()
    neural_system.eval() # Set torch model to evaluation mode
    
    # Optional cap for quick testing
    if max_samples:
        import random
        real_paths = random.sample(real_paths, min(len(real_paths), max_samples//2))
        fake_paths = random.sample(fake_paths, min(len(fake_paths), max_samples//2))

    X, y_labels = [], []
    failed = 0

    all_files = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    
    print(f"\n[*] Extracting 3-Layer Forensics for {len(all_files)} audio clips...")
    for filepath, label in tqdm(all_files, desc="Extracting Features"):
        file_str = str(filepath)
        data = audio_system.load_and_preprocess(file_str)
        if data is None:
            failed += 1
            continue
            
        y, sr = data
        
        try:
            # --- Layer 1 & 2 ---
            l1 = audio_system.layer1_signal_forensics(y, sr)
            l2 = audio_system.layer2_speech_behavior(y, sr)
            
            # --- Layer 3 ---
            # Model heavily expects 16kHz
            if sr != 16000:
                y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                y_16k = y
                
            tensor_input = torch.tensor(y_16k).unsqueeze(0).float()
            with torch.no_grad():
                l3_score = neural_system(tensor_input).item()
                # Assuming OOD extracts a flattened vector representation
                l3_ood_embed = neural_system.extract_embedding_for_ood(tensor_input).numpy().mean()

            # Flatten into a single row
            feature_vector = [
                l1.get('inst_phase_variance', 0),
                l1.get('rt60_estimate', 0),
                l1.get('mfcc_variance', 0),
                l1.get('spectral_flatness_var', 0),
                l1.get('zcr_variance', 0),
                l1.get('codec_banding_score', 0),
                l2.get('pause_ratio', 0),
                l2.get('pitch_drift_over_time', 0),
                l3_score,
                l3_ood_embed
            ]
            
            X.append(feature_vector)
            y_labels.append(label)

        except Exception as e:
            failed += 1
            # print(f"Error on {filepath}: {e}")
            continue

    print(f"\n[+] Extraction complete. Success: {len(X)} | Failed: {failed}")
    return np.array(X), np.array(y_labels)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & CALIBRATION (LightGBM Meta-Ensemble)
# ══════════════════════════════════════════════════════════════════════════════
def train_production_classifier(X, y):
    """
    Trains LightGBM -> Calibrates it (Isotonic/Platt scaling) -> Saves model.
    """
    print("\n[*] Initializing Meta-Ensemble Training...")

    if len(X) < 10:
        print("[!] Too few samples to train safely. Gather more audio.")
        return

    # 1. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Base LightGBM Classifier
    print("  -> Training LightGBM core...")
    base_clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight='balanced'
    )
    base_clf.fit(X_train, y_train)

    # 3. Probability Calibration (Crucial for Audio Forensics 'No binary Fake/Real' rule)
    print("  -> Calibrating probabilities (Isotonic Regression)...")
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv="prefit")
    calibrated_clf.fit(X_test, y_test)

    # 4. Evaluation
    y_pred = base_clf.predict(X_test)
    y_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5  # Happens if test set only has 1 class
        
    ece = brier_score_loss(y_test, y_prob) # Brier serves as a strict proxy for ECE

    print("\n=============================================")
    print(" META-ENSEMBLE TRAINING RESULTS")
    print("=============================================")
    print(f" Accuracy:       {acc:.4f}")
    print(f" AUC-ROC:        {auc:.4f} (Ability to separate Real vs Synthetic)")
    print(f" Brier (ECE):    {ece:.4f} (Target < 0.07. Lower is more calibrated)")
    print("=============================================")

    # 5. Save Artifacts for API
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    with open(model_dir / "audio_lgbm_ensemble.pkl", "wb") as f:
        pickle.dump(calibrated_clf, f)
        
    print(f"[+] Multi-Layer Meta-Model saved to: {model_dir / 'audio_lgbm_ensemble.pkl'}")
    return calibrated_clf

# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser("Audio Forensics System Trainer")
    parser.add_argument("--data-dir", type=str, default="data/audio_forensics/raw", help="Path to raw audio")
    parser.add_argument("--max-samples", type=int, default=1000, help="Limit number of audio clips (for testing)")
    args = parser.parse_args()

    # 1. Look for whatever data is currently on disk
    real_paths, fake_paths = scan_datasets(args.data_dir)
    
    if not real_paths and not fake_paths:
        sys.exit(1)
        
    if not real_paths or not fake_paths:
        print("[!] WARNING: Missing one of the classes (Real or Fake). Cannot train a classifier.")
        print("    Model requires BOTH real and synthetic examples.")
        sys.exit(1)

    # 2. Extract 3-Layer Features
    X, y = extract_all_features(real_paths, fake_paths, max_samples=args.max_samples)

    # 3. Train & Calibrate Ensemble
    train_production_classifier(X, y)

if __name__ == "__main__":
    main()
