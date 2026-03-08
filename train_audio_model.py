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
    """
    Scans raw data directory for real and fake audio.
    Returns files grouped by source subdirectory (dataset name)
    so that --max-per-dataset can sample equally from each source.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found.")
        return {}, {}

    exts = ["*.wav", "*.flac", "*.mp3", "*.ogg"]

    def scan_by_source(root_dir, label):
        """Returns dict: {source_name -> [Path, ...]} grouped by top-level subdir."""
        grouped = {}
        if not root_dir.exists():
            return grouped
        # Group by immediate subdirectory under root_dir
        # e.g. fake/asvspoof/... → 'asvspoof'
        #      fake/wavefake/...  → 'wavefake'
        #      fake/custom_tts/.. → 'custom_tts'
        #      real/librispeech/. → 'librispeech'
        # Files directly in root go into '__root__'
        for ext in exts:
            for fpath in root_dir.rglob(ext):
                try:
                    # relative to root_dir  e.g.  asvspoof/LA/LA/.../foo.flac
                    rel = fpath.relative_to(root_dir)
                    source = rel.parts[0] if len(rel.parts) > 1 else "__root__"
                except ValueError:
                    source = "__root__"
                grouped.setdefault(source, []).append(fpath)
        return grouped

    print("[*] Scanning for REAL audio...")
    real_dir  = base_path / "real"
    real_grouped = scan_by_source(real_dir, "real")
    for src, files in sorted(real_grouped.items()):
        print(f"  -> [{src}] {len(files):,} REAL files")

    print("[*] Scanning for FAKE audio...")
    fake_dir  = base_path / "fake"
    fake_grouped = scan_by_source(fake_dir, "fake")
    for src, files in sorted(fake_grouped.items()):
        print(f"  -> [{src}] {len(files):,} FAKE files")

    total_real = sum(len(v) for v in real_grouped.values())
    total_fake = sum(len(v) for v in fake_grouped.values())
    print(f"[*] Total: {total_real:,} real across {len(real_grouped)} source(s) | "
          f"{total_fake:,} fake across {len(fake_grouped)} source(s)")

    if not real_grouped and not fake_grouped:
        print("\n[!] CRITICAL: No audio files found!")
        print(f"Please put audio in {base_dir}/real/ and {base_dir}/fake/")

    return real_grouped, fake_grouped


def diversity_sample(grouped: dict, max_per_dataset: int, max_total: int = 0):
    """
    Sample up to max_per_dataset files from EACH dataset source,
    then cap the total at max_total if set.
    This prevents any single large dataset from dominating training
    and forces the model to learn generalizable features.

    Args:
        grouped:         {source_name -> [Path, ...]}
        max_per_dataset: max clips taken from any single source (0 = no limit)
        max_total:       overall cap after diversity sampling (0 = no limit)

    Returns:
        flat list of Paths
    """
    import random
    result = []
    for source, paths in sorted(grouped.items()):
        shuffled = paths.copy()
        random.shuffle(shuffled)
        cap = max_per_dataset if max_per_dataset > 0 else len(shuffled)
        selected = shuffled[:cap]
        result.extend(selected)
        print(f"  [{source:25s}] using {len(selected):,} / {len(paths):,} files")

    random.shuffle(result)
    if max_total > 0 and len(result) > max_total:
        result = result[:max_total]
    return result

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def extract_all_features(real_paths, fake_paths, max_samples=None):
    """
    Runs files through Layer 1, Layer 2, and Layer 3 extractors.
    Constructs the X (features) and Y (labels) matrices.
    GPU-accelerated: Wav2Vec2 (Layer 3) runs on CUDA if available.
    Includes checkpoint/resume support for long full-dataset runs.
    """
    # ── Device setup ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory // 1024**3}GB VRAM)")
    else:
        print(" (CPU — install torch+cu118 for GPU acceleration)")

    audio_system = AdvancedAudioForensics()
    neural_system = AudioNeuralDetector()
    neural_system.eval()
    neural_system = neural_system.to(device)  # Move Wav2Vec2 to GPU

    # ── Optional cap for quick testing ────────────────────────────────────────
    if max_samples:
        import random
        real_paths = random.sample(real_paths, min(len(real_paths), max_samples // 2))
        fake_paths = random.sample(fake_paths, min(len(fake_paths), max_samples // 2))

    all_files = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    print(f"[*] Total files to process: {len(all_files):,} "
          f"({len(real_paths):,} real + {len(fake_paths):,} fake)")

    # ── Resume support — load checkpoint if it exists ─────────────────────────
    # We now keep this as a permanent cache. Once extraction finishes, it stays
    # on disk so you never have to re-extract if the classifier crashes later.
    checkpoint_path = Path("models/audio_features_cache.pkl")
    checkpoint_path.parent.mkdir(exist_ok=True)
    X, y_labels, done_paths = [], [], set()

    if checkpoint_path.exists():
        print(f"[*] Found cached features at: {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        X          = ckpt["X"]
        y_labels   = ckpt["y_labels"]
        done_paths = set(ckpt["done_paths"])
        print(f"    Already processed: {len(done_paths):,} files. "
              f"Remaining: {len(all_files) - len(done_paths):,}")


    failed = 0
    SAVE_EVERY = 500   # checkpoint every N files

    print(f"\n[*] Extracting 3-Layer Forensics...")
    for i, (filepath, label) in enumerate(tqdm(all_files, desc="Extracting", unit="file")):
        file_str = str(filepath)

        # Skip already-processed files (resume)
        if file_str in done_paths:
            continue

        data = audio_system.load_and_preprocess(file_str)
        if data is None:
            failed += 1
            done_paths.add(file_str)
            continue

        y, sr = data

        try:
            # --- Layer 1 & 2 (CPU, fast) ---
            l1 = audio_system.layer1_signal_forensics(y, sr)
            l2 = audio_system.layer2_speech_behavior(y, sr)

            # --- Layer 3 (GPU-accelerated Wav2Vec2) ---
            if sr != 16000:
                y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            else:
                y_16k = y

            # Clip to max 10 seconds to keep VRAM usage bounded on GTX 1650
            max_samples_wav = 16000 * 10
            if len(y_16k) > max_samples_wav:
                y_16k = y_16k[:max_samples_wav]

            tensor_input = torch.tensor(y_16k).unsqueeze(0).float().to(device)
            with torch.no_grad():
                # Single GPU pass: Wav2Vec2 runs ONCE, returns both score + embedding
                logits, ood_embed = neural_system.forward_features(tensor_input)
                l3_score     = logits.cpu().item()
                l3_ood_embed = ood_embed.cpu().numpy().mean()

            feature_vector = [
                l1.get('inst_phase_variance',  0),
                l1.get('rt60_estimate',        0),
                l1.get('mfcc_variance',        0),
                l1.get('spectral_flatness_var',0),
                l1.get('zcr_variance',         0),
                l1.get('codec_banding_score',  0),
                l2.get('pause_ratio',          0),
                l2.get('pitch_drift_over_time',0),
                l3_score,
                l3_ood_embed,
            ]

            X.append(feature_vector)
            y_labels.append(label)
            done_paths.add(file_str)

        except Exception as e:
            failed += 1
            done_paths.add(file_str)
            continue

        # ── Save checkpoint every SAVE_EVERY files ────────────────────────────
        if (i + 1) % SAVE_EVERY == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"X": X, "y_labels": y_labels,
                             "done_paths": list(done_paths)}, f)
            real_done = sum(1 for p, l in all_files[:i+1] if l == 0 and str(p) in done_paths)
            fake_done = sum(1 for p, l in all_files[:i+1] if l == 1 and str(p) in done_paths)
            tqdm.write(f"  [Checkpoint] {len(X):,} features saved "
                       f"({real_done:,} real / {fake_done:,} fake | {failed} failed)")

    # Note: We NO LONGER delete the checkpoint file here.
    # It acts as a permanent cache so we don't lose hours of extraction
    # if the downstream classifier training fails.

    print(f"\n[+] Extraction complete. Success: {len(X):,} | Failed: {failed}")
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
    # Sklearn 1.4+ removed cv="prefit". We now use standard 5-fold CV calibration
    # which is actually more robust and trains multiple calibrated models natively.
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=5)
    
    # Needs to be fit on the whole training set (because cv=5 will split it internally)
    calibrated_clf.fit(X_train, y_train)

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
    parser.add_argument("--data-dir",         type=str, default="data/audio_forensics/raw",
                        help="Path to raw audio data")
    parser.add_argument("--max-samples",      type=int, default=0,
                        help="Hard cap on total files (0 = no limit). Old-style shorthand.")
    parser.add_argument("--max-per-dataset",  type=int, default=20000,
                        help="Max clips sampled from EACH dataset source folder. "
                             "Prevents any single dataset from dominating training. "
                             "Default: 20000 per source (good diversity + manageable time). "
                             "Set to 0 for no per-source limit.")
    args = parser.parse_args()

    # 1. Scan — returns files grouped by source
    real_grouped, fake_grouped = scan_datasets(args.data_dir)

    if not real_grouped and not fake_grouped:
        sys.exit(1)
    if not real_grouped or not fake_grouped:
        print("[!] WARNING: Missing one class (Real or Fake). Cannot train classifier.")
        sys.exit(1)

    # 2. Diversity-aware sampling
    print("\n[*] Sampling with diversity constraints...")
    print(f"    max_per_dataset : {args.max_per_dataset if args.max_per_dataset else 'unlimited'}")
    print(f"    max_samples     : {args.max_samples if args.max_samples else 'unlimited'}")
    print("  REAL sources:")
    real_paths = diversity_sample(real_grouped,
                                  max_per_dataset=args.max_per_dataset,
                                  max_total=args.max_samples // 2 if args.max_samples else 0)
    print("  FAKE sources:")
    fake_paths = diversity_sample(fake_grouped,
                                  max_per_dataset=args.max_per_dataset,
                                  max_total=args.max_samples // 2 if args.max_samples else 0)

    # Balance real vs fake (smaller set determines size)
    min_count = min(len(real_paths), len(fake_paths))
    import random
    real_paths = random.sample(real_paths, min_count)
    fake_paths = random.sample(fake_paths, min_count)
    print(f"\n[*] Final balanced dataset: {min_count:,} real + {min_count:,} fake "
          f"= {min_count * 2:,} total")

    # 3. Extract 3-Layer Features
    X, y = extract_all_features(real_paths, fake_paths, max_samples=0)  # already sampled above

    # 4. Train & Calibrate Ensemble
    train_production_classifier(X, y)

if __name__ == "__main__":
    main()
