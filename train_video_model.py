"""
Video Deepfake Training Pipeline
=================================

Three-phase training for video deepfake detection:

Phase A: Frame-Level Model Fine-Tuning
  - Uses DF40 pre-cropped face frames (PNG) + Celeb-DF video frames
  - Fine-tunes EfficientNet-B0 on video-specific face artifacts

Phase B: Signal Feature Extraction
  - Processes training videos through temporal + biological + audio analyzers
  - Extracts feature vectors per video

Phase C: Video Meta-Voter Training
  - Trains GradientBoosting on all features for optimal combination

Usage:
  python train_video_model.py --phase-a        # Fine-tune EfficientNet
  python train_video_model.py --phase-b        # Extract signal features
  python train_video_model.py --phase-c        # Train meta-voter
  python train_video_model.py --all            # Run all phases
"""

import os
import sys
import json
import time
import random
import shutil
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
BASE = Path(r"d:\Devam\Microsoft VS Code\Codes\DeepFake")
DATASETS = BASE / "datasets"
MODELS_DIR = BASE / "models" / "trained"
PROCESSED = DATASETS / "processed"

# DF40 method folders (extracted PNGs)
DF40_REAL_DIRS = ["ff", "cdf"]  # real face datasets within DF40
DF40_FAKE_DIRS = [
    "blendface", "CollabDiff", "ddim", "deepfacelab", "DiT",
    "e4e", "faceswap", "heygen_new", "MidJourney",
]

# Frame model settings
FACE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5


def ensure_deps():
    """Install required packages."""
    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "tqdm": "tqdm",
        "mediapipe": "mediapipe",
    }
    import importlib
    for mod, pkg in required.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"  Installing {pkg}...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE A: Frame-Level EfficientNet Fine-Tuning
# ══════════════════════════════════════════════════════════════════════════════

def collect_frame_dataset() -> Tuple[List[str], List[int]]:
    """
    Collect all face frame images from datasets with labels.
    Returns (paths, labels) where label: 0=real, 1=fake
    """
    paths = []
    labels = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print("  Scanning datasets for training frames...")

    # ── DF40 (already extracted PNG frames) ──────────────────────────────────
    df40_test = DATASETS / "df40" / "test"

    for real_dir in DF40_REAL_DIRS:
        d = df40_test / real_dir
        if d.exists():
            imgs = [f for f in d.rglob("*") if f.suffix.lower() in image_exts]
            paths.extend([str(f) for f in imgs])
            labels.extend([0] * len(imgs))
            print(f"    [-] DF40/{real_dir}: {len(imgs):,} real frames")

    for fake_dir in DF40_FAKE_DIRS:
        d = df40_test / fake_dir
        if d.exists():
            imgs = [f for f in d.rglob("*") if f.suffix.lower() in image_exts]
            paths.extend([str(f) for f in imgs])
            labels.extend([1] * len(imgs))
            print(f"    [-] DF40/{fake_dir}: {len(imgs):,} fake frames")

    # ── Processed video frames (if extracted) ────────────────────────────────
    for proc_dir in PROCESSED.glob("*") if PROCESSED.exists() else []:
        if not proc_dir.is_dir():
            continue
        is_real = "real" in proc_dir.name.lower()
        label = 0 if is_real else 1
        imgs = [f for f in proc_dir.rglob("*") if f.suffix.lower() in image_exts]
        if imgs:
            paths.extend([str(f) for f in imgs])
            labels.extend([label] * len(imgs))
            label_str = "real" if is_real else "fake"
            print(f"    [-] processed/{proc_dir.name}: {len(imgs):,} {label_str} frames")

    print(f"\n  Total: {len(paths):,} images ({labels.count(0):,} real + {labels.count(1):,} fake)")
    return paths, labels


def train_phase_a(epochs: int = 8, max_samples: int = 50000):
    """
    Fine-tune EfficientNet-B0 on video face frames.
    """
    print("\n" + "=" * 60)
    print("  PHASE A: Frame-Level Model Fine-Tuning")
    print("=" * 60)

    ensure_deps()
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Collect data
    all_paths, all_labels = collect_frame_dataset()
    if len(all_paths) < 100:
        print("  ❌ Not enough training data. Need at least 100 images.")
        return

    # Balance + subsample
    real_idx = [i for i, l in enumerate(all_labels) if l == 0]
    fake_idx = [i for i, l in enumerate(all_labels) if l == 1]
    random.shuffle(real_idx)
    random.shuffle(fake_idx)

    per_class = min(max_samples // 2, len(real_idx), len(fake_idx))
    selected = real_idx[:per_class] + fake_idx[:per_class]
    random.shuffle(selected)

    paths = [all_paths[i] for i in selected]
    labels = [all_labels[i] for i in selected]
    print(f"  Using {len(paths):,} balanced images ({per_class:,} per class)")

    # Train/val split
    split = int(len(paths) * 0.8)
    train_paths, val_paths = paths[:split], paths[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Dataset class
    class FrameDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                img = self.transform(img)
                return img, self.labels[idx]
            except Exception:
                # Return a blank image on error
                return torch.zeros(3, 224, 224), self.labels[idx]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FrameDataset(train_paths, train_labels, train_transform)
    val_ds = FrameDataset(val_paths, val_labels, val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    start_epoch = 0
    save_path = MODELS_DIR / "video_efficientnet_b0.pth"
    checkpoint_path = MODELS_DIR / "video_training_checkpoint.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint if interrupted ──────────────────────────────
    if checkpoint_path.exists():
        print(f"  🔄 Resuming from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        print(f"  Resuming from epoch {start_epoch + 1}, best_acc={best_acc:.4f}")

    from tqdm import tqdm

    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} [Train]", unit="batch")
        for images, targets in pbar:
            images = images.to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{100.*correct/total:.1f}%")

        scheduler.step()
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"  Epoch {epoch+1}/{epochs} [Val]  ", unit="batch"):
                images = images.to(device)
                targets = torch.tensor(targets, dtype=torch.long).to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total
        print(f"  → Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Saved best model (val_acc={val_acc:.4f})")

        # Save checkpoint after every epoch (for resume)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved (epoch {epoch+1})")

    # Clean up checkpoint after completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n  ✅ Phase A complete! Best validation accuracy: {best_acc:.4f}")
    print(f"  💾 Model saved: {save_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE B: Signal Feature Extraction from Training Videos
# ══════════════════════════════════════════════════════════════════════════════

def train_phase_b(max_videos: int = 500):
    """
    Extract temporal + biological + audio features from training videos.
    Saves feature vectors + labels to disk for Phase C.
    """
    print("\n" + "=" * 60)
    print("  PHASE B: Signal Feature Extraction")
    print("=" * 60)

    ensure_deps()
    from video_processor import extract_frames_for_analysis, get_video_info
    from temporal_signals import analyze_temporal_signals
    from biological_signals import analyze_biological_signals
    from audio_analyzer import analyze_audio
    from video_detector import analyze_frames_with_model
    from tqdm import tqdm

    video_exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

    # Collect video sources with labels
    sources = []

    # Celeb-DF
    celeb_real = DATASETS / "celeb_df" / "Celeb-real"
    celeb_fake = DATASETS / "celeb_df" / "Celeb-synthesis"
    youtube_real = DATASETS / "celeb_df" / "YouTube-real"

    if celeb_real.exists():
        vids = [f for f in celeb_real.iterdir() if f.suffix.lower() in video_exts]
        sources.extend([(str(v), 0) for v in vids])
        print(f"  ✅ Celeb-Real: {len(vids)} videos")
    if celeb_fake.exists():
        vids = [f for f in celeb_fake.iterdir() if f.suffix.lower() in video_exts]
        sources.extend([(str(v), 1) for v in vids])
        print(f"  ✅ Celeb-Fake: {len(vids)} videos")
    if youtube_real.exists():
        vids = [f for f in youtube_real.iterdir() if f.suffix.lower() in video_exts]
        sources.extend([(str(v), 0) for v in vids])
        print(f"  ✅ YouTube-Real: {len(vids)} videos")

    # FaceForensics++ (if extracted)
    ff_base = DATASETS / "faceforensics"
    for ff_sub in ["real", "Real", "youtube"]:
        ff_dir = ff_base / ff_sub
        if ff_dir.exists():
            vids = [f for f in ff_dir.rglob("*") if f.suffix.lower() in video_exts]
            sources.extend([(str(v), 0) for v in vids])
            print(f"  ✅ FF++/{ff_sub}: {len(vids)} real videos")
    for ff_sub in ["Deepfakes", "deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]:
        ff_dir = ff_base / ff_sub
        if ff_dir.exists():
            vids = [f for f in ff_dir.rglob("*") if f.suffix.lower() in video_exts]
            sources.extend([(str(v), 1) for v in vids])
            print(f"  ✅ FF++/{ff_sub}: {len(vids)} fake videos")

    if not sources:
        print("  ❌ No video sources found. Skipping Phase B.")
        return

    # Subsample if too many
    random.shuffle(sources)
    if max_videos > 0 and len(sources) > max_videos:
        # Balance
        reals = [(p, l) for p, l in sources if l == 0]
        fakes = [(p, l) for p, l in sources if l == 1]
        per_class = min(max_videos // 2, len(reals), len(fakes))
        sources = reals[:per_class] + fakes[:per_class]
        random.shuffle(sources)

    print(f"\n  Processing {len(sources)} videos for feature extraction...")

    # ── Resume support: load partial progress if interrupted ──────────────
    progress_path = MODELS_DIR / "video_features_progress.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    processed_paths = set()
    features_list = []
    labels_list = []
    errors = 0

    if progress_path.exists():
        print(f"  🔄 Resuming from previous progress...")
        with open(progress_path, "rb") as f:
            progress = pickle.load(f)
        features_list = progress["features"]
        labels_list = progress["labels"]
        processed_paths = set(progress.get("processed_paths", []))
        print(f"  Already processed: {len(processed_paths)} videos")

    # Filter out already-processed videos
    remaining = [(p, l) for p, l in sources if p not in processed_paths]
    print(f"  Remaining: {len(remaining)} videos")

    for i, (video_path, label) in enumerate(tqdm(remaining, desc="  Extracting features", unit="vid")):
        try:
            # Get video info
            info = get_video_info(video_path)
            fps = info.fps if info.fps > 0 else 30.0

            # Extract frames
            frames = extract_frames_for_analysis(video_path, sample_fps=2)
            if len(frames) < 5:
                errors += 1
                processed_paths.add(video_path)
                continue

            # Run analyzers
            frame_ai = analyze_frames_with_model(frames)
            temporal = analyze_temporal_signals(frames)
            biological = analyze_biological_signals(frames, fps)
            audio = analyze_audio(video_path, frames, fps)

            # Build feature vector
            feat = {
                "frame_ai_prob": frame_ai.get("mean_prob", 0.5),
                "frame_std": frame_ai.get("std_prob", 0.0),
                "frame_voted_fake": frame_ai.get("voted_fake", 0.0),
                "temporal_score": temporal["temporal_score"],
                "biological_score": biological["biological_score"],
                "audio_score": audio["audio_score"],
                **temporal.get("individual_scores", {}),
                **biological.get("individual_scores", {}),
                **audio.get("individual_scores", {}),
            }
            features_list.append(feat)
            labels_list.append(label)
            processed_paths.add(video_path)

            # Save progress every 10 videos (for resume on interrupt)
            if (i + 1) % 10 == 0:
                with open(progress_path, "wb") as f:
                    pickle.dump({"features": features_list, "labels": labels_list,
                                 "processed_paths": list(processed_paths)}, f)

        except KeyboardInterrupt:
            print(f"\n  ⏸️  Interrupted! Saving progress...")
            with open(progress_path, "wb") as f:
                pickle.dump({"features": features_list, "labels": labels_list,
                             "processed_paths": list(processed_paths)}, f)
            print(f"  💾 Progress saved ({len(features_list)} videos). Re-run to resume.")
            return
        except Exception as e:
            errors += 1
            processed_paths.add(video_path)
            continue

    if not features_list:
        print(f"  ❌ No features extracted (errors: {errors})")
        return

    # Save final features
    output = {
        "features": features_list,
        "labels": labels_list,
        "feature_names": list(features_list[0].keys()),
    }
    save_path = MODELS_DIR / "video_signal_features.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(output, f)

    # Clean up progress file
    if progress_path.exists():
        progress_path.unlink()

    n_real = labels_list.count(0)
    n_fake = labels_list.count(1)
    print(f"\n  ✅ Phase B complete!")
    print(f"  Extracted features from {len(features_list)} videos ({n_real} real + {n_fake} fake)")
    print(f"  Errors: {errors}")
    print(f"  💾 Saved: {save_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE C: Video Meta-Voter Training
# ══════════════════════════════════════════════════════════════════════════════

def train_phase_c():
    """
    Train GradientBoosting meta-voter on extracted features.
    """
    print("\n" + "=" * 60)
    print("  PHASE C: Video Meta-Voter Training")
    print("=" * 60)

    ensure_deps()
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    features_path = MODELS_DIR / "video_signal_features.pkl"
    if not features_path.exists():
        print(f"  ❌ Features file not found: {features_path}")
        print(f"  Run Phase B first: python train_video_model.py --phase-b")
        return

    with open(features_path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    labels = data["labels"]
    feature_names = data["feature_names"]

    print(f"  Loaded {len(features)} samples, {len(feature_names)} features")

    # Build numpy arrays
    X = np.array([
        [f.get(name, 0.0) for name in feature_names]
        for f in features
    ])
    y = np.array(labels)

    print(f"  Real: {(y == 0).sum()}, Fake: {(y == 1).sum()}")

    # Train pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            subsample=0.8,
            random_state=42,
        )),
    ])

    # Cross-validation
    print(f"\n  Running 5-fold cross-validation...")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold: {', '.join(f'{s:.3f}' for s in scores)}")

    # Train on full data
    print(f"\n  Training final model on all data...")
    pipeline.fit(X, y)

    # Save
    save_path = MODELS_DIR / "video_meta_voter.pkl"
    model_data = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "cv_accuracy": float(scores.mean()),
        "n_samples": len(y),
    }
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n  ✅ Phase C complete!")
    print(f"  CV Accuracy: {scores.mean():.4f}")
    print(f"  💾 Saved: {save_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video deepfake model training")
    parser.add_argument("--phase-a", action="store_true", help="Phase A: Fine-tune EfficientNet on video frames")
    parser.add_argument("--phase-b", action="store_true", help="Phase B: Extract signal features from videos")
    parser.add_argument("--phase-c", action="store_true", help="Phase C: Train video meta-voter")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for Phase A")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max images for Phase A")
    parser.add_argument("--max-videos", type=int, default=500, help="Max videos for Phase B")
    args = parser.parse_args()

    if args.all or args.phase_a:
        train_phase_a(epochs=args.epochs, max_samples=args.max_samples)
    if args.all or args.phase_b:
        train_phase_b(max_videos=args.max_videos)
    if args.all or args.phase_c:
        train_phase_c()

    if not any([args.all, args.phase_a, args.phase_b, args.phase_c]):
        parser.print_help()
        print("\nRecommended: python train_video_model.py --all")
