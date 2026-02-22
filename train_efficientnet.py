"""
EfficientNet Retraining Pipeline
=================================

Retrains EfficientNet for DeepFake detection using downloaded datasets.
Supports B0 (recommended for GPUs with ≤6GB VRAM) and B4.

Usage:
    python train_efficientnet.py --from-scratch
    python train_efficientnet.py --epochs 10 --batch-size 16 --from-scratch
    python train_efficientnet.py --resume  (resume from last checkpoint)
    python train_efficientnet.py --model b4 --batch-size 4  (for B4 on small GPU)
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFile, ImageFilter
from pathlib import Path
from collections import Counter

# Allow truncated images (common in web-scraped datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).parent.resolve()


# ============================================================
# Model Architecture (MUST match ensemble_detector.py exactly)
# ============================================================
class EfficientNetDetector(nn.Module):
    """EfficientNet-based AI detector. Supports B0 and B4."""

    def __init__(self, model_name="b0"):
        super().__init__()
        self.model_name = model_name

        if model_name == "b0":
            self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b4(weights=None)

        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )
        self.temperature = nn.Parameter(torch.ones(1))


# ============================================================
# Filter Simulation Augmentation
# ============================================================
class SimulateFilter:
    """Simulates social media filters to make the model robust against them.
    
    This is the KEY augmentation — it teaches the model that filtered photos
    are still REAL, preventing false positives on Snapchat/Instagram photos.
    """

    def __call__(self, img):
        if random.random() < 0.3:  # Apply 30% of the time
            filter_type = random.choice([
                "warm", "cool", "vintage", "blur", "sharpen", "brightness"
            ])

            if filter_type == "warm":
                # Warm color cast (like Instagram)
                r, g, b = img.split()
                r = r.point(lambda x: min(255, int(x * 1.15)))
                b = b.point(lambda x: int(x * 0.85))
                img = Image.merge("RGB", (r, g, b))

            elif filter_type == "cool":
                # Cool color cast
                r, g, b = img.split()
                r = r.point(lambda x: int(x * 0.85))
                b = b.point(lambda x: min(255, int(x * 1.15)))
                img = Image.merge("RGB", (r, g, b))

            elif filter_type == "vintage":
                # Reduced saturation + warm tint
                from PIL import ImageEnhance
                img = ImageEnhance.Color(img).enhance(0.7)
                r, g, b = img.split()
                r = r.point(lambda x: min(255, int(x * 1.1)))
                img = Image.merge("RGB", (r, g, b))

            elif filter_type == "blur":
                # Skin smoothing (beautification)
                img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

            elif filter_type == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)

            elif filter_type == "brightness":
                from PIL import ImageEnhance
                factor = random.uniform(0.8, 1.3)
                img = ImageEnhance.Brightness(img).enhance(factor)

        return img


# ============================================================
# JPEG Compression Augmentation
# ============================================================
class SimulateCompression:
    """Simulates social media JPEG recompression."""

    def __call__(self, img):
        if random.random() < 0.2:  # 20% of the time
            import io
            quality = random.randint(60, 85)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
        return img


# ============================================================
# Dataset
# ============================================================
class DeepFakeDataset(Dataset):
    """Loads images from multiple dataset directories.
    
    Automatically discovers real/fake folders by name patterns.
    Supports: 'real'/'fake', 'REAL'/'FAKE', '0'/'1', 'original'/'synthetic'
    Also recognizes ArtiFact dataset generator-named folders.
    """

    REAL_KEYWORDS = {"real", "original", "authentic", "human", "0_real", "0"}
    FAKE_KEYWORDS = {"fake", "ai", "synthetic", "generated", "deepfake", "1_fake", "1"}

    # ArtiFact dataset: folders named by source, not real/fake
    KNOWN_REAL_SOURCES = {
        "coco", "imagenet", "ffhq", "celebahq", "lsun", "afhq",
        "landscape", "metfaces",
    }
    KNOWN_FAKE_GENERATORS = {
        "big_gan", "cips", "cycle_gan", "ddpm", "denoising_diffusion_gan",
        "diffusion_gan", "face_synthetics", "gansformer", "gau_gan",
        "generative_inpainting", "glide", "lama", "latent_diffusion",
        "mat", "palette", "projected_gan", "pro_gan", "sfhq",
        "stable_diffusion", "star_gan", "stylegan1", "stylegan2",
        "dalle", "midjourney", "sd3", "sdxl",
    }

    def __init__(self, dataset_dirs, transform=None, split="train", val_ratio=0.1, max_per_class=None):
        self.transform = transform
        self.samples = []  # List of (path, label)

        print(f"\n{'='*50}")
        print(f"  Loading {split.upper()} data")
        print(f"{'='*50}")

        all_real = []
        all_fake = []

        for dataset_dir in dataset_dirs:
            path = Path(dataset_dir)
            if not path.exists():
                print(f"  ⚠️  Skipping (not found): {dataset_dir}")
                continue

            real, fake = self._scan_directory(path)
            all_real.extend(real)
            all_fake.extend(fake)
            print(f"  📦 {path.name:25s} → {len(real):>8,} real, {len(fake):>8,} fake")

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(all_real)
        random.shuffle(all_fake)

        # Limit per class to balance and keep training practical
        if max_per_class and max_per_class > 0:
            all_real = all_real[:max_per_class]
            all_fake = all_fake[:max_per_class]

        # Split train/val
        real_split = int(len(all_real) * (1 - val_ratio))
        fake_split = int(len(all_fake) * (1 - val_ratio))

        if split == "train":
            real_set = all_real[:real_split]
            fake_set = all_fake[:fake_split]
        else:
            real_set = all_real[real_split:]
            fake_set = all_fake[fake_split:]

        self.samples = [(p, 0) for p in real_set] + [(p, 1) for p in fake_set]
        random.shuffle(self.samples)

        self.labels = [s[1] for s in self.samples]
        counts = Counter(self.labels)

        print(f"\n  {split.upper()} set: {len(self.samples):,} images")
        print(f"    Real: {counts.get(0, 0):,} | Fake: {counts.get(1, 0):,}")
        
        if len(self.samples) == 0:
            print("  ❌ NO IMAGES FOUND! Check your dataset folder structure.")

    def _scan_directory(self, root_path):
        """Recursively find real and fake image folders.
        
        Uses three strategies:
        1. Keyword matching (real/fake in folder name)
        2. Known real datasets (coco, imagenet, ffhq, etc.)
        3. Known AI generators (stylegan2, stable_diffusion, etc.)
        
        Parent folder labels propagate to ALL child folders.
        e.g. stylegan2/car-part1/ → all images inside are "fake"
        """
        real_images = []
        fake_images = []
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        
        # Track which parent folders have been classified
        # Maps directory path → "real" or "fake"
        classified_dirs = {}

        for dirpath, dirnames, filenames in os.walk(root_path):
            folder = os.path.basename(dirpath).lower().replace(" ", "_")
            
            # Check if any PARENT directory was already classified
            parent = str(Path(dirpath).parent)
            inherited_label = classified_dirs.get(parent)

            # Strategy 1: Keyword matching
            is_real = any(kw in folder for kw in self.REAL_KEYWORDS)
            is_fake = any(kw in folder for kw in self.FAKE_KEYWORDS)

            # Strategy 2: Known real sources (ArtiFact dataset)
            if not is_real and not is_fake:
                if folder in self.KNOWN_REAL_SOURCES:
                    is_real = True
                elif folder in self.KNOWN_FAKE_GENERATORS:
                    is_fake = True

            # Strategy 3: Inherit from parent folder
            if not is_real and not is_fake and inherited_label:
                if inherited_label == "real":
                    is_real = True
                else:
                    is_fake = True

            # Avoid ambiguous folders that match both
            if is_real and is_fake:
                continue

            # Record classification for child inheritance
            if is_real:
                classified_dirs[dirpath] = "real"
                for f in filenames:
                    if Path(f).suffix.lower() in exts:
                        real_images.append(str(Path(dirpath) / f))
            elif is_fake:
                classified_dirs[dirpath] = "fake"
                for f in filenames:
                    if Path(f).suffix.lower() in exts:
                        fake_images.append(str(Path(dirpath) / f))

        return real_images, fake_images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Skip corrupt images
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# Transforms
# ============================================================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        SimulateFilter(),          # Social media filter simulation
        SimulateCompression(),     # JPEG recompression
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.05),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


# ============================================================
# Training
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  EfficientNet-{args.model.upper()} Training Pipeline")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Model:      EfficientNet-{args.model.upper()}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR:         {args.lr}")

    if device.type == "cpu":
        print("\n  ⚠️  Training on CPU — this will be SLOW for large datasets.")
        print("      Consider using --max-per-class 50000 to limit data.")

    # ── Dataset dirs ──
    dataset_base = BASE_DIR / "datasets"
    dataset_dirs = [
        str(d) for d in dataset_base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]

    if not dataset_dirs:
        print("❌ No datasets found in datasets/ folder!")
        sys.exit(1)

    # ── Transforms ──
    train_tf, val_tf = get_transforms()

    # ── Datasets ──
    train_ds = DeepFakeDataset(
        dataset_dirs, transform=train_tf, split="train",
        val_ratio=args.val_ratio, max_per_class=args.max_per_class
    )
    val_ds = DeepFakeDataset(
        dataset_dirs, transform=val_tf, split="val",
        val_ratio=args.val_ratio, max_per_class=args.max_per_class
    )

    if len(train_ds) == 0:
        print("❌ No training images found!")
        sys.exit(1)

    # ── Balanced Sampler ──
    label_counts = Counter(train_ds.labels)
    weights = [1.0 / label_counts[l] for l in train_ds.labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    # Set up DataLoader with safe defaults for Windows
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    # ── Model ──
    print("\n  Initializing model...")
    model = EfficientNetDetector(model_name=args.model)

    # Load pretrained weights if available (fine-tune from existing)
    existing_model = BASE_DIR / "models" / "research" / "research_model_final.pth"
    if existing_model.exists() and not args.from_scratch:
        print(f"  Loading existing weights: {existing_model.name}")
        try:
            checkpoint = torch.load(str(existing_model), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.temperature.data = torch.tensor([checkpoint["temperature"]])
            print("  ✅ Fine-tuning from existing production model")
        except Exception as e:
            print(f"  ⚠️  Can't load existing model ({e})")
            print("  ✅ Falling back to ImageNet pretrained backbone")
            args.from_scratch = True

    if args.from_scratch:
        # Load ImageNet pretrained backbone
        print(f"  Loading ImageNet pretrained EfficientNet-{args.model.upper()}...")
        if args.model == "b0":
            pretrained = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        else:
            pretrained = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Copy backbone weights only (not classifier)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained.state_dict().items()
            if k in model_dict and "classifier" not in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("  ✅ Training from ImageNet pretrained backbone")

    model = model.to(device)

    # ── Optimizer ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume ──
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_path = BASE_DIR / "models" / "training_checkpoint.pth"

    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  ✅ Resumed from epoch {start_epoch}, best acc: {best_val_acc:.2%}")

    # ── Mixed Precision (prevents CUDA OOM) ──
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"  Mixed Precision: {'ON' if use_amp else 'OFF'}")

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"  Starting training...")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model.backbone(images)
                loss = criterion(logits, labels)

            # NaN detection — abort if loss explodes
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  ❌ NaN/Inf loss detected at batch {batch_idx+1}! Aborting epoch.")
                print(f"     Try: --lr 1e-6 or --from-scratch")
                break

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            # Gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # Progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                running_acc = train_correct / train_total
                elapsed = time.time() - epoch_start
                batches_per_sec = (batch_idx + 1) / elapsed
                remaining = (len(train_loader) - batch_idx - 1) / batches_per_sec
                print(
                    f"  Epoch {epoch+1}/{args.epochs} "
                    f"[{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {running_acc:.2%} "
                    f"ETA: {remaining/60:.0f}min"
                )

        train_acc = train_correct / max(train_total, 1)
        train_loss_avg = train_loss / max(train_total, 1)

        # If NaN was detected, skip this epoch's validation and checkpoint
        if train_total == 0:
            print("  ⚠️  Epoch skipped due to NaN loss. Try --from-scratch")
            continue

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model.backbone(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss_avg = val_loss / val_total if val_total > 0 else 0

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{args.epochs} ({epoch_time:.0f}s)")
        print(f"    Train — Loss: {train_loss_avg:.4f}  Acc: {train_acc:.2%}")
        print(f"    Val   — Loss: {val_loss_avg:.4f}  Acc: {val_acc:.2%}")
        print(f"    LR:    {scheduler.get_last_lr()[0]:.6f}")

        # ── Save Checkpoint ──
        os.makedirs(str(BASE_DIR / "models"), exist_ok=True)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }, str(checkpoint_path))

        # ── Save Best Model (production format) ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = BASE_DIR / "models" / "research" / "research_model_v2.pth"
            os.makedirs(str(save_path.parent), exist_ok=True)

            torch.save({
                "model_state": model.state_dict(),
                "temperature": model.temperature.item(),
                "model_name": args.model,
                "epoch": epoch,
                "val_acc": val_acc,
                "train_acc": train_acc,
            }, str(save_path))

            print(f"    ✅ NEW BEST MODEL saved ({val_acc:.2%})")
        print()

    total_time = time.time() - total_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print("=" * 60)
    print(f"  Training Complete!")
    print(f"  Total Time:    {minutes}m {seconds}s")
    print(f"  Model:         EfficientNet-{args.model.upper()}")
    print(f"  Best Val Acc:  {best_val_acc:.2%}")
    print(f"  Model saved:   models/research/research_model_v2.pth")
    print()
    print(f"  To use the new model, rename it:")
    print(f"    1. Backup: rename research_model_final.pth → research_model_v1_backup.pth")
    print(f"    2. Activate: rename research_model_v2.pth → research_model_final.pth")
    print(f"    3. Update ensemble_detector.py to use EfficientNet-{args.model.upper()}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 for DeepFake detection")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 0.00001)")
    parser.add_argument("--model", type=str, default="b0", choices=["b0", "b4"], help="Model: b0 (4GB GPU) or b4 (8GB+ GPU)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (default: 0 for compatibility)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--max-per-class", type=int, default=100000, help="Max images per class (default: 100K, use -1 for all)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch (no existing weights)")

    args = parser.parse_args()
    train(args)
