"""
General Image Detector Training Script
=======================================

Trains the general image detector on diverse AI-generated images:
- StyleGAN faces
- Midjourney scenes
- DALL-E 3 images
- Real photos

This creates a model that detects AI generation in non-face images.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models

    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch not available. Install with: pip install torch torchvision")
    TORCH_AVAILABLE = False
    sys.exit(1)


# Configuration
RESEARCH_DATA_DIR = Path("data/research")
MODEL_OUTPUT_DIR = Path("models/trained")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneralImageDataset(Dataset):
    """Dataset for general image AI detection training."""

    def __init__(
        self, real_dir: Path, synthetic_dirs: list, transform=None, max_per_class=2000
    ):
        self.transform = transform
        self.samples = []  # (path, label) where 0=real, 1=synthetic

        # Load real images
        # Use rglob for recursive search in subdirectories
        real_images = list(real_dir.rglob("*.jpg"))[:max_per_class]
        for img_path in real_images:
            self.samples.append((str(img_path), 0))
        print(f"  Loaded {len(real_images)} real images")

        # Load synthetic images from multiple generators
        total_synthetic = 0
        for syn_dir in synthetic_dirs:
            if syn_dir.exists():
                syn_images = list(syn_dir.rglob("*.jpg"))[
                    : max_per_class // len(synthetic_dirs)
                ]
                for img_path in syn_images:
                    self.samples.append((str(img_path), 1))
                total_synthetic += len(syn_images)
                print(f"  Loaded {len(syn_images)} from {syn_dir.name}")

        print(f"  Total synthetic: {total_synthetic}")

        # Shuffle
        random.shuffle(self.samples)
        print(f"  Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a blank image on error
            if self.transform:
                return torch.zeros(3, 224, 224), label
            return Image.new("RGB", (224, 224)), label


class GeneralDetectorModel(nn.Module):
    """EfficientNet-based general image detector."""

    def __init__(self):
        super().__init__()

        # Use EfficientNet-B0 (lighter weight for faster training)
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(
                f"    Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}"
            )

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def main():
    print("\n" + "=" * 60)
    print("GENERAL IMAGE DETECTOR TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")

    # Setup directories
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    print("\n📊 Loading dataset...")

    DATA_DIR = Path("data/general_detector")
    real_dir = DATA_DIR / "real"
    synthetic_dirs = [
        DATA_DIR / "synthetic" / "sd",
        DATA_DIR / "synthetic" / "mj",
        DATA_DIR / "synthetic" / "dalle",
        DATA_DIR / "synthetic" / "other",
        DATA_DIR / "synthetic" / "stable_diffusion",
        DATA_DIR / "synthetic" / "midjourney",
        DATA_DIR / "synthetic" / "dalle3",
    ]

    # Check if data exists
    if not real_dir.exists():
        print("❌ Training data not found!")
        print("   Run: python download_mega.py")
        return False

    # Create dataset - use all available images
    full_dataset = GeneralImageDataset(
        real_dir=real_dir,
        synthetic_dirs=synthetic_dirs,
        transform=train_transform,
        max_per_class=50000,  # Use more data
    )

    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Override val transform (can't easily do this with random_split, so skip for now)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create model
    print("\n🧠 Creating model...")
    model = GeneralDetectorModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print("\n🚀 Starting training...")
    best_val_acc = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODEL_OUTPUT_DIR / "general_detector_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                },
                save_path,
            )
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")

    # Save final model
    final_path = MODEL_OUTPUT_DIR / "general_detector_final.pth"
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
        },
        final_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")

    # Save training metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "epochs": EPOCHS,
        "best_val_acc": best_val_acc,
        "device": str(DEVICE),
        "model": "EfficientNet-B0",
    }
    with open(MODEL_OUTPUT_DIR / "general_detector_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return True


if __name__ == "__main__":
    main()
