"""
CNN-Based AI Image Detector Training Script
============================================

Uses ResNet-18 transfer learning for improved AI detection accuracy.

Usage:
    python train_cnn.py

Expected Improvement:
    Current GradientBoosting: 73.9% AUC
    Target CNN: 90-95% AUC
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# Configuration
CONFIG = {
    "data_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/data/prepared",
    "output_dir": "d:/Devam/Microsoft VS Code/Codes/DeepFake/models/trained",
    "batch_size": 32,
    "num_epochs": 25,
    "learning_rate": 0.0001,
    "image_size": 224,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
}


class DeepFakeDataset(Dataset):
    """Dataset for AI-generated vs Real images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                return self.transform(Image.new("RGB", (224, 224))), label
            return Image.new("RGB", (224, 224)), label


class CNNDetector(nn.Module):
    """ResNet-18 based AI image detector."""

    def __init__(self, pretrained=True):
        super(CNNDetector, self).__init__()
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # 2 classes: Real (0), AI (1)
        )

    def forward(self, x):
        return self.resnet(x)


def load_dataset(data_dir: str) -> Tuple[list, list]:
    """Load image paths and labels from prepared directory."""
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)

    real_dir = os.path.join(data_dir, "real_unverified")
    fake_dir = os.path.join(data_dir, "synthetic_known")

    image_paths = []
    labels = []

    # Load real images (label = 0)
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(real_dir, filename))
                labels.append(0)

    # Load fake images (label = 1)
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(fake_dir, filename))
                labels.append(1)

    print(f"  Total images: {len(image_paths)}")
    print(f"  Real: {sum(1 for l in labels if l == 0)}")
    print(f"  Fake: {sum(1 for l in labels if l == 1)}")

    return image_paths, labels


def split_data(image_paths, labels, config):
    """Split data into train/val/test sets."""
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATA")
    print("=" * 60)

    np.random.seed(config["random_seed"])
    indices = np.random.permutation(len(image_paths))

    n_train = int(len(indices) * config["train_ratio"])
    n_val = int(len(indices) * config["val_ratio"])

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"  Train: {len(train_paths)}")
    print(f"  Val:   {len(val_paths)}")
    print(f"  Test:  {len(test_paths)}")

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def get_transforms(train=True):
    """Get image transformations with augmentation for training."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of AI class

    avg_loss = running_loss / len(dataloader)
    accuracy = (
        100.0 * np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    )
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc, all_labels, all_preds, all_probs


def train_model():
    """Main training function."""
    print("\n" + "=" * 60)
    print("  CNN-BASED AI IMAGE DETECTOR TRAINING")
    print("=" * 60)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  Training on CPU - this will be slower (~2-3 hours)")

    # Load data
    image_paths, labels = load_dataset(CONFIG["data_dir"])

    # Split data
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        split_data(image_paths, labels, CONFIG)
    )

    # Create datasets
    train_dataset = DeepFakeDataset(
        train_paths, train_labels, transform=get_transforms(train=True)
    )
    val_dataset = DeepFakeDataset(
        val_paths, val_labels, transform=get_transforms(train=False)
    )
    test_dataset = DeepFakeDataset(
        test_paths, test_labels, transform=get_transforms(train=False)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    print("\n" + "=" * 60)
    print("STEP 3: INITIALIZING MODEL")
    print("=" * 60)

    # Create model
    model = CNNDetector(pretrained=True).to(device)
    print("  ✓ ResNet-18 loaded with ImageNet pre-trained weights")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Training loop
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING")
    print("=" * 60)

    best_val_auc = 0.0
    best_model_path = os.path.join(CONFIG["output_dir"], "cnn_best_model.pth")

    for epoch in range(CONFIG["num_epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - start_time

        print(
            f"  Epoch [{epoch+1}/{CONFIG['num_epochs']}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}"
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "config": CONFIG,
                },
                best_model_path,
            )
            print(f"      ✓ New best model saved (Val AUC: {val_auc:.4f})")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("STEP 5: FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_auc, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test AUC: {test_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()

    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn}, FP: {fp}")
    print(f"    FN: {fn}, TP: {tp}")
    print(f"\n  False Negative Rate: {fnr:.2%} (AI images missed)")
    print(f"  False Positive Rate: {fpr:.2%} (Real images wrongly flagged)")

    # Save evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "CNN_ResNet18",
        "test_metrics": {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "false_negative_rate": float(fnr),
            "false_positive_rate": float(fpr),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "config": CONFIG,
    }

    report_path = os.path.join(CONFIG["output_dir"], "cnn_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  ✓ Evaluation report saved to: {report_path}")
    print(f"  ✓ Best model saved to: {best_model_path}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    # Comparison with old model
    print("\n  📊 COMPARISON WITH GRADIENTBOOSTING MODEL:")
    print(f"    Old AUC: 73.9% → New AUC: {test_auc*100:.1f}%")
    print(f"    Old FNR: 17.6% → New FNR: {fnr*100:.1f}%")
    print(f"    Old FPR: 48.6% → New FPR: {fpr*100:.1f}%")

    improvement = ((test_auc - 0.739) / 0.739) * 100
    print(f"\n  🎯 Improvement: +{improvement:.1f}%")

    return model, report


if __name__ == "__main__":
    train_model()
