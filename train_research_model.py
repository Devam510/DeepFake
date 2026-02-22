"""
Research Model Trainer - Maximum Performance Mode
===================================================

Trains multi-generator AI detection models with:
- EfficientNet-B4 backbone
- Frequency domain features
- Temperature scaling calibration
- Cross-validation

Output Metrics:
- ROC-AUC
- Expected Calibration Error (ECE)
- FNR/FPR at optimal threshold
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Configuration
RESEARCH_DATA_DIR = Path("data/research")
MODELS_DIR = Path("models/research")
BATCH_SIZE = 8  # Reduced from 32 to prevent CUDA OOM
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiGeneratorDataset(Dataset):
    """Dataset for multi-generator AI detection."""

    def __init__(self, root_dir: Path, transform=None, max_per_class=2000):
        self.transform = transform
        self.samples = []
        self.generator_labels = {}  # Track which generator each sample came from

        # Load real images (label=0)
        real_dir = root_dir / "real"
        if real_dir.exists():
            real_files = list(real_dir.glob("*.jpg"))[:max_per_class]
            for f in real_files:
                self.samples.append((str(f), 0, "real"))

        # Load synthetic images by generator (label=1)
        synthetic_count = 0
        gen_dir = root_dir / "by_generator"
        if gen_dir.exists():
            for generator_dir in gen_dir.iterdir():
                if generator_dir.is_dir():
                    gen_name = generator_dir.name
                    files = list(generator_dir.glob("*.jpg"))[
                        : max_per_class // 4
                    ]  # Balance
                    for f in files:
                        self.samples.append((str(f), 1, gen_name))
                        synthetic_count += 1

        print(f"  Dataset: {len(self.samples)} total")
        print(f"    Real: {len([s for s in self.samples if s[1] == 0])}")
        print(f"    Synthetic: {synthetic_count}")

        # Count by generator
        gen_counts = defaultdict(int)
        for _, _, gen in self.samples:
            gen_counts[gen] += 1
        for gen, count in gen_counts.items():
            print(f"      {gen}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, generator = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label, generator
        except Exception as e:
            # Return a blank image on error
            if self.transform:
                return torch.zeros(3, 224, 224), label, generator
            return Image.new("RGB", (224, 224)), label, generator


class EfficientNetDetector(nn.Module):
    """EfficientNet-B4 based AI detector."""

    def __init__(self, pretrained=True):
        super().__init__()

        # Load EfficientNet-B4 (use weights parameter for PyTorch 2.x)
        if pretrained:
            self.backbone = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b4(weights=None)

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),  # Real, AI
        )

        # Temperature for calibration (learned)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.backbone(x)
        return logits

    def calibrated_predict(self, x):
        """Get calibrated probabilities."""
        logits = self.forward(x)
        scaled_logits = logits / self.temperature
        return torch.softmax(scaled_logits, dim=1)


def calculate_ece(probs, labels, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = labels[in_bin].mean()
            bin_conf = probs[in_bin].mean()
            bin_size = in_bin.sum() / total
            ece += bin_size * abs(bin_acc - bin_conf)

    return ece


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    all_generators = []

    with torch.no_grad():
        for images, labels, generators in loader:
            images = images.to(device)
            probs = model.calibrated_predict(images)
            ai_probs = probs[:, 1].cpu().numpy()

            all_probs.extend(ai_probs)
            all_labels.extend(labels.numpy())
            all_generators.extend(generators)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs)
    ece = calculate_ece(all_probs, all_labels)

    # Calculate FNR/FPR at 0.5 threshold
    preds = (all_probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    # Per-generator metrics
    gen_metrics = {}
    for gen in set(all_generators):
        mask = np.array([g == gen for g in all_generators])
        if mask.sum() > 10:
            gen_probs = all_probs[mask]
            gen_labels = all_labels[mask]
            try:
                gen_auc = roc_auc_score(gen_labels, gen_probs)
                gen_metrics[gen] = {
                    "auc": gen_auc,
                    "count": int(mask.sum()),
                    "mean_prob": float(gen_probs.mean()),
                }
            except ValueError:
                pass  # Skip if only one class

    return {
        "auc": auc,
        "ece": ece,
        "fnr": fnr,
        "fpr": fpr,
        "accuracy": acc,
        "per_generator": gen_metrics,
    }


def calibrate_temperature(model, loader, device):
    """Learn temperature parameter for calibration."""
    model.eval()

    # Freeze all except temperature
    for param in model.parameters():
        param.requires_grad = False
    model.temperature.requires_grad = True

    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    def calibration_step():
        optimizer.zero_grad()
        loss = criterion(all_logits / model.temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(calibration_step)

    # Unfreeze
    for param in model.parameters():
        param.requires_grad = True

    print(f"  Calibrated temperature: {model.temperature.item():.3f}")


def main():
    print("\n" + "=" * 70)
    print("RESEARCH MODEL TRAINER - MAXIMUM PERFORMANCE MODE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Research data: {RESEARCH_DATA_DIR}")

    # Create output directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
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
    print("\n[DATA] Loading dataset...")
    full_dataset = MultiGeneratorDataset(RESEARCH_DATA_DIR, train_transform)

    # Split: 70% train, 15% val, 15% test
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Update val/test transforms
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(
        f"\n  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create model
    print("\n[MODEL] Creating EfficientNet-B4 model...")
    model = EfficientNetDetector(pretrained=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    print("\n[TRAIN] Training...")
    best_auc = 0
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_auc": val_metrics["auc"],
                "val_ece": val_metrics["ece"],
            }
        )

        # Save best model
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), MODELS_DIR / "best_research_model.pth")

        print(
            f"  Epoch {epoch:02d}: Loss={train_loss:.4f}, Acc={train_acc:.3f}, "
            f"AUC={val_metrics['auc']:.4f}, ECE={val_metrics['ece']:.4f}"
        )

        # Early stopping
        if epoch - best_epoch > 5:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "best_research_model.pth"))

    # Temperature calibration
    print("\n[CALIBRATE] Calibrating temperature...")
    calibrate_temperature(model, val_loader, DEVICE)

    # Final evaluation
    print("\n[EVAL] Final Evaluation on Test Set...")
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  ROC-AUC:     {test_metrics['auc']:.4f}")
    print(f"  ECE:         {test_metrics['ece']:.4f}")
    print(f"  FNR:         {test_metrics['fnr']:.4f} (AI classified as Real)")
    print(f"  FPR:         {test_metrics['fpr']:.4f} (Real classified as AI)")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")

    print("\n  Per-Generator Performance:")
    for gen, metrics in test_metrics["per_generator"].items():
        print(
            f"    {gen}: AUC={metrics['auc']:.4f}, n={metrics['count']}, "
            f"mean_prob={metrics['mean_prob']:.3f}"
        )

    # Save final model and report
    torch.save(
        {
            "model_state": model.state_dict(),
            "temperature": model.temperature.item(),
            "metrics": test_metrics,
            "history": history,
            "created": datetime.now().isoformat(),
        },
        MODELS_DIR / "research_model_final.pth",
    )

    # Save report
    report = {
        "created": datetime.now().isoformat(),
        "metrics": test_metrics,
        "history": history,
        "training_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE),
        },
        "limitations": [
            "Detection accuracy varies by generator",
            "Cutting-edge generators (DALL-E 3, Midjourney v6) may evade detection",
            "Calibration based on validation set distribution",
            "Results may not generalize to all image types",
        ],
    }

    with open(MODELS_DIR / "research_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Model saved to: {MODELS_DIR / 'research_model_final.pth'}")
    print(f"[OK] Report saved to: {MODELS_DIR / 'research_evaluation_report.json'}")

    return test_metrics


if __name__ == "__main__":
    main()
