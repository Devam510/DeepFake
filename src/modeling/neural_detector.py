"""
DeepFake Detection System - Neural Network Detector
Layer 3: Modeling (Ensemble Architecture)

CNN/ViT-based detector for spatial artifact detection.
Architecture designed to detect sub-perceptual generative artifacts.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import json
import os

if TYPE_CHECKING:
    import torch

from .base import BaseDetector, ModelPrediction

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================
# Model Architectures
# ============================================================

if TORCH_AVAILABLE:

    class FrequencyBranch(nn.Module):
        """
        Branch that processes frequency domain features.
        Captures spectral artifacts from generative models.
        """

        def __init__(self, out_features: int = 128):
            super().__init__()
            # Process FFT magnitude spectrum
            self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(128 * 4 * 4, out_features)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class SpatialBranch(nn.Module):
        """
        Branch that processes spatial domain features.
        Based on EfficientNet-style blocks.
        """

        def __init__(self, out_features: int = 256):
            super().__init__()
            # Simplified EfficientNet-like architecture
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
            )

            self.blocks = nn.Sequential(
                self._make_block(32, 64, stride=2),
                self._make_block(64, 128, stride=2),
                self._make_block(128, 256, stride=2),
                self._make_block(256, 512, stride=2),
            )

            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, out_features)

        def _make_block(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class DualBranchDetector(nn.Module):
        """
        Dual-branch detector combining spatial and frequency analysis.

        Architecture:
        - Spatial branch: CNN for texture/edge artifacts
        - Frequency branch: CNN for spectral artifacts
        - Fusion layer: Combines both for final prediction
        """

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.spatial_branch = SpatialBranch(out_features=256)
            self.frequency_branch = FrequencyBranch(out_features=128)

            # Fusion layers
            self.fusion = nn.Sequential(
                nn.Linear(256 + 128, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Classification head
            self.classifier = nn.Linear(128, num_classes)

            # For feature extraction
            self.feature_dim = 128

        def forward(
            self, spatial_input: torch.Tensor, freq_input: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.

            Returns:
                (logits, features) for classification and OOD detection
            """
            spatial_features = self.spatial_branch(spatial_input)
            freq_features = self.frequency_branch(freq_input)

            combined = torch.cat([spatial_features, freq_features], dim=1)
            features = self.fusion(combined)
            logits = self.classifier(features)

            return logits, features

        def get_features(
            self, spatial_input: torch.Tensor, freq_input: torch.Tensor
        ) -> torch.Tensor:
            """Extract features without classification."""
            _, features = self.forward(spatial_input, freq_input)
            return features


class NeuralNetworkDetector(BaseDetector):
    """
    Neural network-based detector for spatial artifacts.

    Uses a dual-branch architecture that analyzes both
    spatial and frequency domain features.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        input_size: int = 224,
    ):
        """
        Args:
            model_path: Path to saved model weights
            device: 'cuda', 'cpu', or 'auto'
            input_size: Input image size
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required: pip install torch torchvision")
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required: pip install opencv-python")

        self.input_size = input_size

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = DualBranchDetector(num_classes=2)
        self.model.to(self.device)
        self.model.eval()

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

        # Normalization parameters (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    @property
    def name(self) -> str:
        return "neural_network_detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_media_types(self) -> List[str]:
        return ["image"]

    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess image for model input.

        Returns:
            (spatial_input, frequency_input) tensors
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (self.input_size, self.input_size))

        # Normalize for spatial branch
        spatial = img.astype(np.float32) / 255.0
        spatial = (spatial - self.mean) / self.std
        spatial = np.transpose(spatial, (2, 0, 1))  # HWC -> CHW
        spatial = torch.from_numpy(spatial).float().unsqueeze(0)

        # Compute FFT for frequency branch
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
        freq = torch.from_numpy(magnitude).float().unsqueeze(0).unsqueeze(0)

        return spatial.to(self.device), freq.to(self.device)

    def predict(self, input_path: str) -> ModelPrediction:
        """Generate prediction for an image."""
        spatial_input, freq_input = self.preprocess(input_path)

        with torch.no_grad():
            logits, features = self.model(spatial_input, freq_input)
            probs = F.softmax(logits, dim=1)

            # Class 0 = real, Class 1 = synthetic
            synthetic_prob = probs[0, 1].item()

        # Compute confidence interval
        lower, upper = self.compute_confidence_interval(
            synthetic_prob, n_samples=100, confidence_level=0.95
        )

        # Feature attribution (simplified)
        attributions = {
            "spatial_features": float(features[0, :64].mean().item()),
            "frequency_features": float(features[0, 64:].mean().item()),
        }

        return ModelPrediction(
            authenticity_score=(1.0 - synthetic_prob) * 100,
            synthetic_probability=synthetic_prob,
            confidence_lower=lower,
            confidence_upper=upper,
            confidence_level=0.95,
            feature_attributions=attributions,
            model_name=self.name,
            model_version=self.version,
        )

    def get_feature_vector(self, input_path: str) -> np.ndarray:
        """Extract feature vector for ensemble."""
        spatial_input, freq_input = self.preprocess(input_path)

        with torch.no_grad():
            features = self.model.get_features(spatial_input, freq_input)
            return features.cpu().numpy().flatten()

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "version": self.version,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.input_size = checkpoint.get("input_size", 224)
        self.model.eval()


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural network detector")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", "-m", help="Path to trained model (optional)")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument("--device", default="auto", help="Device: cuda, cpu, or auto")

    args = parser.parse_args()

    detector = NeuralNetworkDetector(model_path=args.model, device=args.device)
    prediction = detector.predict(args.image)

    result = prediction.to_dict()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Prediction saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))
