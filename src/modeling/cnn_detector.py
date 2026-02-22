"""
CNN-Based Production Detector
==============================

Production-ready CNN detector for AI-generated image detection.
Compatible with existing ProductionFrozenDetector API.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


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


class ProductionCNNDetector:
    """
    Production CNN Detector for AI-Generated Images

    Uses ResNet-18 transfer learning for improved detection accuracy.
    Compatible with existing ProductionFrozenDetector API.
    """

    def __init__(self, model_path=None):
        """Initialize CNN detector with trained model."""
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "models",
                "trained",
                "cnn_best_model.pth",
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNDetector(pretrained=False).to(self.device)

        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print(f"✅ Loaded CNN model from: {model_path}")
            print(f"   Using device: {self.device}")
        else:
            raise FileNotFoundError(
                f"CNN model not found: {model_path}\n"
                "Please train the model first using: python train_cnn.py"
            )

        # Image preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_from_image_path(self, image_path: str) -> dict:
        """
        Predict AI probability from image file path.

        Args:
            image_path: Path to image file

        Returns:
            dict with keys:
                - synthetic_probability: float (0-1)
                - synthetic_likelihood: str (interpretation)
                - confidence_band: str
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return self.predict_from_image(image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def predict_from_image(self, image: Image.Image) -> dict:
        """
        Predict AI probability from PIL Image.

        Args:
            image: PIL Image object

        Returns:
            dict with prediction results
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            ai_prob = probs[0, 1].item()  # Probability of AI class

        # Interpret result
        result = {
            "synthetic_probability": ai_prob,
            "synthetic_likelihood": self._get_likelihood_label(ai_prob),
            "confidence_band": self._get_confidence_band(ai_prob),
            "model_version": "CNN_ResNet18_v1.0.0",
            "semantic_version": "1.0.0",
        }

        return result

    def predict(self, features=None, image_path=None, image=None) -> dict:
        """
        Unified prediction interface compatible with old API.

        Args:
            features: Ignored (CNN doesn't use hand-crafted features)
            image_path: Optional path to image
            image: Optional PIL Image

        Returns:
            dict with prediction results
        """
        if image_path is not None:
            return self.predict_from_image_path(image_path)
        elif image is not None:
            return self.predict_from_image(image)
        else:
            raise ValueError(
                "Either image_path or image must be provided for CNN prediction"
            )

    def _get_likelihood_label(self, prob: float) -> str:
        """Map probability to language label."""
        if prob < 0.2:
            return "LOW synthetic likelihood"
        elif prob < 0.4:
            return "MODERATE-LOW synthetic likelihood"
        elif prob < 0.6:
            return "INCONCLUSIVE"
        elif prob < 0.8:
            return "MODERATE-HIGH synthetic likelihood"
        else:
            return "HIGH synthetic likelihood"

    def _get_confidence_band(self, prob: float) -> str:
        """Map probability to confidence band."""
        # CNN models are generally well-calibrated
        if 0.3 <= prob <= 0.7:
            return "MEDIUM confidence (inconclusive zone)"
        else:
            return "HIGH confidence"


# Convenience function for backward compatibility
def create_detector(use_cnn=True):
    """
    Create detector instance.

    Args:
        use_cnn: If True, use CNN detector. If False, use statistical detector.

    Returns:
        Detector instance
    """
    if use_cnn:
        return ProductionCNNDetector()
    else:
        # Fall back to old detector
        from production_detector import ProductionFrozenDetector

        return ProductionFrozenDetector()
