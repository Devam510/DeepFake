"""
General Image AI Detector
=========================

Detects AI-generated content in non-face images:
- Landscapes
- Objects
- Indoor/outdoor scenes
- Art/illustrations
- Synthetic graphics

This detector is designed to be honest about limitations:
- Returns UNCERTAIN when confidence is low
- Reports domain-specific accuracy
- Never claims universal detection

IMPORTANT: This is a PARALLEL detector to the face system.
          It does NOT replace or modify face detection.
"""

import os
import sys
from dataclasses import dataclass
from typing import Literal, List, Optional, Tuple, Dict
import numpy as np
from PIL import Image

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class GeneralDetectionResult:
    """Result from general image AI detection."""

    synthetic_probability: float  # 0.0 to 1.0
    confidence_interval: Dict[str, float]  # {"lower": x, "upper": y}
    synthetic_likelihood: str  # Interpretation
    confidence_band: str  # HIGH/MEDIUM/LOW
    domain_used: str  # Which domain this was classified as
    detector_version: str
    limitations: str
    is_uncertain: bool  # True if should return UNCERTAIN

    def to_dict(self) -> dict:
        return {
            "synthetic_probability": self.synthetic_probability,
            "confidence_interval": self.confidence_interval,
            "synthetic_likelihood": self.synthetic_likelihood,
            "confidence_band": self.confidence_band,
            "domain_used": self.domain_used,
            "detector_version": self.detector_version,
            "limitations": self.limitations,
            "is_uncertain": self.is_uncertain,
        }


class GeneralImageDetector:
    """
    AI detection for non-face images.

    Uses multiple signals:
    1. Frequency analysis (DCT artifacts)
    2. Texture consistency
    3. Neural features (if available)
    4. Statistical patterns

    Calibrated to degrade gracefully to UNCERTAIN.
    """

    VERSION = "GeneralImageDetector_v1.0.0"

    # Uncertainty thresholds
    UNCERTAIN_LOWER = 0.35
    UNCERTAIN_UPPER = 0.65

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize general image detector.

        Args:
            model_path: Path to trained model weights (optional)
        """
        self.device = None
        self.model = None
        self.model_loaded = False

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._init_neural_model(model_path)

        # Image transforms for neural model
        if TORCH_AVAILABLE:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def _init_neural_model(self, model_path: Optional[str]):
        """Initialize neural network model."""
        try:
            # Use EfficientNet-B0 (matches trained architecture)
            self.model = models.efficientnet_b0(weights=None)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )

            self.model = self.model.to(self.device)

            # Try to find trained model if no path specified
            if model_path is None:
                default_path = os.path.join("models", "trained", "general_detector_best.pth")
                if os.path.exists(default_path):
                    model_path = default_path

            # Load weights if available
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                
                # Handle models saved with 'backbone.' prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        new_state_dict[k[9:]] = v  # Strip 'backbone.' prefix
                    else:
                        new_state_dict[k] = v
                
                self.model.load_state_dict(new_state_dict)
                self.model.eval()
                self.model_loaded = True
                print(f"✅ Loaded general detector model from: {model_path}")
            else:
                # Use pretrained ImageNet weights for basic feature extraction
                self.model.eval()
                self.model_loaded = False
                print(
                    "⚠️ General detector: No trained weights, using signal-based detection"
                )

        except Exception as e:
            print(f"⚠️ Failed to initialize neural model: {e}")
            self.model = None

    def predict(
        self, image_path: str, domain: str = "non_face_photo"
    ) -> GeneralDetectionResult:
        """
        Predict AI generation probability for an image.

        Args:
            image_path: Path to image file
            domain: Domain classification result

        Returns:
            GeneralDetectionResult with probability and confidence
        """
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)
        except Exception as e:
            return self._uncertain_result(domain, f"Failed to load image: {e}")

        # Combine multiple signals
        signals = {}

        # Signal 1: Frequency analysis
        freq_score = self._analyze_frequency(img_array)
        signals["frequency"] = freq_score

        # Signal 2: Noise pattern analysis
        noise_score = self._analyze_noise_patterns(img_array)
        signals["noise"] = noise_score

        # Signal 3: Texture consistency
        texture_score = self._analyze_texture_consistency(img_array)
        signals["texture"] = texture_score

        # Signal 4: Neural features (if available)
        if self.model is not None and self.model_loaded:
            neural_score = self._get_neural_prediction(image)
            signals["neural"] = neural_score

        # Combine signals with domain-specific weighting
        probability, confidence = self._combine_signals(signals, domain)

        # Calculate confidence interval
        interval_width = 0.15 * (1 - confidence)  # Wider when less confident
        lower = max(0.0, probability - interval_width)
        upper = min(1.0, probability + interval_width)

        # Determine if uncertain
        is_uncertain = self._should_be_uncertain(probability, confidence, domain)

        # Get interpretation
        likelihood = self._get_likelihood_label(probability, is_uncertain)
        confidence_band = self._get_confidence_band(probability, confidence)

        # Domain-specific limitations
        limitations = self._get_domain_limitations(domain)

        return GeneralDetectionResult(
            synthetic_probability=probability,
            confidence_interval={"lower": round(lower, 4), "upper": round(upper, 4)},
            synthetic_likelihood=likelihood,
            confidence_band=confidence_band,
            domain_used=domain,
            detector_version=self.VERSION,
            limitations=limitations,
            is_uncertain=is_uncertain,
        )

    def _analyze_frequency(self, img_array: np.ndarray) -> float:
        """
        Analyze frequency spectrum for AI artifacts.

        AI-generated images often have:
        - Unnatural frequency falloff
        - Missing high-frequency details
        - Regular patterns in DCT domain
        """
        try:
            gray = (
                np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            )

            # Compute 2D FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            # Analyze high-frequency energy ratio
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2

            # High-frequency region (outer 30%)
            mask_high = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
            mask_high = r > min(h, w) * 0.35

            high_freq_energy = np.sum(magnitude[mask_high])
            total_energy = np.sum(magnitude)

            ratio = high_freq_energy / total_energy if total_energy > 0 else 0

            # Low high-frequency ratio suggests AI generation
            # Real photos typically have more high-frequency detail
            ai_score = 1.0 - min(1.0, ratio * 3)

            return ai_score

        except Exception:
            return 0.5

    def _analyze_noise_patterns(self, img_array: np.ndarray) -> float:
        """
        Analyze noise patterns for AI signatures.

        AI images often have:
        - Unnaturally smooth noise
        - Consistent noise across regions
        - Missing sensor noise patterns
        """
        try:
            gray = (
                np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            )

            # Extract noise by high-pass filter
            if OPENCV_AVAILABLE:
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                noise = gray.astype(float) - blur.astype(float)
            else:
                # Simple difference filter
                noise = gray[1:, :] - gray[:-1, :]
                noise = noise[:, 1:] - noise[:, :-1]

            # Analyze noise statistics
            noise_std = np.std(noise)
            noise_mean = np.abs(np.mean(noise))

            # AI images often have lower noise variance
            # Real camera photos have sensor noise
            if noise_std < 3:
                return 0.7  # Suspiciously low noise
            elif noise_std > 15:
                return 0.3  # Normal camera noise
            else:
                return 0.5

        except Exception:
            return 0.5

    def _analyze_texture_consistency(self, img_array: np.ndarray) -> float:
        """
        Analyze texture consistency across the image.

        AI images often have:
        - Inconsistent texture detail
        - Repetitive patterns
        - Unnatural smoothing transitions
        """
        try:
            gray = (
                np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            )

            # Divide image into patches and analyze variance consistency
            patch_size = 32
            h, w = gray.shape

            variances = []
            for i in range(0, h - patch_size, patch_size):
                for j in range(0, w - patch_size, patch_size):
                    patch = gray[i : i + patch_size, j : j + patch_size]
                    variances.append(np.var(patch))

            if len(variances) < 4:
                return 0.5

            # Unnaturally consistent variance across patches suggests AI
            variance_of_variances = np.std(variances) / (np.mean(variances) + 1e-6)

            if variance_of_variances < 0.3:
                return 0.65  # Too consistent
            elif variance_of_variances > 1.5:
                return 0.35  # Natural variation
            else:
                return 0.5

        except Exception:
            return 0.5

    def _get_neural_prediction(self, image: Image.Image) -> float:
        """Get prediction from neural network if available."""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                ai_prob = probs[0, 1].item()

            return ai_prob
        except Exception:
            return 0.5

    def _combine_signals(
        self, signals: Dict[str, float], domain: str
    ) -> Tuple[float, float]:
        """
        Combine detection signals with domain-specific weighting.

        Returns:
            Tuple of (probability, confidence)
        """
        weights = {
            "non_face_photo": {
                "frequency": 0.3,
                "noise": 0.25,
                "texture": 0.2,
                "neural": 0.25,
            },
            "art_or_illustration": {
                "frequency": 0.2,
                "noise": 0.15,
                "texture": 0.25,
                "neural": 0.4,
            },
            "synthetic_graphics": {
                "frequency": 0.15,
                "noise": 0.1,
                "texture": 0.15,
                "neural": 0.6,
            },
        }

        domain_weights = weights.get(domain, weights["non_face_photo"])

        # Calculate weighted probability
        total_weight = 0
        weighted_sum = 0

        for signal_name, score in signals.items():
            weight = domain_weights.get(signal_name, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        probability = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Calculate confidence based on signal agreement
        if len(signals) > 1:
            scores = list(signals.values())
            score_std = np.std(scores)
            confidence = max(0.3, 1.0 - score_std * 2)
        else:
            confidence = 0.5

        return probability, confidence

    def _should_be_uncertain(
        self, probability: float, confidence: float, domain: str
    ) -> bool:
        """Determine if result should be UNCERTAIN."""
        # In uncertainty zone
        if self.UNCERTAIN_LOWER <= probability <= self.UNCERTAIN_UPPER:
            return True

        # Low confidence
        if confidence < 0.5:
            return True

        # Art domain is inherently uncertain
        if domain == "art_or_illustration" and confidence < 0.7:
            return True

        return False

    def _get_likelihood_label(self, prob: float, is_uncertain: bool) -> str:
        """Map probability to language label."""
        if is_uncertain:
            return "INCONCLUSIVE - insufficient confidence for determination"

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

    def _get_confidence_band(self, prob: float, confidence: float) -> str:
        """Map probability and confidence to confidence band."""
        if self.UNCERTAIN_LOWER <= prob <= self.UNCERTAIN_UPPER:
            return "LOW - inconclusive zone"
        elif confidence < 0.5:
            return "LOW - insufficient signal agreement"
        elif confidence < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_domain_limitations(self, domain: str) -> str:
        """Get domain-specific limitation text."""
        limitations = {
            "non_face_photo": (
                "General photo detection accuracy varies by scene complexity and AI generator. "
                "Heavily filtered or edited images may produce unreliable results."
            ),
            "art_or_illustration": (
                "Art/illustration detection has inherent limitations. "
                "AI-generated and human-created art share many characteristics. "
                "Results should be interpreted with significant caution."
            ),
            "synthetic_graphics": (
                "UI/graphics detection is less meaningful - these are often synthetic by design. "
                "A high AI probability may not indicate problematic AI generation."
            ),
        }
        return limitations.get(domain, "Unknown domain - results unreliable.")

    def _uncertain_result(self, domain: str, reason: str) -> GeneralDetectionResult:
        """Return an UNCERTAIN result."""
        return GeneralDetectionResult(
            synthetic_probability=0.5,
            confidence_interval={"lower": 0.0, "upper": 1.0},
            synthetic_likelihood=f"INCONCLUSIVE - {reason}",
            confidence_band="LOW",
            domain_used=domain,
            detector_version=self.VERSION,
            limitations=reason,
            is_uncertain=True,
        )


# Convenience function
def detect_general_image(
    image_path: str, domain: str = "non_face_photo"
) -> GeneralDetectionResult:
    """
    Detect AI generation in a general (non-face) image.

    Args:
        image_path: Path to image file
        domain: Domain classification result

    Returns:
        GeneralDetectionResult
    """
    detector = GeneralImageDetector()
    return detector.predict(image_path, domain)


__all__ = [
    "GeneralImageDetector",
    "GeneralDetectionResult",
    "detect_general_image",
]
