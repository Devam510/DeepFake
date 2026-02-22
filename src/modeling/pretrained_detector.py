"""
Pretrained General Image Detector
==================================

Uses pretrained HuggingFace model for AI-generated image detection.
Model: prithivMLmods/deepfake-detector-model-v1 (SigLIP-based)

This provides better generalization across diverse image types
without requiring custom training data.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from PIL import Image

# Try to import transformers
try:
    from transformers import (
        pipeline,
        AutoImageProcessor,
        AutoModelForImageClassification,
    )
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available. Install with: pip install transformers")


@dataclass
class PretrainedDetectionResult:
    """Result from pretrained detector."""

    synthetic_probability: float
    confidence_interval: Dict[str, float]
    synthetic_likelihood: str
    confidence_band: str
    is_uncertain: bool
    model_name: str
    limitations: str

    def to_dict(self) -> dict:
        return {
            "synthetic_probability": self.synthetic_probability,
            "confidence_interval": self.confidence_interval,
            "synthetic_likelihood": self.synthetic_likelihood,
            "confidence_band": self.confidence_band,
            "is_uncertain": self.is_uncertain,
            "model_name": self.model_name,
            "limitations": self.limitations,
        }


class PretrainedGeneralDetector:
    """
    Pretrained AI image detector using HuggingFace models.

    Uses state-of-the-art pretrained models that have been
    trained on diverse AI-generated images from multiple generators.
    """

    MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"
    VERSION = "PretrainedGeneralDetector_v1.0.0"

    # Uncertainty thresholds
    UNCERTAIN_LOWER = 0.35
    UNCERTAIN_UPPER = 0.65

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize pretrained detector.

        Args:
            model_id: HuggingFace model ID (optional, uses default if not provided)
        """
        self.model_id = model_id or self.MODEL_ID
        self.model = None
        self.processor = None
        self.pipeline = None
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu" if TRANSFORMERS_AVAILABLE else "cpu"
        )

        if TRANSFORMERS_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load the pretrained model from HuggingFace."""
        try:
            print(f"📥 Loading pretrained model: {self.model_id}")
            print(f"   Device: {self.device}")

            # Try using pipeline (simpler)
            self.pipeline = pipeline(
                "image-classification",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1,
            )

            print(f"✅ Pretrained model loaded successfully!")

        except Exception as e:
            print(f"⚠️ Failed to load pretrained model: {e}")
            print("   Falling back to signal-based detection")
            self.pipeline = None

    def predict(
        self, image_path: str, domain: str = "general"
    ) -> PretrainedDetectionResult:
        """
        Predict AI generation probability using pretrained model.

        Args:
            image_path: Path to image file
            domain: Domain classification (for context)

        Returns:
            PretrainedDetectionResult
        """
        # Check if model is available
        if self.pipeline is None:
            return self._fallback_predict(image_path, domain)

        try:
            # Load and predict
            image = Image.open(image_path).convert("RGB")
            results = self.pipeline(image)

            # Parse results
            # Model returns labels like "Fake" and "Real"
            fake_score = 0.5
            for result in results:
                label = result["label"].lower()
                score = result["score"]

                if "fake" in label or "synthetic" in label or "ai" in label:
                    fake_score = score
                    break
                elif "real" in label:
                    fake_score = 1.0 - score
                    break

            # Calculate confidence interval
            confidence = abs(fake_score - 0.5) * 2  # Higher when further from 0.5
            interval_width = 0.1 * (1 - confidence)
            lower = max(0.0, fake_score - interval_width)
            upper = min(1.0, fake_score + interval_width)

            # Determine uncertainty
            is_uncertain = self.UNCERTAIN_LOWER <= fake_score <= self.UNCERTAIN_UPPER

            return PretrainedDetectionResult(
                synthetic_probability=round(fake_score, 4),
                confidence_interval={
                    "lower": round(lower, 4),
                    "upper": round(upper, 4),
                },
                synthetic_likelihood=self._get_likelihood_label(
                    fake_score, is_uncertain
                ),
                confidence_band=self._get_confidence_band(fake_score, confidence),
                is_uncertain=is_uncertain,
                model_name=self.model_id,
                limitations=self._get_limitations(domain),
            )

        except Exception as e:
            return PretrainedDetectionResult(
                synthetic_probability=0.5,
                confidence_interval={"lower": 0.0, "upper": 1.0},
                synthetic_likelihood=f"ERROR: {str(e)}",
                confidence_band="LOW",
                is_uncertain=True,
                model_name=self.model_id,
                limitations="Detection failed due to an error.",
            )

    def _fallback_predict(
        self, image_path: str, domain: str
    ) -> PretrainedDetectionResult:
        """Fallback to signal-based detection when model not available."""
        # Import the signal-based detector
        try:
            from src.modeling.general_image_detector import GeneralImageDetector

            detector = GeneralImageDetector()
            result = detector.predict(image_path, domain)

            return PretrainedDetectionResult(
                synthetic_probability=result.synthetic_probability,
                confidence_interval=result.confidence_interval,
                synthetic_likelihood=result.synthetic_likelihood
                + " (signal-based fallback)",
                confidence_band=result.confidence_band,
                is_uncertain=result.is_uncertain,
                model_name="SignalBasedDetector (fallback)",
                limitations=result.limitations,
            )
        except ImportError:
            return PretrainedDetectionResult(
                synthetic_probability=0.5,
                confidence_interval={"lower": 0.0, "upper": 1.0},
                synthetic_likelihood="INCONCLUSIVE - no detector available",
                confidence_band="LOW",
                is_uncertain=True,
                model_name="None",
                limitations="No detection model available.",
            )

    def _get_likelihood_label(self, prob: float, is_uncertain: bool) -> str:
        """Map probability to language label."""
        if is_uncertain:
            return "INCONCLUSIVE - insufficient confidence"
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
        """Map to confidence band."""
        if self.UNCERTAIN_LOWER <= prob <= self.UNCERTAIN_UPPER:
            return "LOW - inconclusive zone"
        elif confidence < 0.5:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_limitations(self, domain: str) -> str:
        """Get domain-specific limitations."""
        base = (
            "This detector uses a pretrained model (SigLIP-based) that has been "
            "trained on diverse AI-generated images. While it generalizes well, "
            "accuracy may vary for novel generators or heavily edited images."
        )

        if domain == "art_or_illustration":
            return base + " Art/illustration detection has inherent uncertainty."
        elif domain == "synthetic_graphics":
            return base + " UI/graphics are often synthetic by design."
        else:
            return base


def detect_with_pretrained(image_path: str, domain: str = "general") -> dict:
    """
    Convenience function for pretrained detection.

    Args:
        image_path: Path to image
        domain: Image domain

    Returns:
        Detection result dict
    """
    detector = PretrainedGeneralDetector()
    result = detector.predict(image_path, domain)
    return result.to_dict()


# Test if run directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pretrained_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    result = detect_with_pretrained(image_path)

    print("\n" + "=" * 60)
    print("PRETRAINED DETECTOR RESULT")
    print("=" * 60)
    print(f"Model:       {result['model_name']}")
    print(f"Probability: {result['synthetic_probability']:.1%}")
    print(f"Likelihood:  {result['synthetic_likelihood']}")
    print(f"Confidence:  {result['confidence_band']}")
    print(f"Uncertain:   {result['is_uncertain']}")
    print("=" * 60)
