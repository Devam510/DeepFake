"""
DeepFake Detection System - Base Model Interface
Layer 3: Modeling (Ensemble Architecture)

This module defines the base interface for all detection models
and common utilities for model outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class ModelPrediction:
    """
    Standard output format for all detection models.

    Per Layer 6 specification: Never output binary labels.
    All outputs are probabilistic with confidence intervals.
    """

    # Core prediction
    authenticity_score: float  # 0-100 scale (100 = definitely real)
    synthetic_probability: float  # 0.0-1.0 probability of being synthetic

    # Confidence
    confidence_lower: float  # Lower bound of confidence interval
    confidence_upper: float  # Upper bound of confidence interval
    confidence_level: float  # e.g., 0.95 for 95% CI

    # Attribution
    feature_attributions: Dict[str, float]  # Which features drove the decision

    # Model metadata
    model_name: str
    model_version: str

    # Optional
    known_model_match: Optional[str] = None  # e.g., "stable-diffusion-xl"
    known_model_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authenticity_score": float(self.authenticity_score),
            "synthetic_probability": float(self.synthetic_probability),
            "confidence_interval": {
                "lower": float(self.confidence_lower),
                "upper": float(self.confidence_upper),
                "level": float(self.confidence_level),
            },
            "feature_attributions": {
                k: float(v) for k, v in self.feature_attributions.items()
            },
            "model_name": self.model_name,
            "model_version": self.model_version,
            "known_model_match": self.known_model_match,
            "known_model_confidence": float(self.known_model_confidence),
        }


@dataclass
class EnsemblePrediction:
    """
    Aggregated prediction from the full ensemble.
    """

    # Aggregated scores
    final_authenticity_score: float  # Weighted ensemble score
    final_synthetic_probability: float

    # Confidence
    ensemble_confidence: float  # Agreement between models
    confidence_lower: float
    confidence_upper: float

    # Individual model predictions
    model_predictions: List[ModelPrediction]

    # OOD detection - INVARIANT 4: tri-state (True/False/None)
    is_out_of_distribution: Optional[bool]  # None = not evaluated
    ood_score: Optional[float]  # None = not computed, higher = more OOD

    # Source identification
    likely_source: Optional[str]  # e.g., "midjourney-v5"
    source_confidence: float

    # Modification detection
    modification_likelihood: float  # Probability of post-processing
    detected_modifications: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_authenticity_score": float(self.final_authenticity_score),
            "final_synthetic_probability": float(self.final_synthetic_probability),
            "ensemble_confidence": float(self.ensemble_confidence),
            "confidence_interval": {
                "lower": float(self.confidence_lower),
                "upper": float(self.confidence_upper),
            },
            "model_predictions": [p.to_dict() for p in self.model_predictions],
            "out_of_distribution": {
                "is_ood": self.is_out_of_distribution,
                "ood_score": float(self.ood_score),
            },
            "source_identification": {
                "likely_source": self.likely_source,
                "confidence": float(self.source_confidence),
            },
            "modification_detection": {
                "likelihood": float(self.modification_likelihood),
                "detected": self.detected_modifications,
            },
        }


class BaseDetector(ABC):
    """
    Abstract base class for all detection models.

    All detectors must implement:
    - predict(): Return probabilistic prediction
    - get_feature_vector(): Return extracted features
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version string."""
        pass

    @property
    @abstractmethod
    def supported_media_types(self) -> List[str]:
        """List of supported media types: ['image', 'video', 'audio']"""
        pass

    @abstractmethod
    def predict(self, input_path: str) -> ModelPrediction:
        """
        Generate prediction for a single input.

        Args:
            input_path: Path to media file

        Returns:
            ModelPrediction with probabilistic scores
        """
        pass

    @abstractmethod
    def get_feature_vector(self, input_path: str) -> np.ndarray:
        """
        Extract feature vector for ensemble aggregation.

        Args:
            input_path: Path to media file

        Returns:
            1D numpy array of features
        """
        pass

    def compute_confidence_interval(
        self, probability: float, n_samples: int = 100, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval using Wilson score interval.

        Args:
            probability: Point estimate of probability
            n_samples: Effective sample size (for calibration)
            confidence_level: Desired confidence level

        Returns:
            (lower_bound, upper_bound)
        """
        from scipy import stats

        z = stats.norm.ppf((1 + confidence_level) / 2)

        # Wilson score interval
        denominator = 1 + z**2 / n_samples
        center = (probability + z**2 / (2 * n_samples)) / denominator
        margin = (
            z
            * np.sqrt(
                (probability * (1 - probability) + z**2 / (4 * n_samples)) / n_samples
            )
            / denominator
        )

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return lower, upper


class FeatureNormalizer:
    """Normalize features for ensemble consistency."""

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, features: np.ndarray) -> None:
        """Fit normalizer to training features."""
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0) + 1e-8
        self.fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted")
        return (features - self.means) / self.stds

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
