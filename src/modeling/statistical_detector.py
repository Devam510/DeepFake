"""
DeepFake Detection System - Statistical Baseline Detector
Layer 3: Modeling (Ensemble Architecture)

Non-neural-network baseline using classical ML on forensic features.
This serves as a sanity check and provides interpretable outputs.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
import pickle

from .base import BaseDetector, ModelPrediction

# Import signal extractors
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extraction.image_signals import ImageSignalExtractor, ImageSignals


class StatisticalBaselineDetector(BaseDetector):
    """
    Statistical baseline detector using classical ML.

    Uses features from Layer 2 signal extractors with
    a simple classifier (logistic regression or SVM).

    This is intentionally simple to:
    1. Provide interpretable baseline
    2. Catch obvious cases NN might miss
    3. Sanity check NN predictions
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to saved model weights (optional)
        """
        self.image_extractor = ImageSignalExtractor()
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_names: List[str] = []
        self.trained = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def name(self) -> str:
        return "statistical_baseline"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_media_types(self) -> List[str]:
        return ["image"]  # Extend for video/audio as needed

    def extract_features(self, input_path: str) -> Dict[str, float]:
        """
        Extract forensic features from input.

        Returns dictionary of named features for interpretability.
        """
        signals = self.image_extractor.extract_all(input_path)

        features = {
            # Frequency domain
            "fft_magnitude_mean": signals.fft_magnitude_mean,
            "fft_magnitude_std": signals.fft_magnitude_std,
            "fft_high_freq_ratio": signals.fft_high_freq_ratio,
            # DCT
            "dct_dc_mean": signals.dct_coefficient_stats.get("dc_mean", 0),
            "dct_ac_mean": signals.dct_coefficient_stats.get("ac_mean", 0),
            "dct_ac_energy": signals.dct_coefficient_stats.get("ac_energy", 0),
            # Entropy
            "entropy_mean": signals.entropy_mean,
            "entropy_std": signals.entropy_std,
            # Edge
            "edge_gradient_mean": signals.edge_gradient_mean,
            "edge_gradient_std": signals.edge_gradient_std,
            "edge_smoothness": signals.edge_smoothness_score,
            # Diffusion
            "diffusion_residue": signals.diffusion_residue_score,
        }

        return features

    def get_feature_vector(self, input_path: str) -> np.ndarray:
        """Extract features as numpy array."""
        features = self.extract_features(input_path)

        if not self.feature_names:
            self.feature_names = sorted(features.keys())

        return np.array([features[name] for name in self.feature_names])

    def predict(self, input_path: str) -> ModelPrediction:
        """
        Generate prediction using statistical model.
        """
        features = self.extract_features(input_path)
        feature_vector = self.get_feature_vector(input_path)

        if self.trained and self.weights is not None:
            # Use trained weights
            logit = np.dot(feature_vector, self.weights) + self.bias
            synthetic_prob = 1.0 / (1.0 + np.exp(-logit))
        else:
            # Heuristic scoring when not trained
            synthetic_prob = self._heuristic_score(features)

        # Compute confidence interval
        lower, upper = self.compute_confidence_interval(
            synthetic_prob, n_samples=50, confidence_level=0.95  # Conservative estimate
        )

        # Feature attribution: normalized feature contributions
        if self.trained and self.weights is not None:
            attributions = {
                name: float(feature_vector[i] * self.weights[i])
                for i, name in enumerate(self.feature_names)
            }
        else:
            attributions = features

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

    def _heuristic_score(self, features: Dict[str, float]) -> float:
        """
        Heuristic scoring when model is not trained.

        Based on known characteristics of synthetic media:
        - Lower entropy = more synthetic
        - Lower edge gradient variance = more synthetic
        - Higher diffusion residue = more synthetic
        """
        score = 0.5  # Start neutral

        # Entropy: low entropy suggests synthetic
        if features["entropy_std"] < 1.0:
            score += 0.1
        elif features["entropy_std"] > 2.0:
            score -= 0.1

        # Edge smoothness: high smoothness suggests synthetic
        if features["edge_smoothness"] > 10:
            score += 0.15

        # Diffusion residue: high score suggests synthetic
        if features["diffusion_residue"] > 5:
            score += 0.1

        # Frequency ratio: unusual ratios suggest synthetic
        if (
            features["fft_high_freq_ratio"] < 0.3
            or features["fft_high_freq_ratio"] > 0.8
        ):
            score += 0.1

        return np.clip(score, 0.0, 1.0)

    def train(
        self,
        feature_vectors: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.01,
    ) -> Dict[str, float]:
        """
        Train logistic regression on labeled data.

        Args:
            feature_vectors: (N, D) array of features
            labels: (N,) array of labels (1 = synthetic, 0 = real)
            learning_rate: SGD learning rate
            n_iterations: Training iterations
            regularization: L2 regularization strength

        Returns:
            Training metrics
        """
        N, D = feature_vectors.shape

        # Normalize features
        self.means = np.mean(feature_vectors, axis=0)
        self.stds = np.std(feature_vectors, axis=0) + 1e-8
        X = (feature_vectors - self.means) / self.stds

        # Initialize weights
        self.weights = np.zeros(D)
        self.bias = 0.0

        losses = []

        for i in range(n_iterations):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Loss
            loss = -np.mean(
                labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8)
            )
            loss += regularization * np.sum(self.weights**2)
            losses.append(loss)

            # Gradients
            error = probs - labels
            grad_w = np.dot(X.T, error) / N + 2 * regularization * self.weights
            grad_b = np.mean(error)

            # Update
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b

        self.trained = True

        # Compute final accuracy
        final_probs = 1.0 / (1.0 + np.exp(-(np.dot(X, self.weights) + self.bias)))
        predictions = (final_probs > 0.5).astype(int)
        accuracy = np.mean(predictions == labels)

        return {
            "final_loss": losses[-1],
            "accuracy": accuracy,
            "n_samples": N,
            "n_features": D,
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias,
            "means": self.means.tolist() if hasattr(self, "means") else None,
            "stds": self.stds.tolist() if hasattr(self, "stds") else None,
            "feature_names": self.feature_names,
            "trained": self.trained,
            "version": self.version,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.weights = np.array(data["weights"]) if data["weights"] else None
        self.bias = data["bias"]
        if data.get("means"):
            self.means = np.array(data["means"])
            self.stds = np.array(data["stds"])
        self.feature_names = data["feature_names"]
        self.trained = data["trained"]


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Statistical baseline detector")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", "-m", help="Path to trained model (optional)")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")

    args = parser.parse_args()

    detector = StatisticalBaselineDetector(model_path=args.model)
    prediction = detector.predict(args.image)

    result = prediction.to_dict()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Prediction saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))
