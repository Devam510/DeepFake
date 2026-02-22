"""
DeepFake Detection System - Out-of-Distribution (OOD) Detector
Layer 3: Modeling (Ensemble Architecture)

Detects inputs that are outside the training distribution.
Critical for handling unknown/future generative models.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import os

from .base import BaseDetector, ModelPrediction


class OODDetector:
    """
    Out-of-Distribution detector using feature-space analysis.

    Methods:
    1. Mahalanobis distance from training distribution
    2. Energy-based scoring
    3. k-NN density estimation

    This is crucial for Layer 4 (Adversarial Robustness) as it
    identifies samples from unknown generators.
    """

    def __init__(self, n_neighbors: int = 5):
        """
        Args:
            n_neighbors: Number of neighbors for k-NN density
        """
        self.n_neighbors = n_neighbors

        # Training distribution statistics
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.training_features: Optional[np.ndarray] = None

        self.fitted = False

    def fit(self, features) -> Dict[str, float]:
        """
        Fit OOD detector to training distribution.

        Args:
            features: (N, D) array of feature vectors from known data (list or array)

        Returns:
            Fitting statistics
        """
        # Convert to numpy array if needed
        if isinstance(features, list):
            features = np.vstack(features)

        N, D = features.shape

        # Compute mean and covariance
        self.mean = np.mean(features, axis=0)
        cov = np.cov(features.T)

        # Regularize covariance for numerical stability
        cov += np.eye(D) * 1e-6

        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.cov_inv = np.linalg.pinv(cov)

        # Store training features for k-NN
        self.training_features = features

        self.fitted = True

        # Compute baseline statistics
        train_distances = self._mahalanobis_batch(features)

        return {
            "n_samples": N,
            "n_features": D,
            "mean_distance": float(np.mean(train_distances)),
            "std_distance": float(np.std(train_distances)),
            "max_distance": float(np.max(train_distances)),
        }

    def _mahalanobis_batch(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for batch of features."""
        diff = features - self.mean
        left = np.dot(diff, self.cov_inv)
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def mahalanobis_distance(self, feature: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from training distribution.

        Higher distance = more OOD.
        """
        if not self.fitted:
            raise ValueError("OOD detector not fitted")

        diff = feature - self.mean
        distance = np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff))
        return float(distance)

    def knn_density(self, feature: np.ndarray) -> float:
        """
        Compute k-NN density score.

        Lower density = more OOD.
        """
        if not self.fitted or self.training_features is None:
            raise ValueError("OOD detector not fitted")

        # Compute distances to all training samples
        distances = np.linalg.norm(self.training_features - feature, axis=1)

        # Get k nearest distances
        k_distances = np.sort(distances)[: self.n_neighbors]

        # Density is inverse of average distance
        avg_distance = np.mean(k_distances) + 1e-8
        density = 1.0 / avg_distance

        return float(density)

    def energy_score(self, logits: np.ndarray, temperature: float = 1.0) -> float:
        """
        Compute energy-based OOD score.

        Based on: "Energy-based Out-of-distribution Detection" (Liu et al., 2020)

        Higher energy = more OOD.
        """
        # Energy = -T * log(sum(exp(logits/T)))
        scaled_logits = logits / temperature
        energy = -temperature * np.log(np.sum(np.exp(scaled_logits)) + 1e-8)
        return float(-energy)  # Negate so higher = more OOD

    def compute_ood_score(
        self,
        feature: np.ndarray,
        logits: Optional[np.ndarray] = None,
        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> Tuple[float, bool]:
        """
        Compute combined OOD score.

        Args:
            feature: Feature vector from model
            logits: Optional classification logits
            weights: (mahalanobis_weight, knn_weight, energy_weight)

        Returns:
            (ood_score, is_ood) where score is 0-100 and is_ood is boolean
        """
        if not self.fitted:
            return None, None  # INVARIANT 3: null when not computed

        # Mahalanobis distance
        maha_dist = self.mahalanobis_distance(feature)

        # k-NN density
        knn_dens = self.knn_density(feature)

        # Normalize scores
        # Mahalanobis: typical range 0-20, map to 0-100
        maha_score = min(maha_dist * 5, 100)

        # k-NN: higher density = less OOD, invert
        knn_score = max(0, 100 - knn_dens * 10)

        # Energy score if logits provided
        if logits is not None:
            energy = self.energy_score(logits)
            energy_score = min(energy * 10, 100)
        else:
            energy_score = 50.0
            weights = (weights[0] + weights[2] / 2, weights[1] + weights[2] / 2, 0)

        # Combine scores
        w_maha, w_knn, w_energy = weights
        combined_score = (
            w_maha * maha_score + w_knn * knn_score + w_energy * energy_score
        ) / sum(weights)

        # Threshold for OOD detection
        is_ood = combined_score > 60.0

        return combined_score, is_ood

    def save(self, path: str) -> None:
        """Save OOD detector state."""
        data = {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "cov_inv": self.cov_inv.tolist() if self.cov_inv is not None else None,
            "training_features": (
                self.training_features.tolist()
                if self.training_features is not None
                else None
            ),
            "n_neighbors": self.n_neighbors,
            "fitted": self.fitted,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load OOD detector state."""
        with open(path, "r") as f:
            data = json.load(f)

        self.mean = np.array(data["mean"]) if data["mean"] else None
        self.cov_inv = np.array(data["cov_inv"]) if data["cov_inv"] else None
        self.training_features = (
            np.array(data["training_features"]) if data["training_features"] else None
        )
        self.n_neighbors = data["n_neighbors"]
        self.fitted = data["fitted"]


class GeneratorClassifier:
    """
    Classifier for identifying the source generator.

    Maps detected features to known generators:
    - Stable Diffusion variants
    - Midjourney versions
    - DALL-E versions
    - etc.
    """

    def __init__(self):
        # Generator signatures (simplified - would be learned from data)
        self.known_generators = {
            "stable-diffusion-xl": {
                "entropy_range": (5.5, 6.5),
                "edge_smoothness_range": (8, 15),
                "diffusion_residue_range": (3, 10),
            },
            "midjourney-v5": {
                "entropy_range": (6.0, 7.0),
                "edge_smoothness_range": (5, 12),
                "diffusion_residue_range": (2, 8),
            },
            "dall-e-3": {
                "entropy_range": (5.8, 6.8),
                "edge_smoothness_range": (6, 14),
                "diffusion_residue_range": (1, 6),
            },
        }

        # Trained classifier weights (placeholder)
        self.classifier_weights: Optional[np.ndarray] = None
        self.label_map: Dict[int, str] = {}

    def classify(self, features: Dict[str, float]) -> Tuple[Optional[str], float]:
        """
        Identify likely source generator.

        Returns:
            (generator_name, confidence) or (None, 0.0) if unknown
        """
        best_match = None
        best_confidence = 0.0

        for gen_name, signature in self.known_generators.items():
            score = 0.0
            checks = 0

            # Check entropy
            if "entropy_mean" in features:
                ent = features["entropy_mean"]
                ent_range = signature["entropy_range"]
                if ent_range[0] <= ent <= ent_range[1]:
                    score += 1.0
                checks += 1

            # Check edge smoothness
            if "edge_smoothness" in features:
                smooth = features["edge_smoothness"]
                smooth_range = signature["edge_smoothness_range"]
                if smooth_range[0] <= smooth <= smooth_range[1]:
                    score += 1.0
                checks += 1

            # Check diffusion residue
            if "diffusion_residue" in features:
                residue = features["diffusion_residue"]
                res_range = signature["diffusion_residue_range"]
                if res_range[0] <= residue <= res_range[1]:
                    score += 1.0
                checks += 1

            if checks > 0:
                confidence = score / checks
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = gen_name

        # Only return match if confidence is high enough
        if best_confidence >= 0.6:
            return best_match, best_confidence
        else:
            return None, None  # FIX B: null, not 0.0 - absence of evidence


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OOD Detector demo")
    parser.add_argument(
        "--demo", action="store_true", help="Run demo with synthetic data"
    )

    args = parser.parse_args()

    if args.demo:
        print("Running OOD detector demo...")

        # Generate synthetic training data
        np.random.seed(42)
        train_features = np.random.randn(100, 64)  # 100 samples, 64 features

        # Fit detector
        ood = OODDetector(n_neighbors=5)
        stats = ood.fit(train_features)
        print(f"Fitted OOD detector: {stats}")

        # Test in-distribution sample
        in_dist = np.random.randn(64)
        score_in, is_ood_in = ood.compute_ood_score(in_dist)
        print(f"In-distribution: score={score_in:.2f}, is_ood={is_ood_in}")

        # Test out-of-distribution sample
        out_dist = np.random.randn(64) * 5 + 10
        score_out, is_ood_out = ood.compute_ood_score(out_dist)
        print(f"Out-of-distribution: score={score_out:.2f}, is_ood={is_ood_out}")
    else:
        parser.print_help()
