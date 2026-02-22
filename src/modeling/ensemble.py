"""
DeepFake Detection System - Ensemble Aggregator
Layer 3: Modeling (Ensemble Architecture)

Combines predictions from multiple models into a single
calibrated output with confidence intervals.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from dataclasses import dataclass

from .base import BaseDetector, ModelPrediction, EnsemblePrediction, FeatureNormalizer
from .ood_detector import OODDetector, GeneratorClassifier


class EnsembleAggregator:
    """
    Aggregates predictions from multiple detectors.

    Features:
    - Weighted voting based on model reliability
    - Confidence calibration
    - OOD detection integration
    - Source identification
    - Disagreement detection
    """

    def __init__(
        self, detectors: List[BaseDetector], weights: Optional[List[float]] = None
    ):
        """
        Args:
            detectors: List of detector instances
            weights: Optional weights for each detector (default: equal)
        """
        self.detectors = detectors

        if weights is None:
            self.weights = [1.0 / len(detectors)] * len(detectors)
        else:
            assert len(weights) == len(detectors)
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # OOD detector
        self.ood_detector = OODDetector()

        # Generator classifier
        self.generator_classifier = GeneratorClassifier()

        # Feature normalizer
        self.normalizer = FeatureNormalizer()

        # Calibration parameters
        self.temperature = 1.0  # For temperature scaling

        # Modification detection thresholds
        self.modification_indicators = {
            "jpeg_artifacts": 0.7,
            "resize_artifacts": 0.6,
            "noise_patterns": 0.5,
        }

    def predict(self, input_path: str) -> EnsemblePrediction:
        """
        Generate ensemble prediction.

        Args:
            input_path: Path to media file

        Returns:
            EnsemblePrediction with aggregated scores
        """
        # Collect predictions from all models
        predictions: List[ModelPrediction] = []
        features: List[np.ndarray] = []

        for detector in self.detectors:
            try:
                pred = detector.predict(input_path)
                predictions.append(pred)

                feat = detector.get_feature_vector(input_path)
                features.append(feat)
            except Exception as e:
                # Log error but continue with other models
                print(f"Warning: {detector.name} failed: {e}")
                continue

        if not predictions:
            raise ValueError("All detectors failed")

        # Aggregate features for OOD detection
        all_features = np.concatenate(features)

        # OOD check - INVARIANT 3: null when not evaluated
        if self.ood_detector.fitted:
            ood_score, is_ood = self.ood_detector.compute_ood_score(all_features)
        else:
            ood_score, is_ood = None, None

        # Weighted aggregation of synthetic probabilities
        synthetic_probs = [p.synthetic_probability for p in predictions]
        weights_used = self.weights[: len(predictions)]
        weights_used = [w / sum(weights_used) for w in weights_used]

        weighted_prob = sum(p * w for p, w in zip(synthetic_probs, weights_used))

        # Apply temperature scaling for calibration
        calibrated_prob = self._calibrate(weighted_prob)

        # Compute confidence (agreement between models)
        prob_std = np.std(synthetic_probs)
        ensemble_confidence = max(0, 100 - prob_std * 200)

        # Confidence interval from bootstrap
        lower, upper = self._bootstrap_confidence_interval(
            synthetic_probs, weights_used
        )

        # Source identification
        # Use attributions from first detector with features
        if predictions[0].feature_attributions:
            likely_source, source_conf = self.generator_classifier.classify(
                predictions[0].feature_attributions
            )
        else:
            likely_source, source_conf = None, None  # FIX 1: null, not 0.0

        # Modification detection
        mod_likelihood, mods = self._detect_modifications(predictions)

        return EnsemblePrediction(
            final_authenticity_score=(1.0 - calibrated_prob) * 100,
            final_synthetic_probability=calibrated_prob,
            ensemble_confidence=ensemble_confidence,
            confidence_lower=lower,
            confidence_upper=upper,
            model_predictions=predictions,
            is_out_of_distribution=is_ood,
            ood_score=ood_score,
            likely_source=likely_source,
            source_confidence=source_conf,
            modification_likelihood=mod_likelihood,
            detected_modifications=mods,
        )

    def _calibrate(self, probability: float) -> float:
        """
        Apply temperature scaling for calibration.

        Calibration ensures that when the model says 90% confident,
        it's actually correct 90% of the time.
        """
        # Convert to logit, scale, convert back
        epsilon = 1e-8
        prob_clipped = np.clip(probability, epsilon, 1 - epsilon)
        logit = np.log(prob_clipped / (1 - prob_clipped))
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(calibrated)

    def _bootstrap_confidence_interval(
        self,
        probs: List[float],
        weights: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval via bootstrap.
        """
        probs = np.array(probs)
        weights = np.array(weights)

        # Bootstrap samples
        bootstrap_means = []
        n = len(probs)

        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            sampled_probs = probs[indices]
            sampled_weights = weights[indices]
            sampled_weights = sampled_weights / sampled_weights.sum()
            bootstrap_means.append(np.sum(sampled_probs * sampled_weights))

        # Percentile interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return float(lower), float(upper)

    def _detect_modifications(
        self, predictions: List[ModelPrediction]
    ) -> Tuple[Optional[float], List[str]]:
        """
        Detect likely post-processing modifications.

        Returns:
            (likelihood, detected_mods) - likelihood is None if no deterministic check succeeded
        """
        detected = []
        total_score = 0.0
        checks_performed = 0

        # Analyze feature attributions for modification signs
        for pred in predictions:
            attrs = pred.feature_attributions

            # Check for compression artifacts
            if "dct_ac_energy" in attrs:
                checks_performed += 1
                if attrs["dct_ac_energy"] < 1000:  # Low AC energy = heavy compression
                    detected.append("jpeg_compression")
                    total_score += 0.3

            # Check for resize artifacts
            if "fft_high_freq_ratio" in attrs:
                checks_performed += 1
                ratio = attrs["fft_high_freq_ratio"]
                if ratio < 0.2:  # Low high-freq = possible downscale
                    detected.append("possible_resize")
                    total_score += 0.2

            # Check for noise injection
            if "entropy_std" in attrs:
                checks_performed += 1
                if attrs["entropy_std"] > 3.0:  # High entropy variance = noise
                    detected.append("noise_injection")
                    total_score += 0.2

        # FIX 1: Only return score if we actually performed checks and found something
        # Absence of evidence is NOT certainty of no modification
        if checks_performed == 0:
            return None, []  # No checks performed = unknown

        return min(total_score, 1.0) if detected else None, list(set(detected))

    def fit_ood_detector(self, feature_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Fit OOD detector on known training data.

        Args:
            feature_list: List of feature vectors from known samples

        Returns:
            Fitting statistics
        """
        features = np.vstack(feature_list)
        return self.ood_detector.fit(features)

    def calibrate_temperature(
        self,
        val_probs: List[float],
        val_labels: List[int],
        lr: float = 0.01,
        n_iter: int = 100,
    ) -> float:
        """
        Learn temperature scaling parameter on validation set.

        Args:
            val_probs: Uncalibrated probabilities
            val_labels: True labels (0 = real, 1 = synthetic)

        Returns:
            Optimal temperature
        """
        probs = np.array(val_probs)
        labels = np.array(val_labels)

        # Grid search for temperature
        best_temp = 1.0
        best_loss = float("inf")

        for temp in np.linspace(0.5, 2.0, 50):
            # Apply temperature scaling
            epsilon = 1e-8
            probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
            logits = np.log(probs_clipped / (1 - probs_clipped))
            scaled_logits = logits / temp
            calibrated = 1.0 / (1.0 + np.exp(-scaled_logits))

            # Cross-entropy loss
            loss = -np.mean(
                labels * np.log(calibrated + epsilon)
                + (1 - labels) * np.log(1 - calibrated + epsilon)
            )

            if loss < best_loss:
                best_loss = loss
                best_temp = temp

        self.temperature = best_temp
        return best_temp

    def save(self, path: str) -> None:
        """Save ensemble configuration."""
        data = {
            "weights": self.weights,
            "temperature": self.temperature,
            "detector_names": [d.name for d in self.detectors],
        }
        with open(path, "w") as f:
            json.dump(data, f)

        # Save OOD detector
        ood_path = path.replace(".json", "_ood.json")
        self.ood_detector.save(ood_path)

    def load(self, path: str) -> None:
        """Load ensemble configuration."""
        with open(path, "r") as f:
            data = json.load(f)

        self.weights = data["weights"]
        self.temperature = data["temperature"]

        # Load OOD detector
        ood_path = path.replace(".json", "_ood.json")
        if os.path.exists(ood_path):
            self.ood_detector.load(ood_path)


# ============================================================
# Factory function
# ============================================================


def create_default_ensemble() -> EnsembleAggregator:
    """
    Create ensemble with default detectors.

    Loads trained weights from models/ if available.
    Falls back to statistical-only mode if PyTorch is unavailable.

    Returns:
        Configured EnsembleAggregator
    """
    from .statistical_detector import StatisticalBaselineDetector

    # Get project root (assumes this file is in src/modeling/)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    models_dir = os.path.join(project_root, "models")

    # Create statistical detector (always available)
    stat_detector = StatisticalBaselineDetector()

    # Load trained weights if available
    stat_model_path = os.path.join(models_dir, "statistical_baseline.pkl")
    if os.path.exists(stat_model_path):
        try:
            stat_detector.load(stat_model_path)
        except Exception:
            pass  # Use untrained if load fails

    # Try to create neural detector (requires PyTorch)
    detectors = [stat_detector]
    weights = [1.0]  # Start with statistical only

    try:
        from .neural_detector import NeuralNetworkDetector

        neural_detector = NeuralNetworkDetector()

        neural_model_path = os.path.join(models_dir, "neural_detector.pt")
        if os.path.exists(neural_model_path):
            try:
                neural_detector.load(neural_model_path)
            except Exception:
                pass  # Use untrained if load fails

        detectors.append(neural_detector)
        # Rebalance weights: statistical=0.3, neural=0.7
        weights = [0.3, 0.7]
    except ImportError:
        # PyTorch not available - use statistical detector only
        pass

    ensemble = EnsembleAggregator(detectors, weights)

    # FIX 3: OOD detector requires 1k+ diverse real samples
    # Cannot be fitted on procedural data - leave unfitted
    # OOD signaling is disabled until proper fitting

    return ensemble


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble aggregator")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")

    args = parser.parse_args()

    try:
        ensemble = create_default_ensemble()
        prediction = ensemble.predict(args.image)

        result = prediction.to_dict()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Prediction saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2))
    except ImportError as e:
        print(f"Error: {e}")
        print("Install dependencies: pip install torch torchvision opencv-python")
