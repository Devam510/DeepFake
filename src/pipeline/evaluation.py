"""
DeepFake Detection System - Evaluation Metrics
Layer 8: Evaluation Metrics (Non-Negotiable)

This module implements comprehensive evaluation beyond simple accuracy:
- False negative rate under attack
- Performance after recompression
- Generalization to unseen models
- Calibration error
- Temporal decay tracking
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""

    run_id: str
    timestamp: datetime
    dataset_name: str
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "metrics": self.metrics,
        }


class CoreMetrics:
    """
    Core detection metrics.

    Per Layer 8: Accuracy alone is insufficient.
    """

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Basic accuracy (for reference only)."""
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        False Negative Rate: Critical for security.

        FNR = FN / (FN + TP)
        High FNR means synthetic media is being missed.
        """
        positives = y_true == 1  # Synthetic samples
        if not positives.any():
            return 0.0

        false_negatives = np.sum((y_pred == 0) & positives)
        true_positives = np.sum((y_pred == 1) & positives)

        denominator = false_negatives + true_positives
        if denominator == 0:
            return 0.0

        return float(false_negatives / denominator)

    @staticmethod
    def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        False Positive Rate: Important for usability.

        FPR = FP / (FP + TN)
        High FPR means real media is being flagged.
        """
        negatives = y_true == 0  # Real samples
        if not negatives.any():
            return 0.0

        false_positives = np.sum((y_pred == 1) & negatives)
        true_negatives = np.sum((y_pred == 0) & negatives)

        denominator = false_positives + true_negatives
        if denominator == 0:
            return 0.0

        return float(false_positives / denominator)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Precision = TP / (TP + FP)"""
        predicted_positive = y_pred == 1
        if not predicted_positive.any():
            return 0.0

        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))

        denominator = true_positives + false_positives
        if denominator == 0:
            return 0.0

        return float(true_positives / denominator)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Recall = TP / (TP + FN) = 1 - FNR"""
        return 1.0 - CoreMetrics.false_negative_rate(y_true, y_pred)

    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        prec = CoreMetrics.precision(y_true, y_pred)
        rec = CoreMetrics.recall(y_true, y_pred)

        if prec + rec == 0:
            return 0.0

        return float(2 * prec * rec / (prec + rec))

    @staticmethod
    def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Area Under ROC Curve.

        Measures discrimination ability across all thresholds.
        """
        # Sort by predicted scores
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        tps = np.concatenate([[0], tps])
        fps = np.concatenate([[0], fps])

        if tps[-1] == 0 or fps[-1] == 0:
            return 0.5

        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # Trapezoidal integration
        auc = np.trapz(tpr, fpr)

        return float(auc)


class CalibrationMetrics:
    """
    Metrics for probability calibration.

    When the model says 90% confident, is it actually
    correct 90% of the time?
    """

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE).

        Lower is better. ECE < 0.05 is well-calibrated.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Samples in this bin
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                avg_confidence = np.mean(y_probs[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                ece += (n_in_bin / len(y_true)) * abs(avg_accuracy - avg_confidence)

        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Maximum Calibration Error (MCE).

        Worst-case calibration across all bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
            n_in_bin = np.sum(in_bin)

            if n_in_bin > 0:
                avg_confidence = np.mean(y_probs[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                mce = max(mce, abs(avg_accuracy - avg_confidence))

        return float(mce)

    @staticmethod
    def brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Brier Score: MSE between predictions and actual outcomes.

        Lower is better. Combines calibration and discrimination.
        """
        return float(np.mean((y_probs - y_true) ** 2))


class RobustnessMetrics:
    """
    Metrics for robustness to attacks and transformations.
    """

    @staticmethod
    def performance_under_attack(
        clean_scores: np.ndarray, attacked_scores: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Measure performance degradation under adversarial attack.

        Returns:
            dict with attack success rate and score shift
        """
        # How much did scores shift toward "real"?
        score_shift = np.mean(attacked_scores - clean_scores)

        # How many became misclassified?
        clean_pred = (clean_scores > threshold).astype(int)
        attack_pred = (attacked_scores > threshold).astype(int)
        flip_rate = np.mean(clean_pred != attack_pred)

        return {
            "mean_score_shift": float(score_shift),
            "classification_flip_rate": float(flip_rate),
            "robust_accuracy": float(1.0 - flip_rate),
        }

    @staticmethod
    def compression_robustness(
        original_scores: np.ndarray,
        compressed_scores: np.ndarray,
        quality_levels: List[int],
    ) -> Dict[str, float]:
        """
        Measure score stability across compression levels.
        """
        score_variance = np.var(compressed_scores - original_scores)
        max_drift = np.max(np.abs(compressed_scores - original_scores))

        return {
            "score_variance": float(score_variance),
            "max_score_drift": float(max_drift),
            "stable": float(max_drift < 0.1),
        }


class GeneralizationMetrics:
    """
    Metrics for generalization to unseen generators.
    """

    @staticmethod
    def cross_generator_performance(
        results_by_generator: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance on each generator separately.

        Args:
            results_by_generator: {generator_name: (y_true, y_pred)}

        Returns:
            Per-generator metrics
        """
        metrics = {}

        for gen_name, (y_true, y_pred) in results_by_generator.items():
            metrics[gen_name] = {
                "accuracy": CoreMetrics.accuracy(y_true, y_pred),
                "fnr": CoreMetrics.false_negative_rate(y_true, y_pred),
                "fpr": CoreMetrics.false_positive_rate(y_true, y_pred),
                "f1": CoreMetrics.f1_score(y_true, y_pred),
            }

        # Overall variance across generators
        accuracies = [m["accuracy"] for m in metrics.values()]
        metrics["_summary"] = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "generalization_gap": float(np.max(accuracies) - np.min(accuracies)),
        }

        return metrics

    @staticmethod
    def unseen_generator_score(
        seen_performance: float, unseen_performance: float
    ) -> float:
        """
        Compute generalization score to unseen generators.

        1.0 = perfect generalization
        0.0 = complete failure on unseen
        """
        if seen_performance == 0:
            return 0.0

        ratio = unseen_performance / seen_performance
        return float(min(ratio, 1.0))


class TemporalDecayTracker:
    """
    Track performance decay over time.

    Critical for detecting when retraining is needed.
    """

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load history from storage."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.history = json.load(f)

    def _save(self) -> None:
        """Save history to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def record_evaluation(self, metrics: Dict[str, float], dataset_name: str) -> None:
        """Record evaluation results."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset_name,
            "metrics": metrics,
        }
        self.history.append(entry)
        self._save()

    def compute_decay(
        self, metric_name: str = "accuracy", window_days: int = 30
    ) -> Dict[str, float]:
        """
        Compute performance decay over time.

        Returns:
            decay statistics
        """
        if len(self.history) < 2:
            return {"decay_rate": 0.0, "samples": 0}

        cutoff = datetime.utcnow() - timedelta(days=window_days)

        recent = [
            h for h in self.history if datetime.fromisoformat(h["timestamp"]) >= cutoff
        ]

        if len(recent) < 2:
            return {"decay_rate": 0.0, "samples": len(recent)}

        # Extract metric values
        values = [
            h["metrics"].get(metric_name, 0)
            for h in recent
            if metric_name in h.get("metrics", {})
        ]

        if len(values) < 2:
            return {"decay_rate": 0.0, "samples": len(values)}

        # Compute trend (negative = decay)
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        return {
            "decay_rate": float(slope),
            "samples": len(values),
            "current_value": float(values[-1]),
            "initial_value": float(values[0]),
            "total_change": float(values[-1] - values[0]),
        }

    def needs_retraining(
        self, metric_name: str = "accuracy", threshold: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Check if model needs retraining based on decay.

        Returns:
            (needs_retraining, reason)
        """
        decay = self.compute_decay(metric_name)

        if decay["samples"] < 5:
            return False, "Insufficient evaluation history"

        if decay["decay_rate"] < -threshold:
            return True, f"Performance decay of {-decay['decay_rate']:.3f}/eval"

        if decay.get("current_value", 1.0) < 0.7:
            return True, f"Absolute performance below 0.7"

        return False, "Performance stable"


class EvaluationSuite:
    """
    Complete evaluation suite.

    Runs all Layer 8 required metrics.
    """

    def __init__(self, history_path: str = "./eval_history.json"):
        self.decay_tracker = TemporalDecayTracker(history_path)

    def full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        dataset_name: str = "test",
        generator_results: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation.

        Returns all Layer 8 required metrics.
        """
        results = {
            "dataset": dataset_name,
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": len(y_true),
            # Core metrics
            "core": {
                "accuracy": CoreMetrics.accuracy(y_true, y_pred),
                "false_negative_rate": CoreMetrics.false_negative_rate(y_true, y_pred),
                "false_positive_rate": CoreMetrics.false_positive_rate(y_true, y_pred),
                "precision": CoreMetrics.precision(y_true, y_pred),
                "recall": CoreMetrics.recall(y_true, y_pred),
                "f1_score": CoreMetrics.f1_score(y_true, y_pred),
                "auc_roc": CoreMetrics.auc_roc(y_true, y_probs),
            },
            # Calibration
            "calibration": {
                "expected_calibration_error": CalibrationMetrics.expected_calibration_error(
                    y_true, y_probs
                ),
                "maximum_calibration_error": CalibrationMetrics.maximum_calibration_error(
                    y_true, y_probs
                ),
                "brier_score": CalibrationMetrics.brier_score(y_true, y_probs),
            },
        }

        # Generalization (if provided)
        if generator_results:
            results["generalization"] = (
                GeneralizationMetrics.cross_generator_performance(generator_results)
            )

        # Temporal tracking
        core_metrics = {k: v for k, v in results["core"].items()}
        self.decay_tracker.record_evaluation(core_metrics, dataset_name)

        results["temporal"] = {
            "decay": self.decay_tracker.compute_decay(),
            "needs_retraining": self.decay_tracker.needs_retraining(),
        }

        return results


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation metrics")
    parser.add_argument("--demo", action="store_true", help="Run demo evaluation")

    args = parser.parse_args()

    if args.demo:
        print("Running evaluation demo...")

        # Generate synthetic data
        np.random.seed(42)
        n = 1000

        y_true = np.random.binomial(1, 0.5, n)
        y_probs = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.2, n)
        y_probs = np.clip(y_probs, 0, 1)
        y_pred = (y_probs > 0.5).astype(int)

        suite = EvaluationSuite("./demo_eval_history.json")
        results = suite.full_evaluation(y_true, y_pred, y_probs, "demo")

        print(json.dumps(results, indent=2))
    else:
        parser.print_help()
