"""
DeepFake Detection System - Adversarial Robustness Loop
Layer 4: Adversarial Robustness

This module implements the continuous adversarial training loop:
1. Red team attacks against the detector
2. Failure collection and feedback
3. Model rotation scheduler
4. Inference-time defenses
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import hashlib
import random

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class AttackResult:
    """Result of an adversarial attack attempt."""

    attack_name: str
    original_path: str
    adversarial_path: str
    original_score: float  # Detector score on original
    adversarial_score: float  # Detector score on adversarial
    success: bool  # Did the attack reduce detection?
    perturbation_magnitude: float  # L2 norm of perturbation
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_name": self.attack_name,
            "original_path": self.original_path,
            "adversarial_path": self.adversarial_path,
            "original_score": float(self.original_score),
            "adversarial_score": float(self.adversarial_score),
            "success": self.success,
            "perturbation_magnitude": float(self.perturbation_magnitude),
            "timestamp": self.timestamp.isoformat(),
        }


class AdversarialAttacks:
    """
    Collection of adversarial attacks for red-teaming.

    These attacks simulate what real attackers might do
    to evade detection.
    """

    def __init__(self):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required")

    def fgsm_attack(
        self, image: np.ndarray, gradient: np.ndarray, epsilon: float = 0.03
    ) -> np.ndarray:
        """
        Fast Gradient Sign Method attack.

        Adds perturbation in the direction of the gradient
        to maximize the detector's error.
        """
        perturbation = epsilon * np.sign(gradient)
        adversarial = image + perturbation
        return np.clip(adversarial, 0, 255).astype(np.uint8)

    def pgd_attack(
        self,
        image: np.ndarray,
        gradient_fn: Callable,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 10,
    ) -> np.ndarray:
        """
        Projected Gradient Descent attack.

        Iteratively applies FGSM and projects back to epsilon-ball.
        """
        adversarial = image.astype(np.float32).copy()
        original = image.astype(np.float32)

        for _ in range(num_iter):
            gradient = gradient_fn(adversarial)
            adversarial = adversarial + alpha * np.sign(gradient)

            # Project to epsilon-ball
            perturbation = adversarial - original
            perturbation = np.clip(perturbation, -epsilon * 255, epsilon * 255)
            adversarial = original + perturbation
            adversarial = np.clip(adversarial, 0, 255)

        return adversarial.astype(np.uint8)

    def noise_injection(self, image: np.ndarray, std: float = 5.0) -> np.ndarray:
        """
        Add Gaussian noise to simulate camera sensor noise.
        """
        noise = np.random.normal(0, std, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def jpeg_compression(self, image: np.ndarray, quality: int = 30) -> np.ndarray:
        """
        Apply heavy JPEG compression to remove artifacts.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to smooth high-frequency artifacts.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def resize_attack(self, image: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """
        Downscale and upscale to disrupt frequency artifacts.
        """
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
        large = cv2.resize(small, (w, h))
        return large

    def style_transfer_simulation(
        self, image: np.ndarray, texture_weight: float = 0.3
    ) -> np.ndarray:
        """
        Simulate style transfer by blending with texture.

        This makes synthetic images look more "photographic".
        """
        # Add film grain
        grain = np.random.normal(0, 10, image.shape).astype(np.float32)

        # Create slight color shift
        color_shift = np.random.uniform(-5, 5, 3).astype(np.float32)

        result = image.astype(np.float32)
        result += grain * texture_weight
        result += color_shift

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_all_attacks(
        self, image_path: str, output_dir: str
    ) -> List[Tuple[str, str]]:
        """
        Apply all attacks to an image.

        Returns:
            List of (attack_name, output_path) tuples
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load: {image_path}")

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        results = []

        attacks = [
            ("noise_low", lambda img: self.noise_injection(img, std=3)),
            ("noise_high", lambda img: self.noise_injection(img, std=10)),
            ("jpeg_q30", lambda img: self.jpeg_compression(img, quality=30)),
            ("jpeg_q50", lambda img: self.jpeg_compression(img, quality=50)),
            ("blur_k3", lambda img: self.gaussian_blur(img, kernel_size=3)),
            ("blur_k5", lambda img: self.gaussian_blur(img, kernel_size=5)),
            ("resize_50", lambda img: self.resize_attack(img, scale=0.5)),
            ("resize_75", lambda img: self.resize_attack(img, scale=0.75)),
            ("style_sim", lambda img: self.style_transfer_simulation(img)),
        ]

        for attack_name, attack_fn in attacks:
            try:
                adversarial = attack_fn(image)
                output_path = os.path.join(output_dir, f"{base_name}_{attack_name}.png")
                cv2.imwrite(output_path, adversarial)
                results.append((attack_name, output_path))
            except Exception as e:
                print(f"Attack {attack_name} failed: {e}")

        return results


class FailureFeedbackPipeline:
    """
    Collects detection failures and feeds them back for retraining.

    This is the core of the adversarial robustness loop.
    """

    def __init__(self, storage_dir: str):
        """
        Args:
            storage_dir: Directory to store failure cases
        """
        self.storage_dir = storage_dir
        self.failures_dir = os.path.join(storage_dir, "failures")
        self.log_path = os.path.join(storage_dir, "failure_log.json")

        os.makedirs(self.failures_dir, exist_ok=True)

        self.failures: List[Dict[str, Any]] = []
        self._load_log()

    def _load_log(self) -> None:
        """Load existing failure log."""
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                self.failures = json.load(f)

    def _save_log(self) -> None:
        """Save failure log."""
        with open(self.log_path, "w") as f:
            json.dump(self.failures, f, indent=2)

    def record_failure(
        self,
        sample_path: str,
        true_label: str,
        predicted_score: float,
        attack_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Record a detection failure for later retraining.

        Args:
            sample_path: Path to the sample that caused failure
            true_label: Ground truth ("real" or "synthetic")
            predicted_score: Detector's authenticity score
            attack_type: Type of adversarial attack if any
            metadata: Additional metadata

        Returns:
            Failure ID
        """
        failure_id = hashlib.md5(
            f"{sample_path}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]

        # Copy sample to failures directory
        ext = os.path.splitext(sample_path)[1]
        stored_path = os.path.join(self.failures_dir, f"{failure_id}{ext}")

        if os.path.exists(sample_path):
            import shutil

            shutil.copy2(sample_path, stored_path)

        failure_record = {
            "failure_id": failure_id,
            "original_path": sample_path,
            "stored_path": stored_path,
            "true_label": true_label,
            "predicted_score": predicted_score,
            "attack_type": attack_type,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self.failures.append(failure_record)
        self._save_log()

        return failure_id

    def get_retraining_batch(
        self, batch_size: int = 100, min_age_hours: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get a batch of failures for retraining.

        Args:
            batch_size: Maximum samples to return
            min_age_hours: Minimum age of failures

        Returns:
            List of failure records
        """
        eligible = []
        cutoff = datetime.utcnow()

        for failure in self.failures:
            failure_time = datetime.fromisoformat(failure["timestamp"])
            age_hours = (cutoff - failure_time).total_seconds() / 3600

            if age_hours >= min_age_hours:
                if os.path.exists(failure["stored_path"]):
                    eligible.append(failure)

        # Sample up to batch_size
        if len(eligible) > batch_size:
            return random.sample(eligible, batch_size)
        return eligible

    def get_attack_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics on attack types."""
        stats = {}

        for failure in self.failures:
            attack = failure.get("attack_type", "unknown")
            if attack not in stats:
                stats[attack] = {"count": 0, "avg_score": 0, "scores": []}

            stats[attack]["count"] += 1
            stats[attack]["scores"].append(failure["predicted_score"])

        for attack, data in stats.items():
            if data["scores"]:
                data["avg_score"] = np.mean(data["scores"])
            del data["scores"]

        return stats


class ModelRotationScheduler:
    """
    Manages periodic rotation of model architectures.

    This prevents attackers from building up knowledge
    about a single model.
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to rotation configuration
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load rotation configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        else:
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default rotation configuration."""
        return {
            "rotation_interval_days": 30,
            "available_architectures": [
                "dual_branch_v1",
                "dual_branch_v2",
                "efficientnet_b0",
                "efficientnet_b4",
                "vit_small",
                "vit_base",
            ],
            "current_architecture": "dual_branch_v1",
            "last_rotation": datetime.utcnow().isoformat(),
            "rotation_history": [],
        }

    def _save_config(self) -> None:
        """Save rotation configuration."""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def check_rotation_needed(self) -> bool:
        """Check if model rotation is needed."""
        last_rotation = datetime.fromisoformat(self.config["last_rotation"])
        days_since = (datetime.utcnow() - last_rotation).days
        return days_since >= self.config["rotation_interval_days"]

    def rotate_architecture(self) -> str:
        """
        Select and activate a new architecture.

        Returns:
            Name of the new architecture
        """
        current = self.config["current_architecture"]
        available = self.config["available_architectures"]

        # Select different architecture
        candidates = [a for a in available if a != current]
        new_arch = random.choice(candidates)

        # Update config
        self.config["rotation_history"].append(
            {
                "from": current,
                "to": new_arch,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.config["current_architecture"] = new_arch
        self.config["last_rotation"] = datetime.utcnow().isoformat()

        self._save_config()

        return new_arch

    def get_current_architecture(self) -> str:
        """Get the current active architecture."""
        return self.config["current_architecture"]


class InferenceDefenses:
    """
    Inference-time defenses against adversarial attacks.

    These are applied during prediction without retraining.
    """

    def __init__(self, dropout_rate: float = 0.1):
        """
        Args:
            dropout_rate: Probability of dropping each feature
        """
        self.dropout_rate = dropout_rate

    def random_feature_dropout(self, features: np.ndarray) -> np.ndarray:
        """
        Randomly drop features during inference.

        This prevents gradient estimation by attackers.
        """
        mask = np.random.binomial(1, 1 - self.dropout_rate, features.shape)
        return features * mask / (1 - self.dropout_rate)

    def input_preprocessing(
        self, image: np.ndarray, jpeg_quality: int = 95
    ) -> np.ndarray:
        """
        Defensive preprocessing to remove adversarial perturbations.
        """
        # Light JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return decoded

    def ensemble_voting_with_noise(
        self, predictions: List[float], noise_std: float = 0.02
    ) -> float:
        """
        Add noise to ensemble voting to prevent exploitation.
        """
        noisy_preds = [p + np.random.normal(0, noise_std) for p in predictions]
        return float(np.clip(np.mean(noisy_preds), 0, 1))

    def confidence_thresholding(
        self, prediction: float, confidence: float, threshold: float = 0.3
    ) -> Tuple[float, bool]:
        """
        Flag low-confidence predictions for manual review.

        Returns:
            (prediction, requires_review)
        """
        # Predictions near 0.5 with low confidence need review
        uncertainty = abs(prediction - 0.5)
        requires_review = confidence < threshold and uncertainty < 0.2

        return prediction, requires_review


# ============================================================
# Orchestrator
# ============================================================


class AdversarialRobustnessOrchestrator:
    """
    Orchestrates the full adversarial robustness loop.
    """

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Base directory for robustness data
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.attacks = AdversarialAttacks()
        self.feedback = FailureFeedbackPipeline(os.path.join(base_dir, "feedback"))
        self.rotation = ModelRotationScheduler(
            os.path.join(base_dir, "rotation_config.json")
        )
        self.defenses = InferenceDefenses()

    def run_red_team_cycle(
        self, test_samples: List[str], detector, output_dir: str  # BaseDetector
    ) -> Dict[str, Any]:
        """
        Run a full red team attack cycle.

        Args:
            test_samples: List of sample paths to attack
            detector: Detection model to attack
            output_dir: Directory for attack outputs

        Returns:
            Cycle statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "total_attacks": 0,
            "successful_attacks": 0,
            "attack_breakdown": {},
            "failures_recorded": 0,
        }

        for sample_path in test_samples:
            # Get baseline score
            baseline_pred = detector.predict(sample_path)
            baseline_score = baseline_pred.authenticity_score

            # Apply all attacks
            attack_results = self.attacks.apply_all_attacks(sample_path, output_dir)

            for attack_name, adv_path in attack_results:
                results["total_attacks"] += 1

                if attack_name not in results["attack_breakdown"]:
                    results["attack_breakdown"][attack_name] = {
                        "attempts": 0,
                        "successes": 0,
                    }
                results["attack_breakdown"][attack_name]["attempts"] += 1

                # Score adversarial
                adv_pred = detector.predict(adv_path)
                adv_score = adv_pred.authenticity_score

                # Success if score changes by >20 points toward "real"
                if adv_score - baseline_score > 20:
                    results["successful_attacks"] += 1
                    results["attack_breakdown"][attack_name]["successes"] += 1

                    # Record failure for retraining
                    self.feedback.record_failure(
                        sample_path=adv_path,
                        true_label="synthetic",
                        predicted_score=adv_score,
                        attack_type=attack_name,
                    )
                    results["failures_recorded"] += 1

        return results

    def check_and_rotate(self) -> Optional[str]:
        """
        Check if rotation is needed and perform if so.

        Returns:
            New architecture name if rotated, None otherwise
        """
        if self.rotation.check_rotation_needed():
            return self.rotation.rotate_architecture()
        return None


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adversarial robustness tools")
    parser.add_argument("--attack", help="Image to attack")
    parser.add_argument(
        "--output", "-o", default="./adversarial_output", help="Output directory"
    )

    args = parser.parse_args()

    if args.attack:
        attacks = AdversarialAttacks()
        results = attacks.apply_all_attacks(args.attack, args.output)

        print(f"Generated {len(results)} adversarial variants:")
        for name, path in results:
            print(f"  {name}: {path}")
    else:
        parser.print_help()
