"""
Video Meta-Voter — Trained Model for Optimal Signal Combination
================================================================

Loads the trained GradientBoosting model from Phase C of training.
Used by video_detector.py for final probability prediction.

Usage:
    from video_meta_voter import VideoMetaVoter
    voter = VideoMetaVoter()
    if voter.is_trained():
        prob = voter.predict(features_dict)
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict

MODEL_PATH = Path(__file__).parent / "models" / "trained" / "video_meta_voter.pkl"


class VideoMetaVoter:
    """Video deepfake meta-voter — trained GradientBoosting model."""

    def __init__(self):
        self.pipeline = None
        self.feature_names = []
        self.cv_accuracy = 0.0
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.pipeline = data["pipeline"]
                self.feature_names = data["feature_names"]
                self.cv_accuracy = data.get("cv_accuracy", 0.0)
                print(f"  [VideoMetaVoter] Loaded (CV acc: {self.cv_accuracy:.4f})")
            except Exception as e:
                print(f"  [VideoMetaVoter] Failed to load: {e}")

    def is_trained(self) -> bool:
        return self.pipeline is not None

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict AI probability from all video signal features.

        Args:
            features: dict with signal scores

        Returns:
            float: AI probability 0.0 - 1.0
        """
        if not self.is_trained():
            # Fallback to weighted average
            weights = {
                "temporal_score": 0.25,
                "biological_score": 0.20,
                "audio_score": 0.20,
                "frame_ai_prob": 0.35,
            }
            total = sum(features.get(k, 0.5) * w for k, w in weights.items())
            return min(1.0, max(0.0, total))

        # Build feature vector in correct order
        x = np.array([[features.get(name, 0.0) for name in self.feature_names]])

        try:
            prob = self.pipeline.predict_proba(x)[0][1]  # probability of class 1 (fake)
            return float(min(1.0, max(0.0, prob)))
        except Exception:
            return 0.5
