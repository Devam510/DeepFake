"""
DeepFake Detection System - Video Signal Extraction
Layer 2: Signal Extraction (Forensics)

This module implements forensic signal extractors for videos:
- Temporal coherence analysis
- Optical flow jitter detection
- Frame-to-frame noise consistency
- Mouth-phoneme synchronization (stub - requires audio alignment)
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import os

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class VideoSignals:
    """Container for extracted video forensic signals."""

    # Temporal coherence
    temporal_consistency_score: float  # Lower = more inconsistent across frames
    frame_ssim_mean: float  # Mean SSIM between adjacent frames
    frame_ssim_std: float  # Std of SSIM (high = temporal artifacts)

    # Optical flow
    flow_magnitude_mean: float
    flow_magnitude_std: float
    flow_jitter_score: float  # Unnatural motion patterns

    # Noise consistency
    noise_variance_mean: float
    noise_variance_std: float
    noise_consistency_score: float  # Similar noise across frames = real

    # Frame-level anomalies
    anomaly_frame_indices: List[int]  # Frames with detected anomalies
    anomaly_scores: List[float]

    # Metadata
    frame_count: int
    fps: float
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temporal_consistency_score": float(self.temporal_consistency_score),
            "frame_ssim_mean": float(self.frame_ssim_mean),
            "frame_ssim_std": float(self.frame_ssim_std),
            "flow_magnitude_mean": float(self.flow_magnitude_mean),
            "flow_magnitude_std": float(self.flow_magnitude_std),
            "flow_jitter_score": float(self.flow_jitter_score),
            "noise_variance_mean": float(self.noise_variance_mean),
            "noise_variance_std": float(self.noise_variance_std),
            "noise_consistency_score": float(self.noise_consistency_score),
            "anomaly_frame_indices": self.anomaly_frame_indices,
            "anomaly_scores": [float(s) for s in self.anomaly_scores],
            "frame_count": self.frame_count,
            "fps": float(self.fps),
            "duration_seconds": float(self.duration_seconds),
        }


class VideoSignalExtractor:
    """
    Extracts forensic signals from videos.

    Analyzes temporal consistency, optical flow patterns,
    and frame-to-frame noise characteristics.
    """

    def __init__(self, sample_rate: int = 1):
        """
        Args:
            sample_rate: Process every Nth frame (1 = all frames)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required: pip install opencv-python")
        self.sample_rate = sample_rate

    def extract_all(self, video_path: str, max_frames: int = 300) -> VideoSignals:
        """
        Extract all forensic signals from a video.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze

        Returns:
            VideoSignals dataclass
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Collect frames
        frames = []
        frame_idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            frame_idx += 1

        cap.release()

        if len(frames) < 2:
            raise ValueError("Video too short for analysis")

        # Extract signals
        temporal_features = self._analyze_temporal_coherence(frames)
        flow_features = self._analyze_optical_flow(frames)
        noise_features = self._analyze_noise_consistency(frames)
        anomalies = self._detect_frame_anomalies(frames)

        return VideoSignals(
            temporal_consistency_score=temporal_features["consistency"],
            frame_ssim_mean=temporal_features["ssim_mean"],
            frame_ssim_std=temporal_features["ssim_std"],
            flow_magnitude_mean=flow_features["magnitude_mean"],
            flow_magnitude_std=flow_features["magnitude_std"],
            flow_jitter_score=flow_features["jitter"],
            noise_variance_mean=noise_features["variance_mean"],
            noise_variance_std=noise_features["variance_std"],
            noise_consistency_score=noise_features["consistency"],
            anomaly_frame_indices=anomalies["indices"],
            anomaly_scores=anomalies["scores"],
            frame_count=len(frames),
            fps=fps,
            duration_seconds=duration,
        )

    def _analyze_temporal_coherence(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze temporal coherence using SSIM between adjacent frames.

        Synthetic videos often have inconsistent quality across frames.
        """
        ssim_scores = []

        for i in range(len(frames) - 1):
            ssim = self._compute_ssim(frames[i], frames[i + 1])
            ssim_scores.append(ssim)

        ssim_scores = np.array(ssim_scores)

        # Consistency: lower variance in SSIM = more consistent
        consistency = 1.0 / (np.std(ssim_scores) + 0.01)
        consistency = min(consistency, 100.0)

        return {
            "consistency": consistency,
            "ssim_mean": np.mean(ssim_scores),
            "ssim_std": np.std(ssim_scores),
        }

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute simplified SSIM between two images."""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return float(np.mean(ssim_map))

    def _analyze_optical_flow(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze optical flow patterns.

        Synthetic videos may have unnatural motion patterns
        or jittery optical flow.
        """
        flow_magnitudes = []
        flow_angles = []

        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(magnitude))
            flow_angles.append(np.std(angle))  # Angle variation

        flow_magnitudes = np.array(flow_magnitudes)

        # Jitter: rapid changes in flow magnitude
        flow_diff = np.abs(np.diff(flow_magnitudes))
        jitter = np.mean(flow_diff) / (np.mean(flow_magnitudes) + 0.01)

        return {
            "magnitude_mean": np.mean(flow_magnitudes),
            "magnitude_std": np.std(flow_magnitudes),
            "jitter": min(jitter * 100, 100.0),
        }

    def _analyze_noise_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze frame-to-frame noise consistency.

        Real camera footage has consistent sensor noise patterns.
        Synthetic videos may have inconsistent noise.
        """
        noise_variances = []

        for frame in frames:
            # High-pass filter to extract noise
            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            noise = frame.astype(np.float32) - blur.astype(np.float32)
            noise_variances.append(np.var(noise))

        noise_variances = np.array(noise_variances)

        # Consistency: similar noise variance across frames
        consistency = 1.0 / (
            np.std(noise_variances) / (np.mean(noise_variances) + 0.01) + 0.01
        )
        consistency = min(consistency, 100.0)

        return {
            "variance_mean": np.mean(noise_variances),
            "variance_std": np.std(noise_variances),
            "consistency": consistency,
        }

    def _detect_frame_anomalies(
        self, frames: List[np.ndarray], threshold: float = 2.0
    ) -> Dict[str, List]:
        """
        Detect frames with anomalous characteristics.

        Uses z-score to find outlier frames.
        """
        # Compute per-frame features
        frame_features = []
        for frame in frames:
            entropy = self._compute_entropy(frame)
            edge_density = self._compute_edge_density(frame)
            frame_features.append([entropy, edge_density])

        features = np.array(frame_features)

        # Compute z-scores
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 0.01
        z_scores = np.abs((features - mean) / std)

        # Find anomalies
        anomaly_scores = np.max(z_scores, axis=1)
        anomaly_indices = np.where(anomaly_scores > threshold)[0].tolist()

        return {"indices": anomaly_indices, "scores": anomaly_scores.tolist()}

    def _compute_entropy(self, frame: np.ndarray) -> float:
        """Compute histogram entropy of a frame."""
        hist, _ = np.histogram(frame, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _compute_edge_density(self, frame: np.ndarray) -> float:
        """Compute edge pixel density."""
        edges = cv2.Canny(frame, 50, 150)
        return np.sum(edges > 0) / edges.size


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract forensic signals from a video"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    parser.add_argument(
        "--max-frames", type=int, default=300, help="Max frames to analyze"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=1, help="Sample every Nth frame"
    )

    args = parser.parse_args()

    extractor = VideoSignalExtractor(sample_rate=args.sample_rate)
    signals = extractor.extract_all(args.video, max_frames=args.max_frames)

    result = signals.to_dict()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Signals saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))
