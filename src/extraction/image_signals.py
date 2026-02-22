"""
DeepFake Detection System - Image Signal Extraction
Layer 2: Signal Extraction (Forensics)

This module implements forensic signal extractors for images:
- Frequency domain analysis (FFT, DCT)
- Texture entropy analysis
- Edge gradient analysis
- PRNU noise pattern extraction (stub)
- Diffusion residue detection
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ImageSignals:
    """Container for extracted image forensic signals."""

    # Frequency domain
    fft_magnitude_mean: float
    fft_magnitude_std: float
    fft_high_freq_ratio: float  # Ratio of high-frequency energy
    dct_coefficient_stats: Dict[str, float]

    # Texture
    entropy_mean: float
    entropy_std: float
    entropy_patches: np.ndarray  # Per-patch entropy map

    # Edge
    edge_gradient_mean: float
    edge_gradient_std: float
    edge_smoothness_score: float  # Higher = more synthetic-like smoothing

    # PRNU (placeholder)
    prnu_correlation: Optional[float]  # Requires camera fingerprint database

    # Diffusion artifacts
    diffusion_residue_score: float  # Checkerboard/grid patterns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fft_magnitude_mean": float(self.fft_magnitude_mean),
            "fft_magnitude_std": float(self.fft_magnitude_std),
            "fft_high_freq_ratio": float(self.fft_high_freq_ratio),
            "dct_coefficient_stats": self.dct_coefficient_stats,
            "entropy_mean": float(self.entropy_mean),
            "entropy_std": float(self.entropy_std),
            "edge_gradient_mean": float(self.edge_gradient_mean),
            "edge_gradient_std": float(self.edge_gradient_std),
            "edge_smoothness_score": float(self.edge_smoothness_score),
            "prnu_correlation": self.prnu_correlation,
            "diffusion_residue_score": float(self.diffusion_residue_score),
        }


class ImageSignalExtractor:
    """
    Extracts forensic signals from images.

    These signals are sub-perceptual and designed to detect
    artifacts left by generative models.
    """

    def __init__(self):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required: pip install opencv-python")

    def extract_all(self, image_path: str) -> ImageSignals:
        """
        Extract all forensic signals from an image.

        Args:
            image_path: Path to the image file

        Returns:
            ImageSignals dataclass with all extracted features
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract all signals
        fft_features = self._extract_fft_features(gray)
        dct_features = self._extract_dct_features(gray)
        entropy_features = self._extract_entropy_features(gray)
        edge_features = self._extract_edge_features(gray)
        diffusion_score = self._detect_diffusion_artifacts(gray)

        return ImageSignals(
            fft_magnitude_mean=fft_features["magnitude_mean"],
            fft_magnitude_std=fft_features["magnitude_std"],
            fft_high_freq_ratio=fft_features["high_freq_ratio"],
            dct_coefficient_stats=dct_features,
            entropy_mean=entropy_features["mean"],
            entropy_std=entropy_features["std"],
            entropy_patches=entropy_features["patches"],
            edge_gradient_mean=edge_features["gradient_mean"],
            edge_gradient_std=edge_features["gradient_std"],
            edge_smoothness_score=edge_features["smoothness"],
            prnu_correlation=None,  # Requires camera database
            diffusion_residue_score=diffusion_score,
        )

    def _extract_fft_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract FFT-based frequency domain features.

        Generative models often leave subtle periodic artifacts
        in the frequency domain.
        """
        # Compute 2D FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        # Overall statistics
        magnitude_mean = np.mean(magnitude)
        magnitude_std = np.std(magnitude)

        # High-frequency ratio
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4

        # Create mask for low-frequency center
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius**2

        low_freq_energy = np.sum(magnitude[mask])
        high_freq_energy = np.sum(magnitude[~mask])
        total_energy = low_freq_energy + high_freq_energy

        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

        return {
            "magnitude_mean": magnitude_mean,
            "magnitude_std": magnitude_std,
            "high_freq_ratio": high_freq_ratio,
        }

    def _extract_dct_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract DCT (Discrete Cosine Transform) features.

        DCT is used in JPEG compression and can reveal
        compression-related artifacts.
        """
        # Resize to multiple of 8 for block DCT
        h, w = gray.shape
        h_new = (h // 8) * 8
        w_new = (w // 8) * 8
        gray_resized = cv2.resize(gray, (w_new, h_new))

        # Compute DCT on 8x8 blocks
        dct_blocks = []
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                block = gray_resized[i : i + 8, j : j + 8].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_blocks.append(dct_block)

        dct_all = np.array(dct_blocks)

        # Extract statistics
        dc_components = dct_all[:, 0, 0]
        ac_components = dct_all[:, 1:, 1:].flatten()

        return {
            "dc_mean": float(np.mean(dc_components)),
            "dc_std": float(np.std(dc_components)),
            "ac_mean": float(np.mean(np.abs(ac_components))),
            "ac_std": float(np.std(ac_components)),
            "ac_energy": float(np.sum(ac_components**2)),
        }

    def _extract_entropy_features(
        self, gray: np.ndarray, patch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Extract patch-level texture entropy.

        Synthetic images often have lower entropy in texture regions
        due to over-smooth generation.
        """
        h, w = gray.shape
        entropies = []

        for i in range(0, h - patch_size, patch_size):
            row_entropies = []
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i : i + patch_size, j : j + patch_size]

                # Compute histogram entropy
                hist, _ = np.histogram(patch, bins=256, range=(0, 256))
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # Remove zeros for log
                entropy = -np.sum(hist * np.log2(hist))

                row_entropies.append(entropy)
            entropies.append(row_entropies)

        entropy_map = np.array(entropies)

        return {
            "mean": np.mean(entropy_map),
            "std": np.std(entropy_map),
            "patches": entropy_map,
        }

    def _extract_edge_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract edge gradient features.

        Generative models often produce over-smooth edges
        compared to real camera images.
        """
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Canny edges for comparison
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = gradient_magnitude[edges > 0]

        # Smoothness score: lower gradient variance at edges = smoother
        if len(edge_pixels) > 0:
            smoothness = 1.0 / (np.std(edge_pixels) + 1e-6)
        else:
            smoothness = 0.0

        return {
            "gradient_mean": np.mean(gradient_magnitude),
            "gradient_std": np.std(gradient_magnitude),
            "smoothness": min(smoothness, 100.0),  # Cap for normalization
        }

    def _detect_diffusion_artifacts(self, gray: np.ndarray) -> float:
        """
        Detect diffusion model artifacts (checkerboard patterns).

        Diffusion upsampling can leave periodic grid patterns
        in the high-frequency components.
        """
        # High-pass filter to isolate high-frequency noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        high_pass = cv2.filter2D(gray.astype(np.float32), -1, kernel)

        # Compute FFT of high-pass result
        f = np.fft.fft2(high_pass)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Look for peaks at specific grid frequencies
        # Checkerboard pattern creates peaks at (h/2, 0), (0, w/2), etc.
        grid_positions = [
            (center_h, 0),
            (center_h, w - 1),
            (0, center_w),
            (h - 1, center_w),
            (center_h + h // 4, center_w + w // 4),
            (center_h - h // 4, center_w - w // 4),
        ]

        grid_energy = sum(
            magnitude[max(0, min(p[0], h - 1)), max(0, min(p[1], w - 1))]
            for p in grid_positions
        )

        total_energy = np.sum(magnitude)

        # Normalize score
        score = grid_energy / (total_energy + 1e-6) * 1000

        return min(score, 100.0)


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Extract forensic signals from an image"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")

    args = parser.parse_args()

    extractor = ImageSignalExtractor()
    signals = extractor.extract_all(args.image)

    result = signals.to_dict()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Signals saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))
