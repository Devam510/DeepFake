"""
Image Processing Level Estimation
=================================

Estimates the intensity of post-processing applied to an image.
Used to adjust confidence and force UNCERTAIN verdict on heavily filtered images.

PRODUCTION-SAFE: Does NOT modify AI probability values.
Only affects interpretation and confidence intervals.

Processing indicators detected:
- JPEG quantization irregularities
- Over-sharpening halos (edge overshoot)
- Frequency spectrum flattening
- Skin texture smoothing loss
- Color LUT banding
- Re-encoding artifacts
"""

import os
from typing import Literal, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Type alias for processing levels
ProcessingLevelType = Literal[
    "minimal_processing", "moderate_processing", "heavy_processing", "unknown"
]


@dataclass
class ProcessingLevelResult:
    """Result of image processing level estimation."""

    level: ProcessingLevelType
    confidence: float  # 0.0 to 1.0
    indicators: List[str]
    scores: dict  # Individual signal scores
    warning: Optional[str]

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "confidence": round(self.confidence, 3),
            "indicators": self.indicators,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "warning": self.warning,
        }


# Thresholds for processing level classification
THRESHOLDS = {
    "minimal": 0.3,
    "moderate": 0.6,
    "heavy": 0.6,  # Above this = heavy
}

# Signal weights for combined score
SIGNAL_WEIGHTS = {
    "jpeg_artifacts": 0.15,
    "sharpening_halos": 0.20,
    "frequency_flattening": 0.15,
    "skin_smoothing": 0.25,
    "color_banding": 0.10,
    "reencoding_artifacts": 0.15,
}


class ImageProcessingAnalyzer:
    """
    Analyzes images for post-processing indicators.

    Uses lightweight, explainable signals - no deep learning required.
    """

    def __init__(self):
        self.cv2_available = CV2_AVAILABLE

    def analyze(self, image_path: str) -> ProcessingLevelResult:
        """
        Estimate processing level of an image.

        Args:
            image_path: Path to image file

        Returns:
            ProcessingLevelResult with level and details
        """
        if not os.path.exists(image_path):
            return ProcessingLevelResult(
                level="unknown",
                confidence=0.0,
                indicators=["File not found"],
                scores={},
                warning="Could not analyze image: file not found",
            )

        try:
            # Load image
            pil_image = Image.open(image_path).convert("RGB")
            img_array = np.array(pil_image)

            # Collect scores from all signals
            scores = {}
            indicators = []

            # 1. JPEG quantization irregularities
            jpeg_score, jpeg_indicators = self._detect_jpeg_artifacts(
                image_path, pil_image
            )
            scores["jpeg_artifacts"] = jpeg_score
            indicators.extend(jpeg_indicators)

            # 2. Over-sharpening halos
            sharp_score, sharp_indicators = self._detect_sharpening_halos(img_array)
            scores["sharpening_halos"] = sharp_score
            indicators.extend(sharp_indicators)

            # 3. Frequency spectrum flattening
            freq_score, freq_indicators = self._detect_frequency_flattening(img_array)
            scores["frequency_flattening"] = freq_score
            indicators.extend(freq_indicators)

            # 4. Skin texture smoothing
            skin_score, skin_indicators = self._detect_skin_smoothing(img_array)
            scores["skin_smoothing"] = skin_score
            indicators.extend(skin_indicators)

            # 5. Color LUT banding
            color_score, color_indicators = self._detect_color_banding(img_array)
            scores["color_banding"] = color_score
            indicators.extend(color_indicators)

            # 6. Re-encoding artifacts
            reenc_score, reenc_indicators = self._detect_reencoding_artifacts(img_array)
            scores["reencoding_artifacts"] = reenc_score
            indicators.extend(reenc_indicators)

            # Calculate weighted combined score
            combined_score = sum(
                scores.get(signal, 0.0) * weight
                for signal, weight in SIGNAL_WEIGHTS.items()
            )

            # Normalize to 0-1 range
            combined_score = min(1.0, max(0.0, combined_score))

            # Determine processing level
            if combined_score < THRESHOLDS["minimal"]:
                level = "minimal_processing"
                warning = None
            elif combined_score < THRESHOLDS["heavy"]:
                level = "moderate_processing"
                warning = "This image shows some post-processing. Detection confidence may be affected."
            else:
                level = "heavy_processing"
                warning = (
                    "This image shows strong post-processing (filters, enhancement, or editing). "
                    "Such processing reduces the reliability of AI-generation detection."
                )

            # Confidence based on signal agreement
            signal_values = list(scores.values())
            if signal_values:
                # Higher confidence when signals agree
                signal_std = np.std(signal_values)
                confidence = 1.0 - min(1.0, signal_std * 2)
            else:
                confidence = 0.5

            return ProcessingLevelResult(
                level=level,
                confidence=confidence,
                indicators=indicators,
                scores=scores,
                warning=warning,
            )

        except Exception as e:
            return ProcessingLevelResult(
                level="unknown",
                confidence=0.0,
                indicators=[f"Analysis error: {str(e)}"],
                scores={},
                warning=f"Could not analyze image processing level: {str(e)}",
            )

    def _detect_jpeg_artifacts(
        self, image_path: str, pil_image: Image.Image
    ) -> Tuple[float, List[str]]:
        """
        Detect JPEG quantization irregularities and double compression.

        Returns: (score 0-1, list of indicator strings)
        """
        indicators = []
        score = 0.0

        try:
            # Check if JPEG format
            is_jpeg = image_path.lower().endswith((".jpg", ".jpeg"))

            if is_jpeg:
                # Analyze DCT block boundaries for compression artifacts
                img_array = np.array(pil_image.convert("L"))

                # Check for 8x8 block patterns (JPEG uses 8x8 DCT blocks)
                h, w = img_array.shape
                if h >= 16 and w >= 16:
                    # Calculate variance at block boundaries vs interior
                    boundary_diffs = []
                    interior_diffs = []

                    for y in range(0, h - 8, 8):
                        for x in range(0, w - 8, 8):
                            # Boundary difference (between adjacent blocks)
                            if x + 8 < w:
                                boundary_diffs.append(
                                    abs(
                                        float(img_array[y, x + 7])
                                        - float(img_array[y, x + 8])
                                    )
                                )
                            # Interior difference (within block)
                            interior_diffs.append(
                                abs(
                                    float(img_array[y, x + 3])
                                    - float(img_array[y, x + 4])
                                )
                            )

                    if boundary_diffs and interior_diffs:
                        boundary_mean = np.mean(boundary_diffs)
                        interior_mean = np.mean(interior_diffs)

                        # High boundary/interior ratio suggests visible JPEG blocking
                        if interior_mean > 0:
                            block_ratio = boundary_mean / interior_mean
                            if block_ratio > 1.5:
                                score += 0.4
                                indicators.append(
                                    "JPEG block boundary artifacts detected"
                                )
                            elif block_ratio > 1.2:
                                score += 0.2
                                indicators.append("Minor JPEG compression artifacts")

                # Check for quality estimation via DCT coefficient analysis
                # Lower quality = more visible artifacts
                img_gray = np.array(pil_image.convert("L"), dtype=np.float32)

                # Simple quality indicator: variance in high-frequency components
                if img_gray.shape[0] >= 32 and img_gray.shape[1] >= 32:
                    # Crop to multiple of 8
                    h_crop = (img_gray.shape[0] // 8) * 8
                    w_crop = (img_gray.shape[1] // 8) * 8
                    img_crop = img_gray[:h_crop, :w_crop]

                    # Use Laplacian to detect high-frequency content
                    if self.cv2_available:
                        laplacian_var = cv2.Laplacian(img_crop, cv2.CV_64F).var()

                        # Very low Laplacian variance suggests over-compression
                        if laplacian_var < 50:
                            score += 0.3
                            indicators.append(
                                "Low high-frequency content (possible over-compression)"
                            )
                        elif laplacian_var < 100:
                            score += 0.1

            # Check for PNG that was likely re-saved from JPEG
            if image_path.lower().endswith(".png"):
                img_array = np.array(pil_image.convert("L"))

                # Look for 8x8 block patterns in PNG (suggests was converted from JPEG)
                if self._has_block_patterns(img_array):
                    score += 0.5
                    indicators.append("Block patterns suggest conversion from JPEG")

        except Exception:
            pass

        return min(1.0, score), indicators

    def _has_block_patterns(self, gray_array: np.ndarray) -> bool:
        """Check for 8x8 block patterns indicating JPEG-to-PNG conversion."""
        try:
            h, w = gray_array.shape
            if h < 64 or w < 64:
                return False

            # Calculate differences at 8-pixel intervals
            h_diffs_8 = []
            h_diffs_other = []

            for y in range(16, h - 16, 16):
                for x in range(16, w - 16):
                    if x % 8 == 0:
                        h_diffs_8.append(
                            abs(float(gray_array[y, x]) - float(gray_array[y, x - 1]))
                        )
                    elif x % 8 == 4:
                        h_diffs_other.append(
                            abs(float(gray_array[y, x]) - float(gray_array[y, x - 1]))
                        )

            if h_diffs_8 and h_diffs_other:
                ratio = np.mean(h_diffs_8) / (np.mean(h_diffs_other) + 1e-6)
                return ratio > 1.3

        except Exception:
            pass

        return False

    def _detect_sharpening_halos(
        self, img_array: np.ndarray
    ) -> Tuple[float, List[str]]:
        """
        Detect over-sharpening halos (edge overshoot).

        Sharpening creates bright/dark halos around edges.
        """
        indicators = []
        score = 0.0

        if not self.cv2_available:
            return 0.0, []

        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            gray = gray.astype(np.float32)

            # Detect edges using Canny
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)

            # Dilate edges to create analysis region
            kernel = np.ones((5, 5), np.uint8)
            edge_region = cv2.dilate(edges, kernel, iterations=1)

            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Calculate second derivative (Laplacian) for overshoot detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)

            # Check for overshoot: high Laplacian values near edges
            edge_mask = edge_region > 0
            if np.sum(edge_mask) > 100:
                # Get Laplacian values in edge regions
                edge_laplacian = np.abs(laplacian[edge_mask])
                non_edge_laplacian = np.abs(laplacian[~edge_mask])

                if len(non_edge_laplacian) > 0:
                    edge_mean = np.mean(edge_laplacian)
                    non_edge_mean = np.mean(non_edge_laplacian)

                    # High ratio indicates sharpening halos
                    if non_edge_mean > 0:
                        overshoot_ratio = edge_mean / non_edge_mean

                        if overshoot_ratio > 3.0:
                            score = 0.8
                            indicators.append("Strong sharpening halos detected")
                        elif overshoot_ratio > 2.0:
                            score = 0.5
                            indicators.append("Moderate sharpening detected")
                        elif overshoot_ratio > 1.5:
                            score = 0.2
                            indicators.append("Light sharpening detected")

        except Exception:
            pass

        return min(1.0, score), indicators

    def _detect_frequency_flattening(
        self, img_array: np.ndarray
    ) -> Tuple[float, List[str]]:
        """
        Detect frequency spectrum flattening from heavy filtering.

        Filters often reduce high-frequency content unevenly.
        """
        indicators = []
        score = 0.0

        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array

            gray = gray.astype(np.float32)

            # Compute 2D FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            # Create frequency masks (low, mid, high)
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2

            # Create distance from center matrix
            y_coords, x_coords = np.ogrid[:h, :w]
            distance = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

            max_dist = np.sqrt(center_y**2 + center_x**2)

            # Define frequency bands
            low_mask = distance < (max_dist * 0.2)
            mid_mask = (distance >= (max_dist * 0.2)) & (distance < (max_dist * 0.5))
            high_mask = distance >= (max_dist * 0.5)

            # Calculate energy in each band
            low_energy = np.mean(magnitude[low_mask])
            mid_energy = np.mean(magnitude[mid_mask])
            high_energy = np.mean(magnitude[high_mask])

            total_energy = low_energy + mid_energy + high_energy + 1e-10

            # Normalized ratios
            high_ratio = high_energy / total_energy

            # Very low high-frequency ratio suggests heavy smoothing
            if high_ratio < 0.05:
                score = 0.8
                indicators.append("Severely reduced high-frequency content")
            elif high_ratio < 0.10:
                score = 0.5
                indicators.append("Reduced high-frequency content (smoothing detected)")
            elif high_ratio < 0.15:
                score = 0.2
                indicators.append("Slightly low high-frequency content")

            # Check for unusual mid/high ratio (filter signature)
            if mid_energy > 0 and high_energy > 0:
                mid_high_ratio = mid_energy / high_energy
                if mid_high_ratio > 10:
                    score = max(score, 0.6)
                    if "filter signature" not in str(indicators):
                        indicators.append(
                            "Frequency distribution suggests filter application"
                        )

        except Exception:
            pass

        return min(1.0, score), indicators

    def _detect_skin_smoothing(self, img_array: np.ndarray) -> Tuple[float, List[str]]:
        """
        Detect skin texture smoothing (beauty mode).

        Beauty filters smooth skin while preserving edges.
        """
        indicators = []
        score = 0.0

        if not self.cv2_available:
            return 0.0, []

        try:
            # Convert to HSV to detect skin-tone regions
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Skin tone detection in HSV
            # Typical skin: H=0-50, S=20-255, V=60-255
            lower_skin = np.array([0, 20, 60], dtype=np.uint8)
            upper_skin = np.array([50, 255, 255], dtype=np.uint8)

            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Erode/dilate to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
            skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = img_array.shape[0] * img_array.shape[1]

            # Only analyze if significant skin area
            skin_ratio = skin_pixels / total_pixels

            if skin_ratio > 0.05:  # At least 5% skin
                # Convert to grayscale for texture analysis
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # Calculate local variance in skin regions
                # Create variance map using a sliding window
                window_size = 15
                gray_float = gray.astype(np.float32)

                # Local mean
                kernel = np.ones((window_size, window_size), np.float32) / (
                    window_size**2
                )
                local_mean = cv2.filter2D(gray_float, -1, kernel)
                local_sq_mean = cv2.filter2D(gray_float**2, -1, kernel)

                # Local variance
                local_var = local_sq_mean - local_mean**2
                local_var = np.maximum(
                    local_var, 0
                )  # Prevent negative due to numerical issues

                # Get variance in skin regions
                skin_variance = local_var[skin_mask > 0]
                non_skin_variance = local_var[skin_mask == 0]

                if len(skin_variance) > 100 and len(non_skin_variance) > 100:
                    skin_var_mean = np.mean(skin_variance)
                    non_skin_var_mean = np.mean(non_skin_variance)

                    # Very low skin variance compared to non-skin suggests smoothing
                    if non_skin_var_mean > 0:
                        var_ratio = skin_var_mean / non_skin_var_mean

                        if var_ratio < 0.2:
                            score = 0.9
                            indicators.append(
                                "Heavy skin smoothing detected (beauty mode)"
                            )
                        elif var_ratio < 0.4:
                            score = 0.6
                            indicators.append("Moderate skin smoothing detected")
                        elif var_ratio < 0.6:
                            score = 0.3
                            indicators.append("Light skin smoothing detected")

        except Exception:
            pass

        return min(1.0, score), indicators

    def _detect_color_banding(self, img_array: np.ndarray) -> Tuple[float, List[str]]:
        """
        Detect color LUT banding from filter color grading.

        LUT application can create gaps in color histogram.
        """
        indicators = []
        score = 0.0

        try:
            # Analyze each color channel
            banding_scores = []

            for channel in range(3):
                channel_data = img_array[:, :, channel].flatten()

                # Create histogram
                hist, _ = np.histogram(channel_data, bins=256, range=(0, 256))

                # Count gaps (zero bins surrounded by non-zero bins)
                gaps = 0
                gap_runs = 0
                in_gap = False

                for i in range(1, 255):
                    is_zero = hist[i] == 0
                    prev_nonzero = hist[i - 1] > 10
                    next_nonzero = hist[i + 1] > 10

                    if is_zero and prev_nonzero and next_nonzero:
                        gaps += 1
                        if not in_gap:
                            gap_runs += 1
                            in_gap = True
                    else:
                        in_gap = False

                # Calculate gap ratio
                total_non_empty = np.sum(hist > 0)
                if total_non_empty > 0:
                    gap_ratio = gaps / total_non_empty
                    banding_scores.append(gap_ratio)

            if banding_scores:
                avg_banding = np.mean(banding_scores)

                if avg_banding > 0.15:
                    score = 0.8
                    indicators.append("Strong color banding (LUT filter detected)")
                elif avg_banding > 0.08:
                    score = 0.5
                    indicators.append("Moderate color banding detected")
                elif avg_banding > 0.04:
                    score = 0.2
                    indicators.append("Minor color quantization detected")

        except Exception:
            pass

        return min(1.0, score), indicators

    def _detect_reencoding_artifacts(
        self, img_array: np.ndarray
    ) -> Tuple[float, List[str]]:
        """
        Detect re-encoding artifacts (mosquito noise, block artifacts).

        Multiple saves create cumulative artifacts.
        """
        indicators = []
        score = 0.0

        if not self.cv2_available:
            return 0.0, []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            gray_float = gray.astype(np.float32)

            # Detect mosquito noise (ringing around edges)
            # Use edge detection + high-pass filter

            # Find edges
            edges = cv2.Canny(gray, 100, 200)

            # Dilate to get region around edges
            kernel = np.ones((7, 7), np.uint8)
            edge_region = cv2.dilate(edges, kernel, iterations=1)
            edge_region_mask = (edge_region > 0) & (
                edges == 0
            )  # Around but not on edges

            if np.sum(edge_region_mask) > 100:
                # Calculate high-frequency noise in edge regions
                # Use Laplacian as high-pass filter
                laplacian = cv2.Laplacian(gray_float, cv2.CV_64F)

                edge_noise = np.std(laplacian[edge_region_mask])
                background_noise = np.std(laplacian[~edge_region_mask & (edges == 0)])

                if background_noise > 0:
                    noise_ratio = edge_noise / background_noise

                    if noise_ratio > 2.0:
                        score = 0.7
                        indicators.append(
                            "Mosquito noise detected (multiple re-encodings)"
                        )
                    elif noise_ratio > 1.5:
                        score = 0.4
                        indicators.append("Some re-encoding artifacts detected")
                    elif noise_ratio > 1.2:
                        score = 0.2
                        indicators.append("Minor compression artifacts")

        except Exception:
            pass

        return min(1.0, score), indicators


# Convenience function matching the API in the implementation plan
def estimate_processing_level(image_path: str) -> ProcessingLevelResult:
    """
    Estimate the post-processing level of an image.

    Args:
        image_path: Path to the image file

    Returns:
        ProcessingLevelResult with:
        - level: "minimal_processing", "moderate_processing", "heavy_processing", or "unknown"
        - confidence: 0.0 to 1.0
        - indicators: List of detected processing signs
        - warning: User-facing warning message if applicable
    """
    analyzer = ImageProcessingAnalyzer()
    return analyzer.analyze(image_path)


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Estimate image post-processing level")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found: {args.image_path}")
        exit(1)

    result = estimate_processing_level(args.image_path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 50)
        print("  IMAGE PROCESSING LEVEL ANALYSIS")
        print("=" * 50)
        print(f"\n  File: {os.path.basename(args.image_path)}")
        print(f"  Processing Level: {result.level.upper()}")
        print(f"  Confidence: {result.confidence:.1%}")

        if result.warning:
            print(f"\n  ⚠️  {result.warning}")

        if result.indicators:
            print("\n  Indicators:")
            for indicator in result.indicators:
                print(f"    • {indicator}")

        print("\n  Signal Scores:")
        for signal, score in result.scores.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"    {signal:25s}: {bar} {score:.2f}")

        print("=" * 50 + "\n")
