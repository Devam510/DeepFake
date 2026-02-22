"""
Domain Classifier Module
========================

Classifies images into domains for routing to appropriate detectors.
This module does NOT determine AI-ness - only which detector should run.

Domains:
- face: Contains human faces → route to face detector (FROZEN)
- non_face_photo: Photographs without faces → route to general detector
- art_or_illustration: Artwork, drawings, illustrations → route to general detector
- synthetic_graphics: UI, screenshots, diagrams → route to general detector
- unknown: Cannot determine → return UNCERTAIN
"""

import os
import sys
from dataclasses import dataclass
from typing import Literal, List, Optional, Tuple
import numpy as np
from PIL import Image

# Domain type definition
DomainType = Literal[
    "face", "non_face_photo", "art_or_illustration", "synthetic_graphics", "unknown"
]


@dataclass
class DomainClassificationResult:
    """Result of domain classification."""

    domain: DomainType
    confidence: float  # 0.0 to 1.0
    indicators: List[str]
    face_count: int  # Number of faces detected (0 if none)
    limitations: str  # Domain-specific limitations text

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "face_count": self.face_count,
            "limitations": self.limitations,
        }


# Try to import face detection libraries
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN
    import torch

    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class DomainClassifier:
    """
    Classifies images into domains for detector routing.

    Uses lightweight signals:
    - Face detection (OpenCV/MTCNN)
    - Edge complexity analysis
    - Color distribution analysis
    - Texture pattern analysis
    - Aspect ratio checks
    """

    def __init__(self, use_mtcnn: bool = False):
        """
        Initialize domain classifier.

        Args:
            use_mtcnn: If True, use MTCNN for more accurate face detection.
                       If False, use OpenCV Haar cascades (faster).
        """
        self.use_mtcnn = use_mtcnn and MTCNN_AVAILABLE

        # Initialize face detector
        if self.use_mtcnn:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mtcnn = MTCNN(keep_all=True, device=device)
            print("Domain Classifier: Using MTCNN for face detection")
        elif OPENCV_AVAILABLE:
            # Load OpenCV Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("Domain Classifier: Using OpenCV Haar cascade for face detection")
        else:
            self.face_cascade = None
            print("Domain Classifier: No face detection available")

    def classify(self, image_path: str) -> DomainClassificationResult:
        """
        Classify an image into a domain.

        Args:
            image_path: Path to the image file

        Returns:
            DomainClassificationResult with domain type and confidence
        """
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)
        except Exception as e:
            return DomainClassificationResult(
                domain="unknown",
                confidence=0.0,
                indicators=[f"Failed to load image: {str(e)}"],
                face_count=0,
                limitations="Cannot classify image that failed to load",
            )

        indicators = []

        # Step 1: Check for faces
        face_count, face_area_ratio = self._detect_faces(img_array)
        
        # Only route to face detector if:
        # 1. At least 1 face detected
        # 2. Face(s) cover at least 5% of image area (not tiny distant figures)
        if face_count > 0 and face_area_ratio > 0.05:
            indicators.append(f"Detected {face_count} prominent face(s) ({face_area_ratio*100:.1f}% of image)")
            return DomainClassificationResult(
                domain="face",
                confidence=min(0.95, 0.7 + face_area_ratio),
                indicators=indicators,
                face_count=face_count,
                limitations="Face detector is specialized for human faces. Results may vary for non-standard angles or occlusions.",
            )
        elif face_count > 0:
            indicators.append(f"Detected {face_count} small face(s) ({face_area_ratio*100:.1f}% of image - too small for face detector)")

        indicators.append("No faces detected")

        # Step 2: Analyze image characteristics
        edge_score = self._analyze_edges(img_array)
        color_score = self._analyze_colors(img_array)
        texture_score = self._analyze_texture(img_array)
        aspect_ratio = image.width / image.height

        indicators.append(f"Edge complexity: {edge_score:.2f}")
        indicators.append(f"Color diversity: {color_score:.2f}")
        indicators.append(f"Texture score: {texture_score:.2f}")

        # Step 3: Classify based on signals
        domain, confidence = self._determine_domain(
            edge_score, color_score, texture_score, aspect_ratio, img_array
        )

        # Get domain-specific limitations
        limitations = self._get_domain_limitations(domain)

        return DomainClassificationResult(
            domain=domain,
            confidence=confidence,
            indicators=indicators,
            face_count=0,
            limitations=limitations,
        )

    def _detect_faces(self, img_array: np.ndarray) -> tuple:
        """Detect faces in image. Returns (count, area_ratio)."""
        image_area = img_array.shape[0] * img_array.shape[1]
        
        if self.use_mtcnn:
            try:
                boxes, _ = self.mtcnn.detect(Image.fromarray(img_array))
                if boxes is None:
                    return 0, 0.0
                # Calculate face area ratio
                face_area = sum((b[2]-b[0]) * (b[3]-b[1]) for b in boxes)
                return len(boxes), face_area / image_area
            except Exception:
                return 0, 0.0
        elif OPENCV_AVAILABLE and self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                # Stricter detection to avoid false positives:
                # - minNeighbors=8 (was 5) - requires more detections to confirm
                # - minSize=(60, 60) (was 30, 30) - ignore tiny faces
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60)
                )
                if len(faces) == 0:
                    return 0, 0.0
                face_area = sum(w * h for (x, y, w, h) in faces)
                return len(faces), face_area / image_area
            except Exception:
                return 0, 0.0
        return 0, 0.0

    def _analyze_edges(self, img_array: np.ndarray) -> float:
        """
        Analyze edge complexity.

        High edge scores indicate:
        - Detailed photos (high)
        - Simple graphics/UI (low)
        - Art/illustrations (medium-high with regularity)
        """
        if not OPENCV_AVAILABLE:
            return 0.5

        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return min(1.0, edge_density * 10)  # Normalize
        except Exception:
            return 0.5

    def _analyze_colors(self, img_array: np.ndarray) -> float:
        """
        Analyze color distribution.

        High color scores indicate:
        - Natural photos (high diversity)
        - Art/illustrations (medium, often stylized)
        - UI/graphics (low, limited palette)
        """
        try:
            # Convert to HSV for better color analysis
            if OPENCV_AVAILABLE:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            else:
                # Simple RGB analysis fallback
                hsv = img_array

            # Calculate color diversity via histogram
            h_hist, _ = np.histogram(hsv[:, :, 0].flatten(), bins=36, range=(0, 180))
            h_hist = h_hist / h_hist.sum() if h_hist.sum() > 0 else h_hist

            # Entropy-based diversity
            h_hist = h_hist[h_hist > 0]
            entropy = -np.sum(h_hist * np.log2(h_hist)) if len(h_hist) > 0 else 0
            normalized_entropy = entropy / np.log2(36)  # Max entropy for 36 bins

            return normalized_entropy
        except Exception:
            return 0.5

    def _analyze_texture(self, img_array: np.ndarray) -> float:
        """
        Analyze texture patterns.

        Uses variance-based texture analysis.
        High texture scores indicate natural photos.
        Low scores indicate flat graphics/UI.
        """
        try:
            gray = img_array.mean(axis=2) if len(img_array.shape) == 3 else img_array

            # Local variance analysis (simplified LBP-like)
            kernel_size = 5
            # Compute local variance
            from scipy import ndimage

            local_mean = ndimage.uniform_filter(gray, size=kernel_size)
            local_sqr_mean = ndimage.uniform_filter(gray**2, size=kernel_size)
            local_var = local_sqr_mean - local_mean**2

            mean_var = np.mean(local_var)
            # Normalize to 0-1 range
            return min(1.0, mean_var / 1000)
        except ImportError:
            # Fallback without scipy
            try:
                gray = (
                    img_array.mean(axis=2) if len(img_array.shape) == 3 else img_array
                )
                return min(1.0, np.std(gray) / 100)
            except Exception:
                return 0.5
        except Exception:
            return 0.5

    def _determine_domain(
        self,
        edge_score: float,
        color_score: float,
        texture_score: float,
        aspect_ratio: float,
        img_array: np.ndarray,
    ) -> Tuple[DomainType, float]:
        """
        Determine domain based on analyzed signals.

        Returns:
            Tuple of (domain type, confidence)
        """
        # Check for synthetic graphics/UI characteristics
        # - Low texture, limited colors, clean edges
        if texture_score < 0.1 and color_score < 0.3:
            return "synthetic_graphics", 0.8

        # Check for UI/screenshot aspect ratios
        common_ui_ratios = [16 / 9, 9 / 16, 4 / 3, 3 / 4, 16 / 10]
        ratio_match = any(abs(aspect_ratio - r) < 0.05 for r in common_ui_ratios)

        # Large flat color regions indicate UI
        flat_regions = self._detect_flat_regions(img_array)
        if flat_regions > 0.4 and ratio_match:
            return "synthetic_graphics", 0.75

        # Check for art/illustration characteristics
        # - Medium-high edges but with regularity
        # - Stylized colors
        if edge_score > 0.3 and color_score < 0.5 and texture_score < 0.3:
            return "art_or_illustration", 0.7

        # Default to non-face photo if has natural characteristics
        # - High color diversity
        # - Natural texture
        if color_score > 0.4 or texture_score > 0.2:
            confidence = min(0.85, (color_score + texture_score) / 2 + 0.3)
            return "non_face_photo", confidence

        # Cannot determine with confidence
        return "unknown", 0.5

    def _detect_flat_regions(self, img_array: np.ndarray) -> float:
        """Detect percentage of image with flat color regions (UI indicator)."""
        try:
            if not OPENCV_AVAILABLE:
                return 0.0

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # Use Laplacian to detect flat regions
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            flat_mask = np.abs(laplacian) < 5
            return np.sum(flat_mask) / flat_mask.size
        except Exception:
            return 0.0

    def _get_domain_limitations(self, domain: DomainType) -> str:
        """Get domain-specific limitation text."""
        limitations = {
            "face": (
                "Face detector is specialized for human faces. "
                "Results may vary for non-standard angles, occlusions, or artistic styles."
            ),
            "non_face_photo": (
                "General image detector handles diverse photo types. "
                "Accuracy varies by generator and image complexity. "
                "Heavily filtered images may produce UNCERTAIN results."
            ),
            "art_or_illustration": (
                "Art/illustration detection has higher uncertainty. "
                "AI-generated art and human art can be difficult to distinguish. "
                "Results should be interpreted with caution."
            ),
            "synthetic_graphics": (
                "UI/graphics are often synthetic by design. "
                "AI detection is less meaningful for this domain. "
                "High synthetic probability may not indicate AI generation."
            ),
            "unknown": (
                "Domain could not be determined with confidence. "
                "Detection results are unreliable. "
                "UNCERTAIN verdict will be returned."
            ),
        }
        return limitations.get(domain, "Unknown domain limitations.")


def classify_domain(
    image_path: str, use_mtcnn: bool = False
) -> DomainClassificationResult:
    """
    Convenience function to classify image domain.

    Args:
        image_path: Path to image file
        use_mtcnn: Use MTCNN for more accurate face detection

    Returns:
        DomainClassificationResult
    """
    classifier = DomainClassifier(use_mtcnn=use_mtcnn)
    return classifier.classify(image_path)


# Export for easy access
__all__ = [
    "DomainClassifier",
    "DomainClassificationResult",
    "DomainType",
    "classify_domain",
]
