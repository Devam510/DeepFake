"""
Social Media Filter Detector
=============================

Detects Instagram, Snapchat, TikTok, and other social media filters
to reduce false positives in AI detection.

Filters create artifacts similar to AI generation:
- Smoothing effects (reduce texture entropy)
- Color grading (changes frequency characteristics)
- Beautification (smooth skin, sharp edges)

This module identifies filtered real photos so the ensemble
can adjust its AI probability accordingly.
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class FilterDetectionResult:
    """Result of social media filter detection."""

    filter_detected: bool
    filter_confidence: float  # 0-1
    filter_type: str  # instagram, snapchat, tiktok, vsco, other, none
    filter_indicators: List[str]
    is_social_media_image: bool

    def to_dict(self) -> Dict:
        return {
            "filter_detected": self.filter_detected,
            "filter_confidence": self.filter_confidence,
            "filter_type": self.filter_type,
            "filter_indicators": self.filter_indicators,
            "is_social_media_image": self.is_social_media_image,
        }


# Known social media app signatures in EXIF/XMP metadata
SOCIAL_MEDIA_SIGNATURES = {
    # Software field markers
    "instagram": ["instagram", "ig_", "iphone photo"],
    "snapchat": ["snapchat", "snap camera", "snap inc"],
    "tiktok": ["tiktok", "bytedance", "musical.ly"],
    "vsco": ["vsco", "vscocam"],
    "snapseed": ["snapseed", "google snapseed"],
    "facetune": ["facetune", "lightricks"],
    "beauty_plus": ["beautyplus", "beauty plus", "meitu"],
    "retrica": ["retrica"],
    "b612": ["b612", "snow camera"],
    "foodie": ["foodie"],
    "ulike": ["ulike"],
}

# Common social media aspect ratios
SOCIAL_MEDIA_RATIOS = {
    (1, 1): "instagram_square",  # Instagram square post
    (4, 5): "instagram_portrait",  # Instagram portrait
    (16, 9): "landscape",  # YouTube/general landscape
    (9, 16): "story_format",  # Stories (Instagram/Snapchat/TikTok)
    (3, 4): "portrait",  # Common phone portrait
}


class SocialMediaFilterDetector:
    """
    Detects social media filters in images to reduce false positives.

    Detection methods:
    1. Metadata analysis (app signatures, XMP data)
    2. Aspect ratio analysis (social media specific)
    3. Visual pattern detection (vignette, color cast, skin smoothing)
    """

    def __init__(self):
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required: pip install opencv-python")

    def detect(self, image_path: str) -> FilterDetectionResult:
        """
        Analyze image for social media filter indicators.

        Args:
            image_path: Path to image file

        Returns:
            FilterDetectionResult with detection details
        """
        indicators = []
        confidence_scores = []
        detected_app = "none"

        try:
            # 0. Check filename for app signatures (Snapchat-*, IMG_*, etc.)
            filename = os.path.basename(image_path).lower()
            filename_apps = {
                "snapchat": ["snapchat-", "snap-", "snap_"],
                "instagram": ["insta_", "instagram_", "ig_"],
                "tiktok": ["tiktok_", "tiktok-"],
            }
            for app_name, prefixes in filename_apps.items():
                for prefix in prefixes:
                    if filename.startswith(prefix):
                        detected_app = app_name
                        indicators.append(f"Filename indicates {app_name}")
                        confidence_scores.append(0.85)  # Strong indicator
                        break
                if detected_app != "none":
                    break

            # 1. Check metadata for app signatures
            metadata_result = self._check_metadata(image_path)
            if metadata_result["app_detected"]:
                detected_app = metadata_result["app_name"]
                indicators.append(f"App signature: {detected_app}")
                confidence_scores.append(0.9)  # Strong indicator

            # 2. Check aspect ratio
            ratio_result = self._check_aspect_ratio(image_path)
            if ratio_result["is_social_media_ratio"]:
                indicators.append(f"Aspect ratio: {ratio_result['ratio_name']}")
                confidence_scores.append(0.3)  # Weak indicator

            # 3. Check for vignette effect
            vignette_result = self._detect_vignette(image_path)
            if vignette_result["has_vignette"]:
                indicators.append(
                    f"Vignette effect detected (strength: {vignette_result['strength']:.1%})"
                )
                confidence_scores.append(min(0.5, vignette_result["strength"]))

            # 4. Check for color cast (warm/cool filters)
            color_cast_result = self._detect_color_cast(image_path)
            if color_cast_result["has_color_cast"]:
                indicators.append(f"Color filter: {color_cast_result['cast_type']}")
                confidence_scores.append(0.4)

            # 5. Check for face beautification (smooth skin with sharp edges)
            beauty_result = self._detect_beautification(image_path)
            if beauty_result["has_beautification"]:
                indicators.append("Face beautification filter detected")
                confidence_scores.append(0.6)

        except Exception as e:
            # If analysis fails, return no filter detected
            return FilterDetectionResult(
                filter_detected=False,
                filter_confidence=0.0,
                filter_type="none",
                filter_indicators=[f"Analysis error: {str(e)}"],
                is_social_media_image=False,
            )

        # Calculate combined confidence
        if confidence_scores:
            # Use max confidence, boosted by number of indicators
            base_confidence = max(confidence_scores)
            indicator_boost = min(0.2, len(indicators) * 0.05)
            combined_confidence = min(1.0, base_confidence + indicator_boost)
        else:
            combined_confidence = 0.0

        filter_detected = combined_confidence > 0.3
        is_social_media = detected_app != "none" or len(indicators) >= 2

        return FilterDetectionResult(
            filter_detected=filter_detected,
            filter_confidence=combined_confidence,
            filter_type=(
                detected_app
                if detected_app != "none"
                else ("other" if filter_detected else "none")
            ),
            filter_indicators=indicators,
            is_social_media_image=is_social_media,
        )

    def _check_metadata(self, image_path: str) -> Dict:
        """Check EXIF/XMP metadata for social media app signatures."""
        result = {"app_detected": False, "app_name": "none", "raw_software": None}

        try:
            from PIL.ExifTags import TAGS

            img = Image.open(image_path)

            # Check EXIF data
            software = ""
            if hasattr(img, "_getexif") and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "Software" and isinstance(value, str):
                        software = value.lower()
                        result["raw_software"] = value
                        break

            # Check PNG text chunks
            if hasattr(img, "info"):
                for key, value in img.info.items():
                    if isinstance(value, str):
                        software += " " + value.lower()

            # Match against known signatures
            for app_name, signatures in SOCIAL_MEDIA_SIGNATURES.items():
                for sig in signatures:
                    if sig in software:
                        result["app_detected"] = True
                        result["app_name"] = app_name
                        return result

        except Exception:
            pass

        return result

    def _check_aspect_ratio(self, image_path: str) -> Dict:
        """Check if aspect ratio matches common social media formats."""
        result = {"is_social_media_ratio": False, "ratio_name": None, "ratio": None}

        try:
            img = Image.open(image_path)
            w, h = img.size

            # Reduce to simplified ratio
            from math import gcd

            divisor = gcd(w, h)
            ratio_w = w // divisor
            ratio_h = h // divisor

            # Check common ratios with some tolerance
            for target_ratio, name in SOCIAL_MEDIA_RATIOS.items():
                target_w, target_h = target_ratio
                # Calculate actual ratio as float
                actual = w / h
                target = target_w / target_h

                if abs(actual - target) < 0.05:  # 5% tolerance
                    result["is_social_media_ratio"] = True
                    result["ratio_name"] = name
                    result["ratio"] = f"{target_w}:{target_h}"
                    break

        except Exception:
            pass

        return result

    def _detect_vignette(self, image_path: str) -> Dict:
        """Detect vignette effect (darkened corners)."""
        result = {"has_vignette": False, "strength": 0.0}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return result

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape

            # Sample corner and center brightness
            corner_size = min(h, w) // 8

            # Get corner regions
            corners = [
                gray[:corner_size, :corner_size],  # Top-left
                gray[:corner_size, -corner_size:],  # Top-right
                gray[-corner_size:, :corner_size],  # Bottom-left
                gray[-corner_size:, -corner_size:],  # Bottom-right
            ]

            # Get center region
            center_y, center_x = h // 2, w // 2
            center = gray[
                center_y - corner_size : center_y + corner_size,
                center_x - corner_size : center_x + corner_size,
            ]

            corner_brightness = np.mean([np.mean(c) for c in corners])
            center_brightness = np.mean(center)

            # Vignette = corners significantly darker than center
            if center_brightness > 10:  # Avoid division issues
                darkness_ratio = corner_brightness / center_brightness

                if darkness_ratio < 0.85:  # Corners are at least 15% darker
                    result["has_vignette"] = True
                    result["strength"] = 1.0 - darkness_ratio

        except Exception:
            pass

        return result

    def _detect_color_cast(self, image_path: str) -> Dict:
        """Detect warm/cool color cast from filters."""
        result = {"has_color_cast": False, "cast_type": None, "strength": 0.0}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return result

            # Split channels
            b, g, r = cv2.split(img.astype(np.float32))

            # Calculate average channel values
            avg_r = np.mean(r)
            avg_g = np.mean(g)
            avg_b = np.mean(b)

            # Neutral gray would have equal R, G, B
            # Warm filter: R > B
            # Cool filter: B > R

            if avg_r > 0 and avg_b > 0:
                warmth_ratio = avg_r / avg_b

                if warmth_ratio > 1.15:  # Warm cast
                    result["has_color_cast"] = True
                    result["cast_type"] = "warm"
                    result["strength"] = min(1.0, (warmth_ratio - 1.0) * 2)
                elif warmth_ratio < 0.85:  # Cool cast
                    result["has_color_cast"] = True
                    result["cast_type"] = "cool"
                    result["strength"] = min(1.0, (1.0 - warmth_ratio) * 2)

        except Exception:
            pass

        return result

    def _detect_beautification(self, image_path: str) -> Dict:
        """
        Detect face beautification filters.

        Beautification creates smooth skin regions while preserving
        sharp edges (eyes, eyebrows, lips).
        """
        result = {"has_beautification": False, "confidence": 0.0}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return result

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Simple heuristic: check for regions with very low texture
            # variance but high edge response at boundaries

            # Compute local variance (texture measure)
            kernel_size = 15
            blur = cv2.GaussianBlur(
                gray.astype(np.float32), (kernel_size, kernel_size), 0
            )
            local_variance = cv2.GaussianBlur(
                (gray.astype(np.float32) - blur) ** 2, (kernel_size, kernel_size), 0
            )

            # Compute edges
            edges = cv2.Canny(gray, 50, 150)

            # Check for pattern: very smooth regions (low variance) with nearby edges
            # This is characteristic of skin smoothing

            smooth_mask = local_variance < np.percentile(local_variance, 25)
            edge_mask = edges > 0

            # Dilate edges to find nearby smooth regions
            kernel = np.ones((5, 5), np.uint8)
            edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=3)

            # Smooth regions near edges
            smooth_near_edges = smooth_mask & (edge_dilated > 0)

            # If significant portion is smooth near edges = beautification
            smooth_ratio = np.sum(smooth_near_edges) / (h * w)

            if smooth_ratio > 0.15:  # 15% of image is smooth near edges
                result["has_beautification"] = True
                result["confidence"] = min(1.0, smooth_ratio * 3)

        except Exception:
            pass

        return result


def detect_social_media_filter(image_path: str) -> FilterDetectionResult:
    """
    Convenience function for filter detection.

    Args:
        image_path: Path to image file

    Returns:
        FilterDetectionResult with detection details
    """
    detector = SocialMediaFilterDetector()
    return detector.detect(image_path)


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Detect social media filters in images"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        exit(1)

    result = detect_social_media_filter(args.image)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n" + "=" * 50)
        print("  SOCIAL MEDIA FILTER DETECTION")
        print("=" * 50)
        print(f"  Filter Detected:     {result.filter_detected}")
        print(f"  Confidence:          {result.filter_confidence:.1%}")
        print(f"  Filter Type:         {result.filter_type}")
        print(f"  Social Media Image:  {result.is_social_media_image}")

        if result.filter_indicators:
            print("\n  Indicators:")
            for indicator in result.filter_indicators:
                print(f"    • {indicator}")

        print("=" * 50 + "\n")
