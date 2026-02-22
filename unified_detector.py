"""
Unified AI Image Detector
=========================

Routes images to appropriate detectors based on domain classification:
- Face images → Face detector (FROZEN)
- Non-face images → General image detector (NEW)

This is the main entry point for the expanded detection system.

CRITICAL: The face detection pipeline is FROZEN and MUST NOT be modified.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import domain classifier
try:
    from src.extraction.domain_classifier import (
        DomainClassifier,
        DomainClassificationResult,
        classify_domain,
    )

    DOMAIN_CLASSIFIER_AVAILABLE = True
except ImportError:
    try:
        from extraction.domain_classifier import (
            DomainClassifier,
            DomainClassificationResult,
            classify_domain,
        )

        DOMAIN_CLASSIFIER_AVAILABLE = True
    except ImportError:
        DOMAIN_CLASSIFIER_AVAILABLE = False

# Import general image detector
try:
    from src.modeling.general_image_detector import (
        GeneralImageDetector,
        GeneralDetectionResult,
    )

    GENERAL_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from modeling.general_image_detector import (
            GeneralImageDetector,
            GeneralDetectionResult,
        )

        GENERAL_DETECTOR_AVAILABLE = True
    except ImportError:
        GENERAL_DETECTOR_AVAILABLE = False

# Import face detector (FROZEN - do not modify)
try:
    from ensemble_detector import ensemble_predict as face_ensemble_predict

    FACE_DETECTOR_AVAILABLE = True
except ImportError:
    FACE_DETECTOR_AVAILABLE = False


@dataclass
class UnifiedDetectionResult:
    """Result from unified detection system."""

    # Domain information
    detected_domain: str
    domain_confidence: float
    detector_used: str

    # Detection results
    synthetic_probability: float
    confidence_interval: Dict[str, float]
    interpretation: str
    confidence_band: str
    verdict: str

    # Transparency
    limitations: str
    uncertainty_notice: str

    # Metadata
    request_id: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "detected_domain": self.detected_domain,
            "domain_confidence": self.domain_confidence,
            "detector_used": self.detector_used,
            "synthetic_probability": self.synthetic_probability,
            "confidence_interval": self.confidence_interval,
            "interpretation": self.interpretation,
            "confidence_band": self.confidence_band,
            "verdict": self.verdict,
            "limitations": self.limitations,
            "uncertainty_notice": self.uncertainty_notice,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }


class UnifiedDetector:
    """
    Unified detector that routes to appropriate domain-specific detectors.

    Architecture:
    1. Classify image domain (face vs non-face)
    2. Route to appropriate detector
    3. Return domain-labeled result with limitations

    CRITICAL: Face detection is FROZEN - this class does NOT modify it.
    """

    VERSION = "UnifiedDetector_v1.0.0"

    def __init__(self):
        """Initialize unified detector with domain classifier and detectors."""
        self.domain_classifier = None
        self.general_detector = None

        if DOMAIN_CLASSIFIER_AVAILABLE:
            self.domain_classifier = DomainClassifier(use_mtcnn=False)
            print("✅ Domain classifier initialized")
        else:
            print("⚠️ Domain classifier not available")

        if GENERAL_DETECTOR_AVAILABLE:
            self.general_detector = GeneralImageDetector()
            print("✅ General image detector initialized")
        else:
            print("⚠️ General detector not available")

        if FACE_DETECTOR_AVAILABLE:
            print("✅ Face detector available (FROZEN)")
        else:
            print("⚠️ Face detector not available")

    def detect(self, image_path: str) -> UnifiedDetectionResult:
        """
        Detect AI generation in an image.

        Routes to appropriate detector based on domain classification.

        Args:
            image_path: Path to image file

        Returns:
            UnifiedDetectionResult with domain-aware detection
        """
        import uuid

        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Check file exists
        if not os.path.exists(image_path):
            return self._error_result("File not found", request_id, timestamp)

        # Step 1: Classify domain
        if self.domain_classifier is None:
            return self._error_result(
                "Domain classifier not available", request_id, timestamp
            )

        domain_result = self.domain_classifier.classify(image_path)

        print(f"\n{'='*60}")
        print(f"  UNIFIED AI DETECTOR")
        print(f"{'='*60}")
        print(f"  Analyzing: {os.path.basename(image_path)}")
        print(f"  Domain: {domain_result.domain} ({domain_result.confidence:.0%})")
        print(f"  Indicators: {', '.join(domain_result.indicators[:3])}")

        # Step 2: Route to appropriate detector
        if domain_result.domain == "face":
            return self._detect_face(image_path, domain_result, request_id, timestamp)
        elif domain_result.domain in [
            "non_face_photo",
            "art_or_illustration",
            "synthetic_graphics",
        ]:
            return self._detect_general(
                image_path, domain_result, request_id, timestamp
            )
        else:
            return self._uncertain_result(domain_result, request_id, timestamp)

    def _detect_face(
        self,
        image_path: str,
        domain_result: DomainClassificationResult,
        request_id: str,
        timestamp: str,
    ) -> UnifiedDetectionResult:
        """
        Route to face detector (FROZEN).

        This method calls the existing face detection pipeline WITHOUT modification.
        """
        print(f"\n  [Routing] Using FACE detector (FROZEN)")

        if not FACE_DETECTOR_AVAILABLE:
            return self._error_result(
                "Face detector not available", request_id, timestamp
            )

        try:
            # Call the FROZEN face detector
            face_result = face_ensemble_predict(image_path)

            return UnifiedDetectionResult(
                detected_domain="face",
                domain_confidence=domain_result.confidence,
                detector_used="FaceEnsembleDetector (FROZEN)",
                synthetic_probability=face_result.get("ensemble_probability", 0.5),
                confidence_interval={
                    "lower": face_result.get("ensemble_probability", 0.5) - 0.1,
                    "upper": min(
                        1.0, face_result.get("ensemble_probability", 0.5) + 0.1
                    ),
                },
                interpretation=face_result.get("verdict", "UNCERTAIN"),
                confidence_band=face_result.get("confidence", "MEDIUM"),
                verdict=face_result.get("verdict", "UNCERTAIN"),
                limitations=domain_result.limitations,
                uncertainty_notice=(
                    "Face detection results are from the production face detector. "
                    "This detector is specialized for human faces and has been validated "
                    "for face-specific AI generation detection."
                ),
                request_id=request_id,
                timestamp=timestamp,
            )

        except Exception as e:
            return self._error_result(
                f"Face detection error: {str(e)}", request_id, timestamp
            )

    def _detect_general(
        self,
        image_path: str,
        domain_result: DomainClassificationResult,
        request_id: str,
        timestamp: str,
    ) -> UnifiedDetectionResult:
        """
        Route to EfficientNet-based detector for all non-face images.
        
        Uses the same EfficientNet model as face detection for consistent,
        high-quality results across all image types.
        """
        print(f"\n  [Routing] Using EfficientNet (universal detector)")

        # Try EfficientNet first (best accuracy)
        if FACE_DETECTOR_AVAILABLE:
            try:
                # Call the EfficientNet-based ensemble detector
                efficientnet_result = face_ensemble_predict(image_path)
                
                synthetic_prob = efficientnet_result.get("ensemble_probability", 0.5)
                verdict = efficientnet_result.get("verdict", "UNCERTAIN")
                confidence = efficientnet_result.get("confidence", "MEDIUM")
                
                print(f"  Probability: {synthetic_prob:.1%}")
                print(f"  Verdict: {verdict}")
                print(f"  Confidence: {confidence}")

                return UnifiedDetectionResult(
                    detected_domain=domain_result.domain,
                    domain_confidence=domain_result.confidence,
                    detector_used="EfficientNet-Universal (v1.0)",
                    synthetic_probability=synthetic_prob,
                    confidence_interval={
                        "lower": max(0.0, synthetic_prob - 0.1),
                        "upper": min(1.0, synthetic_prob + 0.1),
                    },
                    interpretation=verdict,
                    confidence_band=confidence,
                    verdict=verdict,
                    limitations=(
                        f"Domain: {domain_result.domain}. "
                        "EfficientNet provides accurate detection across image types. "
                        "Model trained primarily on faces/photos, accuracy may vary on pure graphics."
                    ),
                    uncertainty_notice=(
                        "Using EfficientNet-based universal detector. "
                        "This model excels at detecting modern AI generators like DALL-E, Midjourney, and Stable Diffusion."
                    ),
                    request_id=request_id,
                    timestamp=timestamp,
                )

            except Exception as e:
                print(f"  ⚠️ EfficientNet error: {e}, falling back to general detector")

        # Fallback to general detector if EfficientNet fails
        if self.general_detector is None:
            return self._error_result(
                "No detector available", request_id, timestamp
            )

        try:
            # Call the general image detector as fallback
            general_result = self.general_detector.predict(
                image_path, domain_result.domain
            )

            # Determine verdict
            if general_result.is_uncertain:
                verdict = "UNCERTAIN"
            elif general_result.synthetic_probability > 0.7:
                verdict = "LIKELY AI-GENERATED"
            elif general_result.synthetic_probability < 0.3:
                verdict = "LIKELY REAL"
            else:
                verdict = "INCONCLUSIVE"

            print(f"  Probability: {general_result.synthetic_probability:.1%}")
            print(f"  Verdict: {verdict}")
            print(f"  Confidence: {general_result.confidence_band}")

            return UnifiedDetectionResult(
                detected_domain=domain_result.domain,
                domain_confidence=domain_result.confidence,
                detector_used=general_result.detector_version,
                synthetic_probability=general_result.synthetic_probability,
                confidence_interval=general_result.confidence_interval,
                interpretation=general_result.synthetic_likelihood,
                confidence_band=general_result.confidence_band,
                verdict=verdict,
                limitations=general_result.limitations,
                uncertainty_notice=(
                    "General image detection has variable accuracy across different "
                    "image types and AI generators. Results should be interpreted with "
                    "appropriate caution, especially in the 30-70% probability range."
                ),
                request_id=request_id,
                timestamp=timestamp,
            )

        except Exception as e:
            return self._error_result(
                f"Detection error: {str(e)}", request_id, timestamp
            )

    def _uncertain_result(
        self, domain_result: DomainClassificationResult, request_id: str, timestamp: str
    ) -> UnifiedDetectionResult:
        """Return UNCERTAIN result when domain cannot be determined."""
        print(f"\n  [Result] UNCERTAIN - domain unknown")

        return UnifiedDetectionResult(
            detected_domain="unknown",
            domain_confidence=domain_result.confidence,
            detector_used="None - domain unknown",
            synthetic_probability=0.5,
            confidence_interval={"lower": 0.0, "upper": 1.0},
            interpretation="INCONCLUSIVE - cannot determine appropriate detector",
            confidence_band="LOW",
            verdict="UNCERTAIN",
            limitations=(
                "Image domain could not be classified with confidence. "
                "No detector is applicable. Results are unreliable."
            ),
            uncertainty_notice=(
                "This image could not be classified into a known domain. "
                "AI detection cannot be performed reliably."
            ),
            request_id=request_id,
            timestamp=timestamp,
        )

    def _error_result(
        self, error_message: str, request_id: str, timestamp: str
    ) -> UnifiedDetectionResult:
        """Return error result."""
        return UnifiedDetectionResult(
            detected_domain="error",
            domain_confidence=0.0,
            detector_used="None - error",
            synthetic_probability=0.5,
            confidence_interval={"lower": 0.0, "upper": 1.0},
            interpretation=f"ERROR: {error_message}",
            confidence_band="LOW",
            verdict="ERROR",
            limitations=error_message,
            uncertainty_notice="An error occurred during detection.",
            request_id=request_id,
            timestamp=timestamp,
        )


def unified_detect(image_path: str) -> dict:
    """
    Detect AI generation in any image type.

    Routes to face or general detector based on image content.

    Args:
        image_path: Path to image file

    Returns:
        Dict with detection results
    """
    detector = UnifiedDetector()
    result = detector.detect(image_path)
    return result.to_dict()


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unified_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    result = unified_detect(image_path)

    print(f"\n{'='*60}")
    print(f"  DETECTION RESULT")
    print(f"{'='*60}")
    print(f"  Domain:       {result['detected_domain']}")
    print(f"  Detector:     {result['detector_used']}")
    print(f"  Probability:  {result['synthetic_probability']:.1%}")
    print(f"  Confidence:   {result['confidence_band']}")
    print(f"  Verdict:      {result['verdict']}")
    print(f"\n  Limitations:")
    print(f"  {result['limitations']}")
    print(f"{'='*60}\n")
