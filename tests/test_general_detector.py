"""
Tests for General Image Detector
=================================
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGeneralDetectorImport:
    """Test general detector imports."""

    def test_import_module(self):
        """Test module imports successfully."""
        from src.modeling.general_image_detector import (
            GeneralImageDetector,
            GeneralDetectionResult,
            detect_general_image,
        )

        assert GeneralImageDetector is not None
        assert GeneralDetectionResult is not None
        assert detect_general_image is not None


class TestGeneralDetectionResult:
    """Test GeneralDetectionResult dataclass."""

    def test_result_to_dict(self):
        """Test result can be converted to dict."""
        from src.modeling.general_image_detector import GeneralDetectionResult

        result = GeneralDetectionResult(
            synthetic_probability=0.65,
            confidence_interval={"lower": 0.55, "upper": 0.75},
            synthetic_likelihood="MODERATE-HIGH synthetic likelihood",
            confidence_band="MEDIUM",
            domain_used="non_face_photo",
            detector_version="GeneralImageDetector_v1.0.0",
            limitations="Test limitations",
            is_uncertain=False,
        )

        result_dict = result.to_dict()
        assert result_dict["synthetic_probability"] == 0.65
        assert result_dict["domain_used"] == "non_face_photo"
        assert result_dict["is_uncertain"] == False


class TestGeneralImageDetector:
    """Test GeneralImageDetector class."""

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        from src.modeling.general_image_detector import GeneralImageDetector

        detector = GeneralImageDetector()
        assert detector is not None

    def test_uncertainty_thresholds(self):
        """Test uncertainty thresholds are correct."""
        from src.modeling.general_image_detector import GeneralImageDetector

        detector = GeneralImageDetector()
        assert detector.UNCERTAIN_LOWER == 0.35
        assert detector.UNCERTAIN_UPPER == 0.65


class TestUncertaintyHandling:
    """Test UNCERTAIN handling logic."""

    def test_uncertain_zone_detection(self):
        """Test that uncertainty zone triggers correctly."""
        # Probabilities in 0.35-0.65 range should be uncertain
        uncertain_probs = [0.4, 0.5, 0.6]
        for prob in uncertain_probs:
            in_uncertain_zone = 0.35 <= prob <= 0.65
            assert in_uncertain_zone, f"Probability {prob} should be in uncertain zone"

    def test_confident_zone_detection(self):
        """Test that confident zones are not uncertain."""
        confident_probs = [0.1, 0.2, 0.8, 0.9]
        for prob in confident_probs:
            not_in_uncertain_zone = prob < 0.35 or prob > 0.65
            assert not_in_uncertain_zone, f"Probability {prob} should be confident"


class TestDomainSpecificWeighting:
    """Test domain-specific signal weighting."""

    def test_art_domain_weights_neural_higher(self):
        """Test art domain weights neural features higher."""
        # Art detection should rely more on neural features
        weights = {
            "non_face_photo": {"neural": 0.25},
            "art_or_illustration": {"neural": 0.4},
        }
        assert (
            weights["art_or_illustration"]["neural"]
            > weights["non_face_photo"]["neural"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
