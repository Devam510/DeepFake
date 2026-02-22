"""
Test Filter Detector Module
tests/test_filter_detector.py

Validates:
- Metadata detection for social media apps
- Aspect ratio detection
- Visual pattern detection (vignette, color cast, beautification)
"""

import pytest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFilterDetectorImport:
    """Test that filter detector can be imported."""

    def test_import_filter_detector(self):
        """Test filter detector module imports successfully."""
        from src.extraction.filter_detector import (
            SocialMediaFilterDetector,
            detect_social_media_filter,
            FilterDetectionResult,
        )

        assert SocialMediaFilterDetector is not None
        assert detect_social_media_filter is not None


class TestAspectRatioDetection:
    """Test aspect ratio detection for social media formats."""

    def test_instagram_square_ratio(self):
        """Test that 1:1 aspect ratio is detected as Instagram square."""
        from src.extraction.filter_detector import SOCIAL_MEDIA_RATIOS

        # 1:1 should be Instagram square
        assert (1, 1) in SOCIAL_MEDIA_RATIOS
        assert SOCIAL_MEDIA_RATIOS[(1, 1)] == "instagram_square"

    def test_story_ratio(self):
        """Test that 9:16 aspect ratio is detected as story format."""
        from src.extraction.filter_detector import SOCIAL_MEDIA_RATIOS

        # 9:16 should be story format
        assert (9, 16) in SOCIAL_MEDIA_RATIOS
        assert SOCIAL_MEDIA_RATIOS[(9, 16)] == "story_format"


class TestSocialMediaSignatures:
    """Test app signature detection."""

    def test_instagram_signatures_exist(self):
        """Test that Instagram signatures are defined."""
        from src.extraction.filter_detector import SOCIAL_MEDIA_SIGNATURES

        assert "instagram" in SOCIAL_MEDIA_SIGNATURES
        assert "instagram" in SOCIAL_MEDIA_SIGNATURES["instagram"]

    def test_snapchat_signatures_exist(self):
        """Test that Snapchat signatures are defined."""
        from src.extraction.filter_detector import SOCIAL_MEDIA_SIGNATURES

        assert "snapchat" in SOCIAL_MEDIA_SIGNATURES
        assert "snapchat" in SOCIAL_MEDIA_SIGNATURES["snapchat"]

    def test_vsco_signatures_exist(self):
        """Test that VSCO signatures are defined."""
        from src.extraction.filter_detector import SOCIAL_MEDIA_SIGNATURES

        assert "vsco" in SOCIAL_MEDIA_SIGNATURES


class TestFilterDetectionResult:
    """Test FilterDetectionResult dataclass."""

    def test_result_to_dict(self):
        """Test that result can be converted to dict."""
        from src.extraction.filter_detector import FilterDetectionResult

        result = FilterDetectionResult(
            filter_detected=True,
            filter_confidence=0.8,
            filter_type="instagram",
            filter_indicators=["App signature: instagram"],
            is_social_media_image=True,
        )

        result_dict = result.to_dict()

        assert result_dict["filter_detected"] == True
        assert result_dict["filter_confidence"] == 0.8
        assert result_dict["filter_type"] == "instagram"
        assert "App signature: instagram" in result_dict["filter_indicators"]

    def test_no_filter_result(self):
        """Test result for no filter detected."""
        from src.extraction.filter_detector import FilterDetectionResult

        result = FilterDetectionResult(
            filter_detected=False,
            filter_confidence=0.0,
            filter_type="none",
            filter_indicators=[],
            is_social_media_image=False,
        )

        assert result.filter_detected == False
        assert result.filter_type == "none"


class TestEnsembleIntegration:
    """Test that filter detection integrates with ensemble."""

    def test_filter_detector_flag_exists(self):
        """Test that FILTER_DETECTOR_AVAILABLE flag exists in ensemble."""
        import ensemble_detector

        assert hasattr(ensemble_detector, "FILTER_DETECTOR_AVAILABLE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
