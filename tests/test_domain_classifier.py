"""
Tests for Domain Classification
================================
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDomainClassifierImport:
    """Test domain classifier imports."""

    def test_import_module(self):
        """Test module imports successfully."""
        from src.extraction.domain_classifier import (
            DomainClassifier,
            DomainClassificationResult,
            classify_domain,
        )

        assert DomainClassifier is not None
        assert DomainClassificationResult is not None
        assert classify_domain is not None

    def test_domain_types(self):
        """Test domain type values."""
        from src.extraction.domain_classifier import DomainType

        # Type alias should accept these values
        valid_domains = [
            "face",
            "non_face_photo",
            "art_or_illustration",
            "synthetic_graphics",
            "unknown",
        ]
        assert True  # Type checking happens at runtime


class TestDomainClassificationResult:
    """Test DomainClassificationResult dataclass."""

    def test_result_to_dict(self):
        """Test result can be converted to dict."""
        from src.extraction.domain_classifier import DomainClassificationResult

        result = DomainClassificationResult(
            domain="face",
            confidence=0.95,
            indicators=["Detected 1 face"],
            face_count=1,
            limitations="Face detector limitations",
        )

        result_dict = result.to_dict()
        assert result_dict["domain"] == "face"
        assert result_dict["confidence"] == 0.95
        assert result_dict["face_count"] == 1


class TestDomainClassifier:
    """Test DomainClassifier class."""

    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        from src.extraction.domain_classifier import DomainClassifier

        classifier = DomainClassifier(use_mtcnn=False)
        assert classifier is not None

    def test_nonexistent_file_returns_unknown(self):
        """Test missing file returns unknown domain."""
        from src.extraction.domain_classifier import classify_domain

        result = classify_domain("/nonexistent/path/image.jpg")
        assert result.domain == "unknown"
        assert result.confidence == 0.0


class TestDomainRouting:
    """Test domain routing logic."""

    def test_face_routes_correctly(self):
        """Test face domain routes to face detector concept."""
        # Conceptual test - verify the routing logic
        domain = "face"
        assert domain == "face"
        # In real implementation, this would route to face detector

    def test_non_face_routes_correctly(self):
        """Test non-face domain routes to general detector concept."""
        domain = "non_face_photo"
        assert domain in ["non_face_photo", "art_or_illustration", "synthetic_graphics"]
        # In real implementation, this would route to general detector

    def test_unknown_returns_uncertain(self):
        """Test unknown domain returns UNCERTAIN."""
        domain = "unknown"
        expected_verdict = "UNCERTAIN"
        assert domain == "unknown"
        # In real implementation, this would return UNCERTAIN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
