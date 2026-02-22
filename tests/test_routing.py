"""
Routing Tests for Unified Detector
===================================

Tests for the domain routing and detector orchestration.
"""

import pytest
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

# Mock the PIL Image for testing
from PIL import Image
import numpy as np

# Import the modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDomainRouting:
    """Tests for domain classification and routing."""

    @pytest.fixture
    def temp_image_dir(self):
        """Create temp directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple test image
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            img_path = os.path.join(tmpdir, "test.jpg")
            img.save(img_path)
            yield tmpdir

    def test_domain_classifier_import(self):
        """Test domain classifier can be imported."""
        from src.extraction.domain_classifier import DomainClassifier, classify_domain
        assert DomainClassifier is not None
        assert classify_domain is not None

    def test_general_detector_import(self):
        """Test general detector can be imported."""
        from src.modeling.general_image_detector import GeneralImageDetector
        assert GeneralImageDetector is not None

    def test_unified_detector_import(self):
        """Test unified detector can be imported."""
        from unified_detector import unified_detect
        assert unified_detect is not None

    def test_routing_non_face_image(self, temp_image_dir):
        """Test that non-face images route to general detector."""
        from src.extraction.domain_classifier import classify_domain
        
        # Create gradient image (no face)
        img = Image.new("RGB", (256, 256))
        pixels = img.load()
        for i in range(256):
            for j in range(256):
                pixels[i, j] = (i, j, 128)
        
        img_path = os.path.join(temp_image_dir, "gradient.jpg")
        img.save(img_path)
        
        result = classify_domain(img_path)
        # classify_domain returns DomainClassificationResult object
        domain = result.domain if hasattr(result, "domain") else result.get("domain", result)
        assert domain != "face", "Gradient image should not be classified as face"

    def test_routing_returns_domain_info(self, temp_image_dir):
        """Test that routing returns domain information."""
        from unified_detector import unified_detect
        
        img_path = os.path.join(temp_image_dir, "test.jpg")
        result = unified_detect(img_path)
        
        # Check result is a dict and has expected structure
        assert isinstance(result, dict), "Result should be a dict"
        # Domain info should be in the result somewhere
        has_domain = "domain" in result or ("domain_info" in result and "domain" in result.get("domain_info", {}))
        assert has_domain or "detected_domain" in result or any("domain" in str(k).lower() for k in result.keys()), \
            f"Result should contain domain info. Keys: {result.keys()}"

    def test_uncertain_result_handling(self, temp_image_dir):
        """Test that UNCERTAIN results are properly handled."""
        from src.modeling.general_image_detector import GeneralImageDetector
        
        detector = GeneralImageDetector()
        
        # The detector should handle edge cases gracefully
        img_path = os.path.join(temp_image_dir, "test.jpg")
        result = detector.predict(img_path)
        
        assert hasattr(result, "is_uncertain"), "Result should have is_uncertain field"
        assert hasattr(result, "synthetic_probability"), "Result should have synthetic_probability"

    def test_domain_limitations_returned(self, temp_image_dir):
        """Test that domain limitations are included in results."""
        from unified_detector import unified_detect
        
        img_path = os.path.join(temp_image_dir, "test.jpg")
        result = unified_detect(img_path)
        
        # Check result structure - limitations may be in different places
        has_limitations = (
            "limitations" in result or 
            "domain_limitations" in result or
            "general_result" in result or
            "uncertainty_notice" in result  # unified_detect uses this key
        )
        assert has_limitations, f"Result should contain limitations/uncertainty info. Keys: {result.keys()}"


class TestAPIResponseExtension:
    """Tests for Phase 4 API response extensions."""

    def test_api_response_has_domain_fields(self):
        """Test APIResponse includes domain detection fields."""
        from src.api.detection_api import APIResponse, AuthenticityScore, SourceIdentification
        from src.api.detection_api import ModificationAnalysis, ProvenanceCheck
        
        response = APIResponse(
            request_id="test",
            timestamp="2026-01-01T00:00:00Z",
            processing_time_ms=100.0,
            media_type="image",
            authenticity=AuthenticityScore(
                score=70.0, confidence_lower=60.0, confidence_upper=80.0,
                confidence_level=0.95, interpretation="test"
            ),
            source=SourceIdentification(
                likely_source=None, confidence=None,
                alternative_sources=[], is_human_created=None
            ),
            modifications=ModificationAnalysis(
                likelihood=None, detected_modifications=[],
                compression_detected=None, resize_detected=None
            ),
            provenance=ProvenanceCheck(
                has_provenance=False, status="not_checked",
                chain_length=0, signed_links=0, original_hash=None
            ),
            models_used=["test"],
            ensemble_agreement=None,
            out_of_distribution=None,
            ood_score=None,
            # Phase 4 fields
            domain_detected="non_face_photo",
            detector_used="general_detector",
            domain_limitations="General detector has lower accuracy on art",
            uncertainty_zone=None,
        )
        
        result = response.to_dict()
        assert "domain_detection" in result
        assert result["domain_detection"]["domain"] == "non_face_photo"
        assert result["domain_detection"]["detector_used"] == "general_detector"
        assert result["domain_detection"]["limitations"] is not None


class TestDetectorVersioning:
    """Tests for detector versioning."""

    def test_general_detector_has_version(self):
        """Test GeneralImageDetector includes version info."""
        from src.modeling.general_image_detector import GeneralDetectionResult
        
        # Version should be in the result
        result = GeneralDetectionResult(
            synthetic_probability=0.5,
            confidence_interval={"lower": 0.3, "upper": 0.7},
            synthetic_likelihood="INCONCLUSIVE",
            confidence_band="MEDIUM",
            domain_used="non_face_photo",
            detector_version="1.0.0",
            limitations="Test",
            is_uncertain=True,
        )
        
        assert result.detector_version == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
