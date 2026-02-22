"""
Test API Output Module
tests/test_api_output.py

Validates:
- CI contains score
- Interpretation matches scientific validity rules
- Null semantics enforced
- No authenticity claims without provenance
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.detection_api import (
    get_interpretation,
    AuthenticityScore,
    SourceIdentification,
    ModificationAnalysis,
    ProvenanceCheck,
)


class TestGetInterpretation:
    """Test interpretation logic matches scientific validity rules."""

    def test_ci_crossing_boundary_is_inconclusive(self):
        """Test that CI crossing 50% boundary returns inconclusive."""
        result = get_interpretation(50.0, 40.0, 60.0)
        assert "Inconclusive" in result
        assert "crosses decision boundary" in result

    def test_ci_crossing_boundary_low_score(self):
        """Test CI crossing with low score."""
        result = get_interpretation(45.0, 30.0, 55.0)
        assert "Inconclusive" in result

    def test_ci_crossing_boundary_high_score(self):
        """Test CI crossing with high score."""
        result = get_interpretation(55.0, 45.0, 70.0)
        assert "Inconclusive" in result

    def test_high_score_does_not_claim_authentic(self):
        """Test that high scores do NOT claim authenticity."""
        result = get_interpretation(90.0, 88.0, 95.0)
        # Must NOT contain "authentic origin" - only provenance can claim that
        assert "authentic origin" not in result.lower()
        # Should express low synthetic probability or require provenance
        assert (
            "synthetic probability" in result.lower() or "provenance" in result.lower()
        )

    def test_moderate_high_score_requires_provenance(self):
        """Test moderate high scores require provenance verification."""
        result = get_interpretation(80.0, 75.0, 85.0)
        assert "authentic origin" not in result.lower()
        # Should mention provenance or be inconclusive
        assert "provenance" in result.lower() or "inconclusive" in result.lower()

    def test_borderline_high_is_inconclusive(self):
        """Test borderline high scores are inconclusive."""
        result = get_interpretation(60.0, 55.0, 65.0)
        assert "authentic" not in result.lower()

    def test_strong_synthetic_detection(self):
        """Test strong evidence of synthetic requires ci_upper <= 15."""
        result = get_interpretation(10.0, 5.0, 12.0)
        assert "Strong evidence of synthetic" in result

    def test_moderate_synthetic_detection(self):
        """Test moderate evidence of synthetic requires ci_upper <= 30."""
        result = get_interpretation(20.0, 15.0, 25.0)
        assert "Moderate evidence of synthetic" in result

    def test_weak_synthetic_for_borderline(self):
        """Test weak evidence for scores just below 50."""
        result = get_interpretation(40.0, 35.0, 45.0)
        assert "synthetic" in result.lower()

    def test_wide_ci_is_uncertain(self):
        """Test that wide CI (>30) returns uncertainty."""
        result = get_interpretation(70.0, 50.0, 90.0)  # Width = 40
        assert "Inconclusive" in result or "Uncertain" in result


class TestNullSemantics:
    """Test that null semantics are enforced correctly."""

    def test_source_identification_null_confidence(self):
        """Test that unknown source has null confidence, not 0.0."""
        source = SourceIdentification(
            likely_source=None,
            confidence=None,
            alternative_sources=[],
            is_human_created=None,
        )
        result = source.to_dict()
        assert result["confidence"] is None

    def test_source_identification_known_source_has_confidence(self):
        """Test that known source has non-null confidence."""
        source = SourceIdentification(
            likely_source="stable_diffusion",
            confidence=0.85,
            alternative_sources=[],
            is_human_created=False,
        )
        result = source.to_dict()
        assert result["confidence"] == 0.85

    def test_modification_analysis_null_likelihood(self):
        """Test that no modifications = null likelihood."""
        mods = ModificationAnalysis(
            likelihood=None,
            detected_modifications=[],
            compression_detected=None,
            resize_detected=None,
        )
        result = mods.to_dict()
        assert result["likelihood"] is None

    def test_authenticity_score_has_required_fields(self):
        """Test AuthenticityScore contains score and CI."""
        auth = AuthenticityScore(
            score=75.0,
            confidence_lower=70.0,
            confidence_upper=80.0,
            confidence_level=0.95,
            interpretation="Low-moderate synthetic probability - provenance verification recommended",
        )
        result = auth.to_dict()

        assert "score" in result
        assert "confidence_interval" in result
        assert result["confidence_interval"]["lower"] <= result["score"]
        assert result["score"] <= result["confidence_interval"]["upper"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
