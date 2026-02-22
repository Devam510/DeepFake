"""
DeepFake Detection System - API Service
Layer 6: System Output Interface

This module implements the public API that:
- Exposes probabilistic scores (never binary)
- Provides confidence intervals
- Returns source identification
- Includes modification detection
"""

import json
import time
import hashlib
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum

# Import core components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MediaType(Enum):
    """Supported media types."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class APIRequest:
    """Incoming API request structure."""

    file_path: str
    request_id: str
    media_type: Optional[MediaType] = None
    check_provenance: bool = True
    detailed_attribution: bool = False
    client_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIRequest":
        return cls(
            file_path=data["file_path"],
            request_id=data.get(
                "request_id", hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
            ),
            media_type=(
                MediaType(data["media_type"]) if data.get("media_type") else None
            ),
            check_provenance=data.get("check_provenance", True),
            detailed_attribution=data.get("detailed_attribution", False),
            client_id=data.get("client_id"),
        )


@dataclass
class AuthenticityScore:
    """
    Primary authenticity score.

    Per Layer 6 spec: Never binary, always probabilistic.
    """

    score: float  # 0-100 scale
    confidence_lower: float  # Lower CI bound
    confidence_upper: float  # Upper CI bound
    confidence_level: float  # e.g., 0.95
    interpretation: str  # Human-readable interpretation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "confidence_interval": {
                "lower": round(self.confidence_lower, 2),
                "upper": round(self.confidence_upper, 2),
                "level": self.confidence_level,
            },
            "interpretation": self.interpretation,
        }


@dataclass
class SourceIdentification:
    """Source/generator identification."""

    likely_source: Optional[str]  # e.g., "midjourney-v5"
    confidence: Optional[float]  # None = not computed, 0.0-1.0 if computed
    alternative_sources: List[Dict[str, float]]
    is_human_created: Optional[bool]  # None = insufficient evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "likely_source": self.likely_source,
            "confidence": (
                round(self.confidence, 3) if self.confidence is not None else None
            ),
            "alternatives": self.alternative_sources,
            "is_human_created": self.is_human_created,
        }


@dataclass
class ModificationAnalysis:
    """Post-processing modification detection."""

    # INVARIANT 2: null if not computed, not 0.0
    likelihood: Optional[float]  # None = not computed, 0.0-1.0 if computed
    detected_modifications: List[str]
    # INVARIANT 4: Tri-state - True/False/None
    compression_detected: Optional[bool]  # None = not checked
    resize_detected: Optional[bool]  # None = not checked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "likelihood": (
                round(self.likelihood, 3) if self.likelihood is not None else None
            ),
            "detected": self.detected_modifications,
            "compression_detected": self.compression_detected,
            "resize_detected": self.resize_detected,
        }


@dataclass
class ProvenanceCheck:
    """Provenance verification results."""

    has_provenance: bool
    status: str  # "verified", "partial", "unverified", "tampered"
    chain_length: int
    signed_links: int
    original_hash: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_provenance": self.has_provenance,
            "status": self.status,
            "chain_length": self.chain_length,
            "signed_links": self.signed_links,
            "original_hash": self.original_hash,
        }


@dataclass
class APIResponse:
    """
    Complete API response structure.

    Per Layer 6: Exposes scores, not binary labels.
    """

    request_id: str
    timestamp: str
    processing_time_ms: float
    media_type: str

    # Core results
    authenticity: AuthenticityScore
    source: SourceIdentification
    modifications: ModificationAnalysis
    provenance: ProvenanceCheck

    # Model ensemble info
    models_used: List[str]
    ensemble_agreement: float  # 0-100

    # OOD detection - INVARIANT 4: tri-state
    out_of_distribution: Optional[bool]  # None = not evaluated
    ood_score: Optional[float]  # None = not computed

    # Phase 4: Domain detection fields
    domain_detected: Optional[str] = None  # face, non_face_photo, art, etc.
    detector_used: Optional[str] = None  # face_detector, general_detector
    domain_limitations: Optional[str] = None  # Domain-specific accuracy notes
    uncertainty_zone: Optional[str] = None  # Explains if in inconclusive range

    # Feature attribution (optional)
    feature_attributions: Optional[Dict[str, float]] = None

    # Warnings/notes
    warnings: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # FIX C: agreement requires per_model_scores with actual values
        # If per_model_scores are not computed, agreement is not valid
        per_model_scores = None
        has_valid_scores = False

        if self.models_used and self.models_used != ["mock"]:
            # Check if we have actual feature data - if not, scores are not computable
            # For now, per_model_scores are not populated, so agreement is not valid
            per_model_scores = None  # Would be {model: score} if computed
            has_valid_scores = False

        # Agreement is only valid if per_model_scores exist
        ensemble_agreement = None
        if has_valid_scores and self.ensemble_agreement is not None:
            ensemble_agreement = round(self.ensemble_agreement, 2)

        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "media_type": self.media_type,
            # Phase 4: Domain detection fields
            "domain_detection": {
                "domain": self.domain_detected,
                "detector_used": self.detector_used,
                "limitations": self.domain_limitations,
                "uncertainty_zone": self.uncertainty_zone,
            },
            "authenticity": self.authenticity.to_dict(),
            "source_identification": self.source.to_dict(),
            "modification_analysis": self.modifications.to_dict(),
            "provenance": self.provenance.to_dict(),
            "ensemble": {
                "models_used": self.models_used,
                "agreement": ensemble_agreement,
                "agreement_status": (
                    "not_computable"
                    if not has_valid_scores and self.models_used != ["mock"]
                    else None
                ),
                "per_model_scores": per_model_scores,
            },
            "out_of_distribution": {
                "is_ood": self.out_of_distribution,
                "score": (
                    round(self.ood_score, 2) if self.ood_score is not None else None
                ),
            },
            # FIX 3: Explicit state instead of null
            "feature_attributions": (
                self.feature_attributions
                if self.feature_attributions
                else "not_available"
            ),
            "warnings": self.warnings or [],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def get_interpretation(score: float, ci_lower: float, ci_upper: float) -> str:
    """
    Generate human-readable interpretation.

    SCIENTIFIC VALIDITY RULES:
    - Detection can ONLY express synthetic likelihood or uncertainty
    - Detection CANNOT assert authenticity (requires provenance)
    - High scores = low synthetic probability, NOT proof of authenticity
    - Language must never outrun evidence
    """
    ci_width = ci_upper - ci_lower

    # RULE 1: CI crossing boundary = Inconclusive
    if ci_lower < 50 and ci_upper > 50:
        return "Inconclusive - confidence interval crosses decision boundary"

    # RULE 2: Wide CI = High uncertainty
    if ci_width > 30:
        if ci_upper <= 50:
            return "Uncertain - weak signal suggesting synthetic origin"
        else:
            return "Inconclusive - insufficient confidence for any claim"

    # RULE 3: Synthetic detection (LOW scores = HIGH synthetic probability)
    if ci_upper <= 15 and ci_width < 10:
        return "Strong evidence of synthetic generation"
    elif ci_upper <= 30 and ci_width < 20:
        return "Moderate evidence of synthetic generation"
    elif ci_upper <= 50:
        return "Weak evidence of synthetic origin - manual review recommended"

    # RULE 4: High scores = low synthetic probability (NOT authenticity)
    # Per scientific validity: cannot claim "authentic" without provenance
    if ci_lower >= 85 and ci_width < 10:
        return (
            "Low synthetic probability - authenticity requires provenance verification"
        )
    elif ci_lower >= 70 and ci_width < 20:
        return (
            "Low-moderate synthetic probability - provenance verification recommended"
        )
    elif ci_lower >= 50:
        return "Inconclusive - no strong synthetic indicators detected"

    return "Inconclusive - manual review recommended"


class DetectionAPI:
    """
    Main API class for the detection system.
    """

    def __init__(self):
        """Initialize API with detection components."""
        self.ensemble = None
        self.provenance_manager = None
        self._initialized = False

    def initialize(self):
        """
        Lazy initialization of detection components.
        """
        if self._initialized:
            return

        try:
            from ..modeling.ensemble import create_default_ensemble

            self.ensemble = create_default_ensemble()
        except ImportError as e:
            print(f"Warning: Ensemble not available ({e}), using mock mode")
            self.ensemble = None

        try:
            from ..utils.provenance import ProvenanceManager

            self.provenance_manager = ProvenanceManager("./provenance")
        except ImportError:
            self.provenance_manager = None

        self._initialized = True

    def detect_media_type(self, file_path: str) -> MediaType:
        """Detect media type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        if ext in image_exts:
            return MediaType.IMAGE
        elif ext in video_exts:
            return MediaType.VIDEO
        elif ext in audio_exts:
            return MediaType.AUDIO
        else:
            return MediaType.UNKNOWN

    def analyze(self, request: APIRequest) -> APIResponse:
        """
        Perform full analysis on a media file.

        Args:
            request: APIRequest object

        Returns:
            APIResponse with all results
        """
        self.initialize()

        start_time = time.time()
        warnings = []

        # Detect media type
        if request.media_type:
            media_type = request.media_type
        else:
            media_type = self.detect_media_type(request.file_path)

        if media_type == MediaType.UNKNOWN:
            warnings.append("Unknown media type - results may be unreliable")

        # Run ensemble detection
        if self.ensemble and media_type == MediaType.IMAGE:
            try:
                ensemble_result = self.ensemble.predict(request.file_path)

                auth_score = ensemble_result.final_authenticity_score
                synth_prob = ensemble_result.final_synthetic_probability
                conf_lower = ensemble_result.confidence_lower
                conf_upper = ensemble_result.confidence_upper
                models_used = [p.model_name for p in ensemble_result.model_predictions]
                agreement = ensemble_result.ensemble_confidence
                is_ood = ensemble_result.is_out_of_distribution
                ood_score = ensemble_result.ood_score
                likely_source = ensemble_result.likely_source
                source_conf = ensemble_result.source_confidence
                mod_likelihood = ensemble_result.modification_likelihood
                mods = ensemble_result.detected_modifications

            except Exception as e:
                warnings.append(f"Ensemble error: {str(e)}")
                auth_score = 50.0
                synth_prob = 0.5
                conf_lower = 0.3
                conf_upper = 0.7
                models_used = []
                agreement = None  # INVARIANT 3: null, not 0.0
                is_ood = None  # INVARIANT 4: null = not evaluated
                ood_score = None  # INVARIANT 3: null, not 50.0
                likely_source = None
                source_conf = None  # INVARIANT 2: null, not 0.0
                mod_likelihood = None  # INVARIANT 3: null placeholder
                mods = []
        else:
            # Mock results for unsupported types or missing ensemble
            # INVARIANT 3: Use null for uncomputed values, not placeholders
            auth_score = 50.0
            synth_prob = 0.5
            conf_lower = 0.3
            conf_upper = 0.7
            models_used = ["mock"]
            agreement = None  # INVARIANT 3: null, not 0.0
            is_ood = None  # INVARIANT 4: null = not evaluated
            ood_score = None  # INVARIANT 3: null, not 50.0
            likely_source = None
            source_conf = None  # INVARIANT 2: null, not 0.0
            mod_likelihood = None  # INVARIANT 3: null placeholder
            mods = []
            warnings.append("Using mock results - ensemble not available")

        # Check provenance
        if request.check_provenance and self.provenance_manager:
            provenance_check = ProvenanceCheck(
                has_provenance=False,
                status="unverified",
                chain_length=0,
                signed_links=0,
                original_hash=None,
            )
        else:
            provenance_check = ProvenanceCheck(
                has_provenance=False,
                status="not_checked",
                chain_length=0,
                signed_links=0,
                original_hash=None,
            )

        # Build response
        # INVARIANT 1: CI must contain the score
        # conf_lower/conf_upper are bounds for synthetic_probability
        # authenticity_score = (1 - synthetic_prob) * 100
        # Therefore CI must be inverted:
        auth_ci_lower = (1.0 - conf_upper) * 100
        auth_ci_upper = (1.0 - conf_lower) * 100

        authenticity = AuthenticityScore(
            score=auth_score,
            confidence_lower=auth_ci_lower,
            confidence_upper=auth_ci_upper,
            confidence_level=0.95,
            interpretation=get_interpretation(auth_score, auth_ci_lower, auth_ci_upper),
        )

        # INVARIANT 3: Use tri-state logic when evidence is insufficient
        # True = strong evidence human, False = strong evidence synthetic, None = inconclusive
        if auth_score > 70:
            is_human = True
        elif auth_score < 30:
            is_human = False
        else:
            is_human = None  # Insufficient evidence

        source = SourceIdentification(
            likely_source=likely_source,
            confidence=source_conf,
            alternative_sources=[],
            is_human_created=is_human,
        )

        # INVARIANT 2,3,4: Use null for uncomputed values, not placeholders
        # mod_likelihood is None when analysis not run (mock mode)
        actual_mod_likelihood = mod_likelihood  # Already None in mock mode

        # INVARIANT 4: Tri-state logic for modification detection
        # None = not checked, False = checked but not found, True = found
        if mod_likelihood is None and not mods:
            # Mock mode: analysis not run
            compression = None
            resize = None
        else:
            # Real analysis: check what was found, null if nothing found (not False)
            compression = True if "compression" in str(mods).lower() else None
            resize = True if "resize" in str(mods).lower() else None

        modifications = ModificationAnalysis(
            likelihood=actual_mod_likelihood,
            detected_modifications=mods,
            compression_detected=compression,
            resize_detected=resize,
        )

        processing_time = (time.time() - start_time) * 1000

        return APIResponse(
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time,
            media_type=media_type.value,
            authenticity=authenticity,
            source=source,
            modifications=modifications,
            provenance=provenance_check,
            models_used=models_used,
            ensemble_agreement=agreement,
            out_of_distribution=is_ood,
            ood_score=ood_score,
            warnings=warnings,
        )


# ============================================================
# Flask/FastAPI-style endpoint handlers
# ============================================================


def handle_analyze_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle an analysis request.

    This is the main entry point for API integration.
    """
    api = DetectionAPI()
    request = APIRequest.from_dict(request_data)
    response = api.analyze(request)
    return response.to_dict()


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepFake Detection API")
    parser.add_argument("file", help="Media file to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--no-provenance", action="store_true", help="Skip provenance check"
    )

    args = parser.parse_args()

    request_data = {"file_path": args.file, "check_provenance": not args.no_provenance}

    result = handle_analyze_request(request_data)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))
