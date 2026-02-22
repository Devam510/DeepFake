"""
Advanced AI Detector - Combined A+B+C+D
=======================================

Combines all detection approaches for robust 2026-era AI detection:
- A: Enhanced Frequency Analysis (DCT/FFT/noise)
- B: Vision Transformer (global context)
- C: Metadata-First Analysis (EXIF/camera data)
- D: Transparent Uncertainty (honest confidence)

This is the main detector for best accuracy on modern AI generators.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import components
try:
    from src.extraction.frequency_analyzer import FrequencyAnalyzer, analyze_frequency
    FREQUENCY_AVAILABLE = True
except ImportError:
    try:
        from extraction.frequency_analyzer import FrequencyAnalyzer, analyze_frequency
        FREQUENCY_AVAILABLE = True
    except ImportError:
        FREQUENCY_AVAILABLE = False

try:
    from src.modeling.vit_detector import ViTDetector, vit_detect
    VIT_AVAILABLE = True
except ImportError:
    try:
        from modeling.vit_detector import ViTDetector, vit_detect
        VIT_AVAILABLE = True
    except ImportError:
        VIT_AVAILABLE = False

# EfficientNet ensemble (existing)
try:
    from ensemble_detector import ensemble_predict
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False

# Metadata analyzer (existing)
try:
    from src.extraction.metadata_extractor import extract_metadata
    METADATA_AVAILABLE = True
except ImportError:
    try:
        from extraction.metadata_extractor import extract_metadata
        METADATA_AVAILABLE = True
    except ImportError:
        METADATA_AVAILABLE = False


@dataclass
class AdvancedDetectionResult:
    """Result from advanced combined detection."""
    
    # Individual model scores
    frequency_score: float
    vit_score: float
    efficientnet_score: float
    metadata_score: float
    
    # Combined result
    final_ai_probability: float
    confidence: str  # HIGH, MEDIUM, LOW
    verdict: str  # AI_GENERATED, REAL, UNCERTAIN
    
    # Model agreement
    model_agreement: float  # 0-1, how much models agree
    strongest_signal: str  # Which model was most decisive
    
    # Transparency (D)
    uncertainty_reason: str
    limitations: List[str]
    
    # Details
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "frequency_score": self.frequency_score,
            "vit_score": self.vit_score,
            "efficientnet_score": self.efficientnet_score,
            "metadata_score": self.metadata_score,
            "final_ai_probability": self.final_ai_probability,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "model_agreement": self.model_agreement,
            "strongest_signal": self.strongest_signal,
            "uncertainty_reason": self.uncertainty_reason,
            "limitations": self.limitations,
            "details": self.details,
        }


class AdvancedDetector:
    """
    Advanced AI detector combining all approaches (A+B+C+D).
    
    Provides the best available detection for 2026-era AI generators
    while maintaining transparency about limitations.
    """
    
    VERSION = "AdvancedDetector_v1.0.0"
    
    # Weights for combining signals
    WEIGHTS = {
        "frequency": 0.15,      # A: DCT/FFT analysis
        "vit": 0.25,            # B: Vision Transformer
        "efficientnet": 0.35,   # Existing best model
        "metadata": 0.25,       # C: Metadata analysis
    }
    
    def __init__(self):
        """Initialize all detection components."""
        self.frequency_analyzer = None
        self.vit_detector = None
        
        print("\n" + "=" * 50)
        print("  ADVANCED DETECTOR - INITIALIZING")
        print("=" * 50)
        
        # Initialize frequency analyzer (A)
        if FREQUENCY_AVAILABLE:
            self.frequency_analyzer = FrequencyAnalyzer()
            print("  ✅ A: Frequency Analyzer (DCT/FFT)")
        else:
            print("  ⚠️ A: Frequency Analyzer not available")
        
        # Initialize ViT (B)
        if VIT_AVAILABLE:
            self.vit_detector = ViTDetector()
            if self.vit_detector.initialized:
                print("  ✅ B: Vision Transformer")
            else:
                print("  ⚠️ B: ViT model not loaded")
        else:
            print("  ⚠️ B: ViT not available")
        
        # Check EfficientNet
        if EFFICIENTNET_AVAILABLE:
            print("  ✅ EfficientNet ensemble")
        else:
            print("  ⚠️ EfficientNet not available")
        
        # Check metadata
        if METADATA_AVAILABLE:
            print("  ✅ C: Metadata Analyzer")
        else:
            print("  ⚠️ C: Metadata not available")
        
        print("  ✅ D: Transparent Uncertainty (always on)")
        print("=" * 50)
    
    def detect(self, image_path: str) -> AdvancedDetectionResult:
        """
        Run all detection methods and combine results.
        
        Args:
            image_path: Path to image file
            
        Returns:
            AdvancedDetectionResult with combined analysis
        """
        if not os.path.exists(image_path):
            return self._error_result(f"File not found: {image_path}")
        
        details = {}
        scores = {}
        
        # A: Frequency Analysis
        freq_score = 0.5
        if self.frequency_analyzer:
            try:
                freq_result = self.frequency_analyzer.analyze(image_path)
                freq_score = freq_result.frequency_ai_probability
                details["frequency"] = freq_result.to_dict()
                print(f"  A: Frequency = {freq_score:.1%}")
            except Exception as e:
                details["frequency_error"] = str(e)
        scores["frequency"] = freq_score
        
        # B: Vision Transformer
        vit_score = 0.5
        if self.vit_detector and self.vit_detector.initialized:
            try:
                vit_result = self.vit_detector.predict(image_path)
                vit_score = vit_result.ai_probability
                details["vit"] = vit_result.to_dict()
                print(f"  B: ViT = {vit_score:.1%}")
            except Exception as e:
                details["vit_error"] = str(e)
        scores["vit"] = vit_score
        
        # EfficientNet (existing)
        eff_score = 0.5
        if EFFICIENTNET_AVAILABLE:
            try:
                eff_result = ensemble_predict(image_path)
                eff_score = eff_result.get("ensemble_probability", 0.5)
                details["efficientnet"] = {
                    "probability": eff_score,
                    "verdict": eff_result.get("verdict", "UNCERTAIN"),
                }
                print(f"  EfficientNet = {eff_score:.1%}")
            except Exception as e:
                details["efficientnet_error"] = str(e)
        scores["efficientnet"] = eff_score
        
        # C: Metadata Analysis
        meta_score = self._analyze_metadata(image_path, details)
        scores["metadata"] = meta_score
        print(f"  C: Metadata = {meta_score:.1%}")
        
        # Combine all scores
        return self._combine_results(scores, details, image_path)
    
    def _analyze_metadata(self, image_path: str, details: Dict) -> float:
        """
        Analyze metadata for AI detection (C).
        
        Strong signals:
        - Missing EXIF = suspicious (AI doesn't have camera)
        - Software tags like "DALL-E", "Midjourney" = definite AI
        - Missing GPS but has camera model = natural
        """
        meta_score = 0.5  # Default uncertain
        
        if not METADATA_AVAILABLE:
            details["metadata"] = {"status": "unavailable"}
            return 0.5
        
        try:
            metadata = extract_metadata(image_path)
            details["metadata"] = metadata
            
            # Check for AI software signatures
            software = metadata.get("software", "").lower()
            ai_keywords = ["dall-e", "midjourney", "stable diffusion", "ai", "generated"]
            
            if any(kw in software for kw in ai_keywords):
                return 0.95  # Very likely AI
            
            # Check for camera data
            has_camera = bool(metadata.get("camera_make") or metadata.get("camera_model"))
            has_gps = bool(metadata.get("gps_latitude"))
            has_datetime = bool(metadata.get("datetime_original"))
            
            # Camera + GPS + datetime = very likely real
            if has_camera and has_gps:
                meta_score = 0.1  # Very likely real
            elif has_camera and has_datetime:
                meta_score = 0.2  # Likely real
            elif has_camera:
                meta_score = 0.3  # Probably real
            elif has_datetime:
                meta_score = 0.4  # Slight real indication
            else:
                # No camera metadata at all = suspicious
                meta_score = 0.7  # Suspicious
            
            return meta_score
            
        except Exception as e:
            details["metadata_error"] = str(e)
            return 0.5
    
    def _combine_results(
        self, scores: Dict[str, float], details: Dict, image_path: str
    ) -> AdvancedDetectionResult:
        """Combine all scores into final result with transparency (D)."""
        
        # Weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for key, weight in self.WEIGHTS.items():
            if key in scores:
                weighted_sum += scores[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            final_prob = weighted_sum / total_weight
        else:
            final_prob = 0.5
        
        # Calculate model agreement
        score_values = list(scores.values())
        agreement = 1.0 - (max(score_values) - min(score_values))
        
        # Find strongest signal
        strongest = max(scores, key=lambda k: abs(scores[k] - 0.5))
        strongest_direction = "AI" if scores[strongest] > 0.5 else "REAL"
        
        # Determine confidence (D: Transparent Uncertainty)
        confidence, uncertainty_reason = self._compute_confidence(
            scores, agreement, final_prob
        )
        
        # Determine verdict
        if confidence == "LOW":
            verdict = "UNCERTAIN"
        elif final_prob > 0.7:
            verdict = "AI_GENERATED"
        elif final_prob < 0.3:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"
        
        # D: Document limitations
        limitations = self._get_limitations(scores, details)
        
        return AdvancedDetectionResult(
            frequency_score=scores["frequency"],
            vit_score=scores["vit"],
            efficientnet_score=scores["efficientnet"],
            metadata_score=scores["metadata"],
            final_ai_probability=final_prob,
            confidence=confidence,
            verdict=verdict,
            model_agreement=agreement,
            strongest_signal=f"{strongest} ({strongest_direction})",
            uncertainty_reason=uncertainty_reason,
            limitations=limitations,
            details=details,
        )
    
    def _compute_confidence(
        self, scores: Dict[str, float], agreement: float, final_prob: float
    ) -> tuple:
        """
        Compute confidence with transparent uncertainty reasoning (D).
        """
        reasons = []
        
        # Check for low agreement
        if agreement < 0.5:
            reasons.append(f"Models disagree (agreement: {agreement:.0%})")
        
        # Check for inconclusive probability
        if 0.35 < final_prob < 0.65:
            reasons.append("Probability in uncertain range (35-65%)")
        
        # Check if any model unavailable
        unavailable = []
        if scores["frequency"] == 0.5 and not FREQUENCY_AVAILABLE:
            unavailable.append("Frequency")
        if scores["vit"] == 0.5 and not VIT_AVAILABLE:
            unavailable.append("ViT")
        if scores["efficientnet"] == 0.5 and not EFFICIENTNET_AVAILABLE:
            unavailable.append("EfficientNet")
        
        if unavailable:
            reasons.append(f"Limited: {', '.join(unavailable)} unavailable")
        
        # Determine confidence level
        if len(reasons) >= 2 or agreement < 0.4:
            confidence = "LOW"
        elif len(reasons) == 1:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"
        
        uncertainty_reason = "; ".join(reasons) if reasons else "High model agreement"
        
        return confidence, uncertainty_reason
    
    def _get_limitations(self, scores: Dict, details: Dict) -> List[str]:
        """Document specific limitations for this detection (D)."""
        limitations = []
        
        # Modern AI limitation
        limitations.append(
            "2026 AI generators (Gemini, GPT-4, etc.) create highly realistic images "
            "that may evade detection"
        )
        
        # Model-specific limitations
        if scores["metadata"] > 0.6:
            limitations.append(
                "No camera metadata found - common for AI but also for screenshots/social media"
            )
        
        if scores["frequency"] < 0.4 and FREQUENCY_AVAILABLE:
            limitations.append(
                "Natural frequency patterns detected - could be real or advanced AI"
            )
        
        # Agreement limitation
        score_values = list(scores.values())
        if max(score_values) - min(score_values) > 0.5:
            limitations.append(
                "Models significantly disagree - manual review recommended"
            )
        
        return limitations
    
    def _error_result(self, message: str) -> AdvancedDetectionResult:
        """Return error result."""
        return AdvancedDetectionResult(
            frequency_score=0.5,
            vit_score=0.5,
            efficientnet_score=0.5,
            metadata_score=0.5,
            final_ai_probability=0.5,
            confidence="LOW",
            verdict="ERROR",
            model_agreement=0.0,
            strongest_signal="None",
            uncertainty_reason=message,
            limitations=[message],
            details={"error": message},
        )


def advanced_detect(image_path: str) -> Dict:
    """
    Convenience function for advanced detection.
    
    Args:
        image_path: Path to image
        
    Returns:
        Dict with all detection results
    """
    detector = AdvancedDetector()
    result = detector.detect(image_path)
    return result.to_dict()


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python advanced_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"\n  Analyzing: {os.path.basename(image_path)}")
    
    detector = AdvancedDetector()
    result = detector.detect(image_path)
    
    print("\n" + "=" * 60)
    print("  ADVANCED DETECTION RESULT")
    print("=" * 60)
    print(f"\n  Individual Scores:")
    print(f"    A: Frequency Analysis:  {result.frequency_score:.1%}")
    print(f"    B: Vision Transformer:  {result.vit_score:.1%}")
    print(f"       EfficientNet:        {result.efficientnet_score:.1%}")
    print(f"    C: Metadata Analysis:   {result.metadata_score:.1%}")
    
    print(f"\n  Combined Result:")
    print(f"    AI Probability:         {result.final_ai_probability:.1%}")
    print(f"    Model Agreement:        {result.model_agreement:.1%}")
    print(f"    Strongest Signal:       {result.strongest_signal}")
    
    print(f"\n  Verdict: {result.verdict}")
    print(f"  Confidence: {result.confidence}")
    
    if result.uncertainty_reason:
        print(f"\n  [D] Uncertainty: {result.uncertainty_reason}")
    
    print(f"\n  Limitations:")
    for lim in result.limitations[:3]:
        print(f"    • {lim[:70]}...")
    
    print("=" * 60)
