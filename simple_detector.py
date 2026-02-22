"""
Simple AI Detector - Clean, ONE Verdict
========================================

Combines all detection approaches but gives ONE clear final answer.
No more confusing multi-verdict outputs.

Logic:
- If EfficientNet is >90% confident → Trust it (AI or Real)
- Otherwise combine all signals and give best guess
- Show probability %, one verdict, done
"""

import os
import sys
from typing import Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import components silently
FREQUENCY_AVAILABLE = False
VIT_AVAILABLE = False
EFFICIENTNET_AVAILABLE = False
METADATA_AVAILABLE = False

try:
    from src.extraction.frequency_analyzer import FrequencyAnalyzer
    FREQUENCY_AVAILABLE = True
except ImportError:
    pass

try:
    from src.modeling.vit_detector import ViTDetector
    VIT_AVAILABLE = True
except ImportError:
    pass

try:
    from ensemble_detector import ensemble_predict
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    pass

try:
    from src.extraction.metadata_extractor import extract_metadata
    METADATA_AVAILABLE = True
except ImportError:
    pass


class SimpleDetector:
    """
    Simple AI detector with ONE clear verdict.
    
    Priority:
    1. EfficientNet (best model) - if >90% confidence, trust it
    2. Combined ensemble - if EfficientNet unsure
    """
    
    def __init__(self, quiet=True):
        """Initialize detector."""
        self.frequency_analyzer = None
        self.vit_detector = None
        self.quiet = quiet
        
        if not quiet:
            print("\n  Initializing AI Detector...")
        
        if FREQUENCY_AVAILABLE:
            self.frequency_analyzer = FrequencyAnalyzer()
        
        if VIT_AVAILABLE:
            self.vit_detector = ViTDetector()
            
        if not quiet:
            print("  Ready.\n")
    
    def detect(self, image_path: str) -> Dict:
        """
        Detect if image is AI-generated.
        
        Returns simple dict with:
        - ai_probability: 0-1
        - verdict: "AI" or "REAL" or "UNCERTAIN"
        - confidence: "HIGH", "MEDIUM", "LOW"
        """
        if not os.path.exists(image_path):
            return {"error": "File not found", "ai_probability": 0.5, "verdict": "ERROR"}
        
        scores = {}
        
        # 1. EfficientNet (primary - best model)
        if EFFICIENTNET_AVAILABLE:
            try:
                eff_result = ensemble_predict(image_path)
                # PRODUCTION FIX: Use RAW EfficientNet probability, NOT ensemble_probability
                # ensemble_probability is adjusted by metadata/filters which causes CLI vs Web mismatch
                individual = eff_result.get("individual_results", {})
                eff_raw = individual.get("efficientnet", {})
                scores["efficientnet"] = eff_raw.get("probability", eff_result.get("ensemble_probability", 0.5))
                
                # PRODUCTION FIX: Also sync METADATA score from Ensemble
                # Ensemble is better at finding metadata (0.3) than SimpleDetector's check (0.5)
                # This ensures we respect the "camera detected" signal found by Ensemble
                if "metadata" in individual:
                    scores["metadata"] = individual["metadata"].get("probability", 0.5)
                
                # STORE ENSEMBLE PROBABILITY for final decision
                # This matches the CLI output exactly (e.g. 30% for real photos even if EffNet is 100%)
                scores["ensemble_main"] = eff_result.get("ensemble_probability", 0.5)
                    
            except:
                scores["efficientnet"] = 0.5
                scores["ensemble_main"] = 0.5
        else:
            scores["efficientnet"] = 0.5
            scores["ensemble_main"] = 0.5
        
        # 2. Frequency Analysis
        if self.frequency_analyzer:
            try:
                freq_result = self.frequency_analyzer.analyze(image_path)
                scores["frequency"] = freq_result.frequency_ai_probability
            except:
                scores["frequency"] = 0.5
        else:
            scores["frequency"] = 0.5
        
        # 3. ViT
        if self.vit_detector and self.vit_detector.initialized:
            try:
                vit_result = self.vit_detector.predict(image_path)
                scores["vit"] = vit_result.ai_probability
            except:
                scores["vit"] = 0.5
        else:
            scores["vit"] = 0.5
        
        # 4. Metadata (Fallback if not already set by Ensemble)
        if "metadata" not in scores:
            scores["metadata"] = self._check_metadata(image_path)
        
        # DECISION LOGIC - Simple and clear
        return self._make_decision(scores)
    
    def _check_metadata(self, image_path: str) -> float:
        """Check metadata for AI signals."""
        if not METADATA_AVAILABLE:
            return 0.5
        
        try:
            metadata = extract_metadata(image_path)
            
            # Check for AI software tags
            software = str(metadata.get("software", "")).lower()
            if any(kw in software for kw in ["dall-e", "midjourney", "stable", "ai"]):
                return 0.95
            
            # Check for camera = real
            has_camera = bool(metadata.get("camera_make") or metadata.get("camera_model"))
            has_gps = bool(metadata.get("gps_latitude"))
            
            if has_camera and has_gps:
                return 0.1  # Very likely real
            elif has_camera:
                return 0.25  # Probably real
            else:
                return 0.6  # No camera data = slightly suspicious
                
        except:
            return 0.5
    
    def _make_decision(self, scores: Dict) -> Dict:
        """
        Make ONE clear decision.
        
        Priority (UPDATED - match CLI Ensemble logic):
        1. Use 'ensemble_main' if available (it handles metadata overrides correctly)
        2. Fallback to EfficientNet/Metadata logic if ensemble missing
        """
        eff = scores.get("efficientnet", 0.5)
        meta = scores.get("metadata", 0.5)
        ensemble = scores.get("ensemble_main", None)
        
        # PRIMARY RULE: Match CLI Ensemble Verdict Logic
        if ensemble is not None:
             verdict = "AI"
             if ensemble < 0.5:
                 verdict = "REAL"
             elif ensemble >= 0.4 and ensemble <= 0.6:
                 verdict = "UNCERTAIN"
                 
             return {
                "ai_probability": ensemble,
                "verdict": verdict,
                "confidence": "HIGH" if (ensemble > 0.8 or ensemble < 0.2) else "MEDIUM",
                "reason": "Ensemble analysis (Metadata + AI Models)",
                "scores": scores,
            }

        # Fallback logic (should rarely be reached if EfficientNet is running)
        freq = scores.get("frequency", 0.5)
        vit = scores.get("vit", 0.5)

        # Weighted average of available signals
        final = eff * 0.4 + freq * 0.2 + vit * 0.2 + meta * 0.2
        
        # Determine verdict - better thresholds
        if final > 0.60:
            verdict = "AI"
            confidence = "HIGH" if final > 0.80 else "MEDIUM"
        elif final < 0.40:
            verdict = "REAL"
            confidence = "HIGH" if final < 0.25 else "MEDIUM"
        else:
            verdict = "UNCERTAIN"
            confidence = "LOW"
        
        return {
            "ai_probability": final,
            "verdict": verdict,
            "confidence": confidence,
            "reason": "Camera metadata found" if meta < 0.4 else "Combined analysis",
            "scores": scores,
        }


def simple_detect(image_path: str) -> Dict:
    """Simple detection - returns dict with verdict."""
    detector = SimpleDetector(quiet=True)
    return detector.detect(image_path)


# CLI interface - CLEAN output
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detector = SimpleDetector(quiet=True)
    result = detector.detect(image_path)
    
    # Get verdict emoji
    emoji = "🤖" if result["verdict"] == "AI" else "📷" if result["verdict"] == "REAL" else "❓"
    
    # Simple, clean output
    print("\n" + "=" * 50)
    print(f"  {emoji} VERDICT: {result['verdict']}")
    print("=" * 50)
    print(f"  AI Probability: {result['ai_probability']:.0%}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Reason: {result['reason']}")
    print("=" * 50 + "\n")
