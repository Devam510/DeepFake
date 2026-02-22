"""
Ensemble AI Detector - Precise Detection
=========================================

Combines multiple detection methods to reduce false positives:
1. EfficientNet-B0 (trained research model, 1.7M images)
2. Statistical frequency analysis
3. Confidence weighting

Usage:
    python ensemble_detector.py <image_path>
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import filter detector for social media filter detection
try:
    from extraction.filter_detector import detect_social_media_filter

    FILTER_DETECTOR_AVAILABLE = True
except ImportError:
    FILTER_DETECTOR_AVAILABLE = False

# Import image processing level estimator
try:
    from extraction.image_processing_level import (
        estimate_processing_level,
        ProcessingLevelResult,
    )

    PROCESSING_LEVEL_AVAILABLE = True
except ImportError:
    PROCESSING_LEVEL_AVAILABLE = False


class EfficientNetDetector(nn.Module):
    """EfficientNet-based AI detector. Supports B0 and B4."""

    def __init__(self, model_name="b0"):
        super().__init__()
        self.model_name = model_name
        if model_name == "b4":
            self.backbone = models.efficientnet_b4(weights=None)
        else:  # default b0
            self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )
        self.temperature = nn.Parameter(torch.ones(1))


# ========================================
# PRODUCTION FIX: Global Model Cache
# ========================================
# Model is loaded ONCE at startup, not on every request.
# This ensures consistent inference and improves performance.

_cached_model = None
_cached_model_path = None


def _get_cached_model():
    """Get or load the cached EfficientNet model."""
    global _cached_model, _cached_model_path
    
    # PRODUCTION FIX: Use absolute path based on this file's location
    # This ensures the model is found regardless of current working directory
    base_dir = Path(__file__).parent.resolve()
    model_path = base_dir / "models" / "research" / "research_model_final.pth"
    
    if not model_path.exists():
        return None, f"Research model not found at {model_path}"
    
    # Check if model needs to be loaded or reloaded
    if _cached_model is None or _cached_model_path != str(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            # Auto-detect architecture from checkpoint (b0 or b4)
            model_name = checkpoint.get("model_name", "b4")  # old models default to b4
            model = EfficientNetDetector(model_name=model_name)
            model.load_state_dict(checkpoint["model_state"])
            model.temperature.data = torch.tensor([checkpoint["temperature"]])
            model.eval()  # Set to eval mode ONCE
            
            _cached_model = model
            _cached_model_path = str(model_path)
            print(f"[MODEL] Loaded and cached EfficientNet-{model_name.upper()} model")
        except Exception as e:
            return None, str(e)
    
    return _cached_model, None

def get_efficientnet_prediction(image_path: str) -> dict:
    """Get prediction from EfficientNet model (uses cached model for consistency)."""
    try:
        # PRODUCTION FIX: Use cached model instead of loading every time
        model, error = _get_cached_model()
        if model is None:
            return {"error": error or "Model not available", "probability": 0.5, "weight": 0.6}

        # Load and preprocess image (SAME as training)
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(img).unsqueeze(0)

        # Single forward pass with cached model (already in eval mode)
        with torch.no_grad():
            logits = model.backbone(img_tensor)
            probs = torch.softmax(logits / model.temperature, dim=1)
            ai_prob = probs[0, 1].item()

        return {
            "probability": ai_prob,
            "uncertainty": 0.0,
            "model": "EfficientNet-B0",
            "weight": 0.6,
        }
    except Exception as e:
        return {"error": str(e), "probability": 0.5, "weight": 0.6}



def get_statistical_prediction(image_path: str) -> dict:
    """Get prediction from statistical frequency analysis."""
    try:
        from extraction.image_signals import ImageSignalExtractor
        from production_detector import ProductionFrozenDetector
        import tempfile

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                img.save(temp_path, "JPEG")

        extractor = ImageSignalExtractor()
        signals = extractor.extract_all(temp_path)
        os.remove(temp_path)

        features = np.array(
            [
                signals.fft_magnitude_mean,
                signals.fft_magnitude_std,
                signals.fft_high_freq_ratio,
                signals.dct_coefficient_stats.get("dc_mean", 0),
                signals.dct_coefficient_stats.get("ac_mean", 0),
                signals.dct_coefficient_stats.get("ac_energy", 0),
                signals.entropy_mean,
                signals.entropy_std,
                signals.edge_gradient_mean,
                signals.edge_gradient_std,
                signals.edge_smoothness_score,
                signals.diffusion_residue_score,
            ]
        )

        detector = ProductionFrozenDetector()
        result = detector.predict(features)

        return {
            "probability": result.synthetic_probability,
            "model": "Statistical (Frequency)",
            "weight": 0.3,  # Secondary model weight
        }
    except Exception as e:
        return {"error": str(e), "probability": 0.5, "weight": 0.3}


def get_metadata_signals(image_path: str) -> dict:
    """Check image metadata for AI indicators."""
    try:
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        exif = img._getexif()

        has_exif = exif is not None and len(exif) > 5
        has_camera_make = False
        has_gps = False

        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "Make":
                    has_camera_make = True
                if tag == "GPSInfo":
                    has_gps = True

        # Real photos usually have rich EXIF data
        # AI images usually have minimal/no EXIF
        real_indicators = sum([has_exif, has_camera_make, has_gps])

        # More real indicators = lower AI probability
        ai_prob = 0.7 - (real_indicators * 0.2)
        ai_prob = max(0.1, min(0.9, ai_prob))

        return {
            "probability": ai_prob,
            "model": "Metadata Analysis",
            "weight": 0.1,  # Low weight, just a hint
            "has_exif": has_exif,
            "has_camera": has_camera_make,
            "has_gps": has_gps,
        }
    except Exception as e:
        return {"error": str(e), "probability": 0.5, "weight": 0.1}


def ensemble_predict(image_path: str) -> dict:
    """Combine all detectors with smart voting."""
    print("\n" + "=" * 60)
    print("  ENSEMBLE AI DETECTOR - PRECISE DETECTION")
    print("=" * 60)
    print(f"\n  Analyzing: {os.path.basename(image_path)}")

    results = {}

    # Get predictions from all models
    print("\n  [1/5] Running EfficientNet-B0...")
    results["efficientnet"] = get_efficientnet_prediction(image_path)

    print("  [2/5] Running Statistical Analysis...")
    results["statistical"] = get_statistical_prediction(image_path)

    print("  [3/5] Checking Metadata...")
    results["metadata"] = get_metadata_signals(image_path)

    # Check for social media filters
    print("  [4/5] Checking for Social Media Filters...")
    filter_result = None
    if FILTER_DETECTOR_AVAILABLE:
        try:
            filter_result = detect_social_media_filter(image_path)
            results["filter"] = {
                "detected": filter_result.filter_detected,
                "confidence": filter_result.filter_confidence,
                "type": filter_result.filter_type,
                "indicators": filter_result.filter_indicators,
            }
        except Exception as e:
            results["filter"] = {"error": str(e), "detected": False}
    else:
        results["filter"] = {
            "error": "Filter detector not available",
            "detected": False,
        }

    # NEW: Estimate image processing level (filters, enhancement, editing)
    print("  [5/5] Analyzing Processing Level...")
    processing_result = None
    processing_level = "unknown"
    processing_warning = None
    if PROCESSING_LEVEL_AVAILABLE:
        try:
            processing_result = estimate_processing_level(image_path)
            processing_level = processing_result.level
            processing_warning = processing_result.warning
            results["processing"] = {
                "level": processing_level,
                "confidence": processing_result.confidence,
                "indicators": processing_result.indicators,
                "warning": processing_warning,
            }
        except Exception as e:
            results["processing"] = {"error": str(e), "level": "unknown"}
    else:
        results["processing"] = {
            "error": "Processing level detector not available",
            "level": "unknown",
        }

    # Print individual results
    print("\n" + "-" * 60)
    print("  INDIVIDUAL MODEL RESULTS")
    print("-" * 60)

    for name, result in results.items():
        if name == "filter":
            if result.get("detected"):
                print(
                    f"  {'Filter Detection':25s}: {result['type']} ({result['confidence']:.0%} confidence)"
                )
            elif "error" not in result:
                print(f"  {'Filter Detection':25s}: No filter detected")
        elif name == "processing":
            # Processing result has 'level' not 'probability' - show separately
            if "error" not in result:
                level = result.get("level", "unknown")
                print(f"  {'Processing Level':25s}: {level}")
        elif "error" not in result:
            prob = result["probability"]
            model = result["model"]
            uncertainty = result.get("uncertainty", 0)
            unc_str = f" (±{uncertainty:.1%})" if uncertainty > 0 else ""
            print(f"  {model:25s}: {prob:.1%}{unc_str}")
        else:
            print(f"  {result.get('model', name):25s}: ERROR - {result['error']}")

    # Smart voting logic
    efficientnet = results.get("efficientnet", {})
    statistical = results.get("statistical", {})
    metadata = results.get("metadata", {})
    filter_info = results.get("filter", {})

    eff_prob = efficientnet.get("probability", 0.5)
    stat_prob = statistical.get("probability", 0.5)

    # Check metadata for strong real indicators and AI probability
    has_camera = metadata.get("has_camera", False)
    has_gps = metadata.get("has_gps", False)
    strong_real_metadata = has_camera or has_gps
    meta_prob = metadata.get("probability", 0.5)  # Metadata's AI probability estimate

    # Check filter detection
    filter_detected = filter_info.get("detected", False)
    filter_confidence = filter_info.get("confidence", 0.0)
    filter_type = filter_info.get("type", "none")

    # Calculate disagreement first
    disagreement = abs(eff_prob - stat_prob)

    # Check if filter is from a KNOWN social media app (not just generic patterns)
    known_filter_apps = [
        "instagram",
        "snapchat",
        "tiktok",
        "vsco",
        "snapseed",
        "facetune",
        "beauty_plus",
        "b612",
    ]
    is_known_app_filter = filter_type in known_filter_apps
    
    # PRODUCTION FIX: Generic "other" filters with multiple visual indicators
    # are very likely social media filters (Snapchat/Instagram) where the
    # filename and EXIF don't identify the app (e.g. saved as "ph.jpg")
    filter_indicators = filter_info.get("indicators", [])
    num_filter_indicators = len(filter_indicators)
    has_beautification = any("beautification" in ind.lower() for ind in filter_indicators)
    is_strong_generic_filter = (
        filter_detected and
        not is_known_app_filter and
        (num_filter_indicators >= 3 or (num_filter_indicators >= 2 and has_beautification))
    )

    # Decision logic (PRODUCTION priority order):
    # 1. Camera/GPS metadata = DEFINITIVE real photo
    # 2. Known social media app filter with HIGH confidence = DEFINITIVE filtered real photo
    #    (Snapchat/Instagram filename or metadata = trust it, NOT AI)
    # 3. EfficientNet 99%+ = Strong AI indicator (only if no real indicators)
    # 4. Known app filter with MEDIUM confidence = Possible filtered photo
    # 5. Model agreement logic
    # 6. Disagreement handling

    if strong_real_metadata:
        # PRIORITY 1: Camera/GPS metadata = definitive real photo
        ensemble_prob = min(eff_prob * 0.3, 0.3)  # Cap at 30%
        decision_source = "Metadata (camera/GPS detected)"
    elif is_known_app_filter and filter_confidence > 0.7:
        # PRIORITY 2: Known social media app filter with HIGH confidence
        # Snapchat/Instagram/TikTok confirmed = this is a filtered REAL photo
        # DO NOT trust AI detection on these - filters cause false positives
        ensemble_prob = min(eff_prob * 0.2, 0.25)  # Cap at 25%
        decision_source = (
            f"Known app: {filter_type} ({filter_confidence:.0%} confidence)"
        )
    elif eff_prob > 0.99 and not is_known_app_filter and not is_strong_generic_filter:
        # PRIORITY 3: EfficientNet EXTREMELY confident it's AI
        # Only trust this if NO filter detected (known or generic with strong evidence)
        ensemble_prob = eff_prob
        decision_source = "EfficientNet (99%+ confidence - modern AI expert)"
    elif is_known_app_filter and filter_confidence > 0.4:
        # PRIORITY 4: Known social media app filter with MEDIUM confidence
        ensemble_prob = min(eff_prob * 0.4, 0.45)  # Cap at 45%
        decision_source = f"Possible {filter_type} filter"
    elif is_strong_generic_filter:
        # PRIORITY 4b: Generic filter with strong visual evidence
        # This catches Snapchat/Instagram photos saved without app-identifying metadata
        # (e.g. saved as "ph.jpg" with no EXIF app signature)
        if has_beautification:
            # Beautification = face filter = very likely social media
            ensemble_prob = min(eff_prob * 0.25, 0.30)  # Cap at 30%
            decision_source = f"Beautification filter detected ({num_filter_indicators} indicators)"
        else:
            # Multiple visual filter indicators but no beautification
            ensemble_prob = min(eff_prob * 0.35, 0.40)  # Cap at 40%
            decision_source = f"Strong filter evidence ({num_filter_indicators} indicators, {filter_confidence:.0%})"
    elif eff_prob > 0.95 and stat_prob > 0.5:
        # BOTH models agree it's likely AI (and no filter detected)
        ensemble_prob = eff_prob
        decision_source = "High agreement (both say AI)"
    elif eff_prob < 0.15 and stat_prob < 0.5:
        # BOTH agree it's likely real
        ensemble_prob = eff_prob
        decision_source = "High agreement (both say real)"
    elif disagreement > 0.6 and filter_detected:
        # HIGH disagreement + filter patterns
        # KEY INSIGHT: AI images (ChatGPT, Midjourney) often have vignette/warm tones
        # naturally — we should not suppress EfficientNet if multiple signals agree it's AI
        meta_also_says_ai = meta_prob > 0.6 and not strong_real_metadata
        eff_clearly_ai = eff_prob > 0.6
        if eff_clearly_ai and meta_also_says_ai:
            # EfficientNet + Metadata both say AI, no camera EXIF
            # Filter is noise (AI images can have vignette/warm characteristics)
            ensemble_prob = (eff_prob * 0.7 + meta_prob * 0.3)  # Trust both signals
            decision_source = f"EfficientNet + Metadata agree AI (filter noise ignored)"
        elif num_filter_indicators >= 3 or filter_confidence > 0.5:
            # Strong filter evidence AND EfficientNet is ambiguous — aggressive reduction
            ensemble_prob = min(eff_prob * 0.3, 0.35)  # Cap at 35%
            decision_source = f"Filter patterns detected ({num_filter_indicators} indicators, {filter_confidence:.0%})"
        else:
            # Moderate filter evidence
            ensemble_prob = min(eff_prob * 0.5, 0.50)  # Cap at 50%
            decision_source = f"Possible filter effects ({filter_confidence:.0%})"
    elif disagreement > 0.6:
        # HIGH disagreement without strong filter evidence
        ensemble_prob = min(eff_prob, stat_prob) * 0.6 + max(eff_prob, stat_prob) * 0.4
        decision_source = f"Conservative (disagreement {disagreement:.0%})"
    else:
        # Moderate agreement - weighted average
        ensemble_prob = eff_prob * 0.6 + stat_prob * 0.4
        decision_source = "Weighted ensemble"

    # Final verdict
    print("\n" + "-" * 60)
    print("  ENSEMBLE VERDICT")
    print("-" * 60)
    print(f"  Combined AI Probability: {ensemble_prob:.1%}")
    print(f"  Decision Source:         {decision_source}")

    if disagreement > 0.4:
        print(f"  Model Disagreement:      {disagreement:.1%}")

    # Determine verdict - be honest about uncertainty
    # CRITICAL: Production-safe logic - force UNCERTAIN when detection is unreliable
    # The AI probability value is NOT modified, only the interpretation

    # HYBRID UNCERTAIN LOGIC (Production-Safe):
    # 1. Heavy processing detected (existing)
    # 2. Strong filter + high disagreement (new)
    force_uncertain_due_to_processing = processing_level == "heavy_processing"

    # PRODUCTION FIX: Strong filter detection + model disagreement = unreliable
    # If models disagree AND visible filter effects detected, be conservative
    # Lowered threshold from 0.5 to 0.35 — real photos with filters at 40% confidence
    # were being missed (e.g. Snapchat selfies saved as "ph.jpg")
    # Whether both EfficientNet and metadata agree it's AI (used to suppress force_uncertain)
    meta_also_says_ai = meta_prob > 0.6 and not strong_real_metadata
    eff_and_meta_agree_ai = eff_prob > 0.6 and meta_also_says_ai

    force_uncertain_due_to_filter = (
        filter_detected and filter_confidence > 0.55 and disagreement > 0.55
        and not strong_real_metadata   # Camera/GPS = DEFINITIVE real
        and not eff_and_meta_agree_ai  # EfficientNet + Metadata both say AI = trust them
    )
    
    # Also force uncertain if beautification detected + high EfficientNet
    force_uncertain_due_to_beautification = (
        has_beautification and eff_prob > 0.85 and disagreement > 0.5
        and not strong_real_metadata   # Camera/GPS = DEFINITIVE real
        and not eff_and_meta_agree_ai  # EfficientNet + Metadata both say AI = trust them
    )

    force_uncertain = (
        force_uncertain_due_to_processing or 
        force_uncertain_due_to_filter or
        force_uncertain_due_to_beautification
    )

    if force_uncertain:
        # PRODUCTION-SAFE: Force UNCERTAIN for unreliable detections
        # DO NOT claim real or fake, DO NOT override probability
        verdict = "UNCERTAIN"
        confidence = "LOW"
        if force_uncertain_due_to_processing:
            print("  [!] Heavy post-processing detected - forcing UNCERTAIN verdict")
            print("      Reason: Filters/enhancement reduce detection reliability")
        else:
            print(
                "  [!] Strong filter effects + model disagreement - forcing UNCERTAIN"
            )
            print(
                f"      Reason: {filter_type} filter ({filter_confidence:.0%}) + {disagreement:.0%} disagreement"
            )
    elif ensemble_prob > 0.85:
        # Very high probability - confident it's AI
        verdict = "AI-GENERATED"
        confidence = "HIGH"
        if disagreement > 0.6:
            print(
                "  [!] Note: Model disagreement detected but probability is very high"
            )
        # Widen confidence for moderate processing
        if processing_level == "moderate_processing":
            confidence = "MEDIUM"
            print("  [!] Note: Moderate processing detected - confidence reduced")
    elif ensemble_prob < 0.25:
        # Very low probability - confident it's real
        verdict = "LIKELY REAL"
        confidence = "HIGH"
        if disagreement > 0.6:
            print("  [!] Note: Model disagreement detected but probability is very low")
        # Widen confidence for moderate processing
        if processing_level == "moderate_processing":
            confidence = "MEDIUM"
            print("  [!] Note: Moderate processing detected - confidence reduced")
    elif disagreement > 0.75 and not strong_real_metadata and not is_known_app_filter:
        # EXTREME disagreement with no clear indicators
        # Only uncertain for mid-range probabilities
        verdict = "UNCERTAIN"
        confidence = "LOW"
        print("  [!] High model conflict - cannot make reliable determination")
    elif ensemble_prob > 0.6:
        verdict = "AI-GENERATED"
        confidence = "MEDIUM" if processing_level == "moderate_processing" else "MEDIUM"
    elif ensemble_prob > 0.4:
        verdict = "POSSIBLY AI"
        confidence = "MEDIUM" if processing_level != "moderate_processing" else "LOW"
    else:
        verdict = "LIKELY REAL"
        confidence = "MEDIUM" if processing_level != "moderate_processing" else "LOW"

    print(f"  Confidence:              {confidence}")
    print(f"\n  FINAL VERDICT: {verdict}")

    # Show metadata hints if available
    if has_camera:
        print("  [Hint] Camera manufacturer detected in EXIF")
    if has_gps:
        print("  [Hint] GPS coordinates found - likely real photo")

    # NEW: Show filter hints
    if filter_detected:
        print(f"  [Hint] Social media filter detected ({filter_type})")
        for indicator in filter_info.get("indicators", [])[:3]:  # Show up to 3
            print(f"         • {indicator}")

    # Show processing level if not minimal
    if processing_level != "minimal_processing" and processing_level != "unknown":
        print(f"  [Processing] Level: {processing_level}")
        if processing_warning:
            print(f"  [Warning] {processing_warning}")
        processing_info = results.get("processing", {})
        for indicator in processing_info.get("indicators", [])[:3]:
            print(f"         • {indicator}")

    print("=" * 60 + "\n")

    return {
        "ensemble_probability": ensemble_prob,
        "verdict": verdict,
        "confidence": confidence,
        "decision_source": decision_source,
        "disagreement": disagreement,
        "filter_detected": filter_detected,
        "filter_type": filter_type,
        "filter_confidence": filter_confidence,
        # NEW: Processing level information
        "image_processing_level": processing_level,
        "processing_warning": processing_warning,
        "individual_results": results,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    result = ensemble_predict(image_path)
