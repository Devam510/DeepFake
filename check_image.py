"""
Local Verification Tool for AI-Generated Image Risk Assessment
==============================================================

Usage:
    python check_image.py <image>                    # Run ALL detectors
    python check_image.py <image> --modern           # Modern AI only (DALL-E 3, Midjourney)
    python check_image.py <image> --cnn              # CNN only (StyleGAN faces)
    python check_image.py <image> --stat             # Statistical only

Detectors:
    - Modern: Best for DALL-E 3, Midjourney v6, Stable Diffusion 3
    - CNN: Best for StyleGAN-generated faces (99.7% accuracy)
    - Statistical: General AI detection (legacy)
"""

import sys
import os
import argparse
import numpy as np
import tempfile
from PIL import Image

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def check_with_modern(image_path: str) -> dict:
    """Check image using modern AI detector (DALL-E 3, Midjourney)."""
    try:
        from src.modeling.modern_ai_detector import detect_modern_ai

        result = detect_modern_ai(image_path)

        return {
            "probability": result.get("combined_probability", 0),
            "interpretation": result.get("verdict", "Unknown"),
            "confidence": (
                "HIGH confidence"
                if result.get("combined_probability", 0) > 0.7
                else "MEDIUM confidence"
            ),
            "model": "Modern AI Detector",
            "best_for": "DALL-E 3, Midjourney v6, Stable Diffusion 3",
            "metadata": result.get("metadata", {}),
        }
    except Exception as e:
        return {"error": str(e), "model": "Modern AI"}


def check_with_cnn(image_path: str) -> dict:
    """Check image using CNN detector (StyleGAN faces)."""
    try:
        from src.modeling.cnn_detector import ProductionCNNDetector

        detector = ProductionCNNDetector()
        result = detector.predict_from_image_path(image_path)
        return {
            "probability": result["synthetic_probability"],
            "interpretation": result["synthetic_likelihood"],
            "confidence": result["confidence_band"],
            "model": "CNN (ResNet-18)",
            "best_for": "StyleGAN face detection",
        }
    except Exception as e:
        return {"error": str(e), "model": "CNN"}


def check_with_statistical(image_path: str) -> dict:
    """Check image using GradientBoosting detector."""
    try:
        from extraction.image_signals import ImageSignalExtractor
        from production_detector import ProductionFrozenDetector

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
            "interpretation": result.synthetic_likelihood,
            "confidence": result.confidence_band,
            "model": "GradientBoosting (Statistical)",
            "best_for": "Legacy AI detection",
        }
    except Exception as e:
        return {"error": str(e), "model": "Statistical"}


def print_result(result: dict, header: str, emoji: str):
    """Print a single detector result."""
    print(f"\n  {emoji} {header}")
    print(f"  " + "-" * 50)

    if "error" in result:
        print(f"  ❌ Error: {result['error']}")
        return

    prob = result["probability"]
    print(f"  Synthetic Probability:  {prob:.1%}")
    print(f"  Interpretation:         {result['interpretation']}")
    print(f"  Best for:               {result['best_for']}")

    # Show metadata if available
    if "metadata" in result and result["metadata"].get("has_ai_watermark"):
        print(f"  ⚠️  AI WATERMARK DETECTED!")
        for indicator in result["metadata"].get("indicators", []):
            print(f"      • {indicator}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Generated Image Detection with multiple detectors."
    )
    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument(
        "--modern",
        action="store_true",
        help="Modern AI detector (DALL-E 3, Midjourney)",
    )
    parser.add_argument(
        "--cnn", action="store_true", help="CNN detector (StyleGAN faces)"
    )
    parser.add_argument(
        "--stat", action="store_true", help="Statistical detector (legacy)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Error: File not found: {args.image_path}")
        sys.exit(1)

    print(f"\n🔍 Analyzing: {os.path.basename(args.image_path)}")

    # Determine which detectors to run
    run_modern = args.modern or (not args.cnn and not args.stat)
    run_cnn = args.cnn
    run_stat = args.stat or (not args.modern and not args.cnn)

    print("\n" + "=" * 60)
    print("  AI-GENERATED IMAGE DETECTOR")
    print("=" * 60)

    results = {}

    # Run selected detectors
    if run_modern:
        results["modern"] = check_with_modern(args.image_path)
        print_result(results["modern"], "MODERN AI DETECTOR", "🚀")

    if run_stat:
        results["stat"] = check_with_statistical(args.image_path)
        print_result(results["stat"], "STATISTICAL DETECTOR", "📊")

    if run_cnn:
        results["cnn"] = check_with_cnn(args.image_path)
        print_result(results["cnn"], "CNN DETECTOR (Faces)", "🧠")

    # Show recommendation
    print("\n" + "=" * 60)
    print("  📋 VERDICT")
    print("=" * 60)

    # NEW: Check processing level
    processing_level = "unknown"
    processing_warning = None
    try:
        from src.extraction.image_processing_level import estimate_processing_level

        processing_result = estimate_processing_level(args.image_path)
        processing_level = processing_result.level
        processing_warning = processing_result.warning

        if processing_level != "minimal_processing":
            print(f"\n  📊 Processing Level: {processing_level.upper()}")
            if processing_warning:
                print(f"     ⚠️  {processing_warning}")
    except Exception:
        pass

    # Get best probability from available results
    max_prob = 0
    best_detector = None
    for name, result in results.items():
        if "error" not in result:
            prob = result.get("probability", 0)
            if prob > max_prob:
                max_prob = prob
                best_detector = result.get("model", name)

    # CRITICAL: Force UNCERTAIN for heavy processing
    if processing_level == "heavy_processing":
        print(f"\n  🤷 UNCERTAIN - Cannot reliably determine")
        print(f"     Reason: Heavy post-processing reduces detection reliability")
        print(f"     Raw probability was: {max_prob:.0%} (NOT to be trusted)")
    elif max_prob > 0.7:
        print(f"\n  🎯 HIGH LIKELIHOOD: AI-GENERATED ({max_prob:.0%})")
        print(f"     Detected by: {best_detector}")
        if processing_level == "moderate_processing":
            print("     ⚠️  Note: Moderate processing detected - confidence reduced")
    elif max_prob > 0.4:
        print(f"\n  ⚠️  MODERATE LIKELIHOOD: Possibly AI ({max_prob:.0%})")
        print(f"     Detected by: {best_detector}")
    else:
        print(f"\n  ✅ LOW LIKELIHOOD: Probably REAL ({max_prob:.0%})")
        if max_prob < 0.2:
            print("     Note: Very confident this is NOT AI-generated")

    print("\n  LIMITATIONS:")
    print("  • Detection is probabilistic, not definitive")
    print("  • Very sophisticated AI can evade detection")
    if processing_level == "heavy_processing":
        print("  • Heavy filtering/editing makes detection unreliable")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
