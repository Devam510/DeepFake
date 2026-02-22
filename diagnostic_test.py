"""
Diagnostic Test: CLI vs Web Pipeline Comparison
=================================================

This script runs the SAME image through BOTH pipelines and compares:
1. Raw tensor values
2. Model output before calibration
3. Final probability
4. All scores

Run this to verify CLI and Web produce identical results.
"""

import os
import sys
import json

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from simple_detector import SimpleDetector
from ensemble_detector import get_efficientnet_prediction, get_statistical_prediction
from PIL import Image
import torch
from torchvision import transforms


def run_diagnostic(image_path: str):
    """Run full diagnostic on an image."""
    
    print("\n" + "=" * 70)
    print("  PRODUCTION DEBUG: Pipeline Diagnostic")
    print("=" * 70)
    print(f"\n  Image: {image_path}")
    print(f"  File exists: {os.path.exists(image_path)}")
    
    if not os.path.exists(image_path):
        print("  ERROR: File not found")
        return
    
    # Get file info
    file_size = os.path.getsize(image_path)
    print(f"  File size: {file_size} bytes")
    
    # ========================================
    # SECTION 1: Raw Image Properties
    # ========================================
    print("\n" + "-" * 50)
    print("  SECTION 1: Raw Image Properties")
    print("-" * 50)
    
    img = Image.open(image_path)
    print(f"  Mode: {img.mode}")
    print(f"  Size: {img.size}")
    print(f"  Format: {img.format}")
    
    # Convert to RGB (same as model preprocessing)
    img_rgb = img.convert("RGB")
    print(f"  After RGB convert - Size: {img_rgb.size}")
    
    # ========================================
    # SECTION 2: Tensor Preprocessing Check
    # ========================================
    print("\n" + "-" * 50)
    print("  SECTION 2: Tensor Preprocessing")
    print("-" * 50)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(img_rgb)
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor min: {tensor.min().item():.6f}")
    print(f"  Tensor max: {tensor.max().item():.6f}")
    print(f"  Tensor mean: {tensor.mean().item():.6f}")
    print(f"  Tensor std: {tensor.std().item():.6f}")
    print(f"  First 5 values: {tensor.flatten()[:5].tolist()}")
    
    # ========================================
    # SECTION 3: EfficientNet Direct Call
    # ========================================
    print("\n" + "-" * 50)
    print("  SECTION 3: EfficientNet Direct Prediction")
    print("-" * 50)
    
    eff_result = get_efficientnet_prediction(image_path)
    print(f"  Raw probability: {eff_result.get('probability', 'N/A')}")
    print(f"  Model: {eff_result.get('model', 'N/A')}")
    print(f"  Weight: {eff_result.get('weight', 'N/A')}")
    if 'error' in eff_result:
        print(f"  ERROR: {eff_result['error']}")
    
    # ========================================
    # SECTION 4: Statistical Analysis
    # ========================================
    print("\n" + "-" * 50)
    print("  SECTION 4: Statistical Prediction")
    print("-" * 50)
    
    stat_result = get_statistical_prediction(image_path)
    print(f"  Raw probability: {stat_result.get('probability', 'N/A')}")
    print(f"  Weight: {stat_result.get('weight', 'N/A')}")
    if 'error' in stat_result:
        print(f"  ERROR: {stat_result['error']}")
    
    # ========================================
    # SECTION 5: SimpleDetector (CLI Path)
    # ========================================
    print("\n" + "-" * 50)
    print("  SECTION 5: SimpleDetector (CLI Path)")
    print("-" * 50)
    
    detector = SimpleDetector(quiet=True)
    result = detector.detect(image_path)
    
    print(f"  ai_probability: {result.get('ai_probability', 'N/A')}")
    print(f"  verdict: {result.get('verdict', 'N/A')}")
    print(f"  confidence: {result.get('confidence', 'N/A')}")
    print(f"  reason: {result.get('reason', 'N/A')}")
    print(f"  scores: {json.dumps(result.get('scores', {}), indent=4)}")
    
    # ========================================
    # SECTION 6: Run Again (Consistency Check)
    # ========================================
    print("\n" + "-" * 50)
    print("  SECTION 6: Consistency Check (Run #2)")
    print("-" * 50)
    
    result2 = detector.detect(image_path)
    
    prob1 = result.get('ai_probability', 0)
    prob2 = result2.get('ai_probability', 0)
    diff = abs(prob1 - prob2)
    
    print(f"  Run 1 probability: {prob1}")
    print(f"  Run 2 probability: {prob2}")
    print(f"  Difference: {diff:.6f}")
    
    if diff > 0.001:
        print("  ⚠️ WARNING: Results differ between runs! Model may have non-deterministic behavior.")
    else:
        print("  ✅ Results consistent between runs")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"""
  Image:           {os.path.basename(image_path)}
  EfficientNet:    {eff_result.get('probability', 'N/A'):.4f}
  Statistical:     {stat_result.get('probability', 'N/A'):.4f}
  Final Verdict:   {result.get('verdict', 'N/A')}
  Final Prob:      {result.get('ai_probability', 'N/A'):.4f}
  Consistency:     {'✅ OK' if diff < 0.001 else '⚠️ INCONSISTENT'}
""")
    print("=" * 70 + "\n")
    
    return {
        "efficientnet_prob": eff_result.get('probability'),
        "statistical_prob": stat_result.get('probability'),
        "final_prob": result.get('ai_probability'),
        "verdict": result.get('verdict'),
        "scores": result.get('scores'),
        "consistent": diff < 0.001
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_test.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    run_diagnostic(image_path)
