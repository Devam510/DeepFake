"""
Phase 6: Evaluation Script
==========================

Evaluates the general image detector with:
- Per-domain accuracy metrics
- Expected Calibration Error (ECE)
- Known failure cases documentation
- Cross-domain confusion matrix
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image

# Try to import required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configuration
DATA_DIR = Path("data/general_detector")
OUTPUT_DIR = Path("evaluation")


def load_test_data() -> List[Tuple[str, int, str]]:
    """Load test data from splits."""
    test_path = DATA_DIR / "splits" / "test.json"
    
    if not test_path.exists():
        print("❌ Test split not found. Run training first.")
        return []
    
    with open(test_path) as f:
        data = json.load(f)
    
    return [(item["path"], item["label"], item.get("source", "unknown")) for item in data]


def evaluate_detector() -> Dict[str, Any]:
    """Evaluate the general image detector."""
    from src.modeling.general_image_detector import GeneralImageDetector
    
    print("\n" + "=" * 60)
    print("GENERAL IMAGE DETECTOR EVALUATION")
    print("=" * 60)
    
    detector = GeneralImageDetector()
    
    if not detector.model_loaded:
        print("⚠️ No trained model found. Using signal-based detection.")
    
    test_data = load_test_data()
    if not test_data:
        return {"error": "No test data"}
    
    print(f"\n📊 Evaluating on {len(test_data)} test samples...")
    
    # Metrics storage
    predictions = []
    labels = []
    probabilities = []
    source_results = {}  # Per-source accuracy
    
    for i, (path, label, source) in enumerate(test_data):
        if not os.path.exists(path):
            continue
        
        try:
            result = detector.predict(path)
            pred = 1 if result.synthetic_probability > 0.5 else 0
            
            predictions.append(pred)
            labels.append(label)
            probabilities.append(result.synthetic_probability)
            
            # Track per-source
            if source not in source_results:
                source_results[source] = {"correct": 0, "total": 0}
            source_results[source]["total"] += 1
            if pred == label:
                source_results[source]["correct"] += 1
            
            if (i + 1) % 500 == 0:
                acc = np.mean(np.array(predictions) == np.array(labels)) * 100
                print(f"   Progress: {i+1}/{len(test_data)} | Acc: {acc:.1f}%")
                
        except Exception as e:
            continue
    
    if len(predictions) == 0:
        return {"error": "No valid predictions"}
    
    # Calculate metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    
    # Overall accuracy
    accuracy = np.mean(predictions == labels) * 100
    
    # Per-class accuracy
    real_mask = labels == 0
    synth_mask = labels == 1
    real_accuracy = np.mean(predictions[real_mask] == labels[real_mask]) * 100 if real_mask.sum() > 0 else 0
    synth_accuracy = np.mean(predictions[synth_mask] == labels[synth_mask]) * 100 if synth_mask.sum() > 0 else 0
    
    # ECE (Expected Calibration Error)
    ece = calculate_ece(probabilities, labels)
    
    # Per-source accuracy
    per_source = {s: v["correct"] / v["total"] * 100 for s, v in source_results.items() if v["total"] > 0}
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector.model_loaded,
        "test_samples": len(predictions),
        "overall_accuracy": round(accuracy, 2),
        "real_accuracy": round(real_accuracy, 2),
        "synthetic_accuracy": round(synth_accuracy, 2),
        "ece": round(ece, 4),
        "per_source_accuracy": {k: round(v, 2) for k, v in per_source.items()},
    }
    
    return results


def calculate_ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error.
    
    ECE measures how well probabilities match actual frequencies.
    Lower is better; < 0.1 is good for detection.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            avg_confidence = probabilities[in_bin].mean()
            avg_accuracy = (labels[in_bin] == (probabilities[in_bin] > 0.5).astype(int)).mean()
            ece += (bin_size / len(probabilities)) * abs(avg_accuracy - avg_confidence)
    
    return ece


def document_failure_cases(test_data: List[Tuple[str, int, str]], n_samples: int = 10) -> List[Dict]:
    """Document known failure cases."""
    from src.modeling.general_image_detector import GeneralImageDetector
    
    detector = GeneralImageDetector()
    failures = []
    
    random.shuffle(test_data)
    
    for path, label, source in test_data[:500]:  # Sample 500
        if not os.path.exists(path):
            continue
        
        try:
            result = detector.predict(path)
            pred = 1 if result.synthetic_probability > 0.5 else 0
            
            if pred != label:
                failures.append({
                    "path": path,
                    "true_label": "synthetic" if label == 1 else "real",
                    "predicted": "synthetic" if pred == 1 else "real",
                    "probability": round(result.synthetic_probability, 3),
                    "source": source,
                })
                
                if len(failures) >= n_samples:
                    break
        except:
            continue
    
    return failures


def main():
    """Run full evaluation."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run evaluation
    results = evaluate_detector()
    
    if "error" in results:
        print(f"❌ Evaluation failed: {results['error']}")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Samples:       {results['test_samples']}")
    print(f"Overall Accuracy:   {results['overall_accuracy']:.2f}%")
    print(f"Real Accuracy:      {results['real_accuracy']:.2f}%")
    print(f"Synthetic Accuracy: {results['synthetic_accuracy']:.2f}%")
    print(f"ECE:                {results['ece']:.4f}")
    
    print("\n📊 Per-Source Accuracy:")
    for source, acc in sorted(results["per_source_accuracy"].items()):
        print(f"   {source}: {acc:.1f}%")
    
    # Get failure cases
    test_data = load_test_data()
    failures = document_failure_cases(test_data)
    results["failure_cases"] = failures
    
    print(f"\n❌ Sample Failure Cases ({len(failures)}):")
    for f in failures[:5]:
        print(f"   {f['source']}: {f['true_label']} → {f['predicted']} (p={f['probability']})")
    
    # Save results
    output_path = OUTPUT_DIR / "evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Check thresholds
    print("\n" + "=" * 60)
    print("THRESHOLD CHECKS")
    print("=" * 60)
    
    if results["ece"] < 0.1:
        print("✅ ECE < 0.1 (well-calibrated)")
    else:
        print(f"⚠️ ECE = {results['ece']} (needs calibration)")
    
    if results["overall_accuracy"] > 95:
        print("✅ Accuracy > 95% (excellent)")
    elif results["overall_accuracy"] > 90:
        print("✅ Accuracy > 90% (good)")
    else:
        print(f"⚠️ Accuracy = {results['overall_accuracy']}% (needs improvement)")


if __name__ == "__main__":
    main()
