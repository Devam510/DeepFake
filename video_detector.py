"""
Video Detector — Main Orchestrator for Video Deepfake Detection
================================================================

Combines all video analysis modules:
1. Frame-level EfficientNet predictions
2. Temporal signals (SSIM, jitter, puppet body, jawline, flow)
3. Biological signals (heartbeat, blinks, skin, micro-expressions)
4. Audio analysis (voice authenticity, lip sync)

Usage:
  python video_detector.py video.mp4
  python video_detector.py video.mp4 --json
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Import our modules
from video_processor import extract_frames_for_analysis, get_video_info
from temporal_signals import analyze_temporal_signals
from biological_signals import analyze_biological_signals
from audio_analyzer import analyze_audio

# Try to import the trained EfficientNet model
try:
    from ensemble_detector import get_efficientnet_prediction
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False

# Try to import video meta-voter
try:
    from video_meta_voter import VideoMetaVoter
    _voter = VideoMetaVoter()
    META_VOTER_AVAILABLE = _voter.is_trained()
except ImportError:
    META_VOTER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME-LEVEL AI DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def analyze_frames_with_model(frames: list, temp_dir: str = None) -> Dict:
    """
    Run EfficientNet on sampled frames to get per-frame AI probability.
    Returns aggregate statistics across all frames.
    """
    if not EFFICIENTNET_AVAILABLE or not frames:
        return {"mean_prob": 0.5, "max_prob": 0.5, "voted_fake": 0.5, "num_frames": 0}

    import tempfile
    temp_dir = temp_dir or tempfile.mkdtemp()
    probabilities = []

    for i, frame in enumerate(frames[:20]):  # Sample up to 20 frames for speed
        path = os.path.join(temp_dir, f"_frame_{i}.jpg")
        cv2.imwrite(path, frame)
        try:
            result = get_efficientnet_prediction(path)
            if result and "probability" in result:
                probabilities.append(result["probability"])
        except Exception:
            pass
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    if not probabilities:
        return {"mean_prob": 0.5, "max_prob": 0.5, "voted_fake": 0.5, "num_frames": 0}

    arr = np.array(probabilities)
    return {
        "mean_prob": round(float(np.mean(arr)), 4),
        "max_prob": round(float(np.max(arr)), 4),
        "median_prob": round(float(np.median(arr)), 4),
        "std_prob": round(float(np.std(arr)), 4),
        "voted_fake": round(float(np.mean(arr > 0.5)), 4),
        "num_frames": len(probabilities),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DETECTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def detect_video(video_path: str) -> Dict:
    """
    Full video deepfake detection pipeline.

    Args:
        video_path: Path to video file

    Returns:
        dict with verdict, probability, and all signal details
    """
    start_time = time.time()

    if not os.path.exists(video_path):
        return {"error": f"File not found: {video_path}"}

    # ── Step 1: Get video metadata ───────────────────────────────────────────
    print(f"  🎬 Analyzing: {os.path.basename(video_path)}")
    info = get_video_info(video_path)
    fps = info.fps or 30.0
    print(f"     Duration: {info.duration:.1f}s | FPS: {fps:.0f} | Size: {info.width}x{info.height}")

    # ── Step 2: Extract frames ───────────────────────────────────────────────
    print(f"  📸 Extracting frames at 2 FPS...")
    frames = extract_frames_for_analysis(video_path, sample_fps=2)
    print(f"     Extracted {len(frames)} frames")

    if not frames:
        return {"error": "Could not extract frames from video"}

    # ── Step 3: Run all analyzers ────────────────────────────────────────────
    print(f"  🔍 Running frame-level AI detection...")
    frame_result = analyze_frames_with_model(frames)

    print(f"  ⏱️  Analyzing temporal signals...")
    temporal_result = analyze_temporal_signals(frames)

    print(f"  💓 Analyzing biological signals...")
    biological_result = analyze_biological_signals(frames, fps)

    print(f"  🔊 Analyzing audio signals...")
    audio_result = analyze_audio(video_path, frames, fps)

    # ── Step 4: Combine scores ───────────────────────────────────────────────
    all_scores = {
        "frame_ai_prob": frame_result["mean_prob"],
        "temporal_score": temporal_result["temporal_score"],
        "biological_score": biological_result["biological_score"],
        "audio_score": audio_result["audio_score"],
    }

    # Check if trained meta-voter is available
    if META_VOTER_AVAILABLE:
        # Use trained meta-voter for optimal combination
        features = {
            **all_scores,
            **temporal_result.get("individual_scores", {}),
            **biological_result.get("individual_scores", {}),
            **audio_result.get("individual_scores", {}),
            **frame_result,
        }
        final_probability = _voter.predict(features)
        method = "trained_meta_voter"
    else:
        # Weighted average fallback
        weights = {
            "frame_ai_prob": 0.35,
            "temporal_score": 0.25,
            "biological_score": 0.20,
            "audio_score": 0.20,
        }
        final_probability = sum(all_scores[k] * weights[k] for k in all_scores)
        method = "weighted_average"

    final_probability = min(1.0, max(0.0, final_probability))

    # ── Step 5: Generate verdict ─────────────────────────────────────────────
    if final_probability >= 0.75:
        verdict = "AI-GENERATED"
        confidence = "HIGH"
    elif final_probability >= 0.55:
        verdict = "LIKELY AI-GENERATED"
        confidence = "MEDIUM"
    elif final_probability >= 0.40:
        verdict = "UNCERTAIN"
        confidence = "LOW"
    elif final_probability >= 0.25:
        verdict = "LIKELY REAL"
        confidence = "MEDIUM"
    else:
        verdict = "REAL"
        confidence = "HIGH"

    elapsed = time.time() - start_time

    # ── Build signal cards (same format as photo detector for web UI) ────────
    signals = []

    # Frame-level signal
    signals.append({
        "name": "Frame AI Detection",
        "score": frame_result["mean_prob"],
        "weight": 0.35,
        "description": f"EfficientNet: {frame_result['num_frames']} frames analyzed, "
                       f"{frame_result['voted_fake']*100:.0f}% voted fake",
        "icon": "🖼️",
    })

    # Temporal signals
    ts = temporal_result.get("individual_scores", {})
    temporal_details = []
    if ts.get("ssim_anomaly", 0) > 0.3:
        temporal_details.append("frame flicker detected")
    if ts.get("puppet_body", 0) > 0.3:
        temporal_details.append("puppet body detected")
    if ts.get("jawline_glow", 0) > 0.3:
        temporal_details.append("jawline glow detected")
    signals.append({
        "name": "Temporal Analysis",
        "score": temporal_result["temporal_score"],
        "weight": 0.25,
        "description": ", ".join(temporal_details) if temporal_details else "No temporal anomalies",
        "icon": "⏱️",
    })

    # Biological signals
    bs = biological_result.get("details", {})
    bio_details = []
    hb = bs.get("heartbeat", {})
    if hb.get("has_pulse"):
        bio_details.append(f"pulse detected ({hb.get('estimated_bpm', 0)} BPM)")
    else:
        bio_details.append("no pulse signal")
    bl = bs.get("blink_rate", {})
    if bl.get("blink_count", 0) > 0:
        bio_details.append(f"{bl.get('blinks_per_min', 0):.0f} blinks/min")
    signals.append({
        "name": "Biological Signals",
        "score": biological_result["biological_score"],
        "weight": 0.20,
        "description": ", ".join(bio_details),
        "icon": "💓",
    })

    # Audio signals
    au = audio_result.get("details", {})
    sync = au.get("lip_sync", {})
    audio_desc = f"Lip sync: {sync.get('lag_ms', 0)}ms lag"
    if sync.get("correlation", 0) > 0.5:
        audio_desc += " (good sync)"
    else:
        audio_desc += " (poor sync)"
    signals.append({
        "name": "Audio Analysis",
        "score": audio_result["audio_score"],
        "weight": 0.20,
        "description": audio_desc,
        "icon": "🔊",
    })

    result = {
        "verdict": verdict,
        "confidence": confidence,
        "probability": round(final_probability, 4),
        "method": method,
        "signals": signals,
        "scores": all_scores,
        "elapsed_seconds": round(elapsed, 1),
        "video_info": {
            "path": video_path,
            "duration": round(info.duration, 1),
            "fps": fps,
            "resolution": f"{info.width}x{info.height}",
            "frames_analyzed": len(frames),
        },
        "details": {
            "frame_analysis": frame_result,
            "temporal": temporal_result,
            "biological": biological_result,
            "audio": audio_result,
        },
    }

    # Print summary
    print(f"\n  {'=' * 50}")
    print(f"  Verdict: {verdict} ({confidence} confidence)")
    print(f"  AI Probability: {final_probability:.1%}")
    print(f"  {'=' * 50}")
    for sig in signals:
        bar = "█" * int(sig["score"] * 20) + "░" * (20 - int(sig["score"] * 20))
        print(f"  {sig['icon']} {sig['name']:25s} {bar} {sig['score']:.2f}")
    print(f"  ⏱️  Analysis time: {elapsed:.1f}s")
    print(f"  {'=' * 50}\n")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_detector.py <video_path> [--json]")
        print("\nSupported formats: .mp4, .avi, .mov, .webm, .mkv")
        sys.exit(1)

    video_path = sys.argv[1]
    output_json = "--json" in sys.argv

    result = detect_video(video_path)

    if output_json:
        print(json.dumps(result, indent=2))
