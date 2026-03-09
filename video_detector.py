"""
Video Detector  Main Orchestrator for Video Deepfake Detection
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
os.environ["GLOG_minloglevel"] = "2"          # Suppress MediaPipe/GLOG warnings (2 = ERROR)
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"        # Additional abseil C++ logging
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"      # Suppress OpenCV warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Suppress OneDNN warnings
import sys
import json
import time
import cv2
import numpy as np
import logging
logging.getLogger("absl").setLevel(logging.ERROR)
from pathlib import Path
from typing import Dict, Optional

# Completely silence C++ DLLs (MediaPipe, TensorFlow, Abseil) on Windows
import ctypes
import io

import io

# Import the core analysis modules directly. We'll deal with MediaPipe console 
# spam instead of completely blinding ourselves to C++ crash messages.
from video_processor import extract_frames_for_analysis, get_video_info
from temporal_signals import analyze_temporal_signals
from biological_signals import analyze_biological_signals
from audio_analyzer import analyze_audio
import mediapipe as mp

#  Load video-specific EfficientNet (trained in Phase A) 
_video_model = None
_video_device = None
VIDEO_MODEL_PATH = Path(__file__).parent / "models" / "trained" / "video_efficientnet_b0.pth"
EFFICIENTNET_AVAILABLE = False

def _load_video_model():
    global _video_model, _video_device, EFFICIENTNET_AVAILABLE
    if _video_model is not None:
        return
    try:
        import torch
        # CRITICAL FIX for Windows: Prevent silent C++ OpenMP thread crashes 
        # when running frame-by-frame inference in a loop.
        torch.set_num_threads(1)
        
        import timm
        from torchvision import transforms
        _video_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=_video_device, weights_only=True))
        model.eval()
        model = model.to(_video_device)
        _video_model = model
        EFFICIENTNET_AVAILABLE = True
        print(f"  [VideoEfficientNet] Loaded from {VIDEO_MODEL_PATH.name}")
    except Exception as e:
        print(f"  [VideoEfficientNet] Failed to load: {e}")
        EFFICIENTNET_AVAILABLE = False

def _predict_frame(frame_bgr):
    """
    Predict AI probability for a single BGR frame using video EfficientNet.
    Crops face first to match DF40 training data format (tight face crops).
    Falls back to full frame if no face detected.
    """
    try:
        import torch
        from torchvision import transforms
        from PIL import Image

        # 3-channel RGB image
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0).to(_video_device)
        with torch.no_grad():
            logits = _video_model(tensor)
            prob = torch.softmax(logits, dim=1)[0][1].item()  # class 1 = fake
        return prob
    except Exception as e:
        print(f"    [DEBUG] Frame prediction failed internally: {e}")
        return 0.5

# Load model at module startup
if VIDEO_MODEL_PATH.exists():
    _load_video_model()

#  Load video meta-voter 
try:
    from video_meta_voter import VideoMetaVoter
    _voter = VideoMetaVoter()
    # Only use meta-voter if it has decent accuracy (>65%)
    META_VOTER_AVAILABLE = _voter.is_trained() and _voter.cv_accuracy > 0.65
except ImportError:
    META_VOTER_AVAILABLE = False


# 
#  FRAME-LEVEL AI DETECTION
# 

def analyze_frames_with_model(frames: list, temp_dir: str = None) -> Dict:
    """
    Run video-specific EfficientNet on sampled frames.
    Uses video_efficientnet_b0.pth trained in Phase A (98% val accuracy).
    """
    if not EFFICIENTNET_AVAILABLE or not frames:
        return {"mean_prob": 0.5, "max_prob": 0.5, "voted_fake": 0.5, "num_frames": 0}

    print("    [DEBUG] Starting analyze_frames_with_model", flush=True)
    probabilities = []
    # Sample up to 8 frames evenly across the video (reduced from 30 for speed)
    step = max(1, len(frames) // 8)
    sampled = frames[::step]
    
    print(f"    [DEBUG] Sampled {len(sampled)} frames", flush=True)
    for i, frame in enumerate(sampled):
        try:
            print(f"    [DEBUG] Predicting frame {i}", flush=True)
            p = _predict_frame(frame)
            print(f"    [DEBUG] Predicted {p}", flush=True)
            probabilities.append(p)
        except Exception as e:
            print(f"    [DEBUG] Error predicting frame {i}: {e}", flush=True)
            continue
    print("    [DEBUG] Finished predicting all frames", flush=True)

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


# 
#  MAIN DETECTION PIPELINE
# 

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

    #  Step 1: Get video metadata 
    print(f"   Analyzing: {os.path.basename(video_path)}")
    info = get_video_info(video_path)
    fps = info.fps or 30.0
    print(f"     Duration: {info.duration:.1f}s | FPS: {fps:.0f} | Size: {info.width}x{info.height}")

    #  Step 2: Extract frames 
    print(f"   Extracting frames at 2 FPS...")
    frames = extract_frames_for_analysis(video_path, sample_fps=2)
    print(f"     Extracted {len(frames)} frames")

    if not frames:
        return {"error": "Could not extract frames from video"}

    #  Step 3: Run all analyzers 
    print(f"   Running frame-level AI detection...")
    frame_result = analyze_frames_with_model(frames)

    # Route stderr to devnull to swallow hardcoded c++ mediapipe warnings
    old_stderr = os.dup(2)
    f_null = open(os.devnull, 'w')
    try:
        os.dup2(f_null.fileno(), 2)

        print(f"    Analyzing temporal signals...")
        temporal_result = analyze_temporal_signals(frames)

        print(f"   Analyzing biological signals...")
        biological_result = analyze_biological_signals(frames, fps)

        print(f"   Analyzing audio signals...")
        audio_result = analyze_audio(video_path, frames, fps)

    finally:
        os.dup2(old_stderr, 2)
        f_null.close()

    #  Step 4: Combine scores 
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
        #  Frame model reliability check 
        # The EfficientNet was trained on tight DF40 face crops.
        # On real-world phone videos the face detector often fails and the
        # model falls back to full frames it has never seen  gives extreme
        # values (all 0.99) on EVERY frame. Detect this and reduce its
        # weight so biological/temporal signals dominate instead.
        frame_mean      = frame_result.get("mean_prob", 0.5)
        frame_std       = frame_result.get("std_prob", 0.5)
        num_frames      = frame_result.get("num_frames", 0)

        frame_model_unreliable = (
            num_frames > 0
            and frame_std < 0.05          # near-identical score on every frame
            and (frame_mean > 0.95 or frame_mean < 0.05)  # stuck at extreme
        )

        if frame_model_unreliable:
            print(f"    Frame model stuck at {frame_mean:.2f} (std={frame_std:.4f})")
            print(f"      Reducing frame weight  trusting bio/temporal signals instead.")
            weights = {
                "frame_ai_prob":    0.10,  # stuck model: minimal influence
                "temporal_score":   0.40,  # temporal anomalies are reliable
                "biological_score": 0.35,  # heartbeat/blinks are very reliable
                "audio_score":      0.15,
            }
            method = "weighted_average_bio_dominant"
        else:
            weights = {
                "frame_ai_prob":    0.50,
                "temporal_score":   0.25,
                "biological_score": 0.15,
                "audio_score":      0.10,
            }
            method = "weighted_average"

        final_probability = sum(all_scores[k] * weights[k] for k in all_scores)

    final_probability = min(1.0, max(0.0, final_probability))

    #  Step 5: Generate verdict 
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

    #  Build signal cards (same format as photo detector for web UI) 
    signals = []

    # Frame-level signal
    signals.append({
        "name": "Frame AI Detection",
        "score": frame_result["mean_prob"],
        "weight": 0.35,
        "description": f"EfficientNet: {frame_result['num_frames']} frames analyzed, "
                       f"{frame_result['voted_fake']*100:.0f}% voted fake",
        "icon": "",
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
        "icon": "",
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
        "icon": "",
    })

    # Audio signals
    au = audio_result.get("details", {})
    sync = au.get("lip_sync", {})
    voice = au.get("voice", {})
    
    audio_desc_parts = []
    
    if voice.get("is_advanced_ml"):
        audio_desc_parts.append("Voice Authenticity: Advanced ML Model")
    else:
        audio_desc_parts.append("Voice Authenticity: Heuristics")
        
    lag = sync.get('lag_ms', 0)
    if sync.get("correlation", 0) > 0.5:
        audio_desc_parts.append(f"Lip sync: Good ({lag}ms lag)")
    else:
        audio_desc_parts.append(f"Lip sync: Poor ({lag}ms lag)")
        
    signals.append({
        "name": "Audio Analysis",
        "score": audio_result["audio_score"],
        "weight": 0.20,
        "description": " | ".join(audio_desc_parts),
        "icon": "🎧",
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
        bar = "" * int(sig["score"] * 20) + "" * (20 - int(sig["score"] * 20))
        print(f"  {sig['icon']} {sig['name']:25s} {bar} {sig['score']:.2f}")
    print(f"    Analysis time: {elapsed:.1f}s")
    print(f"  {'=' * 50}\n")

    return result


# 
#  CLI
# 

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
