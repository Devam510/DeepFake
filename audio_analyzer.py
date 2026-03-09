"""
Audio Analyzer — Voice Authenticity + Lip-Audio Sync Detection
=============================================================

Detects AI-generated audio and audio-visual synchronization issues:
1. Voice Authenticity — MFCC spectral analysis for AI-generated voice
2. Lip-Audio Sync   — Compare lip movement timing with audio energy

Usage:
  from audio_analyzer import analyze_audio
  result = analyze_audio("video.mp4")
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

import pickle
try:
    import torch
    from audio_forensics import AdvancedAudioForensics
    from audio_neural_model import AudioNeuralDetector
    ADVANCED_AUDIO_AVAILABLE = True
except ImportError:
    ADVANCED_AUDIO_AVAILABLE = False

_audio_ensemble_model = None
_audio_system = None
_neural_system = None
_audio_device = None

def _load_lgbm_model():
    """Load just the LightGBM ensemble (fast, always safe in Flask context)."""
    global _audio_ensemble_model
    if _audio_ensemble_model is not None:
        return True

    model_path = Path(__file__).parent / "models" / "audio_lgbm_ensemble.pkl"
    if not model_path.exists():
        return False

    try:
        import joblib
        _audio_ensemble_model = joblib.load(str(model_path))
        print(f"  [AudioEnsemble] LightGBM model loaded successfully")
        return True
    except Exception as e:
        print(f"  [AudioEnsemble] Failed to load LightGBM: {e}")
        return False


import subprocess as _subprocess
import json as _json
from pathlib import Path as _Path

_worker_proc = None  # Persistent audio_worker subprocess


def _start_worker():
    """Launch audio_worker.py as a persistent subprocess (safe on Windows)."""
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return True  # already running

    worker_path = str(_Path(__file__).parent / "audio_worker.py")
    try:
        _worker_proc = _subprocess.Popen(
            [sys.executable, worker_path],
            stdin=_subprocess.PIPE,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Wait for ready signal
        ready_line = _worker_proc.stdout.readline()
        info = _json.loads(ready_line)
        neural_ok = info.get("neural", False)
        print(f"  [AudioWorker] Subprocess started (neural={'yes' if neural_ok else 'no'})")
        return True
    except Exception as e:
        print(f"  [AudioWorker] Failed to start subprocess: {e}")
        _worker_proc = None
        return False


def _get_neural_features(audio_path: str) -> dict:
    """Send audio path to worker subprocess, get back feature dict."""
    if not _start_worker():
        return {}
    try:
        _worker_proc.stdin.write(audio_path + "\n")
        _worker_proc.stdin.flush()
        result_line = _worker_proc.stdout.readline()
        return _json.loads(result_line)
    except Exception as e:
        return {"error": str(e)}


def _load_advanced_audio_models():
    """Load LightGBM and start the neural worker subprocess."""
    lgbm_ok = _load_lgbm_model()
    if lgbm_ok:
        _start_worker()  # non-blocking — spawns in background
    return lgbm_ok


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio(video_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Extract audio from video file using ffmpeg.
    Returns path to extracted WAV file, or None if failed.
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", output_path, "-y"],
            capture_output=True, timeout=60,
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1: Voice Authenticity (MFCC Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_voice_authenticity(audio_path: str) -> Dict:
    """
    Analyze voice for AI-generation artifacts.
    If 'audio_lgbm_ensemble.pkl' exists, uses advanced 3-layer ML detection.
    Otherwise, falls back to basic heuristic MFCC checks.
    """
    if not LIBROSA_AVAILABLE or audio_path is None:
        return {"spectral_flatness": 0.0, "mfcc_variance": 0.0, "voice_score": 0.5}

    # ----- ADVANCED ML DETECTION -----
    if _load_lgbm_model():
        try:
            y_new, sr_new = librosa.load(audio_path, sr=None, mono=True)

            # Get neural features from subprocess worker
            feats = _get_neural_features(os.path.abspath(audio_path))
            neural_ok = feats and "error" not in feats and "l3_score" in feats

            if neural_ok:
                feature_vector = [
                    feats.get("inst_phase_variance",   0),
                    feats.get("rt60_estimate",         0),
                    feats.get("mfcc_variance",         0),
                    feats.get("spectral_flatness_var", 0),
                    feats.get("zcr_variance",          0),
                    feats.get("codec_banding_score",   0),
                    feats.get("pause_ratio",           0),
                    feats.get("pitch_drift_over_time", 0),
                    feats.get("l3_score",              0),
                    feats.get("l3_ood_embed",          0),
                ]
                mfcc_surface = feats.get("mfcc_variance", 0)
                flat_surface = feats.get("spectral_flatness_var", 0)
                zcr_surface  = feats.get("zcr_variance", 0)
            else:
                # 2-layer librosa fallback — no neural features
                mfccs     = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
                mfcc_var  = float(np.mean(np.var(mfccs, axis=1)))
                flatness  = float(np.mean(librosa.feature.spectral_flatness(y=y_new)))
                zcr_var   = float(np.var(librosa.feature.zero_crossing_rate(y_new)))
                feature_vector = [
                    np.nan, np.nan,
                    mfcc_var, flatness, zcr_var,
                    np.nan, np.nan, np.nan, np.nan, np.nan,
                ]
                mfcc_surface = mfcc_var
                flat_surface = flatness
                zcr_surface  = zcr_var

            X_np = np.array([feature_vector])
            prob_fake = float(_audio_ensemble_model.predict_proba(X_np)[0][1])

            return {
                "spectral_flatness": round(flat_surface, 6),
                "mfcc_variance":     round(mfcc_surface, 4),
                "rolloff_variance":  0.0,
                "zcr_variance":      round(zcr_surface, 8),
                "voice_score":       round(prob_fake, 4),
                "is_advanced_ml":    True
            }
        except Exception as e:
            print(f"  [AudioEnsemble] Advanced extraction failed ({e}), falling back to heuristic...")
            
    # ----- FALLBACK BASIC HEURISTIC -----
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        if len(y) < sr:  # less than 1 second
            return {"spectral_flatness": 0.0, "mfcc_variance": 0.0, "voice_score": 0.5}

        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

        # Spectral flatness (tonality measure)
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = float(np.mean(flatness))

        # Spectral rolloff variability
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_var = float(np.var(rolloff))

        # Zero crossing rate variance
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_var = float(np.var(zcr))

        # AI voices tend to have:
        # - Lower MFCC variance (over-smooth)
        # - More uniform spectral flatness
        # - Less rolloff variation
        smoothness_indicators = 0
        if mfcc_var < 50:
            smoothness_indicators += 1
        if mean_flatness > 0.1:
            smoothness_indicators += 1
        if rolloff_var < 1e6:
            smoothness_indicators += 1

        voice_score = smoothness_indicators / 3.0 * 0.7 + 0.15

        return {
            "spectral_flatness": round(mean_flatness, 6),
            "mfcc_variance": round(mfcc_var, 4),
            "rolloff_variance": round(rolloff_var, 2),
            "zcr_variance": round(zcr_var, 8),
            "voice_score": round(voice_score, 4),
            "is_advanced_ml": False
        }
    except Exception as e:
        return {"spectral_flatness": 0.0, "mfcc_variance": 0.0, "voice_score": 0.5, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 2: Lip-Audio Synchronization
# ══════════════════════════════════════════════════════════════════════════════

# MediaPipe lip landmarks
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]


def compute_lip_movement(frames: List[np.ndarray]) -> List[float]:
    """
    Compute lip opening magnitude per frame using MediaPipe.
    Returns list of lip-opening values (0 = closed, 1 = wide open).
    """
    if not MEDIAPIPE_AVAILABLE:
        return []

    lip_openings = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _mp_face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # Vertical lip distance (upper to lower lip)
            upper_y = np.mean([lms[i].y for i in [13, 312, 311, 310]])
            lower_y = np.mean([lms[i].y for i in [14, 317, 402, 318]])
            lip_dist = abs(lower_y - upper_y) * h

            # Normalize by face height
            chin_y = lms[152].y * h
            forehead_y = lms[10].y * h
            face_h = abs(chin_y - forehead_y)

            normalized = lip_dist / max(face_h, 1) * 10  # scale to 0-1 range
            lip_openings.append(min(1.0, normalized))
        else:
            lip_openings.append(0.0)

    return lip_openings


def analyze_lip_sync(video_path: str, frames: List[np.ndarray], fps: float = 30.0) -> Dict:
    """
    Compare lip movement timing with audio energy.

    Real: lips move in sync with voice.
    Deepfake: temporal lag between lip movement and audio peaks.

    Returns:
        dict with correlation, lag_ms, sync_score
    """
    if not LIBROSA_AVAILABLE:
        return {"correlation": 0.0, "lag_ms": 0, "sync_score": 0.5}

    # Extract audio
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return {"correlation": 0.0, "lag_ms": 0, "sync_score": 0.5, "note": "no_audio"}

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Compute audio energy envelope at video frame rate
        hop_length = int(sr / fps)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Compute lip movement signal
        lip_signal = compute_lip_movement(frames)

        if len(lip_signal) < 5 or len(rms) < 5:
            return {"correlation": 0.0, "lag_ms": 0, "sync_score": 0.5}

        # Align lengths
        min_len = min(len(lip_signal), len(rms))
        lip_arr = np.array(lip_signal[:min_len])
        rms_arr = np.array(rms[:min_len])

        # Normalize
        lip_arr = (lip_arr - np.mean(lip_arr)) / max(np.std(lip_arr), 1e-6)
        rms_arr = (rms_arr - np.mean(rms_arr)) / max(np.std(rms_arr), 1e-6)

        # Cross-correlation to find lag
        corr = np.correlate(lip_arr, rms_arr, mode='full')
        lag_idx = np.argmax(corr) - (min_len - 1)
        lag_ms = int(lag_idx / fps * 1000)
        max_corr = float(np.max(corr) / min_len)

        # Score: low correlation or high lag = desync = suspicious
        sync_quality = max(0, min(1.0, max_corr))
        lag_penalty = min(1.0, abs(lag_ms) / 200.0)  # > 200ms lag is very suspicious

        sync_score = (1.0 - sync_quality) * 0.5 + lag_penalty * 0.5

        return {
            "correlation": round(max_corr, 4),
            "lag_ms": lag_ms,
            "sync_score": round(sync_score, 4),
        }
    except Exception as e:
        return {"correlation": 0.0, "lag_ms": 0, "sync_score": 0.5, "error": str(e)}
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_audio(video_path: str, frames: List[np.ndarray], fps: float = 30.0) -> Dict:
    """
    Run all audio analysis on a video.

    Args:
        video_path: Path to video file
        frames: Extracted video frames (for lip analysis)
        fps: Video frame rate

    Returns:
        dict with voice and lip-sync scores + overall audio_score
    """
    # Extract audio once for both analyses
    audio_path = extract_audio(video_path)

    voice = analyze_voice_authenticity(audio_path)
    lip_sync = analyze_lip_sync(video_path, frames, fps)

    # Clean up
    if audio_path and os.path.exists(audio_path):
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    is_ml = voice.get("is_advanced_ml", False)
    voice_score = voice["voice_score"]
    lip_score = lip_sync["sync_score"]

    # Give LightGBM voice model dominant weight (0.7).
    # Lip-sync is a weak signal when face is occluded or audio-only, so 0.3.
    # Old weights (0.4 voice / 0.6 lip) were diluting the ML score back to ~50%.
    overall = voice_score * 0.70 + lip_score * 0.30

    voice_method = "LightGBM+Wav2Vec2" if is_ml else "Heuristics"
    lag_ms = lip_sync.get("lag_ms", 0)

    return {
        "audio_score": round(overall, 4),
        "details": {
            "voice": voice,
            "lip_sync": lip_sync,
        },
        "individual_scores": {
            "voice_authenticity": voice_score,
            "lip_sync": lip_score,
        },
        "description": f"Voice Authenticity: {voice_method} | Lip sync: {'Good' if abs(lag_ms) < 80 else 'Poor'} ({lag_ms}ms lag)",
    }
