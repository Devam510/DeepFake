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
    Analyze voice for AI-generation artifacts using MFCC features.

    Real voice: natural spectral variation, formant transitions.
    AI voice: over-smooth spectral contours, unnatural transitions.

    Returns:
        dict with spectral_flatness, mfcc_variance, voice_score
    """
    if not LIBROSA_AVAILABLE or audio_path is None:
        return {"spectral_flatness": 0.0, "mfcc_variance": 0.0, "voice_score": 0.5}

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

    scores = {
        "voice_authenticity": voice["voice_score"],
        "lip_sync": lip_sync["sync_score"],
    }

    overall = scores["voice_authenticity"] * 0.4 + scores["lip_sync"] * 0.6

    return {
        "audio_score": round(overall, 4),
        "details": {
            "voice": voice,
            "lip_sync": lip_sync,
        },
        "individual_scores": scores,
    }
