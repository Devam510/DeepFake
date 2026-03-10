"""
audio_worker.py — Standalone Neural Audio Inference Worker
===========================================================
Runs as a subprocess. Receives audio file paths on stdin (one per line),
returns JSON results on stdout. This isolates PyTorch from Flask's
Windows thread restrictions that cause WinError 1.

Usage (automatic — called by audio_analyzer.py):
    python audio_worker.py
"""

import sys
import json
import os

# Silence mediapipe/tensorflow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torch
    from audio_forensics import AdvancedAudioForensics
    from audio_neural_model import AudioNeuralDetector
    NEURAL_AVAILABLE = True
except Exception:
    NEURAL_AVAILABLE = False

# Load models once at startup
_system = None
_neural = None
_device = None

def load_models():
    global _system, _neural, _device
    if not NEURAL_AVAILABLE:
        return False
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _system = AdvancedAudioForensics()
        _neural = AudioNeuralDetector()
        _neural.eval()
        _neural = _neural.to(_device)
        return True
    except Exception as e:
        print(json.dumps({"error": f"Model load failed: {e}"}), flush=True)
        return False


def extract_features(audio_path: str) -> dict:
    if not LIBROSA_AVAILABLE:
        return {"error": "librosa not available"}

    try:
        if _system is not None:
            data = _system.load_and_preprocess(audio_path)
            if data is None:
                return {"error": "Failed to load/preprocess audio"}
            y, sr = data
            
            l1 = _system.layer1_signal_forensics(y, sr)
            l2 = _system.layer2_speech_behavior(y, sr)

            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
            y_16k = y_16k[:16000 * 10]  # max 10 seconds

            tensor_in = torch.tensor(y_16k).unsqueeze(0).float().to(_device)
            with torch.no_grad():
                logits, ood_embed = _neural.forward_features(tensor_in)
                l3_score = logits.cpu().item()
                l3_ood   = ood_embed.cpu().numpy().mean()

            return {
                "inst_phase_variance":   float(l1.get("inst_phase_variance", 0)),
                "rt60_estimate":         float(l1.get("rt60_estimate", 0)),
                "mfcc_variance":         float(l1.get("mfcc_variance", 0)),
                "spectral_flatness_var": float(l1.get("spectral_flatness_var", 0)),
                "zcr_variance":          float(l1.get("zcr_variance", 0)),
                "codec_banding_score":   float(l1.get("codec_banding_score", 0)),
                "pause_ratio":           float(l2.get("pause_ratio", 0)),
                "pitch_drift_over_time": float(l2.get("pitch_drift_over_time", 0)),
                "l3_score":              float(l3_score),
                "l3_ood_embed":          float(l3_ood),
            }
        else:
            # 2-layer fallback if AdvancedAudioForensics not available
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return {
                "mfcc_variance":        float(np.mean(np.var(mfccs, axis=1))),
                "spectral_flatness_var":float(np.mean(librosa.feature.spectral_flatness(y=y))),
                "zcr_variance":         float(np.var(librosa.feature.zero_crossing_rate(y))),
            }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    models_ok = load_models()
    # Signal readiness
    print(json.dumps({"status": "ready", "neural": models_ok}), flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        result = extract_features(path)
        print(json.dumps(result), flush=True)
