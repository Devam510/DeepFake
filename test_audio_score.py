"""
test_audio_score.py - Test audio scoring via the online (Flask worker) path.
Uses offline_mode=False which is what the Web UI uses.
"""
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

from audio_analyzer import analyze_voice_authenticity, extract_audio

VIDEO = "datasets/raw_real_videos/VID_20241114201922.mp4"
print(f"Extracting audio from: {VIDEO}")
audio_path = extract_audio(VIDEO)
if audio_path is None:
    print("❌ ERROR: Could not extract audio from video.")
    sys.exit(1)

print(f"Audio extracted to: {audio_path}")
print("Analyzing voice authenticity (Online mode - AudioWorker subprocess)...")
result = analyze_voice_authenticity(audio_path, offline_mode=False)

print("\n" + "=" * 55)
print("AUDIO VOICE AUTHENTICITY RESULT")
print("=" * 55)
score = result.get('voice_score', 'N/A')
print(f"  voice_score:    {score:.4f}  (0=real, 1=fake)")
print(f"  is_advanced_ml: {result.get('is_advanced_ml', False)}")
print(f"  mfcc_variance:  {result.get('mfcc_variance', 'N/A')}")
print("=" * 55)
if result.get("is_advanced_ml"):
    print("✅ Using Advanced ML Model (Wav2Vec2 + LightGBM debiased)")
    if isinstance(score, float) and score < 0.5:
        print("✅ CORRECTLY scored as REAL audio (score < 0.5)!")
    else:
        print(f"⚠️  Scored {score:.2%} fake — may still need investigation.")
else:
    print("⚠️  Using Basic Heuristics — ML model not loaded or worker failed!")
