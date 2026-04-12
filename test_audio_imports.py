import sys
import json

print("Testing imports...")
try:
    import torch
    print("torch imported")
    from audio_forensics import AdvancedAudioForensics
    print("AdvancedAudioForensics imported")
    from audio_neural_model import AudioNeuralDetector
    print("AudioNeuralDetector imported")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
