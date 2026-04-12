import torch
from audio_neural_model import AudioNeuralDetector
import json

print("Attempting to instantiate AudioNeuralDetector...")
try:
    detector = AudioNeuralDetector()
    print("SUCCESS: Detector instantiated")
    
    # Test a dummy input
    dummy_input = torch.randn(1, 16000)
    output = detector(dummy_input)
    print(f"SUCCESS: Forward pass result: {output.shape}")
    
    # Get features
    feats = detector.forward_features(dummy_input)
    print(f"SUCCESS: Features: {feats.keys()}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
