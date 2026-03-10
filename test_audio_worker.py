import sys
import json
import traceback

def test():
    try:
        from audio_worker import extract_features, load_models
        load_models()
        # Test with a dummy short audio file path that might exist, or simulate it.
        # Alternatively, we can let it fail and see the exact exception.
        
        # In fact, we can use librosa to create a dummy wav file
        import librosa
        import numpy as np
        import soundfile as sf
        
        dummy_audio = np.random.randn(16000 * 3) # 3 seconds of noise
        sf.write('test_dummy_audio.wav', dummy_audio, 16000)
        
        print("Testing extraction...")
        res = extract_features('test_dummy_audio.wav')
        print(f"Extraction result: {res}")
        
    except Exception as e:
        print(f"Hard crash: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    test()
