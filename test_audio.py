import sys
import json
from audio_analyzer import analyze_voice_authenticity

def test():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/audio_forensics/raw/fake/wavefake/wavefake_000018.wav"
        
    print(f"Testing audio ML model on {file_path}")
    result = analyze_voice_authenticity(file_path)
    print("Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test()
