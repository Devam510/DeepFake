import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Create directory
dest_dir = Path(r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\audio_forensics\raw\fake\custom_tts")
dest_dir.mkdir(parents=True, exist_ok=True)

# Generate 20 dummy audio files with random sine waves
sr = 16000
for i in range(20):
    duration = np.random.uniform(3.0, 5.0)  # 3 to 5 seconds
    t = np.linspace(0, duration, int(sr * duration), False)
    # Generate some random frequencies to emulate "speech-like" variance
    freq = np.random.uniform(100, 300)
    audio = np.sin(freq * 2 * np.pi * t) * 0.5
    
    # Add some noise
    audio += np.random.normal(0, 0.05, len(audio))
    
    # Save
    sf.write(dest_dir / f"mock_fake_audio_{i:03d}.wav", audio, sr)

print(f"Created 20 mock fake audio files in {dest_dir} for testing.")
