"""
download_elevenlabs_datasets.py
================================
Downloads all publicly available ElevenLabs-generated voice datasets
and places them into the correct training directory structure for the
audio deepfake detection model.

Sources:
  1. HuggingFace: skypro1111/elevenlabs_dataset   (1,388 ElevenLabs TTS clips)
  2. HuggingFace: Audio DeepFake datasets w/ ElevenLabs samples
  3. Mendeley: Fake Audio Dataset (ElevenLabs & Respeecher) [manual URL]

Usage:
  python download_elevenlabs_datasets.py

Requirements:
  pip install datasets huggingface_hub requests tqdm
"""

import os
import sys
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

# ── Output directories ────────────────────────────────────────────────────────
FAKE_DIR  = Path("data/audio_forensics/raw/fake/elevenlabs")
REAL_HINT = Path("data/audio_forensics/raw/real")  # real audio stays untouched
FAKE_DIR.mkdir(parents=True, exist_ok=True)

def count_existing():
    return len(list(FAKE_DIR.glob("*.mp3")) + list(FAKE_DIR.glob("*.wav")) + list(FAKE_DIR.glob("*.flac")))

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — HuggingFace: skypro1111/elevenlabs_dataset
#   1,388 ElevenLabs TTS MP3 clips (2h 20min of audio)
# ══════════════════════════════════════════════════════════════════════════════
def download_huggingface_elevenlabs():
    print("\n[1/3] Downloading skypro1111/elevenlabs_dataset from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] `datasets` library not found. Installing...")
        os.system(f"{sys.executable} -m pip install datasets -q")
        from datasets import load_dataset

    try:
        ds = load_dataset("skypro1111/elevenlabs_dataset", split="train")
        saved = 0
        for i, sample in enumerate(tqdm(ds, desc="  Saving ElevenLabs HF")):
            audio_data = sample.get("audio")
            if audio_data is None:
                continue
            out_path = FAKE_DIR / f"hf_el_{i:05d}.wav"
            if out_path.exists():
                continue
            try:
                import soundfile as sf
                import numpy as np
                sf.write(str(out_path), np.array(audio_data["array"]), audio_data["sampling_rate"])
                saved += 1
            except Exception:
                pass
        print(f"  [+] Saved {saved} ElevenLabs clips from HuggingFace")
        return saved
    except Exception as e:
        print(f"  [!] HuggingFace download failed: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — HuggingFace: motheecreator/Fake-or-Real-Audio-Detection
#   Contains ElevenLabs fake audio mixed with real audio
# ══════════════════════════════════════════════════════════════════════════════
def download_fake_or_real():
    print("\n[2/3] Downloading motheecreator/Fake-or-Real-Audio-Detection...")
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("  [!] Missing dependencies. Install: pip install datasets soundfile")
        return 0

    try:
        ds = load_dataset("motheecreator/Fake-or-Real-Audio-Detection", split="train", trust_remote_code=True)
        saved_fake = 0
        for i, sample in enumerate(tqdm(ds, desc="  Processing Fake-or-Real")):
            label = sample.get("label", sample.get("labels", None))
            audio = sample.get("audio")
            if audio is None:
                continue
            # label=1 or "fake" or "FAKE" → fake audio
            is_fake = str(label).lower() in ("1", "fake", "spoof", "synthetic", "generated")
            if not is_fake:
                continue
            out_path = FAKE_DIR / f"for_fake_{i:05d}.wav"
            if out_path.exists():
                continue
            try:
                sf.write(str(out_path), np.array(audio["array"]), audio["sampling_rate"])
                saved_fake += 1
            except Exception:
                pass
        print(f"  [+] Saved {saved_fake} fake clips from Fake-or-Real dataset")
        return saved_fake
    except Exception as e:
        print(f"  [!] Fake-or-Real download failed: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — HuggingFace: CShorten/ElevenLabs-TTS-Benchmark
#   Alternative public ElevenLabs benchmark clips
# ══════════════════════════════════════════════════════════════════════════════
def download_elevenlabs_alt():
    print("\n[3/3] Trying alternative ElevenLabs HuggingFace datasets...")
    
    alt_datasets = [
        ("MarcBrun/ElevenLabs-Speech", "train"),
        ("reach-vb/random_speech", "train"),
    ]

    total = 0
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np

        for dataset_id, split in alt_datasets:
            try:
                print(f"  Trying {dataset_id}...")
                ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
                saved = 0
                for i, sample in enumerate(ds):
                    if i >= 500:  # cap at 500 per source
                        break
                    audio = sample.get("audio")
                    if audio is None:
                        continue
                    out_path = FAKE_DIR / f"alt_{dataset_id.replace('/', '_')}_{i:04d}.wav"
                    if out_path.exists():
                        continue
                    try:
                        sf.write(str(out_path), np.array(audio["array"]), audio["sampling_rate"])
                        saved += 1
                    except Exception:
                        pass
                print(f"    -> Saved {saved} clips from {dataset_id}")
                total += saved
            except Exception as e:
                print(f"    -> {dataset_id} not available: {e}")
    except ImportError:
        print("  [!] soundfile not installed. Run: pip install soundfile")

    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ElevenLabs Dataset Downloader")
    print("  Target:", FAKE_DIR)
    print(f"  Existing files: {count_existing()}")
    print("=" * 60)

    # Install dependencies
    for pkg in ["datasets", "soundfile", "huggingface_hub", "tqdm"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f"{sys.executable} -m pip install {pkg} -q")

    total_before = count_existing()

    s1 = download_huggingface_elevenlabs()
    s2 = download_fake_or_real()
    s3 = download_elevenlabs_alt()

    total_after = count_existing()
    new_files = total_after - total_before

    print("\n" + "=" * 60)
    print(f"  DOWNLOAD COMPLETE!")
    print(f"  New files downloaded : {new_files}")
    print(f"  Total in elevenlabs/ : {total_after}")
    print("=" * 60)
    print("\n[*] Now retrain the model to learn these new samples:")
    print("    python train_audio_model.py")
