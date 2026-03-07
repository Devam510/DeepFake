"""
Production-Grade Audio Dataset Preparation Pipeline
===================================================

This script sets up the directory structure for training the 3-Layer 
Audio Forensics model. It handles:
1. Directory Initialization
2. Dataset Source Documentation (Links & Tools)
3. Speaker-Isolated Train/Val/Test Splitting Logic

Datasets structured for:
- Real: VoxCeleb (1 & 2), LibriSpeech, Common Voice
- Fake: ASVspoof (19, 21, 24), FakeAVCeleb, WaveFake, Custom (ElevenLabs, Coqui, Bark)
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import shutil

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data/audio_forensics")

DIRS = {
    "raw": DATA_DIR / "raw",
    "processed": DATA_DIR / "processed",
    "splits": DATA_DIR / "splits",
    "real": DATA_DIR / "raw" / "real",
    "fake": DATA_DIR / "raw" / "fake"
}

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
def init_directories():
    """Create the mandatory folder structure."""
    print("[*] Initializing Audio Forensics Dataset Architecture...")
    
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  -> Created: {path}")

    # Sub-folders for specific datasets
    datasets = {
        "real": ["voxceleb", "librispeech", "common_voice"],
        "fake": ["asvspoof", "fakeavceleb", "wavefake", "custom_tts"]
    }
    
    for category, dset_list in datasets.items():
        for dset in dset_list:
            (DIRS[category] / dset).mkdir(exist_ok=True)
            
    print("[+] Directory structure initialized.\n")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET SOURCING (Documentation)
# ══════════════════════════════════════════════════════════════════════════════
def print_dataset_instructions():
    """Outputs instructions/links for acquiring the mandatory datasets."""
    instructions = """
[!] ACQUISITION INSTRUCTIONS 
Please download the following datasets to their respective raw/ folders.
Ensure you have at least 500GB of storage available.

REAL DATASETS:
1. VoxCeleb1 & 2 (data/audio_forensics/raw/real/voxceleb/)
   - Source: https://mm.kaist.ac.kr/datasets/voxceleb/
   - Content: Over 1 million utterances for over 7,000 celebrities.

2. LibriSpeech (data/audio_forensics/raw/real/librispeech/)
   - Source: http://www.openslr.org/12
   - Content: 1000 hours of 16kHz read English speech.

---
SYNTHETIC DATASETS:
1. ASVspoof 2019/2021 (data/audio_forensics/raw/fake/asvspoof/)
   - Source: https://www.asvspoof.org/
   - Content: The academic gold standard for logical access (LA) and deepfake (DF) attacks.

2. FakeAVCeleb (data/audio_forensics/raw/fake/fakeavceleb/)
   - Source: https://github.com/DASH-Lab/FakeAVCeleb
   - Content: Deepfake videos/audio (Wav2Lip, SV2TTS).

3. WaveFake (data/audio_forensics/raw/fake/wavefake/)
   - Source: https://github.com/rubaiyatT/WaveFake
   - Content: Multi-language synthetic dataset using advanced neural vocoders.
   
4. Custom TTS (data/audio_forensics/raw/fake/custom_tts/)
   - Action: Generate ~5,000 samples yourself using ElevenLabs, Coqui TTS, and Bark.
"""
    print(instructions)

# ══════════════════════════════════════════════════════════════════════════════
# STRICT SPEAKER-ISOLATED SPLIT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_speaker_isolated_splits(metadata_file: str, out_prefix: str):
    """
    Reads a metadata JSON file mapping: {"filename.wav": "speaker_ID"}
    Ensures that Speaker A from Train NEVER appears in Val or Test.
    Target Split: 70% Train, 15% Val, 15% Test
    """
    if not os.path.exists(metadata_file):
        print(f"[!] Metadata file not found: {metadata_file}. Skipping split generation.")
        return

    with open(metadata_file, 'r') as f:
        data = json.load(f)

    # Group files by speaker
    speaker_to_files = defaultdict(list)
    for filename, spk_id in data.items():
        speaker_to_files[spk_id].append(filename)

    speakers = list(speaker_to_files.keys())
    random.shuffle(speakers)

    # Calculate target file counts
    total_files = len(data)
    train_target = int(total_files * 0.70)
    val_target = int(total_files * 0.15)
    
    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    # Assign speakers to splits greedy-style
    for spk in speakers:
        spk_files = speaker_to_files[spk]
        num_files = len(spk_files)
        
        if counts["train"] < train_target:
            splits["train"].extend(spk_files)
            counts["train"] += num_files
        elif counts["val"] < val_target:
            splits["val"].extend(spk_files)
            counts["val"] += num_files
        else:
            splits["test"].extend(spk_files)
            counts["test"] += num_files

    # Save manifest files
    for split_name, filenames in splits.items():
        manifest_path = DIRS["splits"] / f"{out_prefix}_{split_name}.txt"
        with open(manifest_path, 'w') as f:
            for fname in filenames:
                f.write(f"{fname}\n")
    
    print(f"[+] Strict Speaker Split [{out_prefix}] Complete:")
    print(f"    Train: {counts['train']} files")
    print(f"    Val:   {counts['val']} files")
    print(f"    Test:  {counts['test']} files\n")

# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=====================================================")
    print(" DeepFake Detector: Audio Dataset Preparation Module ")
    print("=====================================================\n")
    
    init_directories()
    print_dataset_instructions()
    
    # Mock example of how the split generation would be called
    mock_meta = "data/audio_forensics/raw/mock_metadata.json"
    if not os.path.exists(mock_meta):
        # Create a tiny mock metadata file just to demonstrate
        os.makedirs(os.path.dirname(mock_meta), exist_ok=True)
        with open(mock_meta, 'w') as f:
            json.dump({
                "spk1_001.wav": "spk_1", "spk1_002.wav": "spk_1",
                "spk2_001.wav": "spk_2", "spk3_001.wav": "spk_3",
                "spk4_001.wav": "spk_4", "spk4_002.wav": "spk_4"
            }, f)
            
    generate_speaker_isolated_splits(mock_meta, "core_audio")

if __name__ == "__main__":
    main()
