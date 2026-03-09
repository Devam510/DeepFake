import glob
import os
import zipfile
import shutil
from pathlib import Path

cache = os.path.expanduser("~/.cache/huggingface/hub")
zips = glob.glob(cache + "/**/*.zip", recursive=True)
el_zips = [z for z in zips if "elevenlabs" in z.lower()]

if not el_zips:
    print("[!] No elevenlabs ZIP found in HuggingFace cache.")
    print("    Run: python download_elevenlabs_datasets.py first.")
    exit(1)

zip_path = el_zips[0]
print(f"Found ZIP: {zip_path}")
print(f"ZIP size: {round(os.path.getsize(zip_path)/1024/1024, 1)} MB")

out_dir = Path("data/audio_forensics/raw/fake/elevenlabs")
out_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    audio_files = [n for n in zf.namelist() if n.endswith((".mp3", ".wav", ".flac", ".ogg"))]
    print(f"Extracting {len(audio_files)} audio files...")
    saved = 0
    for name in audio_files:
        basename = os.path.basename(name)
        out_path = out_dir / ("hf_" + basename)
        if out_path.exists():
            continue
        with zf.open(name) as src, open(str(out_path), "wb") as dst:
            shutil.copyfileobj(src, dst)
        saved += 1
        if saved % 100 == 0:
            print(f"  Extracted {saved}/{len(audio_files)}...")

total = len(list(out_dir.glob("*")))
print(f"Done! Saved {saved} new files.")
print(f"Total in elevenlabs/ folder: {total}")
print("\nNow retrain: python train_audio_model.py")
