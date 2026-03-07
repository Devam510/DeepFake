"""
Audio Forensics Dataset Downloader — Robust Edition
=====================================================
Auto-downloads, resumes, extracts, and validates all datasets.
Designed for stability and maximum download speed.

Datasets:
  1. LibriSpeech      → OpenSLR (REAL, ~36GB)
  2. ASVspoof 2019 LA → HuggingFace (FAKE, ~7.7GB)
  3. WaveFake         → Zenodo CDN (FAKE, ~70GB)
  4. Common Voice     → HuggingFace (REAL, ~65GB)
  5. VoxCeleb1        → HuggingFace (REAL, ~35GB)
  6. FakeAVCeleb      → HuggingFace (FAKE, ~20GB)

Usage:
  python download_audio_datasets.py --all
  python download_audio_datasets.py --librispeech
  python download_audio_datasets.py --asvspoof
  python download_audio_datasets.py --wavefake
  python download_audio_datasets.py --common_voice
  python download_audio_datasets.py --voxceleb
  python download_audio_datasets.py --fakeavceleb
"""

import os
import sys
import zipfile
import tarfile
import argparse
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\audio_forensics\raw")

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def ensure_deps():
    print("[Setup] Checking packages...")
    for pkg in ["huggingface_hub", "tqdm", "requests"]:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  OK {pkg}")
        except ImportError:
            print(f"  Installing {pkg}...")
            install(pkg)
    print("[Setup] Ready.\n")

# ══════════════════════════════════════════════════════════════════════════════
# ARCHIVE VALIDATION — Catches HTML error pages disguised as zip files
# ══════════════════════════════════════════════════════════════════════════════
def is_valid_archive(filepath: Path) -> bool:
    try:
        with open(filepath, 'rb') as f:
            header = f.read(10)
        # ZIP: PK\x03\x04  |  GZ: \x1f\x8b
        if header[:4] == b'PK\x03\x04' or header[:2] == b'\x1f\x8b':
            return True
        print(f"  [!] File is not a valid archive (likely HTML error page): {filepath.name}")
        filepath.unlink()  # Auto-delete corrupt files
        return False
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════════════════════
# ROBUST HTTP DOWNLOADER — Resume + Retry + Validation
# ══════════════════════════════════════════════════════════════════════════════
def download_http(url: str, dest: Path, filename: str, max_retries: int = 5) -> Path:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / filename

    # Retry session — handles transient connection failures automatically
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    # Get remote file size
    remote_size = 0
    try:
        head = session.head(url, timeout=30, allow_redirects=True)
        remote_size = int(head.headers.get("content-length", 0))
    except Exception:
        pass

    # Check if file is already complete
    if filepath.exists():
        local_size = filepath.stat().st_size
        if remote_size > 0 and local_size >= remote_size:
            if is_valid_archive(filepath):
                print(f"  -> Already complete, skipping: {filename}")
                return filepath
        elif local_size > 0:
            print(f"  -> Resuming from {local_size/1e6:.1f}MB / {remote_size/1e9:.2f}GB...")

    local_size = filepath.stat().st_size if filepath.exists() else 0
    headers = {"Range": f"bytes={local_size}-"} if local_size > 0 else {}

    print(f"  URL:  {url}")
    print(f"  Size: {remote_size/1e9:.2f} GB")
    print(f"  Dest: {filepath}")

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, stream=True, timeout=60, headers=headers)
            mode = "ab" if local_size > 0 else "wb"
            remaining = remote_size - local_size if remote_size > 0 else 0

            with open(filepath, mode) as f, tqdm(
                total=remaining if remaining > 0 else None,
                initial=0,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=filename[:45], ncols=90
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    bar.update(len(chunk))
            break  # Success

        except KeyboardInterrupt:
            saved = filepath.stat().st_size if filepath.exists() else 0
            print(f"\n  Paused at {saved/1e6:.1f}MB. Re-run the same command to resume.")
            sys.exit(0)
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"\n  [Retry {attempt}/{max_retries}] Error: {e}. Waiting {wait}s...")
                time.sleep(wait)
                # Update local_size for next resume attempt
                local_size = filepath.stat().st_size if filepath.exists() else 0
                headers = {"Range": f"bytes={local_size}-"} if local_size > 0 else {}
            else:
                print(f"\n  [FAILED] Could not download {filename} after {max_retries} attempts.")

    return filepath

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-EXTRACT + DELETE
# ══════════════════════════════════════════════════════════════════════════════
def extract_and_delete(filepath: Path, dest: Path):
    if not filepath.exists():
        return
    if not is_valid_archive(filepath):
        return

    print(f"\n  Extracting: {filepath.name} ...")
    try:
        name = filepath.name
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(dest)
        elif name.endswith(".tar.bz2"):
            with tarfile.open(filepath, "r:bz2") as tar:
                tar.extractall(dest)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as z:
                z.extractall(dest)
        else:
            print(f"  [!] Unknown archive type, skipping: {name}")
            return

        filepath.unlink()
        print(f"  OK Extracted and deleted archive: {name}\n")

    except Exception as e:
        print(f"  [!] Extraction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE DOWNLOAD — Concurrent files, auto-resume
# ══════════════════════════════════════════════════════════════════════════════
def hf_download(repo_id: str, dest: Path, repo_type: str = "dataset"):
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    print(f"  Repo:   {repo_id}")
    print(f"  Dest:   {dest}")
    print(f"  Resume: Re-run if interrupted — already-downloaded files are skipped\n")

    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(dest),
                token=token,
                max_workers=8,       # 8 concurrent file downloads
                ignore_patterns=["*.md", "*.txt", "*.json"] if "asvspoof" not in repo_id else [],
            )
            print(f"\n  OK Done: {repo_id}\n")
            return
        except KeyboardInterrupt:
            print(f"\n  Paused. Re-run same command to resume.\n")
            sys.exit(0)
        except Exception as e:
            if attempt < 3:
                print(f"\n  [Retry {attempt}/3] HuggingFace error: {e}. Waiting 5s...")
                time.sleep(5)
            else:
                print(f"\n  [FAILED] {repo_id}: {e}\n")

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def download_librispeech():
    """LibriSpeech — Real human speech. train-clean-100 + train-other-500 (~36GB)"""
    print("\n" + "="*60)
    print("[1/6] LibriSpeech (REAL SPEECH) — OpenSLR")
    print("="*60)
    dest = BASE_DIR / "real" / "librispeech"

    parts = [
        ("https://www.openslr.org/resources/12/train-clean-100.tar.gz", "train-clean-100.tar.gz"),
        ("https://www.openslr.org/resources/12/train-other-500.tar.gz", "train-other-500.tar.gz"),
    ]
    for url, name in parts:
        fp = download_http(url, dest, name)
        extract_and_delete(fp, dest)

    print("[+] LibriSpeech DONE\n")


def download_asvspoof():
    """ASVspoof 2019 LA — HuggingFace mirror of full dataset (~7.7GB)"""
    print("\n" + "="*60)
    print("[2/6] ASVspoof 2019 LA (FAKE) — HuggingFace")
    print("="*60)
    dest = BASE_DIR / "fake" / "asvspoof"
    hf_download("LanceaKing/asvspoof2019", dest)
    print("[+] ASVspoof DONE\n")


def download_wavefake():
    """WaveFake — Multi-vocoder synthetic speech (~72GB). Zenodo CDN."""
    print("\n" + "="*60)
    print("[3/6] WaveFake (FAKE) — Zenodo")
    print("="*60)
    dest = BASE_DIR / "fake" / "wavefake"

    # WaveFake Zenodo record 5642694 — split into LJSpeech + JSUT subsets
    parts = [
        ("https://zenodo.org/record/5642694/files/WaveFake.zip?download=1", "WaveFake.zip"),
    ]
    for url, name in parts:
        fp = download_http(url, dest, name)
        extract_and_delete(fp, dest)

    print("[+] WaveFake DONE\n")


def download_common_voice():
    """Mozilla Common Voice — Crowdsourced real speech (~65GB). HuggingFace."""
    print("\n" + "="*60)
    print("[4/6] Common Voice (REAL) — HuggingFace")
    print("="*60)
    dest = BASE_DIR / "real" / "common_voice"
    hf_download("mozilla-foundation/common_voice_11_0", dest)
    print("[+] Common Voice DONE\n")


def download_voxceleb():
    """VoxCeleb1 — Public Subset (~1.2GB). HuggingFace."""
    print("\n" + "="*60)
    print("[5/6] VoxCeleb1 (REAL) — HuggingFace Public Subset")
    print("="*60)
    dest = BASE_DIR / "real" / "voxceleb"
    # Using a public subset since the original ProgramComputer/voxceleb1 is often gated/removed
    hf_download("sdialog/voices-voxceleb1", dest)
    print("[+] VoxCeleb1 Public Subset DONE\n")


def download_fakeavceleb():
    """FakeAVCeleb is heavily gated and requires manual approval via Google Forms."""
    print("\n" + "="*60)
    print("[6/6] FakeAVCeleb (FAKE) — MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print("  [!] FakeAVCeleb is heavily restricted to prevent deepfake misuse.")
    print("  [!] To get this dataset, you MUST fill out their official request form:")
    print("      https://github.com/DASH-Lab/FakeAVCeleb")
    print("  [!] Once approved and downloaded manually, extract it to:")
    print(f"      {BASE_DIR / 'fake' / 'fakeavceleb'}")
    print("[+] FakeAVCeleb instruction noted\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Audio Forensics Dataset Downloader",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--all",          action="store_true", help="Download ALL datasets (~200GB)")
    parser.add_argument("--librispeech",  action="store_true", help="LibriSpeech (~36GB, REAL)")
    parser.add_argument("--asvspoof",     action="store_true", help="ASVspoof 2019 LA (~7.7GB, FAKE)")
    parser.add_argument("--wavefake",     action="store_true", help="WaveFake (~72GB, FAKE)")
    parser.add_argument("--common_voice", action="store_true", help="Common Voice (~65GB, REAL)")
    parser.add_argument("--voxceleb",     action="store_true", help="VoxCeleb1 (~35GB, REAL)")
    parser.add_argument("--fakeavceleb",  action="store_true", help="FakeAVCeleb (~20GB, FAKE)")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    ensure_deps()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Data root: {BASE_DIR}")
    print(f"  VoxCeleb2 (optional): manual download from https://mm.kaist.ac.kr/datasets/voxceleb/\n")

    if args.all or args.librispeech:   download_librispeech()
    if args.all or args.asvspoof:      download_asvspoof()
    if args.all or args.wavefake:      download_wavefake()
    if args.all or args.common_voice:  download_common_voice()
    if args.all or args.voxceleb:      download_voxceleb()
    if args.all or args.fakeavceleb:   download_fakeavceleb()

    print("\n" + "="*60)
    print(" ALL DOWNLOADS COMPLETE!")
    print(f" Data: {BASE_DIR}")
    print(" Next: python train_audio_model.py --build-features")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
