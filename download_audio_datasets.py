"""
Audio Forensics Dataset Downloader — Robust Edition
=====================================================
Auto-downloads, resumes, extracts, and validates all datasets.
Designed for stability and maximum download speed.

Datasets:
  1. LibriSpeech      → HuggingFace mirror (REAL, ~36GB)
  2. ASVspoof 2019 LA → Kaggle (FAKE, ~7.7GB) [requires kaggle API key]
  3. WaveFake         → HuggingFace datasets library (FAKE, ~30GB)
  4. Common Voice     → HuggingFace (REAL, ~65GB)
  5. VoxCeleb1        → HuggingFace Public Subset (REAL, ~1.2GB)
  6. FakeAVCeleb      → Manual download required (FAKE, ~20GB)

Usage:
  python download_audio_datasets.py --all
  python download_audio_datasets.py --librispeech
  python download_audio_datasets.py --asvspoof
  python download_audio_datasets.py --wavefake
  python download_audio_datasets.py --common_voice
  python download_audio_datasets.py --voxceleb
  python download_audio_datasets.py --fakeavceleb

Notes:
  --asvspoof: Requires Kaggle API. Run: pip install kaggle
              Then set KAGGLE_USERNAME and KAGGLE_KEY env vars,
              or place kaggle.json in ~/.kaggle/
  --wavefake: Uses HuggingFace datasets library (streams & saves audio)
              Run: pip install datasets soundfile
"""

import os
import sys
import zipfile
import tarfile
import argparse
import subprocess
import time
from pathlib import Path

# Enable HuggingFace's Rust-based extreme fast downloader
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

BASE_DIR = Path(r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\audio_forensics\raw")

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def ensure_deps(extra=None):
    print("[Setup] Checking packages...")
    base = ["huggingface_hub", "tqdm", "requests", "hf_transfer"]
    all_pkgs = base + (extra or [])
    for pkg in all_pkgs:
        import_name = pkg.replace("-", "_").replace("soundfile", "soundfile")
        try:
            __import__(import_name)
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
                max_workers=16,       # 16 concurrent file downloads
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
    """LibriSpeech — Real human speech. (~36GB) via fast HuggingFace mirror."""
    print("\n" + "="*60)
    print("[1/6] LibriSpeech (REAL SPEECH) — HuggingFace Mirror")
    print("="*60)
    dest = BASE_DIR / "real" / "librispeech"
    # Using k2-fsa/LibriSpeech mirror since OpenSLR throttles to 60kB/s
    hf_download("k2-fsa/LibriSpeech", dest)
    print("[+] LibriSpeech DONE\n")


def download_asvspoof():
    """ASVspoof 2019 LA — Downloaded via Kaggle API (~7.7GB of real spoofed audio)."""
    print("\n" + "="*60)
    print("[2/6] ASVspoof 2019 LA (FAKE) — Kaggle")
    print("="*60)
    dest = BASE_DIR / "fake" / "asvspoof"
    dest.mkdir(parents=True, exist_ok=True)

    # Check for Kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if not kaggle_json.exists() and not has_env:
        print("  [!] Kaggle credentials not found!")
        print("  To download ASVspoof 2019 LA:")
        print("  1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print("  2. Place the downloaded kaggle.json in: ~/.kaggle/kaggle.json")
        print("     (Windows: C:\\Users\\<You>\\.kaggle\\kaggle.json)")
        print("  3. Run: pip install kaggle")
        print("  4. Run: python download_audio_datasets.py --asvspoof")
        print("  Alternative: https://datashare.is.ed.ac.uk/handle/10283/3336 (Edinburgh DataShare)")
        print("  Or Kaggle web: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset")
        return

    try:
        import kaggle
    except ImportError:
        install("kaggle")
        import kaggle

    print(f"  Dest: {dest}")
    print(f"  Downloading LA partition (~7.7GB)...")
    try:
        # Use Kaggle Python API directly (avoids PATH / subprocess issues)
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        print("  Authenticated with Kaggle OK")
        api.dataset_download_files(
            "awsaf49/asvpoof-2019-dataset",
            path=str(dest),
            unzip=True,
            quiet=False
        )
        print("[+] ASVspoof DONE\n")
    except Exception as e:
        print(f"  [!] Kaggle API download failed: {e}")
        # Fallback: try to find kaggle.exe in common Windows user Scripts locations
        import glob as _glob
        import site
        kaggle_exe = None
        search_dirs = []
        try:
            for s in site.getsitepackages():
                search_dirs.append(os.path.join(os.path.dirname(s), "Scripts"))
        except Exception:
            pass
        try:
            search_dirs.append(os.path.join(os.path.dirname(site.getusersitepackages()), "Scripts"))
        except Exception:
            pass
        for d in search_dirs:
            hits = _glob.glob(os.path.join(d, "kaggle.exe"))
            if hits:
                kaggle_exe = hits[0]
                break

        if kaggle_exe:
            print(f"  Found kaggle CLI at: {kaggle_exe}")
            print("  Retrying with CLI...")
            try:
                import subprocess
                subprocess.run(
                    [kaggle_exe, "datasets", "download",
                     "-d", "awsaf49/asvpoof-2019-dataset",
                     "-p", str(dest), "--unzip"],
                    check=True
                )
                print("[+] ASVspoof DONE\n")
            except Exception as e2:
                print(f"  [!] CLI also failed: {e2}")
                print("  Manual option: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset")
        else:
            print("  [!] Could not locate kaggle.exe on this system.")
            print("  Manual option: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset")


def download_wavefake(max_samples: int = 5000):
    """WaveFake — Multi-vocoder synthetic speech via HuggingFace datasets library (~30GB).
    
    Uses streaming + save to avoid loading everything into RAM.
    Set max_samples to limit how many clips you want (full = ~104k clips).
    """
    print("\n" + "="*60)
    print("[3/6] WaveFake (FAKE) — HuggingFace datasets library")
    print("="*60)
    dest = BASE_DIR / "fake" / "wavefake"
    dest.mkdir(parents=True, exist_ok=True)

    ensure_deps(["datasets", "soundfile"])

    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"  [!] Missing library: {e}")
        print("  Run: pip install datasets soundfile")
        return

    print(f"  Repo:   ajaykarthick/wavefake-audio")
    print(f"  Dest:   {dest}")
    print(f"  Limit:  {max_samples} samples (pass --wavefake-samples N to change)")
    print(f"  Note:   Streams directly, resumable by re-running\n")

    # Count already-saved files for resume
    existing = list(dest.glob("*.wav"))
    start_idx = len(existing)
    if start_idx >= max_samples:
        print(f"  Already have {start_idx} files. Done.")
        print("[+] WaveFake DONE\n")
        return
    if start_idx > 0:
        print(f"  Resuming from {start_idx} existing files...")

    try:
        from tqdm import tqdm
        ds = load_dataset(
            "ajaykarthick/wavefake-audio",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        saved = start_idx
        skipped = 0
        target = max_samples - start_idx

        with tqdm(total=target, desc="Saving WaveFake audio", unit="clip") as bar:
            for i, sample in enumerate(ds):
                if i < start_idx:
                    skipped += 1
                    continue  # Skip already saved
                if saved >= max_samples:
                    break

                try:
                    audio = sample["audio"]
                    arr = np.array(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                    out_path = dest / f"wavefake_{saved:06d}.wav"
                    sf.write(str(out_path), arr, sr, subtype='PCM_16')
                    saved += 1
                    bar.update(1)
                except KeyboardInterrupt:
                    print(f"\n  Paused at {saved} files. Re-run to resume.")
                    sys.exit(0)
                except Exception as e:
                    continue  # Skip bad samples silently

        print(f"\n  Saved {saved} WaveFake clips to {dest}")
        print("[+] WaveFake DONE\n")

    except KeyboardInterrupt:
        print("\n  Paused. Re-run same command to resume.\n")
        sys.exit(0)
    except Exception as e:
        print(f"  [!] WaveFake download failed: {e}")
        print("  Try: pip install datasets soundfile")


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
    parser.add_argument("--all",              action="store_true", help="Download ALL datasets")
    parser.add_argument("--librispeech",      action="store_true", help="LibriSpeech (~36GB, REAL)")
    parser.add_argument("--asvspoof",         action="store_true", help="ASVspoof 2019 LA (~7.7GB, FAKE) [needs Kaggle API]")
    parser.add_argument("--wavefake",         action="store_true", help="WaveFake via HF datasets (~30GB, FAKE)")
    parser.add_argument("--wavefake-samples", type=int, default=5000, metavar="N", help="How many WaveFake clips to save (default: 5000)")
    parser.add_argument("--common_voice",     action="store_true", help="Common Voice (~65GB, REAL)")
    parser.add_argument("--voxceleb",         action="store_true", help="VoxCeleb1 public subset (~1.2GB, REAL)")
    parser.add_argument("--fakeavceleb",      action="store_true", help="FakeAVCeleb — prints manual download instructions")
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
    if args.all or args.wavefake:      download_wavefake(max_samples=args.wavefake_samples)
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
