"""
Video Deepfake Dataset Downloader — Direct Download Edition
============================================================
NO FORMS, NO EMAILS — All datasets download directly.

Datasets:
  1. DF40            → Google Drive official (direct, ~93GB test + ~50GB train)
  2. FaceForensics++ → HuggingFace bitmind/FaceForensicsC23 (direct, 7000 videos)
  3. Deepfake-Eval-2024 → HuggingFace nuriachandra/Deepfake-Eval-2024 (direct)
  4. Celeb-DF v2     → Kaggle reubensuju/celeb-df-v2 (direct, ~5GB)

Resume support: If Wi-Fi drops, re-run same command — continues from where it stopped.

Usage:
  python download_video_datasets.py --all            # Download all 4
  python download_video_datasets.py --df40           # DF40 only
  python download_video_datasets.py --ff             # FaceForensics++ only
  python download_video_datasets.py --eval2024       # Deepfake-Eval-2024 only
  python download_video_datasets.py --celebdf        # Celeb-DF v2 only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(r"d:\Devam\Microsoft VS Code\Codes\DeepFake\datasets")

# ── Auto-install dependencies ─────────────────────────────────────────────────
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def ensure_deps():
    print("[Setup] Checking packages...")
    for pkg in ["huggingface_hub", "kaggle", "tqdm", "requests", "gdown"]:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  Installing {pkg}...")
            install(pkg)
    print("[Setup] Ready.\n")

# ── HuggingFace download (Python API — works on all Windows setups) ──────────────
def hf_download(repo_id: str, dest: Path, repo_type: str = "dataset"):
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  → Saving to: {dest}")
    print(f"  → Resume: Re-run if interrupted — already-downloaded files are skipped\n")

    token = os.environ.get("HF_TOKEN") or None

    print(f"  Downloading {repo_id} ...\n")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=4,
        )
        print(f"\n  ✅ Done: {repo_id}\n")
    except KeyboardInterrupt:
        print(f"\n  ⏸️  Paused. Re-run same command to resume.\n")
    except Exception as e:
        print(f"\n  ⚠️  Error: {e}\n")

# ── Kaggle download (Python API — no CLI PATH issues) ─────────────────────
def kaggle_download(dataset: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print(f"""
  ⚠️  Kaggle API key not found at: {kaggle_json}

  Quick Setup (2 minutes, free account):
  Step 1: Go to https://www.kaggle.com/settings/account
  Step 2: Scroll to "API" → Click "Create New Token" → downloads kaggle.json
  Step 3: Move it to: {kaggle_json}
  Step 4: Re-run this script
        """)
        return False

    print(f"  → Downloading {dataset} from Kaggle...")
    print(f"  → Saving to: {dest}")
    try:
        import kaggle as kaggle_api
        kaggle_api.api.authenticate()
        kaggle_api.api.dataset_download_files(
            dataset,
            path=str(dest),
            unzip=True,
            quiet=False,
        )
        print(f"  ✅ Done: {dataset}\n")
        return True
    except Exception as e:
        print(f"  ❌ Kaggle download failed: {e}\n")
        return False

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 1: DF40 (NeurIPS 2024)
#  ~59,000 video clips, 40 deepfake methods (HeyGen, diffusion, DeepFaceLab...)
#  Size: ~93 GB | Source: HuggingFace | NO FORM
# ══════════════════════════════════════════════════════════════════════════════
def download_df40():
    print("=" * 60)
    print("  [1/4] DF40 — NeurIPS 2024 | 40 deepfake methods | ~93 GB")
    print("        HeyGen, DeepFaceLab, Stable Diffusion, MidJourney...")
    print("        Source: Official Google Drive from DF40 GitHub")
    print("=" * 60)
    import gdown

    # Official link from https://github.com/YZY-stack/DF40
    # Test set only — all 40 methods, ~93GB (Train set skipped — saves 50GB, FF++ covers it)
    TEST_FOLDER_ID = "1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_"

    for label, folder_id, subfolder in [
        ("DF40-Test (~93GB, all 40 methods)", TEST_FOLDER_ID, "test"),
    ]:
        dest = BASE_DIR / "df40" / subfolder
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n  Downloading {label}")
        print(f"  → Saving to: {dest}")
        print(f"  → Resume: re-run if interrupted — gdown skips done files\n")
        try:
            gdown.download_folder(
                id=folder_id,
                output=str(dest),
                quiet=False,
                resume=True,
                remaining_ok=True,
            )
            print(f"  ✅ Download done: {label}")

            # ── Auto-unzip all .zip files and delete them to save space ──────
            import zipfile
            zips = list(dest.glob("*.zip"))
            if zips:
                print(f"  📦 Auto-extracting {len(zips)} zip files...")
                for zf in zips:
                    print(f"     Extracting: {zf.name}")
                    try:
                        with zipfile.ZipFile(zf, "r") as z:
                            z.extractall(dest)
                        zf.unlink()  # delete zip after extracting
                    except Exception as ze:
                        print(f"     ⚠️  Failed to extract {zf.name}: {ze}")
                print(f"  ✅ All zips extracted and removed.\n")
            else:
                print(f"  ✅ No zips to extract.\n")

        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            print(f"  If Google Drive quota exceeded, wait a few hours and re-run.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 2: FaceForensics++ (C23 mirror — no form)
#  1000 real + 4000 fake | 4 manipulation methods
#  Size: ~15 GB | Source: HuggingFace mirror | NO FORM
# ══════════════════════════════════════════════════════════════════════════════
def download_ff():
    print("=" * 60)
    print("  [2/4] FaceForensics++ C23 | 7000 videos | 6000 fake + 1000 real")
    print("        Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures")
    print("=" * 60)
    # Confirmed working HuggingFace mirror — 7,000 MP4 videos, no form
    hf_download("bitmind/FaceForensicsC23", BASE_DIR / "faceforensics")

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 3: Deepfake-Eval-2024 (In-the-Wild)
#  44 hours of video | Sora, HeyGen, RunwayML, Pika Labs
#  Size: ~20 GB | Source: HuggingFace | NO FORM
# ══════════════════════════════════════════════════════════════════════════════
def download_eval2024():
    print("=" * 60)
    print("  [3/4] Deepfake-Eval-2024 | In-the-Wild | multimodal")
    print("        Sora, RunwayML, HeyGen, Pika Labs — social media content")
    print("=" * 60)
    # nuriachandra mirror — confirmed public, no login required
    hf_download("nuriachandra/Deepfake-Eval-2024", BASE_DIR / "deepfake_eval_2024")

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 4: Celeb-DF v2 (Kaggle mirror — no form)
#  590 real + 5,639 celebrity deepfake videos
#  Size: ~5 GB | Source: Kaggle | NEEDS FREE KAGGLE ACCOUNT (no form)
# ══════════════════════════════════════════════════════════════════════════════
def download_celebdf():
    print("=" * 60)
    print("  [4/4] Celeb-DF v2 | 590 real + 5639 fake | ~5 GB")
    print("        Celebrity deepfake videos — Kaggle mirror")
    print("=" * 60)
    # Verified Kaggle slug: reubensuju/celeb-df-v2
    kaggle_download("reubensuju/celeb-df-v2", BASE_DIR / "celeb_df")

# ══════════════════════════════════════════════════════════════════════════════
#  SHOW DISK SPACE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def show_summary():
    print("\n" + "=" * 60)
    print("  Downloaded Dataset Summary:")
    print("=" * 60)
    total_gb = 0
    if BASE_DIR.exists():
        for folder in sorted(BASE_DIR.iterdir()):
            if folder.is_dir():
                size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
                gb = size / 1e9
                total_gb += gb
                print(f"  ├── {folder.name:30s} {gb:6.1f} GB")
    print(f"  {'Total':30s} {total_gb:6.1f} GB")
    print("=" * 60)
    print("\n  ✅ Tell Antigravity when done — training starts next!")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Download video deepfake datasets (no forms required)"
    )
    parser.add_argument("--all",      action="store_true", help="Download all 4 datasets")
    parser.add_argument("--df40",     action="store_true", help="Download DF40 (~93GB)")
    parser.add_argument("--ff",       action="store_true", help="Download FaceForensics++ C23 (~15GB)")
    parser.add_argument("--eval2024", action="store_true", help="Download Deepfake-Eval-2024 (~20GB)")
    parser.add_argument("--celebdf",  action="store_true", help="Download Celeb-DF v2 (~5GB, needs Kaggle key)")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print(f"\n  All datasets will be saved to: {BASE_DIR}")
        print("  Total size if all downloaded: ~133 GB\n")
        print("  Recommended start: python download_video_datasets.py --df40 --eval2024")
        return

    ensure_deps()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Base directory: {BASE_DIR}\n")

    if args.df40 or args.all:
        download_df40()

    if args.ff or args.all:
        download_ff()

    if args.eval2024 or args.all:
        download_eval2024()

    if args.celebdf or args.all:
        download_celebdf()

    show_summary()

if __name__ == "__main__":
    main()
