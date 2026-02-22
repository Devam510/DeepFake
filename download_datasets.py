"""
Dataset Downloader for DeepFake Detection Training
====================================================

Downloads large-scale datasets for retraining EfficientNet to handle:
- Filtered real photos (Snapchat, Instagram, etc.)
- AI-generated images (DALL-E, Midjourney, Stable Diffusion)
- Clean real photos for baseline

Total: ~400K+ images across multiple datasets.

Prerequisites:
    pip install kaggle huggingface_hub gdown tqdm requests

Kaggle Setup:
    1. Go to https://www.kaggle.com/settings → Create New API Token
    2. Save kaggle.json to C:\\Users\\<username>\\.kaggle\\kaggle.json
"""

import os
import sys
import subprocess
import zipfile
import shutil
import argparse
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = BASE_DIR / "datasets"

# Verified datasets with download info
DATASETS = {
    # ── KAGGLE DATASETS ──────────────────────────────────────
    "cifake": {
        "source": "kaggle",
        "id": "birdy654/cifake-real-and-ai-generated-synthetic-images",
        "description": "120K images (60K real CIFAR-10 + 60K AI-generated)",
        "size": "~600 MB",
        "categories": ["real", "ai_generated"],
    },
    "ai_vs_human": {
        "source": "kaggle",
        "id": "alessandrasala79/ai-vs-human-generated-dataset",
        "description": "80K images (40K real + 40K AI) balanced dataset",
        "size": "~2 GB",
        "categories": ["real", "ai_generated"],
    },
    "real_vs_ai_faces": {
        "source": "kaggle",
        "id": "philosopher0808/real-vs-ai-generated-faces-dataset",
        "description": "120K+ face images (real vs AI-generated faces)",
        "size": "~3 GB",
        "categories": ["real_faces", "ai_faces"],
    },
    "artifact_dataset": {
        "source": "kaggle",
        "id": "awsaf49/artifact-dataset",
        "description": "Real and fake images from multiple AI generators (GAN, Diffusion)",
        "size": "~5 GB",
        "categories": ["real", "ai_generated"],
    },
    "detect_ai_vs_human": {
        "source": "kaggle",
        "id": "yanivgorshtein/detect-ai-vs-human-generated-images",
        "description": "100K images (Shutterstock real + AI equivalents)",
        "size": "~4 GB",
        "categories": ["real", "ai_generated"],
    },

    # ── HUGGINGFACE DATASETS ──────────────────────────────────
    "ms_cocoai": {
        "source": "huggingface",
        "id": "InfImagine/MS-COCOAI",
        "description": "96K real+synthetic (SD3, SDXL, DALL-E 3, MidJourney v6)",
        "size": "~8 GB",
        "categories": ["real", "sd3", "sdxl", "dalle3", "midjourney_v6"],
    },
}


def print_banner():
    print("=" * 64)
    print("  DeepFake Detection — Dataset Downloader")
    print("=" * 64)
    print(f"  Download directory: {DATASETS_DIR}")
    print()


def check_dependencies():
    """Check and install required packages."""
    required = ["kaggle", "huggingface_hub", "gdown", "tqdm", "requests"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("  ✅ Dependencies installed\n")
    else:
        print("  ✅ All dependencies available\n")


def check_kaggle_auth():
    """Check if Kaggle API credentials are set up."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        print("  ✅ Kaggle credentials found")
        return True

    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("  ✅ Kaggle credentials found (env vars)")
        return True

    print("  ❌ Kaggle credentials NOT found!")
    print()
    print("  To set up Kaggle API:")
    print("    1. Go to https://www.kaggle.com/settings")
    print("    2. Scroll to 'API' → Click 'Create New API Token'")
    print(f"    3. Save downloaded kaggle.json to: {kaggle_json}")
    print()
    return False


def check_huggingface_auth():
    """Check if HuggingFace credentials are available."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Check if token exists
        token = api.token
        if token:
            print("  ✅ HuggingFace credentials found")
            return True
    except Exception:
        pass

    print("  ⚠️  HuggingFace credentials not found (optional)")
    print("      Some datasets may need: huggingface-cli login")
    return False


def download_kaggle_dataset(dataset_id, output_dir):
    """Download a dataset from Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"    Downloading from Kaggle: {dataset_id}")
    print(f"    Destination: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    api.dataset_download_files(
        dataset_id,
        path=str(output_dir),
        unzip=True,
        quiet=False,
    )

    # Clean up zip files
    for f in Path(output_dir).glob("*.zip"):
        f.unlink()

    return True


def download_huggingface_dataset(dataset_id, output_dir):
    """Download a dataset from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"    Downloading from HuggingFace: {dataset_id}")
    print(f"    Destination: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(output_dir),
    )

    return True


def count_images(directory):
    """Count image files in a directory recursively."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                count += 1
    return count


def list_datasets():
    """List all available datasets."""
    print("\n  Available Datasets:")
    print("  " + "-" * 60)

    total_datasets = 0
    for key, info in DATASETS.items():
        status = "✅ Downloaded" if (DATASETS_DIR / key).exists() else "⬜ Not downloaded"
        print(f"\n  [{key}]")
        print(f"    Source:      {info['source'].upper()}")
        print(f"    ID:          {info['id']}")
        print(f"    Description: {info['description']}")
        print(f"    Size:        {info['size']}")
        print(f"    Status:      {status}")

        if (DATASETS_DIR / key).exists():
            img_count = count_images(DATASETS_DIR / key)
            print(f"    Images:      {img_count:,}")

        total_datasets += 1

    print(f"\n  Total datasets available: {total_datasets}")
    print()


def download_dataset(key):
    """Download a single dataset by key."""
    if key not in DATASETS:
        print(f"  ❌ Unknown dataset: {key}")
        print(f"  Available: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[key]
    output_dir = DATASETS_DIR / key

    if output_dir.exists():
        img_count = count_images(output_dir)
        if img_count > 0:
            print(f"  ⏭️  {key}: Already downloaded ({img_count:,} images)")
            return True

    print(f"\n  📥 Downloading: {key}")
    print(f"     {info['description']}")
    print(f"     Expected size: {info['size']}")
    print()

    try:
        if info["source"] == "kaggle":
            download_kaggle_dataset(info["id"], output_dir)
        elif info["source"] == "huggingface":
            download_huggingface_dataset(info["id"], output_dir)
        else:
            print(f"  ❌ Unknown source: {info['source']}")
            return False

        img_count = count_images(output_dir)
        print(f"\n  ✅ {key}: Downloaded successfully ({img_count:,} images)")
        return True

    except Exception as e:
        print(f"\n  ❌ {key}: Download failed — {e}")
        return False


def download_all():
    """Download all datasets."""
    print("\n  Downloading ALL datasets...")
    print("  This will download ~20+ GB of data.\n")

    results = {}
    for key in DATASETS:
        success = download_dataset(key)
        results[key] = success

    # Summary
    print("\n" + "=" * 64)
    print("  DOWNLOAD SUMMARY")
    print("=" * 64)

    total_images = 0
    for key, success in results.items():
        status = "✅" if success else "❌"
        img_count = count_images(DATASETS_DIR / key) if success else 0
        total_images += img_count
        print(f"  {status} {key:25s} — {img_count:>8,} images")

    print(f"\n  Total images downloaded: {total_images:,}")
    print("=" * 64)


def download_recommended():
    """Download the recommended subset for filter-aware retraining."""
    print("\n  Downloading RECOMMENDED datasets for filter-aware training...")
    print("  Focus: Real faces, AI faces, filtered photos, diverse AI generators\n")

    # Priority order for the user's specific problem
    recommended = [
        "real_vs_ai_faces",    # 120K faces — most relevant for face filter issue
        "cifake",              # 120K general images — diverse real vs AI
        "ai_vs_human",         # 80K balanced — good variety
        "detect_ai_vs_human",  # 100K Shutterstock — high quality real photos
    ]

    results = {}
    for key in recommended:
        success = download_dataset(key)
        results[key] = success

    # Summary
    print("\n" + "=" * 64)
    print("  DOWNLOAD SUMMARY (Recommended)")
    print("=" * 64)

    total_images = 0
    for key, success in results.items():
        status = "✅" if success else "❌"
        img_count = count_images(DATASETS_DIR / key) if success else 0
        total_images += img_count
        print(f"  {status} {key:25s} — {img_count:>8,} images")

    print(f"\n  Total images downloaded: {total_images:,}")
    print(f"  Download location: {DATASETS_DIR}")
    print("=" * 64)


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets for DeepFake detection training"
    )
    parser.add_argument(
        "action",
        choices=["list", "download", "all", "recommended"],
        help=(
            "list: Show available datasets | "
            "download: Download specific dataset | "
            "recommended: Download recommended set (~10GB) | "
            "all: Download everything (~20GB+)"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset key to download (use with 'download' action)",
    )

    args = parser.parse_args()

    print_banner()
    check_dependencies()

    if args.action == "list":
        list_datasets()

    elif args.action == "download":
        if not args.dataset:
            print("  ❌ Must specify --dataset <key>")
            print(f"  Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

        has_kaggle = check_kaggle_auth()
        if DATASETS.get(args.dataset, {}).get("source") == "kaggle" and not has_kaggle:
            sys.exit(1)

        download_dataset(args.dataset)

    elif args.action == "recommended":
        if not check_kaggle_auth():
            sys.exit(1)
        download_recommended()

    elif args.action == "all":
        has_kaggle = check_kaggle_auth()
        has_hf = check_huggingface_auth()

        if not has_kaggle:
            print("  ❌ Kaggle auth required for most datasets. Set up credentials first.")
            sys.exit(1)

        download_all()
