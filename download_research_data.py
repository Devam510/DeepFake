"""
Research Dataset Downloader
============================

Downloads diverse AI-generated image datasets for training
state-of-the-art detection models.

Datasets:
- Hugging Face: ehristoforu/midjourney-images
- Hugging Face: ehristoforu/dalle-3-images
- Existing: data/prepared (StyleGAN faces)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Configuration
RESEARCH_DATA_DIR = Path("data/research")
MAX_IMAGES_PER_SOURCE = 2000  # Limit to manage training time


def setup_directories():
    """Create research data directory structure."""
    print("\n" + "=" * 60)
    print("SETTING UP RESEARCH DATA DIRECTORIES")
    print("=" * 60)

    dirs = [
        RESEARCH_DATA_DIR / "real",
        RESEARCH_DATA_DIR / "synthetic" / "faces",
        RESEARCH_DATA_DIR / "synthetic" / "scenes",
        RESEARCH_DATA_DIR / "synthetic" / "art",
        RESEARCH_DATA_DIR / "by_generator" / "stylegan",
        RESEARCH_DATA_DIR / "by_generator" / "stable_diffusion",
        RESEARCH_DATA_DIR / "by_generator" / "midjourney",
        RESEARCH_DATA_DIR / "by_generator" / "dalle3",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")

    return True


def download_huggingface_dataset(dataset_name: str, target_dir: Path, max_images: int):
    """Download images from Hugging Face dataset."""
    print(f"\n📥 Downloading: {dataset_name}")
    print(f"   Target: {target_dir}")
    print(f"   Max images: {max_images}")

    try:
        from datasets import load_dataset

        # Load dataset
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        count = 0
        for item in dataset:
            if count >= max_images:
                break

            # Get image
            if "image" in item:
                img = item["image"]

                # Save image
                filename = f"{dataset_name.replace('/', '_')}_{count:05d}.jpg"
                filepath = target_dir / filename

                img.save(filepath, "JPEG", quality=95)
                count += 1

                if count % 100 == 0:
                    print(f"   Downloaded: {count}/{max_images}")

        print(f"   ✓ Downloaded {count} images")
        return count

    except ImportError:
        print("   ❌ datasets library not installed. Installing...")
        os.system("pip install datasets")
        return download_huggingface_dataset(dataset_name, target_dir, max_images)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0


def copy_existing_stylegan_data():
    """Copy existing StyleGAN data to research directory."""
    print("\n📋 Copying existing StyleGAN data...")

    src_fake = Path("data/prepared/synthetic_known")
    src_real = Path("data/prepared/real_unverified")

    dst_stylegan = RESEARCH_DATA_DIR / "by_generator" / "stylegan"
    dst_faces = RESEARCH_DATA_DIR / "synthetic" / "faces"
    dst_real = RESEARCH_DATA_DIR / "real"

    copied = 0

    # Copy fake (StyleGAN) images
    if src_fake.exists():
        files = list(src_fake.glob("*.jpg"))[:MAX_IMAGES_PER_SOURCE]
        for f in files:
            shutil.copy2(f, dst_stylegan / f.name)
            shutil.copy2(f, dst_faces / f.name)
            copied += 1
        print(f"   ✓ Copied {len(files)} StyleGAN images")

    # Copy real images
    if src_real.exists():
        files = list(src_real.glob("*.jpg"))[:MAX_IMAGES_PER_SOURCE]
        for f in files:
            shutil.copy2(f, dst_real / f.name)
            copied += 1
        print(f"   ✓ Copied {len(files)} real images")

    return copied


def create_metadata():
    """Create metadata file for research dataset."""
    print("\n📝 Creating metadata...")

    metadata = {
        "created": datetime.now().isoformat(),
        "sources": {},
        "counts": {},
    }

    # Count images in each directory
    for gen_dir in (RESEARCH_DATA_DIR / "by_generator").iterdir():
        if gen_dir.is_dir():
            count = len(list(gen_dir.glob("*.jpg")))
            metadata["counts"][gen_dir.name] = count
            metadata["sources"][gen_dir.name] = str(gen_dir)

    # Count domain directories
    for domain_dir in (RESEARCH_DATA_DIR / "synthetic").iterdir():
        if domain_dir.is_dir():
            count = len(list(domain_dir.glob("*.jpg")))
            metadata["counts"][f"domain_{domain_dir.name}"] = count

    # Real images
    real_count = len(list((RESEARCH_DATA_DIR / "real").glob("*.jpg")))
    metadata["counts"]["real"] = real_count

    # Save
    with open(RESEARCH_DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n📊 Dataset Summary:")
    for key, count in metadata["counts"].items():
        print(f"   {key}: {count}")

    return metadata


def main():
    print("\n" + "=" * 60)
    print("RESEARCH DATASET DOWNLOADER")
    print("=" * 60)

    # Setup directories
    setup_directories()

    # Copy existing data
    copy_existing_stylegan_data()

    # Download Hugging Face datasets
    datasets_to_download = [
        (
            "ehristoforu/midjourney-images",
            RESEARCH_DATA_DIR / "by_generator" / "midjourney",
        ),
        ("ehristoforu/dalle-3-images", RESEARCH_DATA_DIR / "by_generator" / "dalle3"),
    ]

    for dataset_name, target_dir in datasets_to_download:
        download_huggingface_dataset(dataset_name, target_dir, MAX_IMAGES_PER_SOURCE)

    # Create metadata
    metadata = create_metadata()

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nTotal images: {sum(metadata['counts'].values())}")
    print(f"Location: {RESEARCH_DATA_DIR.absolute()}")

    return True


if __name__ == "__main__":
    main()
