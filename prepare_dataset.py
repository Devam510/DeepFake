"""
DFDC Dataset Preparation Script - FULL DATASET
- Copies ALL images from source dataset
- Labels properly:
  - Real -> real_unverified
  - Fake -> synthetic_known
"""

import os
import shutil
import json
from pathlib import Path

# Paths
SOURCE_REAL = (
    r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\deepfake_faces\Final Dataset\Real"
)
SOURCE_FAKE = (
    r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\deepfake_faces\Final Dataset\Fake"
)
DEST_REAL = r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\prepared\real_unverified"
DEST_FAKE = r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\prepared\synthetic_known"
METADATA_FILE = r"d:\Devam\Microsoft VS Code\Codes\DeepFake\data\prepared\metadata.json"


def ensure_dirs():
    """Ensure destination directories exist and are empty."""
    for folder in [DEST_REAL, DEST_FAKE]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)


def get_image_files(folder):
    """Get all image files from a folder."""
    return [
        f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def copy_all_images(source_folder, dest_folder, label):
    """Copy ALL images from source to destination."""
    files = get_image_files(source_folder)

    copied_files = []
    for i, filename in enumerate(files):
        src = os.path.join(source_folder, filename)
        dst = os.path.join(dest_folder, filename)
        shutil.copy2(src, dst)
        copied_files.append({"filename": filename, "label": label})
        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{len(files)} {label} images...")

    print(f"  Copied {len(files)}/{len(files)} {label} images... DONE")
    return copied_files


def main():
    print("=" * 60)
    print("FULL DATASET Preparation - All 12,890 Images")
    print("=" * 60)

    # Clear and recreate directories
    print("\nClearing previous data...")
    ensure_dirs()

    # Check source counts
    real_count = len(get_image_files(SOURCE_REAL))
    fake_count = len(get_image_files(SOURCE_FAKE))
    print(f"\nSource counts:")
    print(f"  Real images: {real_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total: {real_count + fake_count}")

    metadata = {
        "source_dataset": "kshitizbhargava/deepfake-face-images",
        "license": "MIT",
        "dataset_type": "FULL",
        "labels": {
            "real_unverified": "Original human photos - source authenticity unverified",
            "synthetic_known": "StyleGAN-generated synthetic faces",
        },
        "counts": {},
        "files": [],
    }

    # Copy ALL real images -> real_unverified
    print(f"\nCopying ALL Real images -> real_unverified...")
    real_files = copy_all_images(SOURCE_REAL, DEST_REAL, "real_unverified")
    metadata["files"].extend(real_files)
    metadata["counts"]["real_unverified"] = len(real_files)

    # Copy ALL fake images -> synthetic_known
    print(f"\nCopying ALL Fake images -> synthetic_known...")
    fake_files = copy_all_images(SOURCE_FAKE, DEST_FAKE, "synthetic_known")
    metadata["files"].extend(fake_files)
    metadata["counts"]["synthetic_known"] = len(fake_files)

    # Save metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("FULL DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal Dataset State:")
    print(f"  real_unverified:  {len(real_files):,} images")
    print(f"  synthetic_known:  {len(fake_files):,} images")
    print(f"  TOTAL:            {len(real_files) + len(fake_files):,} images")
    print(f"\nMetadata saved to: {METADATA_FILE}")
    print("\n✅ Dataset ready for training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
