"""
DeepFake Detection System - Data Ingestion Pipeline
Layer 1: Data Acquisition

This module handles ingestion of new media samples into the dataset
with validation, hashing, and provenance tracking.
"""

import os
import uuid
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from schema import (
    MediaSample,
    DatasetCategory,
    MediaType,
    DatasetManifest,
    TransformationStep,
)


# Supported file extensions per media type
SUPPORTED_FORMATS = {
    MediaType.IMAGE: {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"},
    MediaType.VIDEO: {".mp4", ".avi", ".mov", ".mkv", ".webm"},
    MediaType.AUDIO: {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
}

# Base data directory
DATA_ROOT = Path(__file__).parent.parent.parent / "data"


class IngestionError(Exception):
    """Raised when ingestion validation fails."""

    pass


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_media_type(file_path: str) -> Optional[MediaType]:
    """Detect media type from file extension."""
    ext = Path(file_path).suffix.lower()
    for media_type, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return media_type
    return None


def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file exists and is a supported format.
    Returns (is_valid, error_message).
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    if not os.path.isfile(file_path):
        return False, f"Not a file: {file_path}"

    media_type = detect_media_type(file_path)
    if media_type is None:
        ext = Path(file_path).suffix
        return False, f"Unsupported format: {ext}"

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "File is empty (0 bytes)"

    return True, ""


def get_category_path(category: DatasetCategory) -> Path:
    """Get the storage path for a dataset category."""
    category_dirs = {
        DatasetCategory.HUMAN_REAL: "human_real",
        DatasetCategory.KNOWN_SYNTH: "known_synth",
        DatasetCategory.UNKNOWN_SYNTH: "unknown_synth",
        DatasetCategory.ADVERSARIAL: "adversarial",
    }
    return DATA_ROOT / category_dirs[category]


def ingest_sample(
    source_path: str,
    category: DatasetCategory,
    source_url: Optional[str] = None,
    source_verified: bool = False,
    generator_model: Optional[str] = None,
    generator_version: Optional[str] = None,
    copy_file: bool = True,
) -> MediaSample:
    """
    Ingest a new media sample into the dataset.

    Args:
        source_path: Path to the source file
        category: Dataset category (HUMAN_REAL, KNOWN_SYNTH, etc.)
        source_url: Optional URL where the file was obtained
        source_verified: Whether the source has been verified
        generator_model: For synthetic media, the generator model name
        generator_version: For synthetic media, the generator version
        copy_file: If True, copy file to data directory. If False, use in-place.

    Returns:
        MediaSample object with full metadata

    Raises:
        IngestionError: If validation fails
    """
    # Validate input
    is_valid, error = validate_file(source_path)
    if not is_valid:
        raise IngestionError(error)

    # Detect media type
    media_type = detect_media_type(source_path)

    # Generate sample ID
    sample_id = str(uuid.uuid4())

    # Compute hash of original
    original_hash = compute_sha256(source_path)

    # Get file info
    file_size = os.path.getsize(source_path)
    file_ext = Path(source_path).suffix.lower().lstrip(".")

    # Determine destination path
    category_path = get_category_path(category)
    category_path.mkdir(parents=True, exist_ok=True)

    if copy_file:
        dest_filename = f"{sample_id}.{file_ext}"
        dest_path = category_path / dest_filename
        shutil.copy2(source_path, dest_path)
        stored_path = str(dest_path)
    else:
        stored_path = source_path

    # Determine label based on category
    if category == DatasetCategory.HUMAN_REAL:
        label = "real"
    else:
        label = "synthetic"

    # Create sample object
    sample = MediaSample(
        sample_id=sample_id,
        category=category,
        media_type=media_type,
        original_path=stored_path,
        original_hash=original_hash,
        original_size_bytes=file_size,
        original_format=file_ext,
        source_url=source_url,
        source_verified=source_verified,
        generator_model=generator_model,
        generator_version=generator_version,
        label=label,
        label_confidence=1.0 if source_verified else 0.8,
    )

    return sample


def create_variant(
    sample: MediaSample,
    variant_path: str,
    operation: str,
    parameters: dict,
    tool_used: Optional[str] = None,
    tool_version: Optional[str] = None,
) -> MediaSample:
    """
    Register a variant (post-processed version) of an existing sample.

    Args:
        sample: The original MediaSample
        variant_path: Path to the variant file
        operation: Name of the transformation (e.g., "recompress", "resize")
        parameters: Operation parameters
        tool_used: Tool name (e.g., "ffmpeg")
        tool_version: Tool version

    Returns:
        Updated MediaSample with variant registered
    """
    # Validate variant file
    is_valid, error = validate_file(variant_path)
    if not is_valid:
        raise IngestionError(f"Variant validation failed: {error}")

    # Compute variant hash
    variant_hash = compute_sha256(variant_path)

    # Create transformation step
    step = TransformationStep(
        step_id=len(sample.transformation_history) + 1,
        operation=operation,
        parameters=parameters,
        timestamp=datetime.utcnow(),
        input_hash=(
            sample.original_hash
            if not sample.variants
            else compute_sha256(sample.variants[-1])
        ),
        output_hash=variant_hash,
        tool_used=tool_used,
        tool_version=tool_version,
    )

    # Update sample
    sample.add_transformation(step)
    sample.add_variant(variant_path)

    return sample


def load_or_create_manifest(category: DatasetCategory) -> DatasetManifest:
    """Load existing manifest or create a new one."""
    category_path = get_category_path(category)
    manifest_path = category_path / "manifest.json"

    if manifest_path.exists():
        return DatasetManifest.load(str(manifest_path))
    else:
        return DatasetManifest(category=category)


def save_manifest(manifest: DatasetManifest) -> None:
    """Save manifest to its category directory."""
    category_path = get_category_path(manifest.category)
    category_path.mkdir(parents=True, exist_ok=True)
    manifest_path = category_path / "manifest.json"
    manifest.save(str(manifest_path))


# ============================================================
# CLI Interface for testing
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest media samples into the dataset"
    )
    parser.add_argument("file", help="Path to the media file")
    parser.add_argument(
        "--category",
        "-c",
        required=True,
        choices=["human_real", "known_synth", "unknown_synth", "adversarial"],
        help="Dataset category",
    )
    parser.add_argument("--source-url", help="Source URL")
    parser.add_argument(
        "--verified", action="store_true", help="Mark source as verified"
    )
    parser.add_argument("--generator", help="Generator model name (for synthetic)")
    parser.add_argument("--generator-version", help="Generator version")

    args = parser.parse_args()

    # Map category string to enum
    category_map = {
        "human_real": DatasetCategory.HUMAN_REAL,
        "known_synth": DatasetCategory.KNOWN_SYNTH,
        "unknown_synth": DatasetCategory.UNKNOWN_SYNTH,
        "adversarial": DatasetCategory.ADVERSARIAL,
    }
    category = category_map[args.category]

    try:
        # Ingest sample
        sample = ingest_sample(
            source_path=args.file,
            category=category,
            source_url=args.source_url,
            source_verified=args.verified,
            generator_model=args.generator,
            generator_version=args.generator_version,
        )

        # Load/create manifest and add sample
        manifest = load_or_create_manifest(category)
        manifest.add_sample(sample)
        save_manifest(manifest)

        print(f"SUCCESS: Ingested sample {sample.sample_id}")
        print(f"  Category: {category.value}")
        print(f"  Hash: {sample.original_hash[:16]}...")
        print(f"  Path: {sample.original_path}")

    except IngestionError as e:
        print(f"ERROR: {e}")
        exit(1)
