"""
DeepFake Detection System - Data Schema Definitions
Layer 1: Data Acquisition

This module defines the database schema for storing media samples
with full provenance tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum
import hashlib
import json


class DatasetCategory(Enum):
    """Four mandatory dataset categories per Layer 1 specification."""
    HUMAN_REAL = "human_real"           # Verified human-authored media
    KNOWN_SYNTH = "known_synth"         # Known generator (SD, Midjourney, etc.)
    UNKNOWN_SYNTH = "unknown_synth"     # Unknown/fine-tuned generators
    ADVERSARIAL = "adversarial"         # Attacked/post-processed samples


class MediaType(Enum):
    """Supported media types."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class TransformationStep:
    """
    Represents a single transformation applied to media.
    Required for audit trail per Layer 1 specification.
    """
    step_id: int
    operation: str                      # e.g., "resize", "recompress", "noise_injection"
    parameters: dict                    # Operation-specific parameters
    timestamp: datetime
    input_hash: str                     # SHA-256 of input
    output_hash: str                    # SHA-256 of output
    tool_used: Optional[str] = None     # e.g., "ffmpeg", "PIL", "opencv"
    tool_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "tool_used": self.tool_used,
            "tool_version": self.tool_version
        }


@dataclass
class MediaSample:
    """
    Core data structure for a media sample.
    Stores original bytes reference and full transformation history.
    """
    # Identification
    sample_id: str                      # UUID
    category: DatasetCategory
    media_type: MediaType
    
    # Original Data (immutable)
    original_path: str                  # Path to original bytes
    original_hash: str                  # SHA-256 of original
    original_size_bytes: int
    original_format: str                # e.g., "png", "mp4", "wav"
    
    # Provenance
    source_url: Optional[str] = None    # Where was this acquired?
    source_verified: bool = False       # Has source been verified?
    generator_model: Optional[str] = None  # For synthetic: "stable-diffusion-xl-1.0"
    generator_version: Optional[str] = None
    
    # Timestamps
    acquisition_timestamp: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    # Transformation Chain
    transformation_history: List[TransformationStep] = field(default_factory=list)
    
    # Variants (paths to post-processed versions)
    variants: List[str] = field(default_factory=list)
    
    # Labels
    label: Literal["real", "synthetic"] = "real"
    label_confidence: float = 1.0       # 0.0 - 1.0
    
    def add_transformation(self, step: TransformationStep) -> None:
        """Append a transformation step to history."""
        self.transformation_history.append(step)
        self.last_modified = datetime.utcnow()
    
    def add_variant(self, variant_path: str) -> None:
        """Register a new variant of this sample."""
        self.variants.append(variant_path)
        self.last_modified = datetime.utcnow()
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "sample_id": self.sample_id,
            "category": self.category.value,
            "media_type": self.media_type.value,
            "original_path": self.original_path,
            "original_hash": self.original_hash,
            "original_size_bytes": self.original_size_bytes,
            "original_format": self.original_format,
            "source_url": self.source_url,
            "source_verified": self.source_verified,
            "generator_model": self.generator_model,
            "generator_version": self.generator_version,
            "acquisition_timestamp": self.acquisition_timestamp.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "transformation_history": [t.to_dict() for t in self.transformation_history],
            "variants": self.variants,
            "label": self.label,
            "label_confidence": self.label_confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MediaSample":
        """Deserialize from dictionary."""
        sample = cls(
            sample_id=data["sample_id"],
            category=DatasetCategory(data["category"]),
            media_type=MediaType(data["media_type"]),
            original_path=data["original_path"],
            original_hash=data["original_hash"],
            original_size_bytes=data["original_size_bytes"],
            original_format=data["original_format"],
            source_url=data.get("source_url"),
            source_verified=data.get("source_verified", False),
            generator_model=data.get("generator_model"),
            generator_version=data.get("generator_version"),
            acquisition_timestamp=datetime.fromisoformat(data["acquisition_timestamp"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            variants=data.get("variants", []),
            label=data.get("label", "real"),
            label_confidence=data.get("label_confidence", 1.0)
        )
        # Reconstruct transformation history
        for t_data in data.get("transformation_history", []):
            step = TransformationStep(
                step_id=t_data["step_id"],
                operation=t_data["operation"],
                parameters=t_data["parameters"],
                timestamp=datetime.fromisoformat(t_data["timestamp"]),
                input_hash=t_data["input_hash"],
                output_hash=t_data["output_hash"],
                tool_used=t_data.get("tool_used"),
                tool_version=t_data.get("tool_version")
            )
            sample.transformation_history.append(step)
        return sample


@dataclass
class DatasetManifest:
    """
    Manifest file for a dataset category.
    Tracks all samples and provides integrity verification.
    """
    category: DatasetCategory
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    sample_count: int = 0
    samples: List[MediaSample] = field(default_factory=list)
    
    def add_sample(self, sample: MediaSample) -> None:
        """Add a sample to the manifest."""
        self.samples.append(sample)
        self.sample_count = len(self.samples)
        self.last_updated = datetime.utcnow()
    
    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        data = {
            "category": self.category.value,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "sample_count": self.sample_count,
            "samples": [s.to_dict() for s in self.samples]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(
            category=DatasetCategory(data["category"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            sample_count=data["sample_count"]
        )
        for s_data in data.get("samples", []):
            manifest.samples.append(MediaSample.from_dict(s_data))
        return manifest
