"""
DeepFake Detection System - Transformation Utilities
Layer 1: Data Acquisition

This module provides utilities for creating adversarial variants
of media samples (noise, blur, recompression, etc.) with full
transformation logging.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Attempt to import image processing libraries
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from schema import MediaSample, TransformationStep, DatasetCategory, MediaType


class TransformationError(Exception):
    """Raised when a transformation fails."""

    pass


def get_ffmpeg_version() -> Optional[str]:
    """Get FFmpeg version string."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            return first_line
        return None
    except Exception:
        return None


def get_opencv_version() -> Optional[str]:
    """Get OpenCV version string."""
    if CV2_AVAILABLE:
        return cv2.__version__
    return None


# ============================================================
# Image Transformations
# ============================================================


def apply_jpeg_recompression(
    input_path: str, output_path: str, quality: int = 50
) -> Dict[str, Any]:
    """
    Recompress an image as JPEG with specified quality.

    Args:
        input_path: Source image path
        output_path: Destination path (should end in .jpg)
        quality: JPEG quality (1-100, lower = more compression)

    Returns:
        Dictionary with operation parameters
    """
    if not CV2_AVAILABLE:
        raise TransformationError(
            "OpenCV not installed. Run: pip install opencv-python"
        )

    img = cv2.imread(input_path)
    if img is None:
        raise TransformationError(f"Failed to read image: {input_path}")

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    return {
        "operation": "jpeg_recompression",
        "parameters": {"quality": quality},
        "tool_used": "opencv",
        "tool_version": get_opencv_version(),
    }


def apply_resize(
    input_path: str,
    output_path: str,
    scale: float = 0.5,
    interpolation: str = "bilinear",
) -> Dict[str, Any]:
    """
    Resize an image by a scale factor.

    Args:
        input_path: Source image path
        output_path: Destination path
        scale: Scale factor (0.5 = half size)
        interpolation: Interpolation method ("nearest", "bilinear", "bicubic")

    Returns:
        Dictionary with operation parameters
    """
    if not CV2_AVAILABLE:
        raise TransformationError("OpenCV not installed")

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }

    img = cv2.imread(input_path)
    if img is None:
        raise TransformationError(f"Failed to read image: {input_path}")

    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR),
    )
    cv2.imwrite(output_path, resized)

    return {
        "operation": "resize",
        "parameters": {
            "scale": scale,
            "interpolation": interpolation,
            "original_size": [w, h],
            "new_size": [new_w, new_h],
        },
        "tool_used": "opencv",
        "tool_version": get_opencv_version(),
    }


def apply_gaussian_noise(
    input_path: str, output_path: str, mean: float = 0, std: float = 25
) -> Dict[str, Any]:
    """
    Add Gaussian noise to an image.

    Args:
        input_path: Source image path
        output_path: Destination path
        mean: Noise mean
        std: Noise standard deviation

    Returns:
        Dictionary with operation parameters
    """
    if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
        raise TransformationError("OpenCV and NumPy required")

    img = cv2.imread(input_path)
    if img is None:
        raise TransformationError(f"Failed to read image: {input_path}")

    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

    return {
        "operation": "gaussian_noise",
        "parameters": {"mean": mean, "std": std},
        "tool_used": "opencv+numpy",
        "tool_version": get_opencv_version(),
    }


def apply_gaussian_blur(
    input_path: str, output_path: str, kernel_size: int = 5
) -> Dict[str, Any]:
    """
    Apply Gaussian blur to an image.

    Args:
        input_path: Source image path
        output_path: Destination path
        kernel_size: Blur kernel size (must be odd)

    Returns:
        Dictionary with operation parameters
    """
    if not CV2_AVAILABLE:
        raise TransformationError("OpenCV not installed")

    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd

    img = cv2.imread(input_path)
    if img is None:
        raise TransformationError(f"Failed to read image: {input_path}")

    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    cv2.imwrite(output_path, blurred)

    return {
        "operation": "gaussian_blur",
        "parameters": {"kernel_size": kernel_size},
        "tool_used": "opencv",
        "tool_version": get_opencv_version(),
    }


# ============================================================
# Video Transformations (using FFmpeg)
# ============================================================


def apply_video_recompression(
    input_path: str, output_path: str, crf: int = 28, codec: str = "libx264"
) -> Dict[str, Any]:
    """
    Recompress a video using FFmpeg.

    Args:
        input_path: Source video path
        output_path: Destination path
        crf: Constant Rate Factor (0-51, higher = more compression)
        codec: Video codec

    Returns:
        Dictionary with operation parameters
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise TransformationError(f"FFmpeg failed: {result.stderr}")

    return {
        "operation": "video_recompression",
        "parameters": {"crf": crf, "codec": codec},
        "tool_used": "ffmpeg",
        "tool_version": get_ffmpeg_version(),
    }


def apply_video_resize(
    input_path: str,
    output_path: str,
    width: int = 640,
    height: int = -1,  # -1 = maintain aspect ratio
) -> Dict[str, Any]:
    """
    Resize a video using FFmpeg.

    Args:
        input_path: Source video path
        output_path: Destination path
        width: Target width
        height: Target height (-1 for auto)

    Returns:
        Dictionary with operation parameters
    """
    scale_filter = f"scale={width}:{height}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        scale_filter,
        "-c:a",
        "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise TransformationError(f"FFmpeg failed: {result.stderr}")

    return {
        "operation": "video_resize",
        "parameters": {"width": width, "height": height},
        "tool_used": "ffmpeg",
        "tool_version": get_ffmpeg_version(),
    }


# ============================================================
# Audio Transformations (using FFmpeg)
# ============================================================


def apply_audio_recompression(
    input_path: str, output_path: str, bitrate: str = "64k", codec: str = "libmp3lame"
) -> Dict[str, Any]:
    """
    Recompress audio using FFmpeg.

    Args:
        input_path: Source audio path
        output_path: Destination path
        bitrate: Target bitrate
        codec: Audio codec

    Returns:
        Dictionary with operation parameters
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise TransformationError(f"FFmpeg failed: {result.stderr}")

    return {
        "operation": "audio_recompression",
        "parameters": {"bitrate": bitrate, "codec": codec},
        "tool_used": "ffmpeg",
        "tool_version": get_ffmpeg_version(),
    }


# ============================================================
# Batch Adversarial Generation
# ============================================================


def generate_adversarial_variants(
    sample: MediaSample, output_dir: str, transformations: list = None
) -> list:
    """
    Generate multiple adversarial variants of a sample.

    Args:
        sample: Source MediaSample
        output_dir: Directory to store variants
        transformations: List of transformation configs, or None for defaults

    Returns:
        List of (variant_path, transformation_info) tuples
    """
    if transformations is None:
        # Default adversarial transformations per media type
        if sample.media_type == MediaType.IMAGE:
            transformations = [
                {"func": apply_jpeg_recompression, "args": {"quality": 30}},
                {"func": apply_jpeg_recompression, "args": {"quality": 50}},
                {"func": apply_resize, "args": {"scale": 0.5}},
                {"func": apply_gaussian_noise, "args": {"std": 15}},
                {"func": apply_gaussian_blur, "args": {"kernel_size": 5}},
            ]
        elif sample.media_type == MediaType.VIDEO:
            transformations = [
                {"func": apply_video_recompression, "args": {"crf": 35}},
                {"func": apply_video_resize, "args": {"width": 480}},
            ]
        elif sample.media_type == MediaType.AUDIO:
            transformations = [
                {"func": apply_audio_recompression, "args": {"bitrate": "64k"}},
                {"func": apply_audio_recompression, "args": {"bitrate": "32k"}},
            ]
        else:
            transformations = []

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, t in enumerate(transformations):
        func = t["func"]
        args = t["args"]

        # Determine output extension
        ext = Path(sample.original_path).suffix
        if func == apply_jpeg_recompression:
            ext = ".jpg"

        output_path = os.path.join(output_dir, f"{sample.sample_id}_variant_{i}{ext}")

        try:
            info = func(sample.original_path, output_path, **args)
            results.append((output_path, info))
        except TransformationError as e:
            print(f"WARNING: Transformation {i} failed: {e}")
            continue

    return results


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply transformations to media files")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "--operation",
        "-o",
        required=True,
        choices=[
            "jpeg_recompress",
            "resize",
            "noise",
            "blur",
            "video_recompress",
            "video_resize",
            "audio_recompress",
        ],
        help="Transformation to apply",
    )
    parser.add_argument("--quality", type=int, default=50, help="JPEG quality (1-100)")
    parser.add_argument("--scale", type=float, default=0.5, help="Resize scale factor")
    parser.add_argument(
        "--noise-std", type=float, default=25, help="Noise std deviation"
    )
    parser.add_argument("--kernel", type=int, default=5, help="Blur kernel size")
    parser.add_argument("--crf", type=int, default=28, help="Video CRF")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--bitrate", default="64k", help="Audio bitrate")

    args = parser.parse_args()

    try:
        if args.operation == "jpeg_recompress":
            info = apply_jpeg_recompression(args.input, args.output, args.quality)
        elif args.operation == "resize":
            info = apply_resize(args.input, args.output, args.scale)
        elif args.operation == "noise":
            info = apply_gaussian_noise(args.input, args.output, std=args.noise_std)
        elif args.operation == "blur":
            info = apply_gaussian_blur(args.input, args.output, args.kernel)
        elif args.operation == "video_recompress":
            info = apply_video_recompression(args.input, args.output, args.crf)
        elif args.operation == "video_resize":
            info = apply_video_resize(args.input, args.output, args.width)
        elif args.operation == "audio_recompress":
            info = apply_audio_recompression(args.input, args.output, args.bitrate)

        print(f"SUCCESS: {info['operation']}")
        print(f"  Parameters: {info['parameters']}")
        print(f"  Tool: {info['tool_used']} {info.get('tool_version', '')}")

    except TransformationError as e:
        print(f"ERROR: {e}")
        exit(1)
