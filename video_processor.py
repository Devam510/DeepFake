"""
Video Processor — Frame Extraction & Face Cropping
===================================================

Extracts frames from video files at configurable FPS,
detects faces, and crops them for downstream analysis.

Usage:
  # Single video
  python video_processor.py path/to/video.mp4

  # Batch extract from dataset
  python video_processor.py --batch datasets/celeb_df/Celeb-synthesis --output datasets/processed/celeb_fake

  # Extract from all datasets
  python video_processor.py --extract-all
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

# ── Face detection ────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    old_stderr = os.dup(2)
    f_null = open(os.devnull, 'w')
    try:
        os.dup2(f_null.fileno(), 2)
        _mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    finally:
        os.dup2(old_stderr, 2)
        f_null.close()
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed. Face cropping will use full frames.")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE = Path(r"d:\Devam\Microsoft VS Code\Codes\DeepFake")
DATASETS = BASE / "datasets"
PROCESSED = DATASETS / "processed"

SAMPLE_FPS = 2        # frames per second to extract
MAX_FRAMES = 100      # max frames per video (caps very long videos)
FACE_PAD = 0.3        # 30% padding around detected face
FACE_SIZE = (224, 224) # output face image size (EfficientNet input)


@dataclass
class VideoInfo:
    """Metadata extracted from a video."""
    path: str
    fps: float = 0.0
    frame_count: int = 0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""


@dataclass
class ExtractionResult:
    """Result of frame extraction from one video."""
    video_path: str
    output_dir: str
    num_frames: int = 0
    num_faces: int = 0
    errors: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_video_info(video_path: str) -> VideoInfo:
    """Extract metadata from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return VideoInfo(path=video_path)

    info = VideoInfo(
        path=video_path,
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        codec=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('ascii', errors='replace'),
    )
    info.duration = info.frame_count / max(info.fps, 1)
    cap.release()
    return info


def detect_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest face in a frame. Returns (x, y, w, h) or None."""
    if not MEDIAPIPE_AVAILABLE:
        return None

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _mp_face.process(rgb)

    if not results.detections:
        return None

    # Pick the largest detection
    best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
    bb = best.location_data.relative_bounding_box

    # Convert relative to absolute with padding
    pad = FACE_PAD
    x1 = max(0, int((bb.xmin - pad * bb.width) * w))
    y1 = max(0, int((bb.ymin - pad * bb.height) * h))
    x2 = min(w, int((bb.xmin + bb.width + pad * bb.width) * w))
    y2 = min(h, int((bb.ymin + bb.height + pad * bb.height) * h))

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_fps: float = SAMPLE_FPS,
    max_frames: int = MAX_FRAMES,
    crop_face: bool = True,
) -> ExtractionResult:
    """
    Extract frames from a video at the specified FPS.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        sample_fps: Frames per second to sample
        max_frames: Maximum number of frames to extract
        crop_face: Whether to detect and crop faces

    Returns:
        ExtractionResult with stats
    """
    result = ExtractionResult(video_path=video_path, output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result.errors.append(f"Cannot open: {video_path}")
        return result

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval
    interval = max(1, int(video_fps / sample_fps))
    frame_idx = 0
    saved = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            # Crop face if requested
            if crop_face:
                face_bbox = detect_face(frame)
                if face_bbox is not None:
                    x, y, w, h = face_bbox
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, FACE_SIZE)
                    out_path = os.path.join(output_dir, f"face_{saved:04d}.jpg")
                    cv2.imwrite(out_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    result.num_faces += 1
                    saved += 1
                else:
                    # No face found — save full frame resized
                    resized = cv2.resize(frame, FACE_SIZE)
                    out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
                    cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
            else:
                resized = cv2.resize(frame, FACE_SIZE)
                out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(out_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        frame_idx += 1

    cap.release()
    result.num_frames = saved
    return result


def extract_frames_for_analysis(video_path: str, sample_fps: float = 2) -> List[np.ndarray]:
    """
    Extract raw frames as numpy arrays (for real-time analysis, not saving to disk).

    Returns list of BGR frames sampled at the given FPS.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(video_fps / sample_fps))
    idx = 0

    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def batch_extract(
    input_dir: str,
    output_dir: str,
    label: str = "unknown",
    sample_fps: float = SAMPLE_FPS,
    max_videos: int = -1,
) -> Dict:
    """
    Batch extract frames from all videos in a directory.

    Args:
        input_dir: Directory containing video files
        output_dir: Base directory for output
        label: Label for progress messages
        max_videos: Max videos to process (-1 = all)

    Returns:
        Dict with stats
    """
    from tqdm import tqdm

    video_exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    videos = [
        f for f in Path(input_dir).rglob("*")
        if f.suffix.lower() in video_exts and f.is_file()
    ]

    if max_videos > 0:
        videos = videos[:max_videos]

    print(f"  [{label}] Processing {len(videos)} videos → {output_dir}")
    total_frames = 0
    total_faces = 0

    for vpath in tqdm(videos, desc=f"  {label}", unit="vid"):
        vid_name = vpath.stem
        vid_out = os.path.join(output_dir, vid_name)
        result = extract_frames(str(vpath), vid_out, sample_fps)
        total_frames += result.num_frames
        total_faces += result.num_faces

    return {
        "videos": len(videos),
        "frames": total_frames,
        "faces": total_faces,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACT ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_datasets():
    """Process all downloaded video datasets into frame images."""
    print("=" * 60)
    print("  Video Dataset Frame Extraction")
    print("=" * 60)

    stats = {}

    # Celeb-DF: has real + fake MP4 videos
    celeb_real = DATASETS / "celeb_df" / "Celeb-real"
    celeb_fake = DATASETS / "celeb_df" / "Celeb-synthesis"
    youtube_real = DATASETS / "celeb_df" / "YouTube-real"

    if celeb_real.exists():
        s = batch_extract(str(celeb_real), str(PROCESSED / "celeb_real"), "Celeb-Real")
        stats["celeb_real"] = s
    if celeb_fake.exists():
        s = batch_extract(str(celeb_fake), str(PROCESSED / "celeb_fake"), "Celeb-Fake")
        stats["celeb_fake"] = s
    if youtube_real.exists():
        s = batch_extract(str(youtube_real), str(PROCESSED / "youtube_real"), "YouTube-Real")
        stats["youtube_real"] = s

    # FaceForensics++ C23 — check for extracted folders
    ff_base = DATASETS / "faceforensics"
    for ff_sub in ["Real", "Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures", "DeepFakeDetection"]:
        ff_dir = ff_base / ff_sub
        if ff_dir.exists():
            label_type = "real" if ff_sub == "Real" else "fake"
            s = batch_extract(str(ff_dir), str(PROCESSED / f"ff_{ff_sub.lower()}"), f"FF-{ff_sub}")
            stats[f"ff_{ff_sub.lower()}"] = s

    print("\n" + "=" * 60)
    print("  Extraction Summary:")
    print("=" * 60)
    total_v = sum(s["videos"] for s in stats.values())
    total_f = sum(s["frames"] for s in stats.values())
    for name, s in stats.items():
        print(f"  {name:30s} {s['videos']:5d} vids → {s['frames']:6d} frames")
    print(f"  {'TOTAL':30s} {total_v:5d} vids → {total_f:6d} frames")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video frame extraction")
    parser.add_argument("video", nargs="?", help="Single video to process")
    parser.add_argument("--batch", help="Batch process a directory of videos")
    parser.add_argument("--output", default="output_frames", help="Output directory")
    parser.add_argument("--extract-all", action="store_true", help="Process all datasets")
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS, help="Sample FPS")
    args = parser.parse_args()

    if args.extract_all:
        extract_all_datasets()
    elif args.batch:
        batch_extract(args.batch, args.output, "batch", args.fps)
    elif args.video:
        info = get_video_info(args.video)
        print(f"Video: {info.path}")
        print(f"  FPS: {info.fps}, Duration: {info.duration:.1f}s, Size: {info.width}x{info.height}")
        result = extract_frames(args.video, args.output, args.fps)
        print(f"  Extracted: {result.num_frames} frames ({result.num_faces} faces)")
    else:
        parser.print_help()
