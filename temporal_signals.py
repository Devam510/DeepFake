"""
Temporal Signal Analyzers — Time-Based Video Artifacts
======================================================

Detects deepfake artifacts that only appear across multiple frames:
1. SSIM Delta      — face swap seam flickering between frames
2. Face Jitter     — unnatural face position jumps
3. Body-Face Ratio — puppet body (face moves, body stays still)
4. Jawline Glow    — diffusion effect around jawline/neck boundary
5. Optical Flow    — incoherent pixel motion patterns

Usage:
  from temporal_signals import analyze_temporal_signals
  result = analyze_temporal_signals("video.mp4")
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import mediapipe as mp
    import os
    old_stderr = os.dup(2)
    f_null = open(os.devnull, 'w')
    try:
        os.dup2(f_null.fileno(), 2)
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
        )
        _mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5
        )
    finally:
        os.dup2(old_stderr, 2)
        f_null.close()
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1: SSIM Delta — Frame-to-Frame Structural Consistency
# ══════════════════════════════════════════════════════════════════════════════

def compute_ssim_deltas(frames: List[np.ndarray]) -> Dict:
    """
    Compute SSIM between consecutive frames in the face region.
    Real videos: high SSIM (smooth transitions ~0.85-0.95).
    Deepfakes: sudden SSIM drops (face swap seams, flicker).

    Returns:
        dict with mean_ssim, min_ssim, ssim_variance, anomaly_score
    """
    if len(frames) < 2:
        return {"mean_ssim": 0.95, "min_ssim": 0.95, "ssim_variance": 0.0, "anomaly_score": 0.0}

    if not SKIMAGE_AVAILABLE:
        return {"mean_ssim": 0.5, "min_ssim": 0.5, "ssim_variance": 0.0, "anomaly_score": 0.5}

    ssim_values = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        # Resize to standard size for fair comparison
        gray1 = cv2.resize(gray1, (224, 224))
        gray2 = cv2.resize(gray2, (224, 224))
        s = ssim(gray1, gray2)
        ssim_values.append(s)

    arr = np.array(ssim_values)
    mean_s = float(np.mean(arr))
    min_s = float(np.min(arr))
    var_s = float(np.var(arr))

    # Anomaly: high variance + low min = face swap flicker
    anomaly = min(1.0, max(0.0, (1.0 - min_s) * 2 + var_s * 10))

    return {
        "mean_ssim": round(mean_s, 4),
        "min_ssim": round(min_s, 4),
        "ssim_variance": round(var_s, 6),
        "anomaly_score": round(anomaly, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 2: Face Position Jitter
# ══════════════════════════════════════════════════════════════════════════════

def compute_face_jitter(frames: List[np.ndarray]) -> Dict:
    """
    Track face center position across frames.
    Real: smooth, gradual movement.
    Deepfake: sudden jumps when face swap fails to track.

    Returns:
        dict with jitter_score, max_jump, mean_displacement
    """
    if not MEDIAPIPE_AVAILABLE or len(frames) < 3:
        return {"jitter_score": 0.0, "max_jump": 0.0, "mean_displacement": 0.0}

    centers = []
    for frame in frames:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _mp_face_mesh.process(rgb)
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0]
            # Use nose tip (landmark 1) as face center proxy
            nose = lms.landmark[1]
            centers.append((nose.x * w, nose.y * h))
        else:
            centers.append(None)

    # Compute displacements between consecutive frames
    displacements = []
    for i in range(1, len(centers)):
        if centers[i] is not None and centers[i - 1] is not None:
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            displacements.append(np.sqrt(dx ** 2 + dy ** 2))

    if not displacements:
        return {"jitter_score": 0.5, "max_jump": 0.0, "mean_displacement": 0.0}

    arr = np.array(displacements)
    mean_d = float(np.mean(arr))
    max_d = float(np.max(arr))
    std_d = float(np.std(arr))

    # Jitter = high std relative to mean (irregular movement)
    jitter = min(1.0, std_d / max(mean_d, 1.0))

    return {
        "jitter_score": round(jitter, 4),
        "max_jump": round(max_d, 2),
        "mean_displacement": round(mean_d, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 3: Body-to-Face Motion Ratio (Puppet Body Detection)
# ══════════════════════════════════════════════════════════════════════════════

def compute_body_face_ratio(frames: List[np.ndarray]) -> Dict:
    """
    Compare face motion to body/shoulder motion.
    Real: face and body move together naturally.
    Puppet deepfake: face animates while body is frozen.

    Returns:
        dict with ratio, face_motion, body_motion, puppet_score
    """
    if not MEDIAPIPE_AVAILABLE or len(frames) < 3:
        return {"ratio": 1.0, "face_motion": 0.0, "body_motion": 0.0, "puppet_score": 0.0}

    face_positions = []
    body_positions = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face landmarks
        fm = _mp_face_mesh.process(rgb)
        if fm.multi_face_landmarks:
            nose = fm.multi_face_landmarks[0].landmark[1]
            face_positions.append((nose.x, nose.y))
        else:
            face_positions.append(None)

        # Pose landmarks (shoulders)
        pm = _mp_pose.process(rgb)
        if pm.pose_landmarks:
            ls = pm.pose_landmarks.landmark[11]  # left shoulder
            rs = pm.pose_landmarks.landmark[12]  # right shoulder
            mid_x = (ls.x + rs.x) / 2
            mid_y = (ls.y + rs.y) / 2
            body_positions.append((mid_x, mid_y))
        else:
            body_positions.append(None)

    # Compute per-frame motion
    face_motions = []
    body_motions = []
    for i in range(1, len(frames)):
        if face_positions[i] and face_positions[i - 1]:
            fd = np.sqrt((face_positions[i][0] - face_positions[i - 1][0]) ** 2 +
                         (face_positions[i][1] - face_positions[i - 1][1]) ** 2)
            face_motions.append(fd)
        if body_positions[i] and body_positions[i - 1]:
            bd = np.sqrt((body_positions[i][0] - body_positions[i - 1][0]) ** 2 +
                         (body_positions[i][1] - body_positions[i - 1][1]) ** 2)
            body_motions.append(bd)

    face_avg = float(np.mean(face_motions)) if face_motions else 0.0
    body_avg = float(np.mean(body_motions)) if body_motions else 0.0

    # Ratio: face moves a lot but body doesn't = puppet
    ratio = face_avg / max(body_avg, 0.001)
    puppet = min(1.0, max(0.0, (ratio - 2.0) / 8.0))  # score > 0 when ratio > 2

    return {
        "ratio": round(ratio, 3),
        "face_motion": round(face_avg, 5),
        "body_motion": round(body_avg, 5),
        "puppet_score": round(puppet, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 4: Jawline Diffusion Glow
# ══════════════════════════════════════════════════════════════════════════════

def compute_jawline_glow(frames: List[np.ndarray]) -> Dict:
    """
    Measure edge sharpness along jawline boundary.
    Real: sharp natural jawline edges.
    Deepfake: blurry/glowing boundary from diffusion blending.

    Returns:
        dict with sharpness_mean, sharpness_std, glow_score
    """
    if len(frames) < 1:
        return {"sharpness_mean": 100.0, "sharpness_std": 0.0, "glow_score": 0.0}

    sharpness_values = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        # Focus on lower face region (jawline area: rows 140-200)
        jaw_region = gray[140:200, 40:184]
        if jaw_region.size == 0:
            continue

        # Compute Laplacian (edge sharpness indicator)
        lap = cv2.Laplacian(jaw_region, cv2.CV_64F)
        sharpness = float(np.var(lap))
        sharpness_values.append(sharpness)

    if not sharpness_values:
        return {"sharpness_mean": 100.0, "sharpness_std": 0.0, "glow_score": 0.0}

    arr = np.array(sharpness_values)
    mean_s = float(np.mean(arr))
    std_s = float(np.std(arr))

    # Low edge sharpness = blurry/glowing = suspicious
    # Typical real: 200-800, deepfake: 50-200
    glow = min(1.0, max(0.0, 1.0 - (mean_s - 50) / 400))

    return {
        "sharpness_mean": round(mean_s, 2),
        "sharpness_std": round(std_s, 2),
        "glow_score": round(glow, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 5: Optical Flow Coherence
# ══════════════════════════════════════════════════════════════════════════════

def compute_optical_flow(frames: List[np.ndarray]) -> Dict:
    """
    Analyze dense optical flow for motion coherence.
    Real: smooth, physically consistent flow.
    Deepfake: discontinuities at face boundary.

    Returns:
        dict with flow_magnitude, flow_variance, incoherence_score
    """
    if len(frames) < 2:
        return {"flow_magnitude": 0.0, "flow_variance": 0.0, "incoherence_score": 0.0}

    flow_magnitudes = []
    flow_variances = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (224, 224))

    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        flow_magnitudes.append(float(np.mean(mag)))
        flow_variances.append(float(np.var(mag)))

        prev_gray = gray

    mean_mag = float(np.mean(flow_magnitudes))
    mean_var = float(np.mean(flow_variances))

    # High variance in flow = incoherent motion (deepfake boundary issues)
    incoherence = min(1.0, mean_var / max(mean_mag + 1.0, 1.0))

    return {
        "flow_magnitude": round(mean_mag, 4),
        "flow_variance": round(mean_var, 4),
        "incoherence_score": round(incoherence, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_temporal_signals(frames: List[np.ndarray]) -> Dict:
    """
    Run all 5 temporal signal analyzers on a list of video frames.

    Args:
        frames: List of BGR numpy arrays (from video_processor.extract_frames_for_analysis)

    Returns:
        dict with all temporal signal scores + overall temporal_score
    """
    ssim_r = compute_ssim_deltas(frames)
    jitter_r = compute_face_jitter(frames)
    body_r = compute_body_face_ratio(frames)
    jaw_r = compute_jawline_glow(frames)
    flow_r = compute_optical_flow(frames)

    # Weighted combination for overall temporal score
    scores = {
        "ssim_anomaly": ssim_r["anomaly_score"],
        "face_jitter": jitter_r["jitter_score"],
        "puppet_body": body_r["puppet_score"],
        "jawline_glow": jaw_r["glow_score"],
        "flow_incoherence": flow_r["incoherence_score"],
    }

    weights = {
        "ssim_anomaly": 0.25,
        "face_jitter": 0.20,
        "puppet_body": 0.20,
        "jawline_glow": 0.15,
        "flow_incoherence": 0.20,
    }

    overall = sum(scores[k] * weights[k] for k in scores)
    overall = min(1.0, max(0.0, overall))

    return {
        "temporal_score": round(overall, 4),
        "details": {
            "ssim": ssim_r,
            "face_jitter": jitter_r,
            "body_face_ratio": body_r,
            "jawline": jaw_r,
            "optical_flow": flow_r,
        },
        "individual_scores": scores,
    }
