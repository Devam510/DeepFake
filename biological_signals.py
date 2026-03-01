"""
Biological Signal Analyzers — Human Physiology Detection
========================================================

Detects deepfake videos by analyzing human physiological signals:
1. rPPG Heartbeat    — real faces show skin-color pulse from heartbeat
2. Blink Rate        — unnatural blink timing = AI face
3. Skin Smoothness   — over-smooth skin texture = AI generation
4. Micro-Expressions — missing shoulder/neck muscle movement

Usage:
  from biological_signals import analyze_biological_signals
  result = analyze_biological_signals(frames)
"""

import cv2
import numpy as np
from typing import List, Dict
from scipy import signal as scipy_signal

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1: rPPG Heartbeat Detection
# ══════════════════════════════════════════════════════════════════════════════

def compute_rppg_heartbeat(frames: List[np.ndarray], fps: float = 30.0) -> Dict:
    """
    Detect pulse signal from forehead skin color changes (remote photoplethysmography).

    Real faces: blood flow causes subtle green-channel oscillation at 0.7-4 Hz (42-240 BPM).
    AI faces: no blood = no pulse signal.

    Returns:
        dict with has_pulse, pulse_strength, estimated_bpm, heartbeat_score
    """
    if len(frames) < 30:  # Need at least 1 second of video
        return {"has_pulse": False, "pulse_strength": 0.0, "estimated_bpm": 0, "heartbeat_score": 0.5}

    if not MEDIAPIPE_AVAILABLE:
        return {"has_pulse": False, "pulse_strength": 0.0, "estimated_bpm": 0, "heartbeat_score": 0.5}

    green_signal = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _mp_face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # Forehead ROI: between eyebrows and hairline
            # Landmarks: 10 (forehead center), 67 (left), 297 (right)
            forehead_pts = [10, 67, 297, 69, 299]
            ys = [int(lms.landmark[i].y * h) for i in forehead_pts]
            xs = [int(lms.landmark[i].x * w) for i in forehead_pts]

            y1, y2 = max(0, min(ys) - 10), max(ys)
            x1, x2 = max(0, min(xs)), min(w, max(xs))

            if y2 > y1 and x2 > x1:
                forehead = frame[y1:y2, x1:x2]
                if forehead.size > 0:
                    # Green channel mean (most sensitive to blood flow)
                    green_mean = float(np.mean(forehead[:, :, 1]))
                    green_signal.append(green_mean)
                    continue

        green_signal.append(0.0)

    if len(green_signal) < 30 or all(v == 0 for v in green_signal):
        return {"has_pulse": False, "pulse_strength": 0.0, "estimated_bpm": 0, "heartbeat_score": 0.7}

    # Detrend + bandpass filter for heart rate range (0.7-4 Hz = 42-240 BPM)
    sig = np.array(green_signal)
    sig = sig - np.mean(sig)  # remove DC

    # Simple moving average detrend
    if len(sig) > 10:
        trend = np.convolve(sig, np.ones(10) / 10, mode='same')
        sig = sig - trend

    # FFT analysis
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_mag = np.abs(np.fft.rfft(sig))

    # Look for peak in heart rate range (0.7-4.0 Hz)
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return {"has_pulse": False, "pulse_strength": 0.0, "estimated_bpm": 0, "heartbeat_score": 0.7}

    hr_fft = fft_mag[mask]
    hr_freqs = freqs[mask]

    peak_idx = np.argmax(hr_fft)
    peak_freq = hr_freqs[peak_idx]
    peak_mag = hr_fft[peak_idx]
    mean_mag = np.mean(hr_fft)

    # SNR: how much the peak stands out
    snr = peak_mag / max(mean_mag, 1e-6)
    has_pulse = snr > 2.0
    bpm = int(peak_freq * 60)

    # Score: strong pulse = real (low score), no pulse = fake (high score)
    pulse_strength = min(1.0, snr / 5.0)
    heartbeat_score = max(0.0, 1.0 - pulse_strength)

    return {
        "has_pulse": has_pulse,
        "pulse_strength": round(pulse_strength, 4),
        "estimated_bpm": bpm if has_pulse else 0,
        "heartbeat_score": round(heartbeat_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 2: Blink Rate Analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_blink_rate(frames: List[np.ndarray], fps: float = 30.0) -> Dict:
    """
    Analyze blink timing using Eye Aspect Ratio (EAR).

    Real: 15-20 blinks/min with natural variation.
    Deepfake: either no blinks, too regular, or too frequent.

    Returns:
        dict with blink_count, blinks_per_min, regularity, blink_score
    """
    if not MEDIAPIPE_AVAILABLE or len(frames) < 10:
        return {"blink_count": 0, "blinks_per_min": 0, "regularity": 0.0, "blink_score": 0.5}

    # MediaPipe eye landmarks for EAR
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(landmarks, eye_indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        # Vertical distances
        v1 = np.sqrt((pts[1][0] - pts[5][0]) ** 2 + (pts[1][1] - pts[5][1]) ** 2)
        v2 = np.sqrt((pts[2][0] - pts[4][0]) ** 2 + (pts[2][1] - pts[4][1]) ** 2)
        # Horizontal distance
        horiz = np.sqrt((pts[0][0] - pts[3][0]) ** 2 + (pts[0][1] - pts[3][1]) ** 2)
        return (v1 + v2) / (2.0 * max(horiz, 1e-6))

    ear_values = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _mp_face_mesh.process(rgb)
        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_ear = eye_aspect_ratio(lms, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ear_values.append((left_ear + right_ear) / 2.0)
        else:
            ear_values.append(None)

    # Detect blinks: EAR drops below threshold
    valid_ears = [e for e in ear_values if e is not None]
    if len(valid_ears) < 5:
        return {"blink_count": 0, "blinks_per_min": 0, "regularity": 0.0, "blink_score": 0.5}

    threshold = np.mean(valid_ears) * 0.75  # blink = 25% below average
    blink_frames = []
    in_blink = False

    for i, ear in enumerate(ear_values):
        if ear is None:
            continue
        if ear < threshold and not in_blink:
            in_blink = True
            blink_frames.append(i)
        elif ear >= threshold:
            in_blink = False

    duration_sec = len(frames) / fps
    blink_count = len(blink_frames)
    blinks_per_min = (blink_count / max(duration_sec, 1)) * 60

    # Regularity: how evenly spaced blinks are
    if len(blink_frames) >= 2:
        intervals = np.diff(blink_frames)
        regularity = float(np.std(intervals) / max(np.mean(intervals), 1))
    else:
        regularity = 1.0

    # Score: abnormal blink rate = suspicious
    # Normal: 15-20 blinks/min
    if blinks_per_min < 5 or blinks_per_min > 40:
        blink_score = 0.7  # abnormal
    elif regularity < 0.2:  # too regular = robotic
        blink_score = 0.6
    else:
        blink_score = 0.2  # normal

    return {
        "blink_count": blink_count,
        "blinks_per_min": round(blinks_per_min, 1),
        "regularity": round(regularity, 3),
        "blink_score": round(blink_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 3: Skin Smoothness Analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_skin_smoothness(frames: List[np.ndarray]) -> Dict:
    """
    Analyze skin texture frequency content.

    Real skin: pores, fine lines, color variation → high-frequency detail.
    AI skin: over-smooth, plastic look → low high-frequency content.

    Returns:
        dict with texture_score, variance_mean, smoothness_score
    """
    if len(frames) < 1:
        return {"texture_score": 0.5, "variance_mean": 0.0, "smoothness_score": 0.5}

    textures = []
    for frame in frames[:30]:  # Sample up to 30 frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        # Focus on cheek/forehead region (typical skin area)
        skin_region = gray[60:160, 60:164]  # mid-face crop
        if skin_region.size == 0:
            continue

        # Laplacian variance = edge/detail density
        lap = cv2.Laplacian(skin_region, cv2.CV_64F)
        variance = float(np.var(lap))
        textures.append(variance)

    if not textures:
        return {"texture_score": 0.5, "variance_mean": 0.0, "smoothness_score": 0.5}

    mean_tex = float(np.mean(textures))

    # Real skin: variance 200-1000, AI skin: 30-150
    texture_score = min(1.0, max(0.0, (mean_tex - 30) / 500))
    smoothness_score = 1.0 - texture_score  # high = too smooth = suspicious

    return {
        "texture_score": round(texture_score, 4),
        "variance_mean": round(mean_tex, 2),
        "smoothness_score": round(smoothness_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 4: Micro-Expressions / Shoulder-Neck Movement
# ══════════════════════════════════════════════════════════════════════════════

def compute_micro_expressions(frames: List[np.ndarray]) -> Dict:
    """
    Analyze subtle shoulder and neck movements.

    Real: natural micro-movements in shoulders, neck when speaking.
    Deepfake: shoulders frozen, only face animates.

    Returns:
        dict with shoulder_motion, neck_motion, micro_score
    """
    if not MEDIAPIPE_AVAILABLE or len(frames) < 5:
        return {"shoulder_motion": 0.0, "neck_motion": 0.0, "micro_score": 0.5}

    _pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    shoulder_positions = []
    for frame in frames[:50]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _pose.process(rgb)
        if result.pose_landmarks:
            ls = result.pose_landmarks.landmark[11]
            rs = result.pose_landmarks.landmark[12]
            shoulder_positions.append(((ls.x, ls.y), (rs.x, rs.y)))
        else:
            shoulder_positions.append(None)

    _pose.close()

    # Compute shoulder micro-motion
    motions = []
    for i in range(1, len(shoulder_positions)):
        if shoulder_positions[i] and shoulder_positions[i - 1]:
            curr = shoulder_positions[i]
            prev = shoulder_positions[i - 1]
            motion = np.sqrt(
                (curr[0][0] - prev[0][0]) ** 2 + (curr[0][1] - prev[0][1]) ** 2 +
                (curr[1][0] - prev[1][0]) ** 2 + (curr[1][1] - prev[1][1]) ** 2
            )
            motions.append(motion)

    if not motions:
        return {"shoulder_motion": 0.0, "neck_motion": 0.0, "micro_score": 0.5}

    shoulder_motion = float(np.mean(motions))

    # Very low shoulder motion while face is presumably moving = puppet
    # Threshold: < 0.002 is suspicious (frozen body)
    micro_score = min(1.0, max(0.0, 1.0 - shoulder_motion / 0.01))

    return {
        "shoulder_motion": round(shoulder_motion, 6),
        "neck_motion": round(shoulder_motion * 0.8, 6),  # approximation
        "micro_score": round(micro_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_biological_signals(frames: List[np.ndarray], fps: float = 30.0) -> Dict:
    """
    Run all 4 biological signal analyzers on video frames.

    Args:
        frames: List of BGR numpy arrays
        fps: Video frame rate (needed for heartbeat + blink timing)

    Returns:
        dict with all biological scores + overall biological_score
    """
    heartbeat = compute_rppg_heartbeat(frames, fps)
    blink = compute_blink_rate(frames, fps)
    skin = compute_skin_smoothness(frames)
    micro = compute_micro_expressions(frames)

    scores = {
        "heartbeat": heartbeat["heartbeat_score"],
        "blink_rate": blink["blink_score"],
        "skin_smoothness": skin["smoothness_score"],
        "micro_expression": micro["micro_score"],
    }

    weights = {
        "heartbeat": 0.30,
        "blink_rate": 0.20,
        "skin_smoothness": 0.25,
        "micro_expression": 0.25,
    }

    overall = sum(scores[k] * weights[k] for k in scores)
    overall = min(1.0, max(0.0, overall))

    return {
        "biological_score": round(overall, 4),
        "details": {
            "heartbeat": heartbeat,
            "blink_rate": blink,
            "skin_smoothness": skin,
            "micro_expressions": micro,
        },
        "individual_scores": scores,
    }
