"""
DeepFake Detection System - Audio Signal Extraction
Layer 2: Signal Extraction (Forensics)

This module implements forensic signal extractors for audio:
- Phase coherence analysis
- Spectral regularization detection
- Breath/silence pattern analysis
- Micro-prosody features
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
import os
import struct
import wave

try:
    import scipy.signal as signal
    from scipy.fft import fft, fftfreq

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class AudioSignals:
    """Container for extracted audio forensic signals."""

    # Phase coherence
    phase_coherence_score: float  # Higher = more natural phase relationships
    phase_discontinuity_count: int  # Number of phase jumps

    # Spectral
    spectral_flatness: float  # Higher = more noise-like
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_regularity_score: float  # Too regular = synthetic

    # Breath/silence
    silence_ratio: float  # Ratio of silent frames
    breath_pattern_detected: bool  # Natural breathing patterns
    silence_distribution_score: float  # Natural silence distribution

    # Micro-prosody
    pitch_variation: float  # Natural speech has high variation
    pitch_flatness_score: float  # Synthetic often has flat pitch

    # Metadata
    sample_rate: int
    duration_seconds: float
    channels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_coherence_score": float(self.phase_coherence_score),
            "phase_discontinuity_count": int(self.phase_discontinuity_count),
            "spectral_flatness": float(self.spectral_flatness),
            "spectral_centroid_mean": float(self.spectral_centroid_mean),
            "spectral_centroid_std": float(self.spectral_centroid_std),
            "spectral_regularity_score": float(self.spectral_regularity_score),
            "silence_ratio": float(self.silence_ratio),
            "breath_pattern_detected": self.breath_pattern_detected,
            "silence_distribution_score": float(self.silence_distribution_score),
            "pitch_variation": float(self.pitch_variation),
            "pitch_flatness_score": float(self.pitch_flatness_score),
            "sample_rate": self.sample_rate,
            "duration_seconds": float(self.duration_seconds),
            "channels": self.channels,
        }


class AudioSignalExtractor:
    """
    Extracts forensic signals from audio.

    Analyzes phase coherence, spectral patterns,
    and prosodic features to detect synthetic audio.
    """

    def __init__(self, frame_size: int = 2048, hop_size: int = 512):
        """
        Args:
            frame_size: FFT frame size
            hop_size: FFT hop size
        """
        self.frame_size = frame_size
        self.hop_size = hop_size

    def extract_all(self, audio_path: str) -> AudioSignals:
        """
        Extract all forensic signals from an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioSignals dataclass
        """
        # Load audio (convert to WAV if needed)
        audio, sample_rate, channels = self._load_audio(audio_path)

        if len(audio) == 0:
            raise ValueError("Audio file is empty")

        duration = len(audio) / sample_rate

        # Extract signals
        phase_features = self._analyze_phase_coherence(audio, sample_rate)
        spectral_features = self._analyze_spectral_features(audio, sample_rate)
        silence_features = self._analyze_silence_patterns(audio, sample_rate)
        prosody_features = self._analyze_prosody(audio, sample_rate)

        return AudioSignals(
            phase_coherence_score=phase_features["coherence"],
            phase_discontinuity_count=phase_features["discontinuities"],
            spectral_flatness=spectral_features["flatness"],
            spectral_centroid_mean=spectral_features["centroid_mean"],
            spectral_centroid_std=spectral_features["centroid_std"],
            spectral_regularity_score=spectral_features["regularity"],
            silence_ratio=silence_features["ratio"],
            breath_pattern_detected=silence_features["breath_detected"],
            silence_distribution_score=silence_features["distribution"],
            pitch_variation=prosody_features["pitch_variation"],
            pitch_flatness_score=prosody_features["flatness"],
            sample_rate=sample_rate,
            duration_seconds=duration,
            channels=channels,
        )

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int, int]:
        """
        Load audio file, converting to WAV if necessary.

        Returns:
            (audio_samples, sample_rate, channels)
        """
        ext = os.path.splitext(path)[1].lower()

        if ext == ".wav":
            return self._load_wav(path)
        else:
            # Convert to WAV using FFmpeg
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    tmp_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=60)

                if result.returncode != 0:
                    raise ValueError(
                        f"FFmpeg conversion failed: {result.stderr.decode()}"
                    )

                return self._load_wav(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def _load_wav(self, path: str) -> Tuple[np.ndarray, int, int]:
        """Load a WAV file."""
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            raw_data = wf.readframes(n_frames)

            # Convert to numpy array
            if wf.getsampwidth() == 2:
                audio = (
                    np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                )
            elif wf.getsampwidth() == 1:
                audio = (
                    np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0
                    - 1.0
                )
            else:
                raise ValueError(f"Unsupported sample width: {wf.getsampwidth()}")

            # Convert stereo to mono
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            return audio, sample_rate, channels

    def _analyze_phase_coherence(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Analyze phase coherence in the frequency domain.

        Synthetic audio may have unnatural phase relationships
        between frequency components.
        """
        # Compute STFT
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        phases = []

        for i in range(min(num_frames, 100)):  # Limit frames for speed
            start = i * self.hop_size
            frame = audio[start : start + self.frame_size]

            # Apply window
            window = np.hanning(len(frame))
            frame = frame * window

            # FFT
            spectrum = np.fft.rfft(frame)
            phase = np.angle(spectrum)
            phases.append(phase)

        phases = np.array(phases)

        # Phase coherence: consistency of phase differences
        phase_diffs = np.diff(phases, axis=0)

        # Wrap to [-pi, pi]
        phase_diffs = np.angle(np.exp(1j * phase_diffs))

        # Coherence: low variance in phase differences = high coherence
        coherence = 1.0 / (np.std(phase_diffs) + 0.01)
        coherence = min(coherence, 100.0)

        # Detect phase discontinuities
        discontinuities = np.sum(np.abs(phase_diffs) > np.pi / 2)

        return {"coherence": coherence, "discontinuities": int(discontinuities)}

    def _analyze_spectral_features(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, float]:
        """
        Analyze spectral characteristics.

        Synthetic audio often has over-regular spectral structure.
        """
        centroids = []
        flatnesses = []

        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1

        for i in range(min(num_frames, 200)):
            start = i * self.hop_size
            frame = audio[start : start + self.frame_size]

            # FFT
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1 / sample_rate)

            # Spectral centroid
            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                centroids.append(centroid)

            # Spectral flatness (geometric mean / arithmetic mean)
            spectrum_pos = spectrum[spectrum > 0]
            if len(spectrum_pos) > 0:
                geo_mean = np.exp(np.mean(np.log(spectrum_pos + 1e-10)))
                arith_mean = np.mean(spectrum_pos)
                flatness = geo_mean / (arith_mean + 1e-10)
                flatnesses.append(flatness)

        centroids = np.array(centroids) if centroids else np.array([0])
        flatnesses = np.array(flatnesses) if flatnesses else np.array([0])

        # Regularity: low variation in spectral features = synthetic
        regularity = 1.0 / (np.std(centroids) / (np.mean(centroids) + 1) + 0.01)
        regularity = min(regularity, 100.0)

        return {
            "flatness": np.mean(flatnesses),
            "centroid_mean": np.mean(centroids),
            "centroid_std": np.std(centroids),
            "regularity": regularity,
        }

    def _analyze_silence_patterns(
        self, audio: np.ndarray, sample_rate: int, threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Analyze silence and breath patterns.

        Natural speech has specific patterns of silence and breathing.
        """
        # Frame-level energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop = int(0.010 * sample_rate)  # 10ms hop

        energies = []
        for i in range(0, len(audio) - frame_length, hop):
            frame = audio[i : i + frame_length]
            energy = np.mean(frame**2)
            energies.append(energy)

        energies = np.array(energies)

        # Silence detection
        silence_mask = energies < threshold
        silence_ratio = np.mean(silence_mask)

        # Analyze silence segment lengths
        silence_segments = []
        current_length = 0
        for is_silent in silence_mask:
            if is_silent:
                current_length += 1
            else:
                if current_length > 0:
                    silence_segments.append(current_length)
                current_length = 0

        if current_length > 0:
            silence_segments.append(current_length)

        # Breath detection: silences of 0.3-1.0 seconds
        hop_duration = hop / sample_rate
        breath_lengths = [
            s for s in silence_segments if 0.3 / hop_duration < s < 1.0 / hop_duration
        ]
        breath_detected = len(breath_lengths) > 2

        # Silence distribution: natural speech has varied silence lengths
        if silence_segments:
            distribution_score = np.std(silence_segments) / (
                np.mean(silence_segments) + 1
            )
        else:
            distribution_score = 0.0

        return {
            "ratio": silence_ratio,
            "breath_detected": breath_detected,
            "distribution": min(distribution_score, 10.0),
        }

    def _analyze_prosody(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """
        Analyze micro-prosody (pitch variation).

        Synthetic speech often has flattened pitch patterns.
        """
        # Simple autocorrelation-based pitch detection
        frame_length = int(0.040 * sample_rate)  # 40ms
        hop = int(0.010 * sample_rate)  # 10ms

        pitches = []

        for i in range(0, len(audio) - frame_length, hop):
            frame = audio[i : i + frame_length]

            # Skip low-energy frames
            if np.mean(frame**2) < 0.001:
                continue

            # Autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]

            # Find first peak after 2ms (avoid zero-lag peak)
            min_lag = int(0.002 * sample_rate)
            max_lag = int(0.020 * sample_rate)  # Max period = 50Hz

            if max_lag < len(corr):
                search_region = corr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    if corr[peak_idx] > 0.3 * corr[0]:
                        pitch = sample_rate / peak_idx
                        if 50 < pitch < 500:  # Human voice range
                            pitches.append(pitch)

        if not pitches:
            return {"pitch_variation": 0.0, "flatness": 100.0}

        pitches = np.array(pitches)

        # Pitch variation
        variation = np.std(pitches) / np.mean(pitches)

        # Flatness: low variation = synthetic
        flatness = 1.0 / (variation + 0.01)
        flatness = min(flatness, 100.0)

        return {"pitch_variation": variation, "flatness": flatness}


# ============================================================
# CLI Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract forensic signals from audio")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")

    args = parser.parse_args()

    extractor = AudioSignalExtractor()
    signals = extractor.extract_all(args.audio)

    result = signals.to_dict()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Signals saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))
