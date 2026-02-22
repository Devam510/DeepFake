"""
Enhanced Frequency Analyzer for AI Detection
=============================================

Analyzes frequency domain patterns to detect AI-generated images.
Modern AI generators leave subtle but detectable patterns in:
- DCT (Discrete Cosine Transform) coefficients
- FFT (Fast Fourier Transform) spectrum
- High-frequency artifacts

These patterns are often invisible in spatial domain but reveal
AI generation fingerprints in frequency domain.
"""

import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from scipy import fftpack
    from scipy.ndimage import uniform_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class FrequencyAnalysisResult:
    """Result of frequency domain analysis."""
    
    # DCT analysis
    dct_anomaly_score: float  # 0-1, higher = more AI-like
    dct_block_uniformity: float  # AI images often have uniform DCT blocks
    
    # FFT analysis  
    fft_spectrum_score: float  # 0-1, higher = more AI-like
    fft_radial_profile: str  # "natural", "synthetic", "uncertain"
    
    # High-frequency analysis
    high_freq_energy: float  # Natural images have more high-freq noise
    noise_pattern: str  # "natural_noise", "synthetic_smooth", "uncertain"
    
    # Combined score
    frequency_ai_probability: float  # 0-1, final frequency-based AI probability
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    
    def to_dict(self) -> dict:
        return {
            "dct_anomaly_score": self.dct_anomaly_score,
            "dct_block_uniformity": self.dct_block_uniformity,
            "fft_spectrum_score": self.fft_spectrum_score,
            "fft_radial_profile": self.fft_radial_profile,
            "high_freq_energy": self.high_freq_energy,
            "noise_pattern": self.noise_pattern,
            "frequency_ai_probability": self.frequency_ai_probability,
            "confidence": self.confidence,
        }


class FrequencyAnalyzer:
    """
    Analyzes images in frequency domain to detect AI generation patterns.
    
    Key detection methods:
    1. DCT Block Analysis - AI compressors leave uniform block patterns
    2. FFT Radial Profile - AI images have unusual frequency distribution
    3. High-Frequency Noise - Natural photos have characteristic noise
    """
    
    VERSION = "FrequencyAnalyzer_v1.0.0"
    
    def __init__(self):
        """Initialize frequency analyzer."""
        self.initialized = SCIPY_AVAILABLE or OPENCV_AVAILABLE
        if not self.initialized:
            print("⚠️ FrequencyAnalyzer: scipy/opencv not available")
    
    def analyze(self, image_path: str) -> FrequencyAnalysisResult:
        """
        Perform comprehensive frequency analysis on image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            FrequencyAnalysisResult with all frequency metrics
        """
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image, dtype=np.float32)
            gray = np.mean(img_array, axis=2)
        except Exception as e:
            return self._error_result(f"Failed to load image: {e}")
        
        # Run all analyses
        dct_result = self._analyze_dct(gray)
        fft_result = self._analyze_fft(gray)
        noise_result = self._analyze_noise(gray)
        
        # Combine scores
        combined_score = self._combine_scores(dct_result, fft_result, noise_result)
        
        return FrequencyAnalysisResult(
            dct_anomaly_score=dct_result["anomaly_score"],
            dct_block_uniformity=dct_result["block_uniformity"],
            fft_spectrum_score=fft_result["spectrum_score"],
            fft_radial_profile=fft_result["profile_type"],
            high_freq_energy=noise_result["high_freq_energy"],
            noise_pattern=noise_result["pattern"],
            frequency_ai_probability=combined_score["probability"],
            confidence=combined_score["confidence"],
        )
    
    def _analyze_dct(self, gray: np.ndarray) -> Dict:
        """
        Analyze DCT (Discrete Cosine Transform) patterns.
        
        AI-generated images often show:
        - Unusual DCT coefficient distributions
        - More uniform block patterns
        - Missing natural JPEG artifacts
        """
        if not SCIPY_AVAILABLE:
            return {"anomaly_score": 0.5, "block_uniformity": 0.5}
        
        try:
            # Apply 2D DCT
            dct = fftpack.dct(fftpack.dct(gray.T, norm='ortho').T, norm='ortho')
            
            # Analyze block uniformity (8x8 blocks like JPEG)
            h, w = gray.shape
            block_size = 8
            block_vars = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = dct[i:i+block_size, j:j+block_size]
                    block_vars.append(np.var(block))
            
            if len(block_vars) == 0:
                return {"anomaly_score": 0.5, "block_uniformity": 0.5}
            
            # Uniformity: low variance in block variances = synthetic
            uniformity = 1.0 / (1.0 + np.std(block_vars) / (np.mean(block_vars) + 1e-10))
            
            # Check DCT coefficient distribution
            # Natural images have specific falloff patterns
            dct_flat = np.abs(dct).flatten()
            dct_sorted = np.sort(dct_flat)[::-1]
            
            # AI images often have more energy in mid-frequencies
            mid_energy = np.mean(dct_sorted[len(dct_sorted)//4:len(dct_sorted)//2])
            low_energy = np.mean(dct_sorted[:len(dct_sorted)//4])
            
            mid_ratio = mid_energy / (low_energy + 1e-10)
            
            # Anomaly score based on unusual mid-frequency energy
            anomaly = min(1.0, mid_ratio / 0.5) if mid_ratio > 0.3 else 0.3
            
            return {
                "anomaly_score": float(anomaly),
                "block_uniformity": float(uniformity),
            }
            
        except Exception:
            return {"anomaly_score": 0.5, "block_uniformity": 0.5}
    
    def _analyze_fft(self, gray: np.ndarray) -> Dict:
        """
        Analyze FFT (Fast Fourier Transform) spectrum.
        
        AI images show:
        - Different radial frequency distribution
        - Missing natural camera noise patterns
        - Unusual high-frequency characteristics
        """
        try:
            # Compute 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            
            # Compute radial profile (frequency vs distance from center)
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
            
            # Bin by radius
            max_r = min(center_x, center_y)
            radial_profile = np.zeros(max_r)
            counts = np.zeros(max_r)
            
            for i in range(h):
                for j in range(w):
                    radius = int(np.sqrt((i - center_y)**2 + (j - center_x)**2))
                    if radius < max_r:
                        radial_profile[radius] += magnitude[i, j]
                        counts[radius] += 1
            
            radial_profile = radial_profile / (counts + 1e-10)
            
            # Analyze profile shape
            # Natural images: smooth falloff with noise
            # AI images: often smoother or with unusual peaks
            
            # High-frequency to low-frequency ratio
            low_freq = np.mean(radial_profile[:max_r//4])
            high_freq = np.mean(radial_profile[3*max_r//4:])
            
            ratio = high_freq / (low_freq + 1e-10)
            
            # Natural photos have more high-freq energy (noise, texture)
            if ratio < 0.05:
                profile_type = "synthetic"
                spectrum_score = 0.7
            elif ratio > 0.2:
                profile_type = "natural"
                spectrum_score = 0.3
            else:
                profile_type = "uncertain"
                spectrum_score = 0.5
            
            return {
                "spectrum_score": float(spectrum_score),
                "profile_type": profile_type,
            }
            
        except Exception:
            return {"spectrum_score": 0.5, "profile_type": "uncertain"}
    
    def _analyze_noise(self, gray: np.ndarray) -> Dict:
        """
        Analyze noise patterns in image.
        
        Natural photos have:
        - Sensor noise (Gaussian + Poisson)
        - Film grain characteristics
        - Compression artifacts
        
        AI images often:
        - Too smooth (no sensor noise)
        - Uniform noise patterns (if added artificially)
        - Missing natural imperfections
        """
        try:
            # Extract high-frequency component (noise estimate)
            if OPENCV_AVAILABLE:
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            else:
                # Manual blur approximation
                from scipy.ndimage import gaussian_filter
                blurred = gaussian_filter(gray, sigma=2)
            
            noise = gray - blurred
            
            # Noise characteristics
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # Natural noise has specific variance patterns
            # AI images often too smooth or with artificial noise
            
            # High-frequency energy
            high_freq_energy = noise_std / 255.0
            
            # Check noise distribution
            # Natural noise is roughly Gaussian
            noise_flat = noise.flatten()
            
            # Kurtosis-like measure (natural noise has specific kurtosis)
            m4 = np.mean((noise_flat - np.mean(noise_flat))**4)
            m2 = np.var(noise_flat)
            kurtosis = m4 / (m2**2 + 1e-10) - 3
            
            # Natural images: kurtosis near 0 (Gaussian)
            # AI smooth: very low variance, undefined kurtosis
            # AI with fake noise: often higher kurtosis
            
            if noise_std < 2.0:
                pattern = "synthetic_smooth"
            elif abs(kurtosis) > 3.0:
                pattern = "uncertain"
            else:
                pattern = "natural_noise"
            
            return {
                "high_freq_energy": float(high_freq_energy),
                "pattern": pattern,
            }
            
        except Exception:
            return {"high_freq_energy": 0.5, "pattern": "uncertain"}
    
    def _combine_scores(
        self,
        dct_result: Dict,
        fft_result: Dict,
        noise_result: Dict,
    ) -> Dict:
        """Combine all frequency analysis scores into final probability."""
        
        # Weight each signal
        weights = {
            "dct": 0.25,
            "fft": 0.35,
            "noise": 0.40,
        }
        
        # Convert noise pattern to score
        noise_scores = {
            "natural_noise": 0.2,
            "uncertain": 0.5,
            "synthetic_smooth": 0.8,
        }
        noise_score = noise_scores.get(noise_result["pattern"], 0.5)
        
        # Combine
        combined = (
            weights["dct"] * dct_result["anomaly_score"] +
            weights["fft"] * fft_result["spectrum_score"] +
            weights["noise"] * noise_score
        )
        
        # Determine confidence
        # Higher confidence if signals agree
        scores = [
            dct_result["anomaly_score"],
            fft_result["spectrum_score"],
            noise_score,
        ]
        score_std = np.std(scores)
        
        if score_std < 0.15:
            confidence = "HIGH"
        elif score_std < 0.25:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            "probability": float(np.clip(combined, 0.0, 1.0)),
            "confidence": confidence,
        }
    
    def _error_result(self, message: str) -> FrequencyAnalysisResult:
        """Return error result."""
        return FrequencyAnalysisResult(
            dct_anomaly_score=0.5,
            dct_block_uniformity=0.5,
            fft_spectrum_score=0.5,
            fft_radial_profile="uncertain",
            high_freq_energy=0.5,
            noise_pattern="uncertain",
            frequency_ai_probability=0.5,
            confidence="LOW",
        )


def analyze_frequency(image_path: str) -> Dict:
    """
    Convenience function to analyze image frequency patterns.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict with frequency analysis results
    """
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(image_path)
    return result.to_dict()


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python frequency_analyzer.py <image_path>")
        sys.exit(1)
    
    result = analyze_frequency(sys.argv[1])
    
    print("\n" + "=" * 50)
    print("  FREQUENCY ANALYSIS")
    print("=" * 50)
    print(f"  DCT Anomaly Score:     {result['dct_anomaly_score']:.2f}")
    print(f"  DCT Block Uniformity:  {result['dct_block_uniformity']:.2f}")
    print(f"  FFT Spectrum Score:    {result['fft_spectrum_score']:.2f}")
    print(f"  FFT Radial Profile:    {result['fft_radial_profile']}")
    print(f"  High-Freq Energy:      {result['high_freq_energy']:.3f}")
    print(f"  Noise Pattern:         {result['noise_pattern']}")
    print("-" * 50)
    print(f"  AI Probability:        {result['frequency_ai_probability']:.1%}")
    print(f"  Confidence:            {result['confidence']}")
    print("=" * 50)
