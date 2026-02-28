"""
Forensic Signal Analyzers
=========================

4 physics-based forensic analyzers for AI image detection:
1. LightingAnalyzer - Shadow/illumination consistency
2. NoisePatternAnalyzer - Sensor noise (PRNU) uniformity
3. ReflectionAnalyzer - Specular highlight consistency
4. GANFingerprintAnalyzer - AI model frequency artifacts

Usage:
    from forensic_signals import analyze_forensics
    result = analyze_forensics("image.jpg")
    print(result["probability"])  # 0.0 (real) to 1.0 (AI)
"""

import os
import sys
import numpy as np
from PIL import Image

# Try to import OpenCV (optional but recommended)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] opencv-python not installed. Some forensic signals will be limited.")
    print("       Install with: pip install opencv-python")


class LightingAnalyzer:
    """
    Detects inconsistent lighting/shadow directions.
    
    Real photos have a consistent light source — shadows all point
    the same way. AI images often have physically impossible lighting
    where different parts of the image have different shadow directions.
    """
    
    def analyze(self, image_path: str) -> dict:
        """Analyze lighting consistency across image quadrants."""
        try:
            img = np.array(Image.open(image_path).convert("L").resize((256, 256)))
            
            # Compute gradient direction (where light comes from)
            grad_x = np.gradient(img.astype(float), axis=1)  # horizontal gradient
            grad_y = np.gradient(img.astype(float), axis=0)  # vertical gradient
            
            # Get dominant light direction per quadrant
            h, w = img.shape
            quadrants = [
                (0, h//2, 0, w//2),       # top-left
                (0, h//2, w//2, w),        # top-right
                (h//2, h, 0, w//2),        # bottom-left
                (h//2, h, w//2, w),        # bottom-right
            ]
            
            directions = []
            for y1, y2, x1, x2 in quadrants:
                qx = grad_x[y1:y2, x1:x2]
                qy = grad_y[y1:y2, x1:x2]
                
                # Magnitude-weighted average direction
                magnitude = np.sqrt(qx**2 + qy**2)
                mask = magnitude > np.percentile(magnitude, 75)  # strong edges only
                
                if mask.sum() > 10:
                    avg_angle = np.arctan2(
                        np.mean(qy[mask] * magnitude[mask]),
                        np.mean(qx[mask] * magnitude[mask])
                    )
                    directions.append(avg_angle)
            
            if len(directions) < 3:
                return {"probability": 0.5, "details": "insufficient_edges"}
            
            # Compare all quadrant directions
            # Real photos: directions should be similar (same light source)
            # AI images: directions may be inconsistent
            angle_diffs = []
            for i in range(len(directions)):
                for j in range(i+1, len(directions)):
                    diff = abs(directions[i] - directions[j])
                    diff = min(diff, 2*np.pi - diff)  # wrap around
                    angle_diffs.append(diff)
            
            avg_inconsistency = np.mean(angle_diffs)
            max_inconsistency = max(angle_diffs)
            
            # Map to AI probability
            # Small inconsistency (< 0.3 rad) = likely real
            # Large inconsistency (> 1.0 rad) = likely AI
            if avg_inconsistency < 0.3:
                ai_prob = 0.2
            elif avg_inconsistency < 0.6:
                ai_prob = 0.4
            elif avg_inconsistency < 1.0:
                ai_prob = 0.6
            else:
                ai_prob = 0.8
            
            return {
                "probability": ai_prob,
                "avg_inconsistency": float(avg_inconsistency),
                "max_inconsistency": float(max_inconsistency),
                "num_quadrants": len(directions),
            }
            
        except Exception as e:
            return {"probability": 0.5, "error": str(e)}


class NoisePatternAnalyzer:
    """
    Analyzes sensor noise consistency across image patches.
    
    Real cameras produce a consistent noise pattern (PRNU) across the
    entire sensor. AI-generated images either have no noise or
    inconsistent noise patterns across different regions.
    """
    
    def analyze(self, image_path: str) -> dict:
        """Analyze noise pattern uniformity across patches."""
        try:
            img = np.array(Image.open(image_path).convert("L").resize((256, 256)), dtype=np.float64)
            
            # Extract noise residual using high-pass filter
            # Subtract a blurred version to get just the noise
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(img, sigma=2.0)
            noise = img - smoothed
            
            # Split into patches and measure noise variance per patch
            patch_size = 32
            variances = []
            means = []
            
            for y in range(0, 256 - patch_size + 1, patch_size):
                for x in range(0, 256 - patch_size + 1, patch_size):
                    patch = noise[y:y+patch_size, x:x+patch_size]
                    variances.append(np.var(patch))
                    means.append(np.mean(np.abs(patch)))
            
            variances = np.array(variances)
            means = np.array(means)
            
            # Real photos: variance is consistent across patches (uniform sensor noise)
            # AI images: variance varies wildly (smooth areas vs textured areas)
            variance_of_variances = np.var(variances) / (np.mean(variances) + 1e-8)
            coefficient_of_variation = np.std(variances) / (np.mean(variances) + 1e-8)
            
            # Also check if noise is too uniform (suspiciously clean = AI)
            avg_noise_level = np.mean(means)
            
            # Scoring
            # Low CoV (< 0.3) = consistent noise = likely real camera
            # High CoV (> 0.8) = inconsistent noise = likely AI
            # Very low noise level (< 0.5) = suspiciously clean = likely AI
            
            if avg_noise_level < 0.5:
                # Almost no noise at all — very suspicious (AI tends to be too clean)
                ai_prob = 0.75
            elif coefficient_of_variation < 0.3:
                ai_prob = 0.2  # Very consistent noise = real sensor
            elif coefficient_of_variation < 0.5:
                ai_prob = 0.35
            elif coefficient_of_variation < 0.8:
                ai_prob = 0.5
            elif coefficient_of_variation < 1.2:
                ai_prob = 0.65
            else:
                ai_prob = 0.8  # Very inconsistent noise
            
            return {
                "probability": ai_prob,
                "noise_variance_cv": float(coefficient_of_variation),
                "avg_noise_level": float(avg_noise_level),
                "variance_of_variances": float(variance_of_variances),
                "num_patches": len(variances),
            }
            
        except Exception as e:
            return {"probability": 0.5, "error": str(e)}


class ReflectionAnalyzer:
    """
    Analyzes specular highlight consistency.
    
    Real photos have physically consistent reflections — bright spots
    come from actual light sources and follow physics rules. AI images
    often have highlights that don't match the scene's lighting.
    """
    
    def analyze(self, image_path: str) -> dict:
        """Analyze specular highlight distribution and consistency."""
        try:
            img = np.array(Image.open(image_path).convert("RGB").resize((256, 256)))
            gray = np.mean(img, axis=2)
            
            # Find specular highlights (very bright spots)
            threshold = np.percentile(gray, 98)  # Top 2% brightest pixels
            highlights = gray > threshold
            
            num_highlight_pixels = highlights.sum()
            
            if num_highlight_pixels < 10:
                # No significant highlights — can't analyze
                return {"probability": 0.5, "details": "no_highlights"}
            
            # Get positions of highlight clusters
            from scipy.ndimage import label
            labeled, num_features = label(highlights)
            
            if num_features < 2:
                # Single highlight or none — inconclusive
                return {"probability": 0.5, "num_highlights": num_features}
            
            # For each highlight cluster, get centroid and intensity
            centroids = []
            intensities = []
            sizes = []
            
            for i in range(1, min(num_features + 1, 20)):  # max 20 highlights
                mask = labeled == i
                size = mask.sum()
                if size < 3:
                    continue
                    
                ys, xs = np.where(mask)
                centroid = (np.mean(ys), np.mean(xs))
                intensity = np.mean(gray[mask])
                
                centroids.append(centroid)
                intensities.append(intensity)
                sizes.append(size)
            
            if len(centroids) < 2:
                return {"probability": 0.5, "num_highlights": len(centroids)}
            
            # Check intensity consistency
            # Real photos: highlights from same light source have similar intensity
            intensity_variation = np.std(intensities) / (np.mean(intensities) + 1e-8)
            
            # Check size consistency  
            # Real photos: similar-distance highlights have similar sizes
            size_variation = np.std(sizes) / (np.mean(sizes) + 1e-8)
            
            # Check spatial distribution
            # Real photos: highlights tend to align with light source direction
            centroid_arr = np.array(centroids)
            distances = []
            for i in range(len(centroid_arr)):
                for j in range(i+1, len(centroid_arr)):
                    d = np.linalg.norm(centroid_arr[i] - centroid_arr[j])
                    distances.append(d)
            
            spread = np.std(distances) / (np.mean(distances) + 1e-8) if distances else 0
            
            # Combined scoring
            # High intensity variation + high size variation = inconsistent highlights = AI
            inconsistency = (intensity_variation * 0.4 + size_variation * 0.3 + spread * 0.3)
            
            if inconsistency < 0.3:
                ai_prob = 0.25
            elif inconsistency < 0.5:
                ai_prob = 0.4
            elif inconsistency < 0.7:
                ai_prob = 0.55
            else:
                ai_prob = 0.75
            
            return {
                "probability": ai_prob,
                "num_highlights": len(centroids),
                "intensity_variation": float(intensity_variation),
                "size_variation": float(size_variation),
                "spatial_spread": float(spread),
                "inconsistency_score": float(inconsistency),
            }
            
        except Exception as e:
            return {"probability": 0.5, "error": str(e)}


class GANFingerprintAnalyzer:
    """
    Detects GAN-specific frequency artifacts in images.
    
    Many AI generators (especially GANs) leave distinctive periodic
    patterns in the frequency domain. These appear as spectral peaks
    at specific frequencies that don't occur in natural images.
    """
    
    def analyze(self, image_path: str) -> dict:
        """Analyze FFT spectrum for GAN-specific artifacts."""
        try:
            img = np.array(Image.open(image_path).convert("L").resize((256, 256)), dtype=np.float64)
            
            # 2D FFT
            fft = np.fft.fft2(img)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            
            # Log magnitude for better analysis
            log_mag = np.log1p(magnitude)
            
            # Remove DC component (center)
            h, w = log_mag.shape
            cy, cx = h // 2, w // 2
            log_mag[cy-2:cy+3, cx-2:cx+3] = np.median(log_mag)
            
            # 1. Check for periodic peaks (GAN grid artifacts)
            # Compute radial profile
            Y, X = np.ogrid[:h, :w]
            r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
            
            radial_profile = np.zeros(min(cy, cx))
            radial_counts = np.zeros(min(cy, cx))
            
            for radius in range(len(radial_profile)):
                mask = r == radius
                if mask.any():
                    radial_profile[radius] = np.mean(log_mag[mask])
                    radial_counts[radius] = mask.sum()
            
            # Normalize profile
            if radial_profile.max() > 0:
                norm_profile = radial_profile / radial_profile.max()
            else:
                return {"probability": 0.5, "details": "flat_spectrum"}
            
            # 2. Find peaks in radial profile (potential GAN artifacts)
            # Natural images have smooth falloff; GANs have peaks
            peak_count = 0
            peak_strengths = []
            
            for i in range(5, len(norm_profile) - 5):
                # Is this a local peak?
                local_region = norm_profile[max(0, i-5):min(len(norm_profile), i+6)]
                local_mean = np.mean(local_region)
                
                if norm_profile[i] > local_mean * 1.3:  # 30% above local mean
                    peak_count += 1
                    peak_strengths.append(norm_profile[i] / local_mean)
            
            # 3. Check high-frequency energy ratio
            mid_freq = len(radial_profile) // 3
            high_freq = 2 * len(radial_profile) // 3
            
            low_energy = np.mean(radial_profile[5:mid_freq]) if mid_freq > 5 else 1
            mid_energy = np.mean(radial_profile[mid_freq:high_freq]) if high_freq > mid_freq else 0
            high_energy = np.mean(radial_profile[high_freq:]) if len(radial_profile) > high_freq else 0
            
            # Natural images: energy drops smoothly (low > mid >> high)
            # AI images: may have unusual energy distribution
            energy_ratio = (mid_energy + high_energy) / (low_energy + 1e-8)
            
            # 4. Check for cross-shaped or grid patterns (common in upsampled images)
            center_row = log_mag[cy, :]
            center_col = log_mag[:, cy]
            
            row_peak = np.max(center_row) / (np.mean(center_row) + 1e-8)
            col_peak = np.max(center_col) / (np.mean(center_col) + 1e-8)
            
            cross_pattern = max(row_peak, col_peak)
            
            # Combined scoring
            # More peaks + higher energy ratio + cross pattern = more likely AI
            if peak_count >= 5 and energy_ratio > 0.5:
                ai_prob = 0.8
            elif peak_count >= 3 or cross_pattern > 3.0:
                ai_prob = 0.65
            elif peak_count >= 1 and energy_ratio > 0.3:
                ai_prob = 0.55
            elif energy_ratio < 0.15:
                ai_prob = 0.25  # Natural smooth falloff
            else:
                ai_prob = 0.4
            
            return {
                "probability": ai_prob,
                "spectral_peaks": peak_count,
                "peak_strengths": [float(s) for s in peak_strengths[:5]],
                "energy_ratio": float(energy_ratio),
                "cross_pattern_strength": float(cross_pattern),
            }
            
        except Exception as e:
            return {"probability": 0.5, "error": str(e)}


# ============================================================
# MAIN API
# ============================================================

def analyze_forensics(image_path: str) -> dict:
    """
    Run all 4 forensic analyzers and return combined result.
    
    Returns:
        dict with "probability" (0.0-1.0), individual scores, and details
    """
    results = {}
    
    # Run all analyzers
    lighting = LightingAnalyzer().analyze(image_path)
    results["lighting"] = lighting
    
    noise = NoisePatternAnalyzer().analyze(image_path)
    results["noise"] = noise
    
    reflection = ReflectionAnalyzer().analyze(image_path)
    results["reflection"] = reflection
    
    gan_fp = GANFingerprintAnalyzer().analyze(image_path)
    results["gan_fingerprint"] = gan_fp
    
    # Weighted combination
    weights = {
        "lighting": 0.25,
        "noise": 0.30,        # Noise is most reliable
        "reflection": 0.15,   # Reflection is niche (not all images have highlights)
        "gan_fingerprint": 0.30,
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for key, weight in weights.items():
        prob = results[key].get("probability", 0.5)
        # Skip inconclusive results (exactly 0.5) — don't let them dilute
        if prob != 0.5 or "error" not in results[key]:
            weighted_sum += prob * weight
            total_weight += weight
    
    if total_weight > 0:
        combined_prob = weighted_sum / total_weight
    else:
        combined_prob = 0.5
    
    results["probability"] = combined_prob
    results["model"] = "Forensic Analysis"
    results["weight"] = 0.15  # Weight in ensemble
    
    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python forensic_signals.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  FORENSIC SIGNAL ANALYSIS")
    print(f"{'='*60}")
    print(f"\n  Analyzing: {os.path.basename(image_path)}\n")
    
    result = analyze_forensics(image_path)
    
    print(f"  {'Signal':<25s} {'AI Prob':>8s}  Details")
    print(f"  {'-'*55}")
    
    for key in ["lighting", "noise", "reflection", "gan_fingerprint"]:
        data = result[key]
        prob = data.get("probability", 0.5)
        
        # Pick one detail to show
        detail = ""
        if key == "lighting":
            detail = f"inconsistency={data.get('avg_inconsistency', '?'):.2f}" if isinstance(data.get('avg_inconsistency'), float) else data.get('details', '')
        elif key == "noise":
            detail = f"CV={data.get('noise_variance_cv', '?'):.2f}" if isinstance(data.get('noise_variance_cv'), float) else data.get('error', '')
        elif key == "reflection":
            detail = f"highlights={data.get('num_highlights', '?')}" if data.get('num_highlights') else data.get('details', '')
        elif key == "gan_fingerprint":
            detail = f"peaks={data.get('spectral_peaks', '?')}, energy={data.get('energy_ratio', '?'):.2f}" if isinstance(data.get('energy_ratio'), float) else ''
        
        label = key.replace("_", " ").title()
        print(f"  {label:<25s} {prob:>7.1%}  {detail}")
    
    print(f"\n  {'Combined Forensic':25s} {result['probability']:>7.1%}")
    
    # Verdict
    prob = result["probability"]
    if prob > 0.6:
        verdict = "LIKELY AI (forensic signals)"
    elif prob < 0.4:
        verdict = "LIKELY REAL (forensic signals)"
    else:
        verdict = "INCONCLUSIVE (forensic signals)"
    
    print(f"\n  Forensic Verdict: {verdict}")
    print(f"{'='*60}\n")
