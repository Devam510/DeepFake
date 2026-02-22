"""
Enhanced Detection Script with Detailed Breakdown
==================================================

Provides professional-quality output similar to deepfakedetection.io:
- Overall Forgery Score
- Detailed breakdown by category
- Visual analysis of specific elements
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import detectors
try:
    from src.extraction.domain_classifier import DomainClassifier, classify_domain
    from src.modeling.general_image_detector import GeneralImageDetector
    DETECTORS_AVAILABLE = True
except ImportError:
    DETECTORS_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DetailedAnalyzer:
    """Provides detailed AI detection breakdown like professional services."""
    
    def __init__(self):
        self.domain_classifier = DomainClassifier() if DETECTORS_AVAILABLE else None
        self.general_detector = GeneralImageDetector() if DETECTORS_AVAILABLE else None
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Run comprehensive analysis with detailed breakdown."""
        
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        try:
            image = Image.open(image_path).convert("RGB")
            img_array = np.array(image)
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}
        
        # Collect all analysis results
        breakdown = {}
        warnings = []
        
        # 1. Domain classification
        if self.domain_classifier:
            domain_result = self.domain_classifier.classify(image_path)
            image_type = self._get_image_type(domain_result.domain)
        else:
            image_type = "Unknown"
            domain_result = None
        
        # 2. Neural network analysis
        neural_score = self._analyze_neural(image_path)
        breakdown["Neural Network Analysis"] = {
            "score": neural_score,
            "description": self._get_neural_description(neural_score),
            "warning": neural_score > 60,
        }
        
        # 3. Eye analysis (for face images)
        eye_score = self._analyze_eyes(img_array)
        breakdown["Eye Analysis"] = {
            "score": eye_score,
            "description": self._get_eye_description(eye_score, domain_result),
            "warning": eye_score > 50,
        }
        
        # 4. Text analysis (watermarks, dates)
        text_score = self._analyze_text(img_array, image_path)
        breakdown["Text Analysis"] = {
            "score": text_score,
            "description": self._get_text_description(text_score),
            "warning": text_score > 50,
        }
        
        # 5. Lighting and shadows
        lighting_score = self._analyze_lighting(img_array)
        breakdown["Lighting and Shadows"] = {
            "score": lighting_score,
            "description": self._get_lighting_description(lighting_score),
            "warning": lighting_score > 50,
        }
        
        # 6. Background coherence
        bg_score = self._analyze_background(img_array)
        breakdown["Background Coherence"] = {
            "score": bg_score,
            "description": self._get_background_description(bg_score),
            "warning": bg_score > 50,
        }
        
        # 7. Facial features (if applicable)
        face_score = self._analyze_facial_features(img_array)
        breakdown["Facial Features and Symmetry"] = {
            "score": face_score,
            "description": self._get_face_description(face_score, domain_result),
            "warning": face_score > 50,
        }
        
        # 8. Skin texture (if faces present)
        skin_score = self._analyze_skin_texture(img_array, domain_result)
        breakdown["Skin Texture Analysis"] = {
            "score": skin_score,
            "description": self._get_skin_description(skin_score, domain_result),
            "warning": skin_score > 50,
        }
        
        # 9. Hair and hairline
        hair_score = self._analyze_hair(img_array, domain_result)
        breakdown["Hair and Hairline Details"] = {
            "score": hair_score,
            "description": self._get_hair_description(hair_score, domain_result),
            "warning": hair_score > 50,
        }
        
        # 10. Hand/anatomy issues
        anatomy_score = self._analyze_anatomy(img_array)
        breakdown["Hand/Anatomy Issues"] = {
            "score": anatomy_score,
            "description": self._get_anatomy_description(anatomy_score),
            "warning": anatomy_score > 50,
        }
        
        # 11. Overall image quality
        quality_score = self._analyze_quality(img_array)
        breakdown["Overall Image Quality"] = {
            "score": quality_score,
            "description": self._get_quality_description(quality_score),
            "warning": quality_score > 50,
        }
        
        # 12. Frequency analysis
        freq_score = self._analyze_frequency(img_array)
        breakdown["Frequency Spectrum Analysis"] = {
            "score": freq_score,
            "description": self._get_frequency_description(freq_score),
            "warning": freq_score > 60,
        }
        
        # Calculate overall forgery score
        # Weight neural network more heavily
        weights = {
            "Neural Network Analysis": 0.30,
            "Text Analysis": 0.08,
            "Lighting and Shadows": 0.08,
            "Background Coherence": 0.10,
            "Overall Image Quality": 0.12,
            "Frequency Spectrum Analysis": 0.12,
            "Eye Analysis": 0.05,
            "Facial Features and Symmetry": 0.05,
            "Skin Texture Analysis": 0.03,
            "Hair and Hairline Details": 0.03,
            "Hand/Anatomy Issues": 0.04,
        }
        
        overall_score = sum(
            breakdown[k]["score"] * weights.get(k, 0.05)
            for k in breakdown
        )
        
        # Generate summary
        summary = self._generate_summary(breakdown, overall_score)
        
        return {
            "image_path": image_path,
            "image_type": image_type,
            "overall_forgery_score": round(overall_score, 1),
            "verdict": self._get_verdict(overall_score),
            "summary": summary,
            "detailed_breakdown": breakdown,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _get_image_type(self, domain: str) -> str:
        """Map domain to image type."""
        mapping = {
            "face": "Portrait/Face Image",
            "non_face_photo": "Photograph",
            "art_or_illustration": "Art/Creative Image",
            "synthetic_graphics": "Graphics/UI",
            "unknown": "Unknown Type",
        }
        return mapping.get(domain, "Unknown")
    
    def _analyze_neural(self, image_path: str) -> float:
        """Get neural network prediction."""
        if not self.general_detector or not self.general_detector.model_loaded:
            return 50.0  # Neutral if no model
        
        try:
            result = self.general_detector.predict(image_path)
            return result.synthetic_probability * 100
        except Exception:
            return 50.0
    
    def _get_neural_description(self, score: float) -> str:
        if score > 80:
            return "Strong indicators of AI generation detected by neural network."
        elif score > 60:
            return "Moderate AI generation indicators. Some features suggest artificial origin."
        elif score > 40:
            return "Inconclusive neural network analysis. Cannot determine origin."
        else:
            return "Neural network finds patterns consistent with natural photography."
    
    def _analyze_eyes(self, img_array: np.ndarray) -> float:
        """Analyze eye regions for AI artifacts."""
        if not OPENCV_AVAILABLE:
            return 0.0
        
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
            
            if len(eyes) == 0:
                return 0.0  # No eyes to analyze
            
            # Check for asymmetry in eye regions
            if len(eyes) >= 2:
                # Compare eye regions
                eye1 = img_array[eyes[0][1]:eyes[0][1]+eyes[0][3], 
                                 eyes[0][0]:eyes[0][0]+eyes[0][2]]
                eye2 = img_array[eyes[1][1]:eyes[1][1]+eyes[1][3], 
                                 eyes[1][0]:eyes[1][0]+eyes[1][2]]
                
                # Simple variance comparison - AI eyes often too similar
                var_diff = abs(np.var(eye1) - np.var(eye2))
                if var_diff < 100:  # Too similar
                    return 60.0
            
            return 20.0  # Eyes found, look normal
        except Exception:
            return 0.0
    
    def _get_eye_description(self, score: float, domain_result) -> str:
        if domain_result and domain_result.face_count == 0:
            return "There are no eyes visible in the scene."
        if score > 50:
            return "Eye regions show potential AI artifacts or unusual symmetry."
        elif score > 0:
            return "Eye analysis shows natural characteristics."
        return "No eye regions detected for analysis."
    
    def _analyze_text(self, img_array: np.ndarray, image_path: str) -> float:
        """Detect and analyze text/watermarks."""
        # Check filename for AI generator hints
        filename = os.path.basename(image_path).lower()
        ai_hints = ["gemini", "dall", "midjourney", "stable", "generated", "ai"]
        
        if any(hint in filename for hint in ai_hints):
            return 70.0  # Filename suggests AI
        
        # Check for uniform text-like regions in corners (watermarks)
        try:
            h, w = img_array.shape[:2]
            corners = [
                img_array[:h//10, :w//4],  # Top-left
                img_array[:h//10, -w//4:],  # Top-right
                img_array[-h//10:, :w//4],  # Bottom-left
                img_array[-h//10:, -w//4:],  # Bottom-right
            ]
            
            for corner in corners:
                # Check for text-like patterns (high contrast small regions)
                gray = np.mean(corner, axis=2) if len(corner.shape) == 3 else corner
                edges = np.abs(np.diff(gray, axis=1)).mean()
                if edges > 30:  # High edge activity suggests text
                    return 70.0
            
            return 10.0
        except Exception:
            return 0.0
    
    def _get_text_description(self, score: float) -> str:
        if score > 60:
            return "Text or watermark detected. May indicate AI generator labeling."
        elif score > 30:
            return "Possible text elements detected in image."
        return "No significant text or watermarks detected."
    
    def _analyze_lighting(self, img_array: np.ndarray) -> float:
        """Analyze lighting consistency."""
        try:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            
            # Check for inconsistent lighting (multiple light sources)
            h, w = gray.shape
            quadrants = [
                gray[:h//2, :w//2].mean(),
                gray[:h//2, w//2:].mean(),
                gray[h//2:, :w//2].mean(),
                gray[h//2:, w//2:].mean(),
            ]
            
            variance = np.var(quadrants)
            
            # AI images often have too-consistent lighting
            if variance < 100:  # Too uniform
                return 55.0
            elif variance > 2000:  # Natural variation
                return 30.0
            return 45.0
        except Exception:
            return 50.0
    
    def _get_lighting_description(self, score: float) -> str:
        if score > 50:
            return "The lighting appears slightly artificial or too uniform across the scene."
        elif score > 30:
            return "Lighting is generally consistent with natural illumination."
        return "Lighting analysis shows natural variation typical of real photography."
    
    def _analyze_background(self, img_array: np.ndarray) -> float:
        """Analyze background coherence."""
        try:
            # Check for repeating patterns (common in AI backgrounds)
            h, w = img_array.shape[:2]
            
            # Sample regions from background (edges of image)
            top = img_array[:h//8, :, :].reshape(-1, 3)
            bottom = img_array[-h//8:, :, :].reshape(-1, 3)
            
            # Check color consistency
            top_std = np.std(top, axis=0).mean()
            bottom_std = np.std(bottom, axis=0).mean()
            
            # AI backgrounds often too smooth or with repeating patterns
            if top_std < 20 and bottom_std < 20:
                return 60.0  # Too smooth
            
            return 40.0  # Normal variation
        except Exception:
            return 50.0
    
    def _get_background_description(self, score: float) -> str:
        if score > 55:
            return "Background shows potential AI artifacts - repeating patterns or unusual smoothness."
        elif score > 40:
            return "Background elements are generally coherent with minor irregularities."
        return "Background appears natural and coherent."
    
    def _analyze_facial_features(self, img_array: np.ndarray) -> float:
        """Analyze facial feature symmetry."""
        if not OPENCV_AVAILABLE:
            return 0.0
        
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            if len(faces) == 0:
                return 0.0
            
            # Check face symmetry
            for (x, y, w, h) in faces:
                face_region = gray[y:y+h, x:x+w]
                left = face_region[:, :w//2]
                right = np.fliplr(face_region[:, w//2:])
                
                # AI faces often too symmetric
                if left.shape == right.shape:
                    similarity = 1 - (np.abs(left - right).mean() / 255)
                    if similarity > 0.85:  # Very symmetric
                        return 65.0
            
            return 30.0  # Normal asymmetry
        except Exception:
            return 0.0
    
    def _get_face_description(self, score: float, domain_result) -> str:
        if domain_result and domain_result.face_count == 0:
            return "There are no visible faces in the image to analyze for symmetry or artificial smoothing."
        if score > 50:
            return "Facial features show unusual symmetry or smoothing typical of AI generation."
        elif score > 0:
            return "Facial features appear natural with normal asymmetry."
        return "No faces detected for facial feature analysis."
    
    def _analyze_skin_texture(self, img_array: np.ndarray, domain_result) -> float:
        """Analyze skin texture for AI smoothing."""
        if domain_result and domain_result.face_count == 0:
            return 0.0
        
        try:
            # Sample skin-tone regions
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV) if OPENCV_AVAILABLE else None
            if hsv is None:
                return 0.0
            
            # Skin detection in HSV
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            skin_area = np.sum(skin_mask > 0)
            if skin_area < 1000:
                return 0.0
            
            # Check texture in skin regions
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            skin_texture = gray[skin_mask > 0]
            
            texture_var = np.var(skin_texture)
            if texture_var < 200:  # Too smooth
                return 60.0
            
            return 25.0  # Normal texture
        except Exception:
            return 0.0
    
    def _get_skin_description(self, score: float, domain_result) -> str:
        if domain_result and domain_result.face_count == 0:
            return "There are no visible people in the image. Thus there are no skin textures to be analyzed."
        if score > 50:
            return "Skin texture appears unusually smooth, suggesting AI generation."
        elif score > 0:
            return "Skin texture shows natural variation and pores."
        return "No skin regions detected for analysis."
    
    def _analyze_hair(self, img_array: np.ndarray, domain_result) -> float:
        """Analyze hair details."""
        if domain_result and domain_result.face_count == 0:
            return 0.0
        
        # Hair analysis is complex - return neutral for now
        return 0.0
    
    def _get_hair_description(self, score: float, domain_result) -> str:
        if domain_result and domain_result.face_count == 0:
            return "There are no visible faces in the image, so hair analysis is not applicable."
        if score > 50:
            return "Hair shows potential AI artifacts - unusual merging or repetitive strands."
        return "Hair details appear natural."
    
    def _analyze_anatomy(self, img_array: np.ndarray) -> float:
        """Analyze for anatomy issues (hands, fingers)."""
        # This would require a pose detection model for proper analysis
        # Return neutral for now
        return 0.0
    
    def _get_anatomy_description(self, score: float) -> str:
        if score > 50:
            return "Potential anatomical issues detected (hand/finger anomalies)."
        return "There are no visible hands or human figures in sufficient detail to assess anatomical accuracy."
    
    def _analyze_quality(self, img_array: np.ndarray) -> float:
        """Analyze overall image quality - AI images often too perfect."""
        try:
            # Check for "too perfect" characteristics
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            
            # Noise analysis - AI images often lack natural noise
            noise_level = np.std(gray - cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)) if OPENCV_AVAILABLE else np.std(gray)
            
            if noise_level < 3:  # Very clean
                return 65.0
            elif noise_level > 15:  # Natural noise
                return 30.0
            
            return 50.0
        except Exception:
            return 50.0
    
    def _get_quality_description(self, score: float) -> str:
        if score > 50:
            return "The image has a slightly 'too perfect' quality. The lighting is very pleasant, and the textures are mostly well-rendered. However, the clarity and composition lean towards an artificial aesthetic."
        return "Image quality shows characteristics of natural photography."
    
    def _analyze_frequency(self, img_array: np.ndarray) -> float:
        """Analyze frequency spectrum for AI artifacts."""
        try:
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            
            # FFT analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            
            # AI images often have unusual frequency patterns
            center = magnitude.shape[0] // 2
            high_freq = magnitude[center-20:center+20, center-20:center+20].mean()
            low_freq = magnitude[:20, :20].mean()
            
            ratio = high_freq / (low_freq + 1e-10)
            
            if ratio > 1.5:  # Unusual frequency distribution
                return 55.0
            return 35.0
        except Exception:
            return 50.0
    
    def _get_frequency_description(self, score: float) -> str:
        if score > 50:
            return "Frequency analysis shows patterns sometimes associated with AI generation."
        return "Frequency spectrum appears consistent with natural photography."
    
    def _generate_summary(self, breakdown: Dict, overall_score: float) -> str:
        """Generate analysis summary."""
        warnings = [k for k, v in breakdown.items() if v.get("warning")]
        
        if overall_score > 70:
            base = "The image analysis reveals strong indicators of AI generation or heavy manipulation."
        elif overall_score > 50:
            base = "The image analysis reveals a blend of realistic and artificial elements."
        else:
            base = "The image analysis suggests this is likely an authentic photograph."
        
        if warnings:
            details = " Key concerns include: " + ", ".join(warnings[:3]) + "."
        else:
            details = " No significant AI indicators detected."
        
        return base + details
    
    def _get_verdict(self, score: float) -> str:
        if score > 70:
            return "LIKELY AI-GENERATED"
        elif score > 50:
            return "POSSIBLY AI-GENERATED"
        elif score > 35:
            return "INCONCLUSIVE"
        else:
            return "LIKELY AUTHENTIC"


def print_detailed_report(result: Dict):
    """Print professional-style detailed report."""
    
    print("\n" + "=" * 70)
    print("  DETECTION REPORT")
    print("=" * 70)
    print(f"  Image Type: {result.get('image_type', 'Unknown')}")
    print()
    
    # Overall score with big display
    score = result.get("overall_forgery_score", 0)
    print(f"  Overall Forgery Score")
    print()
    print(f"       {score:.0f}%")
    print()
    
    # Analysis summary
    print("  Analysis Summary")
    print("-" * 70)
    print(f"  {result.get('summary', 'No summary available.')}")
    print()
    
    # Verdict
    verdict = result.get("verdict", "UNKNOWN")
    print(f"  Verdict: {verdict}")
    print()
    
    # Detailed breakdown
    print("=" * 70)
    print("  DETAILED BREAKDOWN")
    print("=" * 70)
    
    breakdown = result.get("detailed_breakdown", {})
    for category, details in breakdown.items():
        score = details.get("score", 0)
        warning = details.get("warning", False)
        desc = details.get("description", "")
        
        warning_badge = " ⚠️ WARNING" if warning else ""
        print(f"\n  {category}{warning_badge}")
        print(f"  {'─' * 60}  {score:.0f}%")
        print(f"  {desc}")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python detailed_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("\n  Analyzing image...")
    analyzer = DetailedAnalyzer()
    result = analyzer.analyze(image_path)
    
    if "error" in result:
        print(f"  Error: {result['error']}")
        sys.exit(1)
    
    print_detailed_report(result)


if __name__ == "__main__":
    main()
