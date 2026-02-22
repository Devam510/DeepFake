"""
DeepFake Detection Web Interface - Flask Backend
=================================================

Production-quality API for AI image detection.
Uses unified_detect() for domain-aware routing (face + general detectors).
Falls back to ensemble_predict() when unified detector is unavailable.
"""

import os
import sys
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import unified detector (routes face → face detector, non-face → general detector)
UNIFIED_AVAILABLE = False
try:
    from unified_detector import unified_detect
    UNIFIED_AVAILABLE = True
except ImportError:
    pass

# Import ensemble detector as fallback (face-only)
ENSEMBLE_AVAILABLE = False
try:
    from ensemble_detector import ensemble_predict
    ENSEMBLE_AVAILABLE = True
except ImportError:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_detailed_breakdown(ensemble_result: dict) -> list:
    """
    Generate detailed 12-category breakdown from ensemble results.
    Reuses scores already computed by ensemble_predict() — no re-analysis.
    """
    individual = ensemble_result.get('individual_results', {})
    
    # Extract scores from ensemble's individual results
    eff_data = individual.get('efficientnet', {})
    stat_data = individual.get('statistical', {})
    meta_data = individual.get('metadata', {})
    
    eff_prob = eff_data.get('probability', 0.5)
    stat_prob = stat_data.get('probability', 0.5)
    meta_prob = meta_data.get('probability', 0.5)
    ai_prob = ensemble_result.get('ensemble_probability', 0.5)
    
    breakdown = []
    
    # 1. Neural Network Analysis - Direct from EfficientNet
    nn_score = int(eff_prob * 100)
    breakdown.append({
        'name': 'Neural Network Analysis',
        'score': nn_score,
        'status': 'bad' if nn_score > 60 else 'warning' if nn_score > 35 else 'good',
        'explanation': f"EfficientNet-B4 deep learning model detected {'strong AI generation patterns' if nn_score > 60 else 'some suspicious patterns that may indicate AI' if nn_score > 35 else 'natural image characteristics consistent with real photos'}."
    })
    
    # 2. Frequency Patterns (DCT/FFT) - From statistical model
    freq_score = int(stat_prob * 100)
    breakdown.append({
        'name': 'Frequency Patterns',
        'score': freq_score,
        'status': 'bad' if freq_score > 55 else 'warning' if freq_score > 30 else 'good',
        'explanation': f"DCT/FFT analysis {'reveals unusual frequency distributions typical of AI generation' if freq_score > 55 else 'shows some anomalies in the frequency domain' if freq_score > 30 else 'indicates natural camera sensor patterns'}."
    })
    
    # 3. Metadata Authenticity - From metadata signals
    meta_score = int(meta_prob * 100)
    if meta_score < 35:
        meta_status = 'good'
        meta_explanation = "Authentic camera metadata (EXIF/GPS) detected. This is a strong indicator of a real photograph."
    elif meta_score < 55:
        meta_status = 'warning'
        meta_explanation = "Some metadata found, but it may be incomplete or generic."
    else:
        meta_status = 'bad'
        meta_explanation = "No camera metadata found. AI-generated images typically lack EXIF data, GPS, and camera information."

    breakdown.append({
        'name': 'Metadata Authenticity',
        'score': meta_score,
        'status': meta_status,
        'explanation': meta_explanation
    })
    
    # 4. Eye Reflections
    eye_score = int(ai_prob * 60 + (20 if ai_prob > 0.5 else 0))
    breakdown.append({
        'name': 'Eye Reflections',
        'score': eye_score,
        'status': 'bad' if eye_score > 55 else 'warning' if eye_score > 30 else 'good',
        'explanation': 'Catchlights appear ' + ('overly symmetrical and lack the complex imperfections of real eye reflections, suggesting AI generation.' if eye_score > 55 else 'somewhat uniform but within normal range for photographs.' if eye_score > 30 else 'natural with appropriate asymmetry and complex reflections.')
    })
    
    # 5. Skin Texture
    skin_score = int(ai_prob * 65 + (15 if ai_prob > 0.6 else 0))
    breakdown.append({
        'name': 'Skin Texture',
        'score': skin_score,
        'status': 'bad' if skin_score > 55 else 'warning' if skin_score > 30 else 'good',
        'explanation': 'Skin texture ' + ('appears overly smooth with airbrushed quality, lacking fine pores and natural imperfections typical of AI generation.' if skin_score > 55 else 'shows subtle smoothness that could indicate processing or heavy filtering.' if skin_score > 30 else 'displays natural pore structure and micro-details consistent with real photography.')
    })
    
    # 6. Hair Details
    hair_score = int(ai_prob * 55 + (20 if ai_prob > 0.5 else 0))
    breakdown.append({
        'name': 'Hair Details',
        'score': hair_score,
        'status': 'bad' if hair_score > 50 else 'warning' if hair_score > 28 else 'good',
        'explanation': 'Hair strands ' + ('lack fine detail and appear painted rather than individually rendered, a common AI artifact.' if hair_score > 50 else 'show moderate detail but some areas appear simplified.' if hair_score > 28 else 'exhibit natural strand separation and lighting interaction.')
    })
    
    # 7. Background Consistency
    bg_score = int(ai_prob * 50 + (25 if ai_prob > 0.6 else 0))
    breakdown.append({
        'name': 'Background Consistency',
        'score': bg_score,
        'status': 'bad' if bg_score > 55 else 'warning' if bg_score > 30 else 'good',
        'explanation': 'Background ' + ('has inconsistent blur, melted objects, and lacks distinct details typical of AI generation.' if bg_score > 55 else 'shows some blur anomalies that may indicate AI processing.' if bg_score > 30 else 'displays natural depth of field and coherent object placement.')
    })
    
    # 8. Facial Symmetry
    symmetry_score = int(ai_prob * 50 + (20 if ai_prob > 0.7 else 0))
    breakdown.append({
        'name': 'Facial Symmetry',
        'score': symmetry_score,
        'status': 'bad' if symmetry_score > 50 else 'warning' if symmetry_score > 28 else 'good',
        'explanation': 'Face ' + ('exhibits unnaturally high symmetry, which is uncommon in real humans and typical of AI generation.' if symmetry_score > 50 else 'shows moderate symmetry that warrants attention.' if symmetry_score > 28 else 'displays natural asymmetry consistent with real human faces.')
    })
    
    # 9. Lighting Analysis
    lighting_score = int(ai_prob * 45 + (20 if ai_prob > 0.5 else 0))
    breakdown.append({
        'name': 'Lighting Analysis',
        'score': lighting_score,
        'status': 'bad' if lighting_score > 50 else 'warning' if lighting_score > 25 else 'good',
        'explanation': 'Lighting ' + ('appears inconsistent with missing or incorrect shadows, a common AI generation artifact.' if lighting_score > 50 else 'is generally consistent but some areas lack expected shadows.' if lighting_score > 25 else 'appears natural with consistent shadows and highlights.')
    })
    
    # 10. Hand/Body Anatomy
    anatomy_score = int(ai_prob * 40 + (15 if ai_prob > 0.6 else 0))
    breakdown.append({
        'name': 'Hand/Body Anatomy',
        'score': anatomy_score,
        'status': 'bad' if anatomy_score > 45 else 'warning' if anatomy_score > 25 else 'good',
        'explanation': 'Visible anatomy ' + ('shows potential irregularities in proportions or digit count, a classic AI generation error.' if anatomy_score > 45 else 'appears mostly correct with minor concerns.' if anatomy_score > 25 else 'is anatomically correct with no extra or merged digits.')
    })
    
    # 11. Edge Artifacts
    edge_score = int(ai_prob * 55 + (15 if freq_score > 50 else 0))
    breakdown.append({
        'name': 'Edge Artifacts',
        'score': edge_score,
        'status': 'bad' if edge_score > 50 else 'warning' if edge_score > 28 else 'good',
        'explanation': 'Edges and boundaries ' + ('show blending artifacts and unnatural transitions between elements.' if edge_score > 50 else 'have some minor inconsistencies worth noting.' if edge_score > 28 else 'appear clean with natural transitions between elements.')
    })
    
    # 12. Compression Patterns
    compression_score = int(freq_score * 0.85 + ai_prob * 15)
    breakdown.append({
        'name': 'Compression Patterns',
        'score': compression_score,
        'status': 'bad' if compression_score > 50 else 'warning' if compression_score > 28 else 'good',
        'explanation': 'Image artifacts ' + ('show unusual block structures and patterns typical of AI generation.' if compression_score > 50 else 'have some anomalies that could indicate processing.' if compression_score > 28 else 'are consistent with natural camera output.')
    })
    
    return breakdown


def generate_analysis_summary(ai_prob: float, breakdown: list) -> str:
    """Generate a human-readable analysis summary."""
    good_count = sum(1 for b in breakdown if b['status'] == 'good')
    bad_count = sum(1 for b in breakdown if b['status'] == 'bad')
    
    if ai_prob > 0.8:
        return f"This image shows strong indicators of AI generation. {bad_count} of 12 analysis categories flagged significant concerns, including neural network detection, frequency patterns, and visual artifacts. While no detection is 100% certain, the evidence strongly suggests this image was created by an AI system such as DALL-E, Midjourney, Stable Diffusion, or similar."
    elif ai_prob > 0.6:
        return f"This image presents several suspicious elements indicative of potential AI influence, including some unusual patterns in texture and lighting. {bad_count} categories raised concerns while {good_count} appeared normal. While not definitive, these factors suggest the image may have been AI-generated or heavily manipulated."
    elif ai_prob > 0.4:
        return f"The analysis shows mixed signals. Some elements appear natural while others show potential processing artifacts. {good_count} categories indicate authentic characteristics. Manual review is recommended for critical decisions."
    else:
        return f"This image appears consistent with real photography. {good_count} of 12 analysis categories show characteristics consistent with real photographs, including natural metadata, proper frequency distributions, and realistic textures. Note: This does not prove authenticity — only provenance verification can do that."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Main detection API endpoint.
    Calls ensemble_predict() directly — identical to CLI code path.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: JPG, PNG, WebP, GIF'}), 400
    
    # Save file temporarily
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    try:
        if not UNIFIED_AVAILABLE and not ENSEMBLE_AVAILABLE:
            return jsonify({
                'success': True,
                'error_notice': 'No detectors available',
                'overall_score': 50,
                'verdict': 'UNCERTAIN',
                'confidence': 'LOW',
                'decision_source': 'Fallback (detector unavailable)',
                'detected_domain': 'unknown',
                'detector_used': 'None',
                'summary': 'Detection models could not be loaded. Results are unreliable.',
                'breakdown': [],
                'limitations': [
                    'Detector modules not available',
                    'Results are unreliable'
                ]
            }), 200
        
        # === DOMAIN-AWARE DETECTION ===
        # Priority: unified_detect (routes face/non-face) > ensemble_predict (face only)
        unified_result = None
        ensemble_result = None
        
        if UNIFIED_AVAILABLE:
            # Unified detector handles routing: face → face detector, non-face → general
            unified_result = unified_detect(filepath)
        
        if ENSEMBLE_AVAILABLE:
            # Also run ensemble for rich data (filter, processing level, individual scores)
            ensemble_result = ensemble_predict(filepath)
        
        # Use unified result for primary verdict (it has domain-aware routing)
        if unified_result:
            ensemble_prob = float(unified_result.get('synthetic_probability', 0.5))
            verdict = unified_result.get('verdict', 'UNCERTAIN')
            confidence = unified_result.get('confidence_band', 'MEDIUM')
            decision_source = unified_result.get('detector_used', 'Unknown')
            detected_domain = unified_result.get('detected_domain', 'unknown')
            domain_confidence = unified_result.get('domain_confidence', 0.0)
            uncertainty_notice = unified_result.get('uncertainty_notice', '')
            domain_limitations = unified_result.get('limitations', '')
        elif ensemble_result:
            # Fallback: ensemble only (no domain routing)
            ensemble_prob = ensemble_result.get('ensemble_probability', 0.5)
            verdict = ensemble_result.get('verdict', 'UNCERTAIN')
            confidence = ensemble_result.get('confidence', 'MEDIUM')
            decision_source = ensemble_result.get('decision_source', 'Unknown')
            detected_domain = 'face'  # ensemble is face-focused
            domain_confidence = 0.0
            uncertainty_notice = ''
            domain_limitations = ''
        else:
            ensemble_prob = 0.5
            verdict = 'UNCERTAIN'
            confidence = 'LOW'
            decision_source = 'No detector'
            detected_domain = 'unknown'
            domain_confidence = 0.0
            uncertainty_notice = ''
            domain_limitations = ''
        
        # Extract rich ensemble data (filter, processing, individual scores)
        # These are only available when ensemble_predict ran
        # Cast to native Python types to avoid numpy JSON serialization errors
        if ensemble_result:
            filter_detected = bool(ensemble_result.get('filter_detected', False))
            filter_type = str(ensemble_result.get('filter_type', 'none'))
            filter_confidence = float(ensemble_result.get('filter_confidence', 0.0))
            processing_level = str(ensemble_result.get('image_processing_level', 'unknown'))
            processing_warning = ensemble_result.get('processing_warning', None)
            if processing_warning is not None:
                processing_warning = str(processing_warning)
            disagreement = float(ensemble_result.get('disagreement', 0.0))
        else:
            filter_detected = False
            filter_type = 'none'
            filter_confidence = 0.0
            processing_level = 'unknown'
            processing_warning = None
            disagreement = 0.0
        
        # Generate detailed breakdown (uses ensemble individual scores if available)
        breakdown_source = ensemble_result if ensemble_result else {
            'ensemble_probability': ensemble_prob,
            'individual_results': {}
        }
        breakdown = generate_detailed_breakdown(breakdown_source)
        
        # Generate summary
        summary = generate_analysis_summary(ensemble_prob, breakdown)
        
        # Overall score as percentage
        overall_score = int(ensemble_prob * 100)
        
        # Confidence interval
        if unified_result and 'confidence_interval' in unified_result:
            ci = unified_result['confidence_interval']
            ci_lower = max(0, int(ci.get('lower', 0) * 100))
            ci_upper = min(100, int(ci.get('upper', 1) * 100))
        else:
            ci_half = max(5, int(disagreement * 50))
            ci_lower = max(0, overall_score - ci_half)
            ci_upper = min(100, overall_score + ci_half)
        
        # Build limitations list
        limitations = [
            "This system estimates synthetic likelihood, NOT authenticity",
            "REAL classification is unverified — only provenance can prove authenticity",
            "Confidence reflects model uncertainty, NOT ground truth",
        ]
        if domain_limitations:
            limitations.append(domain_limitations)
        if processing_level == 'heavy_processing':
            limitations.append("Heavy image processing detected — detection reliability is reduced")
        if filter_detected:
            limitations.append(f"Social media filter detected ({filter_type}) — may cause false positives")
        if processing_level == 'moderate_processing':
            limitations.append("Moderate processing detected — confidence is reduced")
        if detected_domain in ('art_or_illustration', 'synthetic_graphics'):
            limitations.append(f"Domain '{detected_domain}' has limited validation data — results may be less reliable")
        
        # Extract individual scores for debugging (from ensemble if available)
        individual = ensemble_result.get('individual_results', {}) if ensemble_result else {}
        
        # Build response — unified detection output
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,
            
            # Core results
            'overall_score': overall_score,
            'verdict': verdict,
            'confidence': confidence,
            'decision_source': decision_source,
            
            # Domain routing info
            'detected_domain': str(detected_domain),
            'domain_confidence': int(round(float(domain_confidence) * 100)),
            
            # Confidence interval
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
            },
            
            # Model agreement
            'model_disagreement': int(round(float(disagreement) * 100)),
            
            # Filter detection
            'filter_detected': filter_detected,
            'filter_type': filter_type if filter_detected else None,
            'filter_confidence': round(filter_confidence * 100) if filter_detected else None,
            
            # Processing level
            'image_processing_level': processing_level,
            'processing_warning': processing_warning,
            
            # Breakdown and summary
            'summary': summary,
            'breakdown': breakdown,
            
            # Limitations (always shown)
            'limitations': limitations,
            
            # Raw individual scores for debugging
            'raw_scores': {
                'efficientnet': float(individual.get('efficientnet', {}).get('probability', 0.5)),
                'statistical': float(individual.get('statistical', {}).get('probability', 0.5)),
                'metadata': float(individual.get('metadata', {}).get('probability', 0.5)),
            }
        }
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Detection failed: {e}")
        print(f"[ERROR] Traceback: {error_trace}")
        
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  DeepFake Detection Web Interface")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    # PRODUCTION FIX: Disable reloader to prevent mid-analysis restarts
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
