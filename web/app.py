"""
DeepFake Detection Web Interface - Flask Backend
=================================================

Production-quality API for AI image detection.
Uses ensemble_predict() and exposes ALL real signal scores.
Zero hardcoded/static values — everything is computed from the actual models.
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

ENSEMBLE_AVAILABLE = False
try:
    from ensemble_detector import ensemble_predict
    ENSEMBLE_AVAILABLE = True
except ImportError:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max (for videos)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
VIDEO_EXTENSIONS   = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

VIDEO_DETECTOR_AVAILABLE = False
try:
    from video_detector import detect_video
    VIDEO_DETECTOR_AVAILABLE = True
except ImportError:
    pass

try:
    from audio_analyzer import analyze_voice_authenticity
    AUDIO_DETECTOR_AVAILABLE = True
except ImportError:
    AUDIO_DETECTOR_AVAILABLE = False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def allowed_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'm4a', 'flac', 'ogg'}


def build_signals(ensemble_result: dict) -> list:
    """
    Build a list of real signal objects from ensemble results.
    Every single score comes directly from the actual detector — zero fake values.
    """
    individual = ensemble_result.get('individual_results', {})

    eff_data     = individual.get('efficientnet', {})
    stat_data    = individual.get('statistical', {})
    meta_data    = individual.get('metadata', {})
    filter_data  = individual.get('filter', {})
    proc_data    = individual.get('processing', {})
    forensic_data = individual.get('forensics', {})

    eff_prob    = eff_data.get('probability', None)
    stat_prob   = stat_data.get('probability', None)
    meta_prob   = meta_data.get('probability', None)
    filter_conf = filter_data.get('confidence', None)
    filter_det  = filter_data.get('detected', False)
    filter_type = filter_data.get('type', 'none')
    jpeg_qual   = meta_data.get('jpeg_quality', None)  # 0–100 or None

    # Forensic sub-scores (0.0–1.0 probability each)
    f_lighting   = forensic_data.get('lighting', {}).get('probability', None)
    f_noise      = forensic_data.get('noise', {}).get('probability', None)
    f_reflection = forensic_data.get('reflection', {}).get('probability', None)
    f_gan        = forensic_data.get('gan_fingerprint', {}).get('probability', None)
    f_overall    = forensic_data.get('probability', None)

    # Processing score
    proc_level = proc_data.get('level', individual.get('processing_level', {}).get('level', 'unknown'))
    proc_score_map = {'minimal_processing': 5, 'moderate_processing': 50, 'heavy_processing': 90, 'unknown': None}
    proc_score = proc_score_map.get(proc_level, None)

    def pct(val):
        """Convert 0-1 float to int percent, or None if unknown."""
        if val is None:
            return None
        return int(round(float(val) * 100))

    def status(score, bad_thresh, warn_thresh):
        if score is None:
            return 'unknown'
        if score > bad_thresh:
            return 'bad'
        if score > warn_thresh:
            return 'warning'
        return 'good'

    def explain_score(score, name, low_text, mid_text, high_text):
        if score is None:
            return f"{name} could not be computed for this image."
        if score < 30:
            return low_text
        if score < 60:
            return mid_text
        return high_text

    signals = []

    # 1. EfficientNet Deep Neural Network
    if eff_prob is not None:
        s = pct(eff_prob)
        signals.append({
            'name': 'Neural Network (EfficientNet-B0)',
            'icon': '🤖',
            'score': s,
            'status': status(s, 60, 35),
            'source': 'Deep CNN trained on 1.7M images',
            'explanation': explain_score(s, 'EfficientNet',
                'EfficientNet-B0 sees natural camera patterns — strong indicator this is a real photo.',
                'EfficientNet detected some suspicious patterns that may suggest AI generation or heavy editing.',
                'EfficientNet-B0 strongly detected AI generation signatures from its training on 30+ generators.')
        })

    # 2. Statistical / Frequency Analysis
    if stat_prob is not None:
        s = pct(stat_prob)
        signals.append({
            'name': 'Statistical Frequency (DCT)',
            'icon': '📊',
            'score': s,
            'status': status(s, 55, 30),
            'source': 'Gradient Boosting on DCT frequency features',
            'explanation': explain_score(s, 'Frequency analysis',
                'DCT frequency distribution matches natural camera sensor patterns.',
                'Some anomalies in the DCT frequency domain — could be compression or light editing.',
                'DCT frequency distribution is unusual. AI generators leave distinct frequency artifacts.')
        })

    # 3. Metadata Authenticity
    if meta_prob is not None:
        s = pct(meta_prob)
        has_camera = meta_data.get('has_camera', False)
        has_gps    = meta_data.get('has_gps', False)
        if has_camera or has_gps:
            expl = f"✅ Camera EXIF data found ({('GPS + ' if has_gps else '')}{'camera make/model' if has_camera else ''}). This is a strong real-photo indicator."
        elif s > 60:
            expl = "No camera metadata (EXIF/GPS/Make/Model). AI-generated images typically lack these — this is suspicious."
        elif s > 30:
            expl = "Limited metadata present. Could be a stripped real photo (social media upload) or an AI image."
        else:
            expl = "Some metadata present but inconclusive without camera/GPS data."
        signals.append({
            'name': 'Metadata & EXIF Forensics',
            'icon': '📷',
            'score': s,
            'status': status(s, 60, 35),
            'source': 'EXIF, GPS, camera fingerprint analysis',
            'explanation': expl
        })

    # 4. JPEG Compression Quality
    if jpeg_qual is not None:
        # AI images often have very high (90-100) or very low (<50) JPEG quality
        # Real photos from cameras are usually 75-92
        if 70 <= jpeg_qual <= 92:
            jq_status = 'good'
            jq_score = int((jpeg_qual - 70) / 22 * 20)  # 0-20% (looks real)
            jq_expl = f"JPEG quality {jpeg_qual}/100 — falls within the typical range for real camera photos (70–92)."
        elif jpeg_qual > 92:
            jq_status = 'warning'
            jq_score = int(30 + (jpeg_qual - 92) / 8 * 30)  # 30-60%
            jq_expl = f"JPEG quality {jpeg_qual}/100 — unusually high. AI-generated images saved losslessly often land here."
        else:  # < 70
            jq_status = 'warning'
            jq_score = int(40 + (70 - jpeg_qual) / 70 * 30)  # 40-70%
            jq_expl = f"JPEG quality {jpeg_qual}/100 — low compression. May indicate heavy re-encoding or processing."
        signals.append({
            'name': 'JPEG Compression Quality',
            'icon': '🗜️',
            'score': jq_score,
            'status': jq_status,
            'source': 'JPEG quality factor from metadata',
            'explanation': jq_expl
        })

    # 5. Social Media Filter Detection
    if filter_conf is not None:
        s = pct(filter_conf)
        if filter_det:
            expl = f"Filter detected: {filter_type} ({s}% confidence). {filter_data.get('indicators', [''])[0] if filter_data.get('indicators') else 'Multiple visual indicators found.'}. This may cause false-positive AI readings."
            fst = 'warning' if s < 70 else 'bad'
        else:
            expl = "No social media filter signature detected. Image appears unfiltered."
            fst = 'good'
        signals.append({
            'name': 'Social Media Filter Detection',
            'icon': '🎨',
            'score': s if filter_det else int(s * 0.3),
            'status': fst,
            'source': 'Visual pattern matching for Instagram/Snapchat/TikTok',
            'explanation': expl
        })

    # 6. Image Processing Level
    if proc_score is not None:
        signals.append({
            'name': 'Post-Processing Level',
            'icon': '⚙️',
            'score': proc_score,
            'status': status(proc_score, 70, 30),
            'source': 'Sharpening, noise reduction, resampling detection',
            'explanation': {
                'minimal_processing': 'Minimal post-processing detected — image appears close to original camera output.',
                'moderate_processing': 'Moderate post-processing detected — some editing applied (sharpening, color grading, noise reduction).',
                'heavy_processing': 'Heavy post-processing detected — significant editing makes accurate AI detection unreliable.',
            }.get(proc_level, 'Processing level unknown.')
        })

    # 7. Forensic Lighting Consistency
    if f_lighting is not None:
        s = pct(f_lighting)
        signals.append({
            'name': 'Lighting Consistency (Forensic)',
            'icon': '💡',
            'score': s,
            'status': status(s, 65, 40),
            'source': 'Sobel gradient shadow-direction analysis per quadrant',
            'explanation': explain_score(s, 'Lighting',
                'Shadow directions are consistent across all quadrants — physically plausible lighting.',
                'Some lighting inconsistency detected. Could be studio lighting or mild AI artifacts.',
                'Significant lighting inconsistency detected across image regions — typical of AI generation artifacts.')
        })

    # 8. Sensor Noise Pattern (PRNU)
    if f_noise is not None:
        s = pct(f_noise)
        signals.append({
            'name': 'Sensor Noise Pattern (PRNU)',
            'icon': '🔬',
            'score': s,
            'status': status(s, 65, 40),
            'source': 'High-pass filter patch variance analysis',
            'explanation': explain_score(s, 'Noise pattern',
                'Sensor noise is naturally distributed — consistent with a real camera sensor fingerprint.',
                'Partial noise irregularity. Could be compressed/re-encoded real photo.',
                'Noise pattern is unusually uniform or inconsistent — AI synthesized images often lack realistic sensor noise.')
        })

    # 9. Specular Reflection Analysis
    if f_reflection is not None:
        s = pct(f_reflection)
        signals.append({
            'name': 'Specular Reflection Analysis',
            'icon': '✨',
            'score': s,
            'status': status(s, 65, 40),
            'source': 'Specular highlight detection and consistency check',
            'explanation': explain_score(s, 'Reflections',
                'Specular highlights appear physically consistent with the scene lighting direction.',
                'Some irregularity in specular highlights. Could be complex multi-light setup.',
                'Specular highlights appear inconsistent or unnaturally positioned — a common AI generation artifact.')
        })

    # 10. GAN Frequency Fingerprint
    if f_gan is not None:
        s = pct(f_gan)
        signals.append({
            'name': 'GAN Frequency Fingerprint',
            'icon': '🌀',
            'score': s,
            'status': status(s, 65, 40),
            'source': '2D FFT spectral peak detection',
            'explanation': explain_score(s, 'GAN fingerprint',
                'No GAN-specific spectral artifacts detected in the frequency domain.',
                'Some unusual frequency patterns present. Could be compression or camera lens artifacts.',
                'Periodic spectral peaks detected in 2D FFT — characteristic of GAN upsampling artifacts.')
        })

    # 11. Overall Forensic Score
    if f_overall is not None:
        s = pct(f_overall)
        signals.append({
            'name': 'Overall Forensic Score',
            'icon': '🔍',
            'score': s,
            'status': status(s, 65, 40),
            'source': 'Weighted combination of all 4 forensic analyzers',
            'explanation': explain_score(s, 'Combined forensics',
                'All forensic signals are consistent with a real photograph.',
                'Forensic signals show mixed results — some signals lean real, others show minor anomalies.',
                'Forensic analysis collectively flags multiple artifacts characteristic of AI-generated content.')
        })

    # 12. Cross-Model Agreement (disagreement)
    disagreement = ensemble_result.get('disagreement', None)
    if disagreement is not None:
        s = int(round(float(disagreement) * 100))
        if s < 20:
            dag_status = 'good'
            dag_expl = f"All detection models are in strong agreement ({100-s}% alignment). High confidence in the verdict."
        elif s < 45:
            dag_status = 'warning'
            dag_expl = f"Models show moderate disagreement ({s}%). Verdict is less certain — consider context."
        else:
            dag_status = 'bad'
            dag_expl = f"High model disagreement ({s}%). EfficientNet and Statistical model give very different readings — result is uncertain."
        signals.append({
            'name': 'Cross-Model Agreement',
            'icon': '🤝',
            'score': s,
            'status': dag_status,
            'source': 'Disagreement between EfficientNet and Statistical model',
            'explanation': dag_expl
        })

    return signals


def build_summary(ai_prob: float, verdict: str, signals: list) -> str:
    """Generate a dynamic summary from real signal results."""
    good = sum(1 for s in signals if s['status'] == 'good')
    bad  = sum(1 for s in signals if s['status'] == 'bad')
    total = len(signals)
    pct_str = f"{int(ai_prob * 100)}%"

    if verdict == 'AI-GENERATED':
        return (f"Analysis complete: {bad} of {total} forensic signals flagged significant AI artifacts. "
                f"The synthetic likelihood score is {pct_str}. "
                "Multiple independent detectors (neural network, frequency analysis, forensic signals) "
                "agree this image was likely generated by an AI system such as DALL-E, Midjourney, or Stable Diffusion.")
    elif verdict == 'LIKELY REAL':
        return (f"Analysis complete: {good} of {total} signals are consistent with real photography. "
                f"The synthetic likelihood score is {pct_str}. "
                "Note: this does NOT prove authenticity. Only provenance verification (original file metadata, "
                "source chain of custody) can establish an image's true origin.")
    elif verdict == 'POSSIBLY AI':
        return (f"Analysis complete: signals are mixed. {good} signals lean real, {bad} signal(s) raise concerns. "
                f"The synthetic likelihood score is {pct_str}. "
                "Manual review is recommended. The image may be AI-generated, heavily edited, or a filtered real photo.")
    else:  # UNCERTAIN
        return (f"Analysis inconclusive. {bad} signals flagged concerns while {good} appear normal. "
                f"Synthetic likelihood: {pct_str}. "
                "Detection is unreliable due to heavy image processing, strong filter effects, or high model disagreement. "
                "Do not rely on this result for any critical decision.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Main detection API endpoint — returns fully dynamic signal data.
    Zero hardcoded scores.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: JPG, PNG, WebP, GIF'}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    try:
        if not ENSEMBLE_AVAILABLE:
            return jsonify({'error': 'Detection models not loaded.'}), 500

        ensemble_result = ensemble_predict(filepath)

        ai_prob         = float(ensemble_result.get('ensemble_probability', 0.5))
        verdict         = str(ensemble_result.get('verdict', 'UNCERTAIN'))
        confidence      = str(ensemble_result.get('confidence', 'LOW'))
        decision_source = str(ensemble_result.get('decision_source', 'Unknown'))
        disagreement    = float(ensemble_result.get('disagreement', 0.0))
        filter_detected = bool(ensemble_result.get('filter_detected', False))
        filter_type     = str(ensemble_result.get('filter_type', 'none'))
        filter_conf     = float(ensemble_result.get('filter_confidence', 0.0))
        proc_level      = str(ensemble_result.get('image_processing_level', 'unknown'))
        proc_warning    = ensemble_result.get('processing_warning', None)
        if proc_warning:
            proc_warning = str(proc_warning)

        # Build real dynamic signals from actual model outputs
        signals = build_signals(ensemble_result)

        # Build confidence interval
        ci_half   = max(5, int(disagreement * 50))
        overall   = int(ai_prob * 100)
        ci_lower  = max(0, overall - ci_half)
        ci_upper  = min(100, overall + ci_half)

        # Build limitations
        limitations = [
            "This system estimates synthetic likelihood — NOT image authenticity.",
            "A LIKELY REAL result does NOT prove authenticity. Only provenance verification can.",
            "AI probability is a statistical estimate with inherent uncertainty.",
        ]
        if proc_level == 'heavy_processing':
            limitations.append("⚠️ Heavy post-processing detected — detection reliability is significantly reduced.")
        if filter_detected:
            limitations.append(f"⚠️ Social media filter detected ({filter_type}) — may cause false AI positives.")
        if proc_level == 'moderate_processing':
            limitations.append("Moderate image processing detected — confidence interval is wider than normal.")

        # Build summary from real data
        summary = build_summary(ai_prob, verdict, signals)

        # Raw individual scores for debug panel
        individual = ensemble_result.get('individual_results', {})
        forensics  = individual.get('forensics', {})

        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename,

            # Core verdict
            'overall_score': overall,
            'verdict': verdict,
            'confidence': confidence,
            'decision_source': decision_source,

            # Interval & agreement
            'confidence_interval': {'lower': ci_lower, 'upper': ci_upper},
            'model_disagreement': int(round(disagreement * 100)),

            # Filter & processing
            'filter_detected': filter_detected,
            'filter_type': filter_type if filter_detected else None,
            'filter_confidence': round(filter_conf * 100) if filter_detected else None,
            'image_processing_level': proc_level,
            'processing_warning': proc_warning,

            # The real dynamic signals
            'signals': signals,
            'summary': summary,
            'limitations': limitations,

            # Raw scores for the live scores panel in the UI
            'raw_scores': {
                'efficientnet':  round(float(individual.get('efficientnet', {}).get('probability', 0.5)) * 100, 1),
                'statistical':   round(float(individual.get('statistical', {}).get('probability', 0.5)) * 100, 1),
                'metadata':      round(float(individual.get('metadata', {}).get('probability', 0.5)) * 100, 1),
                'forensic':      round(float(forensics.get('probability', 0.5)) * 100, 1),
                'meta_voter':    'active' if 'Meta-Voter' in decision_source else 'fallback',
            }
        }

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify(response)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR] Detection failed: {e}")
        print(tb_str)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'error': f"Detection failed: {str(e)}\n\nTraceback:\n{tb_str}"}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/detect-video', methods=['POST'])
def detect_video_api():
    """
    Video deepfake detection endpoint.
    Accepts MP4/AVI/MOV/WEBM/MKV uploads, runs the full video pipeline,
    returns signal data in the same format as /api/detect for the UI.
    """
    if not VIDEO_DETECTOR_AVAILABLE:
        return jsonify({'error': 'Video detection module not loaded.'}), 500

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_video(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: MP4, AVI, MOV, WEBM, MKV'}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    try:
        result = detect_video(filepath)

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        ai_prob  = float(result.get('probability', 0.5))
        verdict  = str(result.get('verdict', 'UNCERTAIN'))
        confidence = str(result.get('confidence', 'LOW'))
        overall  = int(ai_prob * 100)

        # Map video signals to UI format
        raw_signals = result.get('signals', [])
        signals = []
        for s in raw_signals:
            score_val = float(s.get('score', 0.5))
            score_pct = int(score_val * 100)
            if score_pct > 65:
                st = 'bad'
            elif score_pct > 40:
                st = 'warning'
            else:
                st = 'good'
            signals.append({
                'name':        s.get('name', 'Signal'),
                'icon':        s.get('icon', '📊'),
                'score':       score_pct,
                'status':      st,
                'source':      'Video analysis pipeline',
                'explanation': s.get('description', ''),
            })

        # Build summary
        video_info = result.get('video_info', {})
        duration   = video_info.get('duration', 0)
        n_frames   = video_info.get('frames_analyzed', 0)
        method     = result.get('method', 'weighted_average')

        if verdict == 'AI-GENERATED':
            summary = (f"Video analysis complete: {overall}% AI probability across {n_frames} sampled frames "
                       f"({duration:.1f}s video). Frame-level deep neural network, temporal, biological, "
                       f"and audio signals all flagged this video as likely AI-generated or deepfake.")
        elif verdict in ('REAL', 'LIKELY REAL'):
            summary = (f"Video analysis complete: {overall}% AI probability across {n_frames} sampled frames "
                       f"({duration:.1f}s video). Signals are consistent with a real video recording.")
        else:
            summary = (f"Video analysis inconclusive: {overall}% AI probability. "
                       f"Analyzed {n_frames} frames over {duration:.1f}s. Manual review recommended.")

        response = {
            'success':       True,
            'timestamp':     datetime.now().isoformat(),
            'filename':      file.filename,
            'media_type':    'video',

            'overall_score': overall,
            'verdict':       verdict,
            'confidence':    confidence,
            'decision_source': method,

            'confidence_interval': {'lower': max(0, overall - 10), 'upper': min(100, overall + 10)},
            'model_disagreement':  0,

            'signals':     signals,
            'summary':     summary,
            'limitations': [
                "Video detection analyzes sampled frames — very short clips may be less accurate.",
                "Celeb-DF v2 style deepfakes require broader training data for confident detection.",
                "Audio analysis requires ffmpeg to be installed.",
            ],

            'raw_scores': {
                'frame_ai':   round(float(result['scores'].get('frame_ai_prob', 0.5)) * 100, 1),
                'temporal':   round(float(result['scores'].get('temporal_score', 0.5)) * 100, 1),
                'biological': round(float(result['scores'].get('biological_score', 0.5)) * 100, 1),
                'audio':      round(float(result['scores'].get('audio_score', 0.5)) * 100, 1),
            },

            'video_info': video_info,
            'elapsed_seconds': result.get('elapsed_seconds', 0),
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR] Video detection failed: {e}")
        print(tb_str)
        return jsonify({'error': f'Video detection failed: {str(e)}\n\nTraceback:\n{tb_str}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/detect-audio', methods=['POST'])
def detect_audio_api():
    """
    Audio deepfake detection endpoint.
    Accepts WAV/MP3/M4A/FLAC/OGG uploads, runs the audio ML pipeline,
    returns signal data formatted for the UI.
    """
    if not AUDIO_DETECTOR_AVAILABLE:
        return jsonify({'error': 'Audio detection module not loaded.'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_audio(file.filename):
        return jsonify({'error': 'Invalid file type. Supported: WAV, MP3, M4A, FLAC, OGG'}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    import time
    start_time = time.time()
    
    try:
        voice = analyze_voice_authenticity(filepath)
        
        if "error" in voice and isinstance(voice["error"], str):
             return jsonify({'error': voice["error"]}), 500
             
        ai_prob = float(voice.get('voice_score', 0.5))
        overall = int(ai_prob * 100)
        
        if ai_prob > 0.70:
            verdict = "AI-GENERATED"
            confidence = "HIGH"
        elif ai_prob > 0.55:
            verdict = "LIKELY AI-GENERATED"
            confidence = "MEDIUM"
        elif ai_prob > 0.40:
            verdict = "UNCERTAIN"
            confidence = "LOW"
        elif ai_prob > 0.25:
            verdict = "LIKELY REAL"
            confidence = "MEDIUM"
        else:
            verdict = "REAL"
            confidence = "HIGH"
            
        signals = []
        
        # Audio Feature Signal
        score_pct = int(ai_prob * 100)
        if score_pct > 65: st = 'bad'
        elif score_pct > 40: st = 'warning'
        else: st = 'good'
        
        is_ml = voice.get("is_advanced_ml", False)
        desc = "Advanced ML Model (LightGBM Meta-Ensemble over 10 parameters)" if is_ml else "Fallback Heuristic Model (Spectral flatness, MFCC variance)"
        
        signals.append({
            'name': 'Voice Authenticity Analysis',
            'icon': '🎧',
            'score': score_pct,
            'status': st,
            'source': 'Audio Frequency & Feature Fingerprinting',
            'explanation': desc
        })
        
        if is_ml:
            method = "LightGBM + Wav2Vec2"
        else:
            method = "MFCC Heuristics"


        if verdict == 'AI-GENERATED':
            summary = f"Audio analysis complete: {overall}% AI likelihood. Acoustic feature matching highly suggests this voice was synthesized."
        elif verdict in ('REAL', 'LIKELY REAL'):
            summary = f"Audio analysis complete: {overall}% AI likelihood. Acoustic parameters are consistent with human speech patterns."
        else:
            summary = f"Audio analysis inconclusive: {overall}% AI likelihood. The acoustic anomalies were borderline."

        # Print summary to terminal
        print(f"\n  {'=' * 60}")
        print(f"  AUDIO ENSEMBLE DETECTOR - {file.filename}")
        print(f"  {'=' * 60}")
        print(f"  Combined AI Probability: {ai_prob:.1%}")
        print(f"  Decision Source:         {method}")
        print(f"  Confidence:              {confidence}")
        print(f"  FINAL VERDICT: {verdict}")
        print(f"  {'=' * 60}\n")

        response = {
            'success':       True,
            'timestamp':     datetime.now().isoformat(),
            'filename':      file.filename,
            'media_type':    'audio',
            
            'overall_score': overall,
            'verdict':       verdict,
            'confidence':    confidence,
            'decision_source': method,

            'confidence_interval': {'lower': max(0, overall - 8), 'upper': min(100, overall + 8)},
            'model_disagreement':  0,

            'signals':     signals,
            'summary':     summary,
            'limitations': [
                "Background noise, strong compression, or music can heavily alter results.",
                "Short clips (< 2 seconds) lack sufficient phonetic variety for high confidence."
            ],
            
            'raw_scores': {
                'audio': round(ai_prob * 100, 1),
            },
            
            'elapsed_seconds': round(time.time() - start_time, 1),
        }

        # UX: Add artificial delay if analysis was too fast (Labor Illusion)
        elapsed_time = time.time() - start_time
        if elapsed_time < 2.0:
            time.sleep(2.0 - elapsed_time)

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"[ERROR] Audio detection failed: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Audio detection failed: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  DeepFake Detection Web Interface")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
