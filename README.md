# 🔍 DeepFake Detection System

A comprehensive multi-modal AI-powered authentication system that detects AI-generated **Images, Videos, and Audio** using an ensemble of deep learning models, frequency analysis, temporal/biological signals, and cross-model voting.

---

## ✨ Features

- 🖼️ **Image Deepfake Detection** — Retrained EfficientNet-B0 (1.7M+ images), Statistical Frequency Analysis, Metadata Forensics, and Social Media Filter Detection.
- 🎬 **Video Deepfake Detection** — Frame-by-frame analysis combined with Temporal Signals (SSIM, jitter) and Biological Signals (heartbeat, blinks, skin, micro-expressions).
- 🎙️ **Audio Deepfake Detection** — Voice Authenticity (LightGBM on Wav2Vec2/MFCC spectral features) + Lip-Audio Synchronization detection (MediaPipe tracking).
- 🔬 **Forensic Signal Analysis** — Lighting consistency, sensor noise patterns (PRNU), reflection analysis, and GAN frequency fingerprints.
- 🧠 **Cross-Model Meta-Voters** — Trained gradient boosting meta-voters for both image and video modalities to learn optimal signal weighting.
- 🌐 **Web Interface** — Upload images, videos, or audio via browser for instant analysis.
- ⚡ **GPU Accelerated** — CUDA support for fast inference across all models.

---

## 🏗️ Architecture

```
ensemble_detector.py          ← Image orchestrator (7-step pipeline)
video_detector.py             ← Video orchestrator (Temporal, Biological, Audio, Frames)
audio_analyzer.py             ← Audio orchestrator (Voice Authenticity & Lip Sync)

├── [Image] EfficientNet-B0, Statistical, Metadata, Filter, Forensics
├── [Video] Temporal Signals (jitter, SSIM), Biological Signals (heartbeat, blink rate)
└── [Audio] Voice Authenticity (Wav2Vec2, MFCC), Lip-Audio Sync (MediaPipe)

forensic_signals.py           ← 4 physics-based forensic analyzers for images
├── LightingAnalyzer          ← Shadow direction consistency per quadrant
├── NoisePatternAnalyzer      ← Sensor noise (PRNU) uniformity across patches
└── GANFingerprintAnalyzer    ← 2D FFT spectral peak detection

meta_voter.py                 ← Image cross-model voting meta-learner
video_meta_voter.py           ← Video cross-model voting meta-learner

web/app.py                    ← Flask web server
src/
├── extraction/               ← Feature extractors (filter, metadata, frequency)
└── modeling/                 ← Statistical model training
datasets/                     ← 1.7M training images (local, not pushed)
models/trained/               ← Trained model weights
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision pillow numpy flask scikit-learn scipy tqdm
```

### Run Web Interface
```bash
python web/app.py
# Open http://localhost:5000
```

### Run CLI Prediction (Image)
```bash
python ensemble_detector.py path/to/image.jpg
```

### Run CLI Prediction (Video & Audio)
```bash
python video_detector.py path/to/video.mp4
```

### Run Forensic Analysis Only (Image)
```bash
python forensic_signals.py path/to/image.jpg
```

### Train Cross-Model Meta-Voter
```bash
python meta_voter.py --train
# Trains on ~110K images, takes ~30-60 min
# Saves to models/trained/meta_voter.pkl
```

---

## 📊 Model Performance

| Model | Training Data | Accuracy | AUC |
|-------|--------------|----------|-----|
| EfficientNet-B0 (v2) | 1.7M images, 30+ generators | **80.57%** | — |
| Statistical Baseline | Frequency features | ~60-70% | — |
| Cross-Model Meta-Voter | 110K images, 10 features | **68.7%** | **0.7693** |
| **Ensemble (all combined)** | All signals | **Highest precision** | — |

### Feature Importance (Meta-Voter)

| Feature | Importance |
|---------|-----------|
| EfficientNet score | 32.0% |
| Statistical score | 27.9% |
| JPEG quality | 16.7% |
| Forensic noise | 8.8% |
| Metadata analysis | 4.8% |
| Model disagreement | 4.7% |
| Forensic reflections | 4.2% |
| Forensic lighting | 0.9% |

---

## 🎯 Detection Capabilities

| Image Type | Result |
|---|---|
| Real camera photo (with EXIF) | ✅ LIKELY REAL (2.5%) |
| Real photo via WhatsApp (no EXIF) | ✅ LIKELY REAL (15.4%) |
| Real photo with Instagram filter | ✅ LIKELY REAL |
| ChatGPT / DALL-E generated | ✅ AI-GENERATED (74.3%) |
| Midjourney / Stable Diffusion | ✅ AI-GENERATED |
| Gemini generated | ✅ AI-GENERATED |
| Ambiguous (no clear signal) | ⚠️ UNCERTAIN |

---

## 🗂️ Datasets Used for Training

| Dataset | Images | Source |
|---------|--------|--------|
| ArtiFact | ~1.1M | Kaggle |
| 140K Real & Fake Faces | 140K | Kaggle |
| CiFAKE | 120K | Kaggle |
| Real vs AI Faces | 242K | Kaggle |
| DALL-E Recognition | 21K | Kaggle |
| Deepfake & Real | 190K | Kaggle |
| **Total** | **~1.7M** | |

---

## ⚙️ Training Your Own Model

```bash
# Download datasets first
python download_datasets.py

# Train EfficientNet-B0 on all data (requires GPU)
python train_efficientnet.py --epochs 10 --batch-size 16 --max-per-class -1 --from-scratch

# Resume if interrupted
python train_efficientnet.py --epochs 10 --batch-size 16 --max-per-class -1 --resume

# Train the cross-model meta-voter (after EfficientNet is trained)
python meta_voter.py --train
```

---

## 📁 Project Structure

```
DeepFake/
├── ensemble_detector.py      # Main 7-step ensemble predictor
├── forensic_signals.py       # 4 forensic analyzers (lighting, noise, reflection, GAN)
├── meta_voter.py             # Cross-model meta-voter training & inference
├── train_efficientnet.py     # EfficientNet training pipeline
├── download_datasets.py      # Dataset downloader (Kaggle)
├── check_data.py             # Dataset verification
├── web/
│   ├── app.py               # Flask web server
│   └── templates/           # HTML templates
├── src/
│   └── extraction/          # Feature extractors
├── models/
│   └── trained/             # Trained model weights (EfficientNet, meta-voter)
└── datasets/                # Training data (local only)
```

---

## 🔧 Technical Details

- **Backbone**: EfficientNet-B0 with custom classifier (Dropout → Linear 512 → ReLU → Linear 2)
- **Training**: Mixed precision (float16), AdamW optimizer, cosine LR schedule
- **Augmentation**: Filter simulation (warm/cool/vintage/blur), JPEG compression, color jitter
- **Temperature Scaling**: Calibrated probability outputs (T=1.157)
- **Forensic Signals**: Sobel gradients, Gaussian high-pass filter, 2D FFT, specular highlight detection
- **Meta-Voter**: GradientBoosting (200 trees, depth 4) trained on 10 signal features from 110K images
- **Inference**: GPU accelerated with cached model loading


