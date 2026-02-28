# 🔍 DeepFake Detection System

An AI-powered image authentication system that detects AI-generated images using an ensemble of deep learning models, frequency analysis, forensic signal analysis, and cross-model voting.

---

## ✨ Features

- 🤖 **EfficientNet-B0** — Retrained on **1.7M+ images** from 30+ AI generators (StyleGAN, Stable Diffusion, DALL-E, Midjourney, ChatGPT, Gemini, etc.)
- 📊 **Statistical Frequency Analysis** — Gradient Boosting on DCT frequency features
- 📷 **Metadata Forensics** — EXIF, GPS, camera fingerprint detection
- 🎨 **Social Media Filter Detection** — Detects Instagram/Snapchat/TikTok filters to prevent false positives
- 🔬 **Forensic Signal Analysis** — Lighting consistency, sensor noise patterns (PRNU), reflection analysis, and GAN frequency fingerprints
- 🧠 **Cross-Model Meta-Voter** — GradientBoosting trained on 110K images learns optimal signal weighting (replaces hand-tuned rules)
- 🌐 **Web Interface** — Upload images via browser for instant analysis
- ⚡ **GPU Accelerated** — CUDA support for fast inference

---

## 🏗️ Architecture

```
ensemble_detector.py          ← Main orchestrator (7-step pipeline)
├── [1/7] EfficientNet-B0     ← Deep CNN (primary detector, 80.57% accuracy)
├── [2/7] Statistical Model   ← Gradient Boosting on frequency features
├── [3/7] Metadata Analyzer   ← EXIF/GPS/JPEG quality forensics
├── [4/7] Filter Detector     ← Social media filter detection
├── [5/7] Processing Level    ← Heavy post-processing detection
├── [6/7] Forensic Signals    ← Lighting, noise, reflections, GAN fingerprints
└── [7/7] Cross-Model Voter   ← Trained meta-voter (GradientBoosting)

forensic_signals.py           ← 4 physics-based forensic analyzers
├── LightingAnalyzer          ← Shadow direction consistency per quadrant
├── NoisePatternAnalyzer      ← Sensor noise (PRNU) uniformity across patches
├── ReflectionAnalyzer        ← Specular highlight consistency
└── GANFingerprintAnalyzer    ← 2D FFT spectral peak detection

meta_voter.py                 ← Cross-model voting meta-learner
├── Trains on 110K labeled images
├── Compares GradientBoosting vs RandomForest vs LogisticRegression
├── Checkpoint + resume support for long training runs
└── Falls back to hand-tuned logic if not trained

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

### Run CLI
```bash
python ensemble_detector.py path/to/image.jpg
```

### Run Forensic Analysis Only
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


