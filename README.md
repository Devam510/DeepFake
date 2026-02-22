# 🔍 DeepFake Detection System

An AI-powered image authentication system that detects AI-generated images using an ensemble of deep learning models, frequency analysis, and metadata forensics.

---

## ✨ Features

- 🤖 **EfficientNet-B0** — Retrained on **1.7M+ images** from 30+ AI generators (StyleGAN, Stable Diffusion, DALL-E, Midjourney, ChatGPT, Gemini, etc.)
- 📊 **Statistical Frequency Analysis** — Gradient Boosting on DCT frequency features
- 📷 **Metadata Forensics** — EXIF, GPS, camera fingerprint detection
- 🎨 **Social Media Filter Detection** — Detects Instagram/Snapchat/TikTok filters to prevent false positives
- 🧠 **Smart Ensemble Voting** — Priority-based decision logic combining all signals
- 🌐 **Web Interface** — Upload images via browser for instant analysis
- ⚡ **GPU Accelerated** — CUDA support for fast inference

---

## 🏗️ Architecture

```
ensemble_detector.py          ← Main orchestrator
├── EfficientNet-B0           ← Deep CNN (primary detector, 80.57% accuracy)
├── Statistical Baseline      ← Gradient Boosting on frequency features
├── Metadata Analyzer         ← EXIF/GPS forensics
├── Filter Detector           ← Social media filter detection
└── Processing Level          ← Heavy post-processing detection

web/app.py                    ← Flask web server
src/
├── extraction/               ← Feature extractors (filter, metadata, frequency)
└── modeling/                 ← Statistical model training
datasets/                     ← 1.7M training images (local, not pushed)
models/research/              ← Trained model weights
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision pillow numpy flask scikit-learn tqdm
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

---

## 📊 Model Performance

| Model | Training Data | Accuracy |
|-------|--------------|----------|
| EfficientNet-B0 (v2) | 1.7M images, 30+ generators | **80.57%** |
| Statistical Baseline | Frequency features | ~60-70% |
| Ensemble (combined) | All signals | **Higher precision** |

---

## 🎯 Detection Capabilities

| Image Type | Result |
|---|---|
| Real camera photo (with EXIF) | ✅ LIKELY REAL |
| Real photo with Instagram filter | ✅ LIKELY REAL |
| ChatGPT / DALL-E generated | ✅ AI-GENERATED |
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
```

---

## 📁 Project Structure

```
DeepFake/
├── ensemble_detector.py      # Main ensemble predictor
├── train_efficientnet.py     # Model training pipeline
├── download_datasets.py      # Dataset downloader (Kaggle)
├── check_data.py             # Dataset verification
├── web/
│   ├── app.py               # Flask web server
│   └── templates/           # HTML templates
├── src/
│   └── extraction/          # Feature extractors
├── models/
│   └── research/            # Trained model weights
└── datasets/                # Training data (local only)
```

---

## 🔧 Technical Details

- **Backbone**: EfficientNet-B0 with custom classifier (Dropout → Linear 512 → ReLU → Linear 2)
- **Training**: Mixed precision (float16), AdamW optimizer, cosine LR schedule
- **Augmentation**: Filter simulation (warm/cool/vintage/blur), JPEG compression, color jitter
- **Temperature Scaling**: Calibrated probability outputs (T=1.157)
- **Inference**: GPU accelerated with cached model loading

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
