# Limitations

*This page describes known limitations of the Synthetic Image Risk Estimator. Please read carefully before using the service.*

---

## ⚠️ Critical Understanding

This tool provides **probabilistic estimates**, not definitive answers.

All results are uncertain and may be incorrect. This page explains why, and how to interpret results responsibly.

---

## Training Data Limitations

### Limited Dataset

The model was trained on a **single dataset** of face images:
- StyleGAN-generated synthetic faces
- Real face photographs (source: publicly available datasets)

This means:
- The model may not recognize patterns from other AI generators (Stable Diffusion, DALL-E, Midjourney, etc.)
- Performance on non-face images is unknown and likely degraded
- New AI generation methods may not be detectable

### Unverified "Real" Images

⚠️ **Important**

Images labeled as "real" in the training data **were not verified as authentic**. They were assumed to be non-synthetic based on dataset labels, but:
- Some may have been AI-generated
- Some may have been heavily edited
- None were forensically verified

**The model has no ground truth for what "authentic" means.**

---

## Known Failure Modes

### Borderline Cases (30%–70% Probability)

Results in this range are **unreliable**:
- The model cannot distinguish confidently
- Interpretation is not meaningful
- These results should be treated as "no information"

### Image Processing Effects

The following may cause unexpected results:
- Heavy JPEG compression
- Resizing or upscaling
- Screenshots of images
- Social media re-encoding
- Color correction or filters
- Cropping or rotation

### Unusual Images

The model may perform poorly on:
- Non-face images
- Artwork or illustrations
- Low-resolution images
- Images with text overlays
- Collages or composites
- Scanned photographs

---

## False Positives

The model may incorrectly flag **non-AI images** as having high synthetic likelihood:
- Professionally retouched photographs
- Images with unusual lighting
- High-dynamic-range (HDR) images
- Images with artificial backgrounds
- Stock photography with heavy editing

**A high probability does not prove an image is AI-generated.**

---

## False Negatives

The model may fail to detect **AI-generated images**:
- From generators not represented in training
- That have been post-processed or edited
- That are low quality or heavily compressed
- From newer generation techniques

**A low probability does not prove an image is authentic.**

---

## Calibration Limitations

The model's probability estimates are calibrated based on **training distribution**:
- Calibration may not hold for images outside this distribution
- Probability values should be interpreted as relative, not absolute
- Confidence intervals reflect statistical uncertainty, not real-world accuracy

---

## What This Means for You

### Do NOT Use This Tool To:

❌ Make legal determinations  
❌ Accuse individuals of deception  
❌ Moderate content without human review  
❌ Make employment or admission decisions  
❌ Verify identity documents  
❌ Authenticate evidence  

### DO Use This Tool To:

✅ Get an initial probabilistic signal  
✅ Inform (not replace) further investigation  
✅ Understand limitations of AI detection  
✅ Learn about synthetic image characteristics  

---

## The Fundamental Problem

Detecting AI-generated images is an **inherently uncertain task**:
- There is no universal signature of AI generation
- Detection methods are always playing catch-up with generation methods
- Perfect detection is theoretically impossible
- Any tool claiming certainty is misleading you

We have built this tool to be honest about these limitations.

---

## Summary

| Limitation | Impact |
|------------|--------|
| Single training dataset | May miss other AI generators |
| Unverified "real" labels | No ground truth for authenticity |
| Borderline results | 30–70% range is uninformative |
| Image processing | Compression, resizing reduce reliability |
| False positives | Non-AI images may be flagged |
| False negatives | AI images may be missed |
| Calibration scope | Only valid for similar images |

---

*Please proceed with appropriate caution and always combine results with human judgment.*
