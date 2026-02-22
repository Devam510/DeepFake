# DeepFake Detection System - Threat Model (Layer 0)

> [!IMPORTANT]
> This document defines the adversarial landscape. The system is designed assuming these capabilities are present in the attacker's toolkit.

## 1. Attacker Profile
**Goal:** Bypass detection to disseminate synthetic media as authentic.
**Resources:** High (State-level actors or well-funded organizations) to Low (Script kiddies using public tools).
**Knowledge:** Gray Box (Attacker knows the general architecture: Ensembles + Signal Extraction, but not specific model weights or rotation schedules).

## 2. Attacker Capabilities

### A. Generator Access (Source Generation)
Attackers have access to and can fine-tune:
*   **Text-to-Image:** Stable Diffusion XL, Midjourney v6+, DALL-E 3.
*   **Text-to-Video:** Sora, Runway Gen-2/3, Pika Labs.
*   **Audio:** ElevenLabs, Voice cloning OSS models.
*   **Deepfakes (Face Swaps):** Roop, DeepFaceLab (high quality).

### B. Post-Processing (Sanitization)
Attackers will attempt to scrub "generative artifacts" using:
*   **Re-compression:** Saving as heavily compressed JPEG/MP4 to hide high-frequency artifacts.
*   **Resizing/Cropping:** To disrupt frequency-based detectors (grids, checkerboard artifacts).
*   **Noise Injection:** Adding Gaussian or film grain to simulate camera sensor noise.
*   **Metadata Stripping:** Removing EXIF/XMP data using tools like `exiftool`.
*   **Format Shifting:** Converting WebP -> PNG -> JPEG.

### C. Adversarial Attacks (Model Evasion)
*   **Gradient-based Attacks:** FGSM / PGD attacks if they can estimate the model gradient.
*   **Style Transfer:** Applying "real photo" style (texture) over synthetic content.
*   **In-painting:** Manually fixing hands/eyes to remove obvious visual cues.

## 3. Vulnerability Analysis & Mitigations

| Vulnerability | Attack Vector | Mitigation Strategy |
| :--- | :--- | :--- |
| **High-Freq Artifacts** | Blurring / Downsampling | Multi-scale analysis + Semantic inconsistency checks (Layer 2) |
| **Metadata Reliance** | Stripping Metadata | Do not rely on metadata for *detection*, only for *provenance* (Layer 5) |
| **Model Stagnation** | Training against frozen model | Periodic Model Rotation + Red Teaming (Layer 4) |
| **Unknown Generators** | New SOTA model released | OOD (Out-of-Distribution) Detector (Layer 3) |
| **Binary Classification** | "Just barely real" score | Probabilistic confidence intervals (Layer 6) |

## 4. Operational Security
*   **Model Weights:** Must be kept private.
*   **API Rate Limiting:** Prevent attackers from "mapping" the decision boundary via high-volume queries.
*   **Log Analysis:** Monitor for spikes in near-boundary decisions (indicating probing).

## 5. Critical Assumption
**"The Attacker will always eventually win the perception game."**
Therefore, the system must fallback to **Cryptographic Provenance** (Layer 5) when signal extraction becomes ambiguous.
