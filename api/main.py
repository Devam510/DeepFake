"""
AI-Generated Image Detector - Production API
=============================================

⚠️ PRODUCTION FREEZE MODE ACTIVE
- Model weights: FROZEN (v1.0.0)
- Calibration: FROZEN (v1.0.0)
- Semantics: FROZEN (v1.0.0)
- Training: DISABLED

Single Responsibility:
"What is the estimated probability that this image is AI-generated,
with uncertainty and limitations clearly stated?"

Endpoints:
- POST /analyze/image - Analyze single image
- GET /health - Service health check
- GET /capabilities - What system can/cannot do
"""

import os
import sys
import io
import uuid
import time
import pickle
import hashlib
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add src to path for feature extraction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import processing level estimator
try:
    from extraction.image_processing_level import estimate_processing_level

    PROCESSING_LEVEL_AVAILABLE = True
except ImportError:
    PROCESSING_LEVEL_AVAILABLE = False

# ============================================================
# VERSION CONSTANTS (FROZEN)
# ============================================================

API_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"
SEMANTIC_VERSION = "1.0.0"
CALIBRATION_VERSION = "1.0.0"

# ============================================================
# CONFIGURATION (FROZEN)
# ============================================================

MAX_FILE_SIZE_MB = 10
MAX_RESOLUTION = 4096
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
REQUEST_TIMEOUT_SECONDS = 30

# ============================================================
# GLOBAL STATE (Loaded once at startup)
# ============================================================


class ModelState:
    """Singleton model state - loaded once at startup."""

    model = None
    scaler = None
    model_name = None
    loaded = False
    load_time = None
    startup_time = None


# ============================================================
# FROZEN SEMANTIC MAPPINGS (v1.0.0)
# ============================================================


def probability_to_interpretation(prob: float) -> str:
    """
    FROZEN: Map probability to interpretation.
    Version: 1.0.0 - DO NOT MODIFY
    """
    if prob < 0.2:
        return "LOW synthetic likelihood"
    elif prob < 0.4:
        return "MODERATE-LOW synthetic likelihood"
    elif prob < 0.6:
        return "INCONCLUSIVE - insufficient evidence"
    elif prob < 0.8:
        return "MODERATE-HIGH synthetic likelihood"
    else:
        return "HIGH synthetic likelihood"


def get_confidence_interval(prob: float, n_samples: int = 1000) -> tuple:
    """
    FROZEN: Compute confidence interval.
    Version: 1.0.0 - DO NOT MODIFY
    """
    # Wilson score interval approximation
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / n_samples
    center = (prob + z**2 / (2 * n_samples)) / denominator
    margin = (
        z
        * np.sqrt((prob * (1 - prob) + z**2 / (4 * n_samples)) / n_samples)
        / denominator
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return round(lower, 4), round(upper, 4)


# ============================================================
# FEATURE EXTRACTION
# ============================================================


def extract_features(image: Image.Image) -> Optional[np.ndarray]:
    """Extract features from PIL Image."""
    try:
        import tempfile
        from extraction.image_signals import ImageSignalExtractor

        # Save to temp file for extractor (Windows-compatible)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path, "JPEG")

        extractor = ImageSignalExtractor()
        signals = extractor.extract_all(temp_path)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        features = np.array(
            [
                signals.fft_magnitude_mean,
                signals.fft_magnitude_std,
                signals.fft_high_freq_ratio,
                signals.dct_coefficient_stats.get("dc_mean", 0),
                signals.dct_coefficient_stats.get("ac_mean", 0),
                signals.dct_coefficient_stats.get("ac_energy", 0),
                signals.entropy_mean,
                signals.entropy_std,
                signals.edge_gradient_mean,
                signals.edge_gradient_std,
                signals.edge_smoothness_score,
                signals.diffusion_residue_score,
            ]
        )

        return features
    except Exception as e:
        return None


# ============================================================
# RESPONSE MODELS
# ============================================================


class AnalysisResponse(BaseModel):
    """Response schema for /analyze/image endpoint."""

    synthetic_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: dict = Field(...)
    interpretation: str = Field(...)
    uncertainty_notice: str = Field(...)
    # NEW: Image processing level detection
    image_processing_level: str = Field(
        ...,
        description="Processing level: minimal_processing, moderate_processing, heavy_processing, or unknown",
    )
    processing_warning: Optional[str] = Field(
        None, description="Warning message when heavy processing is detected"
    )
    model_version: str = Field(...)
    semantic_version: str = Field(...)
    api_version: str = Field(...)
    request_id: str = Field(...)
    timestamp: str = Field(...)


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    service_status: str = Field(...)
    model_loaded: bool = Field(...)
    uptime_seconds: float = Field(...)
    model_version: str = Field(...)
    semantic_version: str = Field(...)
    api_version: str = Field(...)


class CapabilitiesResponse(BaseModel):
    """Response schema for /capabilities endpoint."""

    can_do: list = Field(...)
    cannot_do: list = Field(...)
    known_limitations: list = Field(...)
    intended_use: str = Field(...)
    model_version: str = Field(...)
    semantic_version: str = Field(...)
    api_version: str = Field(...)


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(...)
    message: str = Field(...)
    request_id: str = Field(...)


# ============================================================
# LIFESPAN: Model loads ONCE at startup
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup, cleanup on shutdown."""
    # Startup
    print("=" * 60)
    print("  PRODUCTION API STARTING")
    print("  ⚠️ PRODUCTION FREEZE MODE ACTIVE")
    print("=" * 60)

    ModelState.startup_time = time.time()

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "trained", "best_classifier.pkl"
    )

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        ModelState.model = data["model"]
        ModelState.scaler = data["scaler"]
        ModelState.model_name = data["name"]
        ModelState.loaded = True
        ModelState.load_time = datetime.now().isoformat()
        print(f"  ✅ Model loaded: {ModelState.model_name} v{MODEL_VERSION}")
    else:
        print(f"  ❌ Model not found: {model_path}")
        ModelState.loaded = False

    print("  ✅ API ready for inference")
    print("=" * 60)

    yield

    # Shutdown
    print("\n  Shutting down API...")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI-Generated Image Detector API",
    description="Production inference-only API for estimating synthetic image probability",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# ENDPOINT 1: POST /analyze/image
# ============================================================


@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single image for synthetic probability.

    Returns probability estimate with uncertainty and limitations.
    Does NOT return binary verdicts or claims of authenticity.
    """
    request_id = str(uuid.uuid4())

    # Guard: Model must be loaded
    if not ModelState.loaded:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error="SERVICE_UNAVAILABLE",
                message="Model not loaded. Service is not ready.",
                request_id=request_id,
            ).model_dump(),
        )

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="INVALID_CONTENT_TYPE",
                message=f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}",
                request_id=request_id,
            ).model_dump(),
        )

    # Read file
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="FILE_READ_ERROR",
                message="Could not read uploaded file",
                request_id=request_id,
            ).model_dump(),
        )

    # Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="FILE_TOO_LARGE",
                message=f"Maximum file size: {MAX_FILE_SIZE_MB} MB",
                request_id=request_id,
            ).model_dump(),
        )

    # Decode image
    try:
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="IMAGE_DECODE_ERROR",
                message="Could not decode image. Ensure valid JPEG/PNG/WebP.",
                request_id=request_id,
            ).model_dump(),
        )

    # Validate resolution
    if max(image.size) > MAX_RESOLUTION:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error="RESOLUTION_TOO_HIGH",
                message=f"Maximum resolution: {MAX_RESOLUTION}x{MAX_RESOLUTION}",
                request_id=request_id,
            ).model_dump(),
        )

    # Extract features
    features = extract_features(image)
    if features is None:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="FEATURE_EXTRACTION_ERROR",
                message="Could not extract features from image",
                request_id=request_id,
            ).model_dump(),
        )

    # Inference (thread-safe: sklearn models are read-only after training)
    try:
        features_scaled = ModelState.scaler.transform(features.reshape(1, -1))
        prob = float(ModelState.model.predict_proba(features_scaled)[0, 1])
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="INFERENCE_ERROR",
                message="Model inference failed",
                request_id=request_id,
            ).model_dump(),
        )

    # Compute response (FROZEN semantics)
    lower, upper = get_confidence_interval(prob)

    # NEW: Compute image processing level (for filtered/edited images)
    processing_level = "unknown"
    processing_warning = None
    if PROCESSING_LEVEL_AVAILABLE:
        try:
            import tempfile

            # Save image to temp file for processing level analysis
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                image.save(temp_path, "JPEG")

            processing_result = estimate_processing_level(temp_path)
            processing_level = processing_result.level
            processing_warning = processing_result.warning

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            processing_level = "unknown"
            processing_warning = None

    # Adjust interpretation for heavy processing
    interpretation = probability_to_interpretation(prob)
    if processing_level == "heavy_processing":
        interpretation = "INCONCLUSIVE - heavy post-processing detected"

    return AnalysisResponse(
        synthetic_probability=round(prob, 4),
        confidence_interval={
            "lower": lower,
            "upper": upper,
            "level": 0.95,
        },
        interpretation=interpretation,
        uncertainty_notice=(
            "This probability estimate reflects model uncertainty, not ground truth. "
            "Images labeled as 'real_unverified' in training data were not proven to be non-synthetic. "
            "High confidence does not guarantee correctness."
        ),
        image_processing_level=processing_level,
        processing_warning=processing_warning,
        model_version=MODEL_VERSION,
        semantic_version=SEMANTIC_VERSION,
        api_version=API_VERSION,
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
    )


# ============================================================
# ENDPOINT 2: GET /health
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check.

    No model inference performed.
    """
    uptime = time.time() - ModelState.startup_time if ModelState.startup_time else 0

    return HealthResponse(
        service_status="healthy" if ModelState.loaded else "degraded",
        model_loaded=ModelState.loaded,
        uptime_seconds=round(uptime, 2),
        model_version=MODEL_VERSION,
        semantic_version=SEMANTIC_VERSION,
        api_version=API_VERSION,
    )


# ============================================================
# ENDPOINT 3: GET /capabilities
# ============================================================


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    """
    Describe system capabilities and limitations.

    This endpoint protects users by clearly stating what the system can and cannot do.
    """
    return CapabilitiesResponse(
        can_do=[
            "Estimate probability that an image is AI-generated (synthetic)",
            "Provide confidence intervals for probability estimates",
            "Detect StyleGAN-generated face images with moderate accuracy",
            "Process JPEG, PNG, and WebP images up to 10MB",
        ],
        cannot_do=[
            "Prove an image is authentic or human-made",
            "Provide binary real/fake verdicts",
            "Guarantee 100% accuracy",
            "Detect all types of AI-generated images",
            "Analyze videos or audio",
            "Detect deepfakes in video format",
        ],
        known_limitations=[
            "Model trained on single StyleGAN dataset only",
            "REAL training images were unverified - authenticity unknown",
            "Hand-crafted features may miss generator-specific artifacts",
            "Calibration valid only within training distribution",
            "Performance varies by image resolution and compression",
            "Augmented images may produce unexpected results",
        ],
        intended_use=(
            "This API provides probability estimates for research and informational purposes. "
            "It should NOT be used as sole evidence for legal, journalistic, or high-stakes decisions. "
            "Always combine with human review and additional verification methods."
        ),
        model_version=MODEL_VERSION,
        semantic_version=SEMANTIC_VERSION,
        api_version=API_VERSION,
    )


# ============================================================
# ERROR HANDLERS
# ============================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler - never expose internals."""
    request_id = str(uuid.uuid4())
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again.",
            "request_id": request_id,
        },
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
