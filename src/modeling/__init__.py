"""
DeepFake Detection System - Modeling Package
Layer 3: Modeling (Ensemble Architecture)
"""

from .base import BaseDetector, ModelPrediction, EnsemblePrediction, FeatureNormalizer

from .statistical_detector import StatisticalBaselineDetector
from .neural_detector import NeuralNetworkDetector
from .ood_detector import OODDetector, GeneratorClassifier
from .ensemble import EnsembleAggregator, create_default_ensemble

__all__ = [
    # Base
    "BaseDetector",
    "ModelPrediction",
    "EnsemblePrediction",
    "FeatureNormalizer",
    # Detectors
    "StatisticalBaselineDetector",
    "NeuralNetworkDetector",
    # OOD
    "OODDetector",
    "GeneratorClassifier",
    # Ensemble
    "EnsembleAggregator",
    "create_default_ensemble",
]
