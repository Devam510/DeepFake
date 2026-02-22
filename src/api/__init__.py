"""
DeepFake Detection System - API Package
"""

from .detection_api import DetectionAPI, APIRequest, APIResponse, handle_analyze_request

__all__ = ["DetectionAPI", "APIRequest", "APIResponse", "handle_analyze_request"]
